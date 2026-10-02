import os
import uuid
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='keras')

# Container optimization
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from tensorflow.keras.models import load_model

# Application setup
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
SAMPLE_FOLDER = os.path.join(BASE_DIR, 'Sample MRI images')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SAMPLE_FOLDER'] = SAMPLE_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Build unified Functional model architecture
def build_brain_tumor_detector():
    """
    Constructs the VGG-16 MRI classification network with explicit input tensor
    and unified Functional topology. This avoids legacy Keras 2/3 nested sequential
    deserialization issues and guarantees 100% cross-platform compatibility.
    """
    inputs = tf.keras.Input(shape=(128, 128, 3))
    vgg_base = tf.keras.applications.VGG16(include_top=False, weights=None, input_tensor=inputs)
    x = tf.keras.layers.Flatten(name='flatten')(vgg_base.output)
    x = tf.keras.layers.Dropout(0.5, name='dropout')(x)
    x = tf.keras.layers.Dense(128, activation='relu', name='dense')(x)
    x = tf.keras.layers.Dropout(0.5, name='dropout_1')(x)
    outputs = tf.keras.layers.Dense(4, activation='softmax', name='dense_1')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='mri_vgg16_detector')

# Model, weights, and Grad-CAM runtime state
WEIGHTS_PATH = os.path.join(BASE_DIR, 'models', 'detector_weights.weights.h5')
LEGACY_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.h5')

# Class labels mapped directly to model output indices:
# Index 0: Glioma | Index 1: No Tumor | Index 2: Pituitary | Index 3: Meningioma
CLASS_LABELS = ['glioma', 'notumor', 'pituitary', 'meningioma']
DISPLAY_NAMES = {
    'glioma': 'Glioma',
    'notumor': 'No Tumor',
    'pituitary': 'Pituitary',
    'meningioma': 'Meningioma'
}

# Pre-defined reference scans for validated diagnostic testing
SAMPLE_SCANS = [
    {'name': 'Glioma Case', 'type': 'glioma', 'filename': 'Te-gl_0015.jpg', 'tag': 'Astrocytoma / GBM'},
    {'name': 'Meningioma Case', 'type': 'meningioma', 'filename': 'Te-meTr_0001.jpg', 'tag': 'Dural Base Mass'},
    {'name': 'Pituitary Case', 'type': 'pituitary', 'filename': 'Te-piTr_0003.jpg', 'tag': 'Sellar Region'},
    {'name': 'Healthy Control', 'type': 'notumor', 'filename': 'Te-noTr_0004.jpg', 'tag': 'No Tumor Detected'}
]

_MODEL_INITIALIZED = False
model = None
grad_model = None
GRADCAM_SUPPORTED = False
compute_gradcam_tensor = None
JET_LUT = None

def init_model():
    """
    Initializes TensorFlow model, Grad-CAM graph, and warm-up compilation inside
    the worker process. This avoids Gunicorn pre-fork deadlock on Linux where
    TensorFlow C++ threadpool mutexes get permanently locked across fork().
    """
    global _MODEL_INITIALIZED, model, grad_model, GRADCAM_SUPPORTED, compute_gradcam_tensor, JET_LUT
    if _MODEL_INITIALIZED and model is not None:
        return True

    print("[NeuroScan] Initializing TensorFlow model inside worker process...")
    model = build_brain_tumor_detector()
    if os.path.exists(WEIGHTS_PATH):
        model.load_weights(WEIGHTS_PATH)
    elif os.path.exists(LEGACY_MODEL_PATH):
        model.load_weights(LEGACY_MODEL_PATH)

    try:
        last_conv_layer = model.get_layer('block5_conv3')
        grad_model = tf.keras.Model(inputs=model.inputs, outputs=[last_conv_layer.output, model.output])
        GRADCAM_SUPPORTED = True

        @tf.function(reduce_retracing=True)
        def _compute_gc(img_tensor):
            with tf.GradientTape() as tape:
                conv_outputs, preds = grad_model(img_tensor, training=False)
                top_idx = tf.argmax(preds[0])
                top_val = preds[:, top_idx]

            grads = tape.gradient(top_val, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0.0)
            return heatmap / (tf.math.reduce_max(heatmap) + 1e-10)

        compute_gradcam_tensor = _compute_gc
    except Exception as e:
        print(f"Warning: Grad-CAM models could not be initialized: {e}")
        GRADCAM_SUPPORTED = False

    # Precompute static Jet colormap lookup table (256x3 uint8)
    JET_LUT = (matplotlib.colormaps['jet'](np.arange(256))[:, :3] * 255).astype(np.uint8)

    # Worker warmup pass
    try:
        _warmup_input = tf.zeros((1, 128, 128, 3), dtype=tf.float32)
        _ = model(_warmup_input, training=False)
        if GRADCAM_SUPPORTED and compute_gradcam_tensor is not None:
            _ = compute_gradcam_tensor(_warmup_input)
        print("[NeuroScan] TensorFlow model and Grad-CAM compilation warm.")
    except Exception as _w_err:
        print(f"Warmup notice: {_w_err}")

    _MODEL_INITIALIZED = True
    return True


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def verify_is_brain_mri(image_path):
    """
    Clinical Domain Validation Gate for Brain MRI Scans.
    Rejects:
      - Color images (photos of people, objects, animals, landscapes, UI screenshots)
      - Document scans, charts, and text pages
      - Computer screenshots (code, browser windows, desktop)
      - Blank, solid, or uniform noise images
      - Non-cranial objects
    Returns: (is_valid: bool, rejection_info: dict or None)
    
    Rejection info contains plain, user-friendly language designed for intuitive user experience:
      - 'detected_type': Simple category label (e.g. 'Color Photo or Screenshot')
      - 'simple_reason': Clear 1-sentence explanation without technical jargon
      - 'tip': Helpful action recommendation
    """
    try:
        with Image.open(image_path) as raw_img:
            orig_w, orig_h = raw_img.size
            if orig_w < 64 or orig_h < 64:
                return False, {
                    'detected_type': 'Low-Resolution Image',
                    'simple_reason': 'The image is too small to identify brain anatomical structures.',
                    'tip': 'Please upload a larger brain scan (recommended 128x128 or higher).'
                }
            # Resize image down to max 256x256 before converting to numpy
            # This protects against high-resolution memory spikes and OOM crashes
            raw_img.thumbnail((256, 256), Image.Resampling.LANCZOS)
            img = raw_img.convert('RGB')
    except Exception:
        return False, {
            'detected_type': 'Unreadable File',
            'simple_reason': 'This file could not be opened as a readable image.',
            'tip': 'Please ensure your file is a valid image (JPG, PNG, or WEBP).'
        }

    arr = np.array(img, dtype=np.float32)
    h, w, _ = arr.shape

    # 1. Color / Chromatic Divergence Check
    # Authentic clinical brain MRI scans are acquired as monochromatic grayscale modalities.
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    chroma_diff = np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(r - b))
    
    # Allow small tolerance (< 16.0) for minor JPEG compression / browser export tint
    if chroma_diff > 16.0:
        return False, {
            'detected_type': 'Color Photo or Screenshot',
            'simple_reason': 'This image contains color. Genuine brain MRI scans are black-and-white (grayscale).',
            'tip': 'Please upload an authentic black-and-white brain MRI scan or choose a sample scan below.'
        }

    # Convert to grayscale luminance
    gray = np.mean(arr, axis=2)

    # 2. Ambient Air / Perimeter Background Check
    # Axial brain MRIs are centered inside scanner bore; outer edges are ambient dark air (< 45).
    # Take a 5% margin around the frame perimeter.
    m_h, m_w = max(2, int(h * 0.05)), max(2, int(w * 0.05))
    top_edge = gray[:m_h, :]
    bottom_edge = gray[-m_h:, :]
    left_edge = gray[:, :m_w]
    right_edge = gray[:, -m_w:]
    border_pixels = np.concatenate([top_edge.flatten(), bottom_edge.flatten(), left_edge.flatten(), right_edge.flatten()])
    
    dark_border_ratio = np.mean(border_pixels < 45.0)
    # Documents, light screenshots, camera photos have bright borders (< 25% dark)
    if dark_border_ratio < 0.25:
        return False, {
            'detected_type': 'Document or Webpage Screenshot',
            'simple_reason': 'The image background is bright. Genuine MRI scans always have a dark black background surrounding the brain.',
            'tip': 'Please upload a scan that has the natural dark scanner background.'
        }

    # 3. Central Tissue Contrast & Signal Presence
    # Center 50% must contain actual tissue with realistic soft-tissue contrast
    center = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    center_mean = float(np.mean(center))
    center_std = float(np.std(center))

    if center_mean < 15.0 or center_std < 10.0:
        return False, {
            'detected_type': 'Blank or Flat Image',
            'simple_reason': 'The image is nearly completely black or lacks visible anatomical details.',
            'tip': 'Please provide an MRI scan with clear cross-sectional brain tissue.'
        }
    if center_mean > 230.0:
        return False, {
            'detected_type': 'Overexposed White Image',
            'simple_reason': 'The image is solid white or washed out.',
            'tip': 'Please upload a standard MRI scan with balanced contrast.'
        }

    # 4. Cranial Tissue Coverage & Structural Topology
    tissue_mask = gray > 20.0
    tissue_ratio = float(np.mean(tissue_mask))
    
    if tissue_ratio < 0.08:
        return False, {
            'detected_type': 'Non-Anatomical Image',
            'simple_reason': 'No recognizable brain tissue was detected in this image.',
            'tip': 'Please upload an axial head or brain scan.'
        }

    # 5. Centroid Centering (Anatomical alignment)
    y_idx, x_idx = np.where(tissue_mask)
    if len(y_idx) > 0:
        cy, cx = float(np.mean(y_idx)), float(np.mean(x_idx))
        offset_y = abs(cy - h / 2) / h
        offset_x = abs(cx - w / 2) / w
        if offset_x > 0.32 or offset_y > 0.32:
            return False, {
                'detected_type': 'Cropped or Off-Center Image',
                'simple_reason': 'The subject is cut off or located far from the center of the frame.',
                'tip': 'Please center the brain MRI scan in the image frame.'
            }

    # 6. Gradient / Edge Directional Uniformity (Screenshots vs Natural Brain Anatomy)
    gx = np.abs(gray[:, 1:] - gray[:, :-1])
    gy = np.abs(gray[1:, :] - gray[:-1, :])
    min_h, min_w = min(gx.shape[0], gy.shape[0]), min(gx.shape[1], gy.shape[1])
    gx, gy = gx[:min_h, :min_w], gy[:min_h, :min_w]
    
    sum_gx, sum_gy = np.sum(gx), np.sum(gy)
    if sum_gx > 0 and sum_gy > 0:
        axis_ratio = max(sum_gx, sum_gy) / min(sum_gx, sum_gy)
        if axis_ratio > 3.0:
            return False, {
                'detected_type': 'Code or UI Screenshot',
                'simple_reason': 'Straight lines and text rows typical of a computer screen or code editor were detected.',
                'tip': 'Please upload an MRI scan instead of a screenshot of your screen.'
            }

    return True, None


def generate_gradcam_overlay(image_path, output_path):
    """Generates a Grad-CAM heatmap overlaid on the MRI scan with strict memory bounds."""
    if not _MODEL_INITIALIZED or model is None:
        init_model()

    if not GRADCAM_SUPPORTED or compute_gradcam_tensor is None:
        return False

    try:
        with Image.open(image_path) as raw_img:
            img = raw_img.convert('RGB')
            # Cap maximum dimension to 512px to eliminate memory spikes
            if img.width > 512 or img.height > 512:
                img.thumbnail((512, 512), Image.Resampling.BILINEAR)
            base_img = img.copy()
            model_img = img.resize((128, 128))

        img_array = np.expand_dims(np.array(model_img, dtype=np.float32) / 255.0, axis=0)
        img_tensor = tf.convert_to_tensor(img_array)

        heatmap = compute_gradcam_tensor(img_tensor).numpy()

        # Apply precomputed Jet colormap lookup table
        jet_heatmap = JET_LUT[(heatmap * 255).astype(np.uint8)]
        jet_heatmap_img = Image.fromarray(jet_heatmap).resize(
            (base_img.width, base_img.height), Image.Resampling.BILINEAR
        )

        superimposed = Image.blend(base_img, jet_heatmap_img, alpha=0.45)
        superimposed.save(output_path, quality=90)
        return True
    except Exception as err:
        print(f"Error computing Grad-CAM: {err}")
        return False


def predict_tumor(image_path):
    """
    Loads, preprocesses, and predicts class for an MRI image.
    Returns: (result_text, confidence_float, tumor_type_str, probabilities_dict, gradcam_filename)
    """
    if not _MODEL_INITIALIZED or model is None:
        init_model()

    IMAGE_SIZE = 128
    with Image.open(image_path) as raw_img:
        img = raw_img.convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    raw_preds = model(img_array, training=False).numpy()[0]
    predicted_class_index = int(np.argmax(raw_preds))
    confidence_score = float(raw_preds[predicted_class_index])
    tumor_type = CLASS_LABELS[predicted_class_index]

    # Calculate full 4-class percentage distribution
    probabilities = {
        'Glioma': round(float(raw_preds[0]) * 100, 2),
        'No Tumor': round(float(raw_preds[1]) * 100, 2),
        'Pituitary': round(float(raw_preds[2]) * 100, 2),
        'Meningioma': round(float(raw_preds[3]) * 100, 2),
    }

    # Generate Grad-CAM visualization
    base_name = os.path.basename(image_path)
    gradcam_filename = f"gradcam_{os.path.splitext(base_name)[0]}.jpg"
    gradcam_full_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
    gradcam_success = generate_gradcam_overlay(image_path, gradcam_full_path)
    if not gradcam_success:
        gradcam_filename = None

    if tumor_type == 'notumor':
        return "No Tumor", confidence_score, None, probabilities, gradcam_filename
    else:
        return f"Tumor: {tumor_type}", confidence_score, tumor_type, probabilities, gradcam_filename


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if user clicked a quick-test sample
        sample_choice = request.form.get('sample_choice')
        if sample_choice:
            sample_file_path = os.path.join(app.config['SAMPLE_FOLDER'], sample_choice)
            if os.path.exists(sample_file_path):
                # Copy/save to uploads so it is accessible via standard upload URL
                unique_name = f"sample_{uuid.uuid4().hex[:6]}_{secure_filename(sample_choice)}"
                dest_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
                with open(sample_file_path, 'rb') as src, open(dest_path, 'wb') as dst:
                    dst.write(src.read())

                try:
                    result, confidence, tumor_type, probabilities, gradcam_file = predict_tumor(dest_path)
                    gradcam_path = f"/uploads/{gradcam_file}" if gradcam_file else None
                    return render_template(
                        'index.html',
                        result=result,
                        confidence=f"{confidence * 100:.2f}",
                        file_path=f"/uploads/{unique_name}",
                        tumor_type=tumor_type,
                        probabilities=probabilities,
                        gradcam_path=gradcam_path,
                        sample_scans=SAMPLE_SCANS,
                        active_filename=sample_choice
                    )
                except Exception as ex:
                    return render_template(
                        'index.html',
                        result=None,
                        error_message=f"Analysis failed: {str(ex)}",
                        sample_scans=SAMPLE_SCANS
                    )

        # Standard file upload
        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template(
                'index.html',
                result=None,
                error_message="Please select or drop an MRI scan image before analyzing.",
                sample_scans=SAMPLE_SCANS
            )

        if not allowed_file(file.filename):
            return render_template(
                'index.html',
                result=None,
                error_message="Unsupported file format. Please upload an image file (JPG, PNG, JPEG, WEBP).",
                sample_scans=SAMPLE_SCANS
            )

        try:
            # Secure and deduplicate filename
            clean_name = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex[:6]}_{clean_name}"
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_location)

            # Validate that the file is indeed a valid readable image
            try:
                with Image.open(file_location) as test_img:
                    test_img.verify()
            except Exception:
                if os.path.exists(file_location):
                    os.remove(file_location)
                return render_template(
                    'index.html',
                    result=None,
                    error_message="The uploaded file is not a valid image scan. Please provide a genuine MRI image.",
                    sample_scans=SAMPLE_SCANS
                )

            # Reopen and bound maximum resolution to 1024x1024 to eliminate memory spikes
            with Image.open(file_location) as raw_upload:
                if raw_upload.width > 1024 or raw_upload.height > 1024:
                    raw_upload.thumbnail((1024, 1024), Image.Resampling.BILINEAR)
                    raw_upload.convert('RGB').save(file_location, quality=92)

            # Clinical Domain Validation Gate: Verify image is an authentic Brain MRI scan
            is_valid_mri, rejection_info = verify_is_brain_mri(file_location)
            if not is_valid_mri:
                return render_template(
                    'index.html',
                    result=None,
                    is_non_mri=True,
                    rejection_info=rejection_info,
                    rejection_reason=rejection_info.get('simple_reason', 'Non-Brain MRI detected'),
                    file_path=f"/uploads/{unique_filename}",
                    sample_scans=SAMPLE_SCANS,
                    active_filename=clean_name
                )

            # Perform prediction
            result, confidence, tumor_type, probabilities, gradcam_file = predict_tumor(file_location)
            gradcam_path = f"/uploads/{gradcam_file}" if gradcam_file else None

            return render_template(
                'index.html',
                result=result,
                confidence=f"{confidence * 100:.2f}",
                file_path=f"/uploads/{unique_filename}",
                tumor_type=tumor_type,
                probabilities=probabilities,
                gradcam_path=gradcam_path,
                sample_scans=SAMPLE_SCANS,
                active_filename=clean_name
            )

        except Exception as ex:
            return render_template(
                'index.html',
                result=None,
                error_message=f"Error analyzing image: {str(ex)}",
                sample_scans=SAMPLE_SCANS
            )

    return render_template('index.html', result=None, sample_scans=SAMPLE_SCANS)


@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/samples/<filename>')
def get_sample_file(filename):
    return send_from_directory(app.config['SAMPLE_FOLDER'], filename)


if __name__ == '__main__':
    init_model()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
