import os
import uuid
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

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

# Load the trained VGG16 model
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.h5')
model = load_model(MODEL_PATH)

# Class labels mapped directly to model output indices:
# Index 0: Glioma | Index 1: No Tumor | Index 2: Pituitary | Index 3: Meningioma
CLASS_LABELS = ['glioma', 'notumor', 'pituitary', 'meningioma']
DISPLAY_NAMES = {
    'glioma': 'Glioma',
    'notumor': 'No Tumor',
    'pituitary': 'Pituitary',
    'meningioma': 'Meningioma'
}

# Pre-defined sample scans for quick live panel demonstration
SAMPLE_SCANS = [
    {'name': 'Glioma Scan', 'type': 'glioma', 'filename': 'Te-gl_0015.jpg', 'icon': '🧠'},
    {'name': 'Meningioma Scan', 'type': 'meningioma', 'filename': 'Te-meTr_0001.jpg', 'icon': '🔬'},
    {'name': 'Pituitary Scan', 'type': 'pituitary', 'filename': 'Te-piTr_0003.jpg', 'icon': '⚡'},
    {'name': 'Healthy (No Tumor)', 'type': 'notumor', 'filename': 'Te-noTr_0004.jpg', 'icon': '🛡️'}
]

# Build Grad-CAM computation sub-models for convolutional explainability
try:
    vgg_base = model.layers[0]
    last_conv_layer = vgg_base.get_layer('block5_conv3')
    last_conv_model = tf.keras.Model(inputs=vgg_base.inputs, outputs=last_conv_layer.output)

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    x = vgg_base.get_layer('block5_pool')(x)
    for layer in model.layers[1:]:
        x = layer(x)
    classifier_model = tf.keras.Model(classifier_input, x)
    GRADCAM_SUPPORTED = True
except Exception as e:
    print(f"Warning: Grad-CAM models could not be initialized: {e}")
    GRADCAM_SUPPORTED = False


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_gradcam_overlay(image_path, output_path):
    """Generates a Grad-CAM heatmap overlaid on the MRI scan."""
    if not GRADCAM_SUPPORTED:
        return False

    try:
        img = load_img(image_path, target_size=(128, 128))
        img_array = np.expand_dims(img_to_array(img) / 255.0, axis=0)

        with tf.GradientTape() as tape:
            conv_outputs = last_conv_model(img_array)
            tape.watch(conv_outputs)
            preds = classifier_model(conv_outputs)
            top_idx = tf.argmax(preds[0])
            top_val = preds[:, top_idx]

        grads = tape.gradient(top_val, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        heatmap = heatmap.numpy()

        # Resize heatmap to original image dimensions
        orig_img = Image.open(image_path).convert('RGB')
        orig_w, orig_h = orig_img.size

        cmap = matplotlib.colormaps['jet']
        jet_colors = cmap(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[(heatmap * 255).astype(np.uint8)]
        jet_heatmap_img = Image.fromarray((jet_heatmap * 255).astype(np.uint8)).resize(
            (orig_w, orig_h), Image.Resampling.BILINEAR
        )

        superimposed = Image.blend(orig_img, jet_heatmap_img, alpha=0.45)
        superimposed.save(output_path, quality=95)
        return True
    except Exception as err:
        print(f"Error computing Grad-CAM: {err}")
        return False


def predict_tumor(image_path):
    """
    Loads, preprocesses, and predicts class for an MRI image.
    Returns: (result_text, confidence_float, tumor_type_str, probabilities_dict, gradcam_filename)
    """
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    raw_preds = model.predict(img_array, verbose=0)[0]
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
                os.remove(file_location)
                return render_template(
                    'index.html',
                    result=None,
                    error_message="The uploaded file is not a valid image scan. Please provide a genuine MRI image.",
                    sample_scans=SAMPLE_SCANS
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


if __name__ == '__main__':
    app.run(debug=True)
