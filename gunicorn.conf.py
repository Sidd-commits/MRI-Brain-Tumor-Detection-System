import os

# Render injects the PORT environment variable dynamically (defaults to 10000)
port = os.environ.get("PORT", "10000")
bind = f"0.0.0.0:{port}"

# Free-tier memory & concurrency optimization (512 MB RAM)
# Single synchronous worker eliminates thread contention with TensorFlow C++ runtime
workers = 1
threads = 1
timeout = 120
keepalive = 2
preload_app = False
accesslog = "-"
errorlog = "-"
loglevel = "info"
