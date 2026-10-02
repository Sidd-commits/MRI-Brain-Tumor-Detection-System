import os

# Render injects the PORT environment variable dynamically (defaults to 10000)
port = os.environ.get("PORT", "10000")
bind = f"0.0.0.0:{port}"

# Free-tier memory constraints (512 MB RAM)
workers = 1
threads = 2
timeout = 120
keepalive = 5
preload_app = False
accesslog = "-"
errorlog = "-"
loglevel = "info"
