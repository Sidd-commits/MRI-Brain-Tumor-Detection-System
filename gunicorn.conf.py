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

def post_fork(server, worker):
    """
    Hook executed in worker process after fork().
    Initializes TensorFlow cleanly inside the worker process to avoid
    inheriting deadlocked C++ mutexes from master process pre-fork state.
    """
    server.log.info("Worker %s spawned: initializing TensorFlow runtime...", worker.pid)
    try:
        import main
        main.init_model()
        server.log.info("Worker %s: Model initialization and warmup complete.", worker.pid)
    except Exception as e:
        server.log.error("Worker %s model initialization error: %s", worker.pid, e)
