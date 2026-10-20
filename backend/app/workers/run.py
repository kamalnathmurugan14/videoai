"""Standalone worker: `python -m app.workers.run` (set WORKER_ENABLED=false on the API process)."""
import logging

from .worker import WorkerThread

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    t = WorkerThread()
    t.start()
    t.join()
