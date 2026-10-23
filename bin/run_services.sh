#!/bin/bash
# Start the Prefect server in the background and the prediction API in the foreground.
set -e

prefect server start --host 0.0.0.0 --port 4201 &
exec uvicorn src.web_service.main:app --host 0.0.0.0 --port 8001
