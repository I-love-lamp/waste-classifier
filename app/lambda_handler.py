"""AWS Lambda entrypoint: wraps the FastAPI app with Mangum's ASGI adapter.

`lifespan="auto"` runs FastAPI's startup (which loads the model) once per Lambda
execution environment, on cold start. Warm invocations reuse the loaded model.
"""
from mangum import Mangum

from app.main import app

handler = Mangum(app, lifespan="auto")
