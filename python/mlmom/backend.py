"""Backend boundary. Training operations deliberately return HTTP 501."""

from fastapi import FastAPI

from mlmom.contracts import ApiError, Capabilities, EvaluateRequest, Health, History, ProfileRequest, TrainRequest
from mlmom.http import error_response, install_validation_handler

app = FastAPI(title="MLmom Backend", version="1.0.0")
install_validation_handler(app)
errors = {422: {"model": ApiError}, 501: {"model": ApiError}}


@app.get("/health", response_model=Health)
def health():
    return Health()


@app.get("/v1/capabilities", response_model=Capabilities)
def capabilities():
    return Capabilities()


@app.get("/v1/history", response_model=History)
def history():
    return History()


@app.post("/v1/profile", response_model=ApiError, status_code=501, responses=errors)
def profile(request: ProfileRequest):
    return error_response(501, "NOT_IMPLEMENTED", "Data profiling is not implemented")


@app.post("/v1/train", response_model=ApiError, status_code=501, responses=errors)
def train(request: TrainRequest):
    return error_response(501, "NOT_IMPLEMENTED", "Training is not implemented")


@app.post("/v1/evaluate", response_model=ApiError, status_code=501, responses=errors)
def evaluate(request: EvaluateRequest):
    return error_response(501, "NOT_IMPLEMENTED", "Evaluation is not implemented")
