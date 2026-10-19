"""Shared error envelope, including validation errors."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from mlmom.contracts import ApiError


def error_response(status: int, code: str, message: str, request_id=None, *, mock=False, retryable=False):
    error = ApiError(code=code, message=message, request_id=request_id, mock=mock, retryable=retryable)
    return JSONResponse(status_code=status, content=error.model_dump(mode="json"))


def install_validation_handler(app: FastAPI, *, mock=False):
    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        body = exc.body
        request_id = body.get("request_id") if isinstance(body, dict) else None
        if not isinstance(request_id, str):
            request_id = None
        return error_response(422, "VALIDATION_ERROR", "Request does not match the API schema", request_id, mock=mock)
