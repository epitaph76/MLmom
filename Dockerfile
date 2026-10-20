FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml requirements.txt solution.py ./
COPY mlmom ./mlmom
COPY skills ./skills
COPY profiles ./profiles
COPY tests ./tests
# Dependencies are installed at image build time, never while solving a task.
RUN pip install --no-cache-dir .
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["python", "solution.py"]
