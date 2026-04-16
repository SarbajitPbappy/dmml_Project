FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY retrox /app/retrox
COPY artifacts /app/artifacts

ENV RETROX_ARTIFACTS_DIR=/app/artifacts

EXPOSE 8000
CMD ["uvicorn", "retrox.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

