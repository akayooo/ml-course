FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY Примеры/ /app/Примеры/
CMD ["uvicorn", "m15_api:app", "--app-dir", "Примеры", "--host", "0.0.0.0", "--port", "8000"]
