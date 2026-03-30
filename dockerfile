FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

ENV API_BASE_URL=http://localhost:7860
ENV MODEL_NAME=gpt-4o-mini

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]