FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn[standard] httpx[socks] python-dotenv

COPY freebuff2api/ freebuff2api/
COPY main.py .

EXPOSE 8000

CMD ["python", "main.py"]
