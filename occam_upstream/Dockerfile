FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY solver/ solver/
COPY viewer/ viewer/

RUN pip install --no-cache-dir .

EXPOSE 8080

CMD ["occam", "run", "--host", "0.0.0.0", "--port", "8080"]
