FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# 🔥 Installer Torch CPU depuis PyTorch (stable)
RUN pip install --no-cache-dir --default-timeout=1000 \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Installer le reste
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]