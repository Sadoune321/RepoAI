FROM python:3.10-bullseye

WORKDIR /app

# Pas besoin d'apt-get, Python 3.10-bullseye a déjà les libs système

COPY requirements.txt .

RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchvision==0.16.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]