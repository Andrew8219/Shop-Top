FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Ядро приложения (без PyTorch). Для семантического поиска соберите образ
# с аргументом --build-arg WITH_ML=1 (образ станет заметно больше).
ARG WITH_ML=0
COPY requirements.txt requirements-ml.txt ./
RUN pip install -r requirements.txt \
    && if [ "$WITH_ML" = "1" ]; then \
         pip install --index-url https://download.pytorch.org/whl/cpu torch \
         && pip install -r requirements-ml.txt; \
       fi

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
