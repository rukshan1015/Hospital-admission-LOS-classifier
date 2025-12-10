FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /home/app

# pip use PyPI only (avoiding any CUDA extra indexes)
ENV PIP_INDEX_URL=https://pypi.org/simple \
    PIP_EXTRA_INDEX_URL= \
    PIP_FIND_LINKS=

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential gcc libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy project

COPY ml  ./ml
COPY local_ui  ./local_ui
COPY data ./data
COPY models ./models

# Running single-inference UI by default
EXPOSE 7860
CMD ["python", "local_ui/single_inference.py"]


