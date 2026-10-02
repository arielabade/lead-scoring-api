# Multi-stage: the build stage compiles wheels, the runtime stage ships without
# a compiler. Keeps the image smaller and the attack surface narrower.
FROM python:3.11-slim AS build

WORKDIR /build
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-api.txt


FROM python:3.11-slim AS runtime

# libgomp is LightGBM's OpenMP runtime: needed at inference, not just build.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Run as a non-root user.
RUN useradd --create-home --uid 10001 scoring
WORKDIR /srv

COPY --from=build /install /usr/local
COPY src/ src/
COPY app/ app/
COPY models/ models/

USER scoring
ENV PYTHONPATH=/srv/src PYTHONUNBUFFERED=1
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
