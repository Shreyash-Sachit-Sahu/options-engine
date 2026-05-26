# ── Stage 1: Build C++ pricing core ──────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y \
    cmake g++ ninja-build git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir pybind11 && \
    pip install --no-cache-dir -r requirements.txt

COPY CMakeLists.txt .
COPY src/ src/

RUN pip install pybind11[global] && \
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
          -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())") && \
    cmake --build build --config Release

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=builder /build/build/ build/
COPY . .

# Copy compiled .so from build stage
RUN find build/ -name "pricer*.so" -exec cp {} . \; || true

ENV PYTHONPATH=/app
ENV MODEL_PATH=agent/models/best_heston/best_model
ENV VNORM_PATH=agent/models/vec_normalize_heston.pkl
ENV API_URL=http://localhost:8000
ENV DEVICE=cpu

EXPOSE 8000 8501