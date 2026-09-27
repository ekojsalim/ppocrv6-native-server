# CUDA/cuDNN development tools only; no inference framework in any stage.
ARG BUILD_IMAGE=docker.io/nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04
FROM ${BUILD_IMAGE} AS builder
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/cargo/bin:/root/.cargo/bin:${PATH}
RUN if ! command -v cargo >/dev/null || ! test -f /usr/include/polyclipping/clipper.hpp; then \
      apt-get update && apt-get install -y --no-install-recommends build-essential cmake curl ca-certificates libopencv-core-dev libopencv-imgproc-dev libpolyclipping-dev pkg-config && \
      rm -rf /var/lib/apt/lists/*; \
    fi; \
    if ! command -v cargo >/dev/null; then curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.96.0; fi
WORKDIR /workspace
COPY native native
COPY server/Cargo.toml server/Cargo.lock server/
COPY server/src server/src
ARG CUDA_ARCH=120
RUN cmake -S native -B /compiled-build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} && \
    cmake --build /compiled-build -j4 --target ppocrv6_native && \
    cargo build --release --locked --manifest-path server/Cargo.toml
# Copy the OpenCV/Clipper dependency closure, retaining the final base's libc
# and C++ runtime.
RUN mkdir /postprocess-libs && \
    ldd /compiled-build/libppocrv6_native.so | awk '/=> \// {print $3}' | \
    while read -r lib; do \
      case "$(basename "$lib")" in libc.so*|libm.so*|libstdc++*|libgcc_s*|libpthread*|libdl.so*|librt.so*|libcudart*) ;; \
      *) cp -L "$lib" /postprocess-libs/ ;; esac; \
    done

FROM ${BUILD_IMAGE} AS math
RUN mkdir /math-libs && cp -L /usr/lib/x86_64-linux-gnu/libcudnn*.so.9 /math-libs/ && \
    cp -L /usr/local/cuda/targets/x86_64-linux/lib/libcublas.so.13 \
          /usr/local/cuda/targets/x86_64-linux/lib/libcublasLt.so.13 \
          /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.13 \
          /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so.13 \
          /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc-builtins.so.* \
          /usr/local/cuda/targets/x86_64-linux/lib/libnvJitLink.so.13 /math-libs/
FROM docker.io/library/ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 zlib1g && rm -rf /var/lib/apt/lists/*
COPY --from=math /math-libs/ /opt/native-libs/
COPY --from=builder /postprocess-libs/ /opt/native-libs/
COPY --from=builder /compiled-build/libppocrv6_native.so /opt/ppocrv6/lib/libppocrv6_native.so
COPY --from=builder /workspace/server/target/release/ppocrv6-native-server /opt/ppocrv6/bin/
ENV LD_LIBRARY_PATH=/opt/native-libs:/usr/lib64
# Model operators use a leased 64 MiB workspace. Keep cuBLAS's otherwise
# unused per-handle default pool small (8 x 16 KiB).
ENV CUBLAS_WORKSPACE_CONFIG=:16:8
EXPOSE 8184
ENTRYPOINT ["/opt/ppocrv6/bin/ppocrv6-native-server"]
CMD ["--model-dir", "/models", "--listen", "0.0.0.0:8184"]
