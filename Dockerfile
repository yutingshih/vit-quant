FROM nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04

# Leverage BuildKit cache mounts to accelerate APT fetch operations
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    DEBIAN_FRONTEND=noninteractive \
    apt-get update && apt-get install -y --no-install-recommends \
    clang-format \
    clang-tidy \
    cmake \
    curl \
    git \
    libdw1 \
    ninja-build \
    python3.12-dev \
    tree \
    vim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Install Gemini CLI
COPY --from=node:24-slim /usr/local/ /usr/local/
RUN npm install -g @google/gemini-cli

ENV color_prompt=yes
WORKDIR /root
