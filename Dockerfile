FROM nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    DEBIAN_FRONTEND=noninteractive \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    tree \
    vim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV color_prompt=yes
WORKDIR /root
