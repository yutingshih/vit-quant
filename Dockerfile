FROM nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    vim \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="~/.local/bin:$PATH"

ENV color_prompt=yes
WORKDIR /root
