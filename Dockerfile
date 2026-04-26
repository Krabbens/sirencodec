FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    git-lfs \
    openssh-client \
    libsndfile1 \
    tar \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

ENV SIRENCODEC_WORKDIR=/workspace \
    SIRENCODEC_REPO_URL=https://github.com/Krabbens/sirencodec.git

WORKDIR /workspace

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY tools ./tools
COPY tests ./tests

RUN uv sync --frozen --python python3

COPY . .

RUN install -m 0755 scripts/docker_git_sync.sh /usr/local/bin/sirencodec-sync \
    && install -m 0755 scripts/download_train_clean_360.sh /usr/local/bin/download-train-clean-360 \
    && git lfs install --system --skip-repo \
    && git config --global --add safe.directory /workspace \
    && git config --global pull.ff only

CMD ["uv", "run", "train", "--help"]
