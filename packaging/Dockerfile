# Окружение собирается по тому же абсолютному пути, по которому будет лежать
# на сервере, поэтому shebang'и в .venv/bin и pyvenv.cfg остаются валидными
# после распаковки.
#
# Ubuntu 24.04 повторяет glibc 2.39 целевого сервера. Образ берётся из зеркала
# AWS ECR, потому что Docker Hub с этой машины недоступен.
FROM public.ecr.aws/ubuntu/ubuntu:24.04

ARG PY_VERSION=3.10
ARG TORCH_VERSION=2.6.0
ARG TORCHVISION_VERSION=0.21.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124

ENV PREFIX=/workspace/musubi-tuner \
    UV_PYTHON_INSTALL_DIR=/workspace/musubi-tuner/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_LINK_MODE=copy \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates pigz \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR ${PREFIX}

# Интерпретатор кладём внутрь PREFIX, чтобы на сервере не требовался
# системный Python подходящей версии.
RUN uv python install ${PY_VERSION}

COPY pyproject.toml README.md ${PREFIX}/
COPY src ${PREFIX}/src

RUN uv venv --relocatable --python ${PY_VERSION} ${PREFIX}/.venv

ENV VIRTUAL_ENV=${PREFIX}/.venv

# torch ставится отдельно и с явными версиями: иначе резолвер может подтянуть
# сборку под другую CUDA, несовместимую с драйвером 535 на целевом сервере.
RUN uv pip install --index-url ${TORCH_INDEX} \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"

# Ставим и dev-группу целиком: она выглядит как инструменты разработки, но
# tensorboard в ней нужен в рантайме - без него accelerate молча выкидывает
# трекер, и логи обучения никуда не пишутся. Выбирать пакеты поштучно значит
# однажды снова угадать неверно.
RUN uv pip install -e . --group dev ipykernel

# opencv-python тянет из системы libGL и libglib, которых на сервере может
# не оказаться. GUI-функции OpenCV здесь не используются, headless-сборка
# делает архив независимым от системных библиотек.
RUN uv pip uninstall opencv-python && \
    uv pip install "opencv-python-headless==4.10.0.84"

COPY . ${PREFIX}

RUN uv cache clean
