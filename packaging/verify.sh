#!/usr/bin/env bash
# Проверяет собранный архив: распаковывает его в чистом контейнере и запускает
# интерпретатор из архива. Контейнер без GPU, поэтому cuda_available здесь
# всегда False - это ожидаемо, реальная проверка CUDA только на сервере.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PACK_NAME="${PACK_NAME:-musubi-tuner-cu124}"
PACK_OUT_DIR="${PACK_OUT_DIR:-$REPO_ROOT/../out_build}"
PACK_OUT_DIR="$(cd "$PACK_OUT_DIR" && pwd)"

docker run --rm -i \
    -v "$PACK_OUT_DIR:/in:ro" \
    -e PACK_NAME="$PACK_NAME" \
    public.ecr.aws/ubuntu/ubuntu:24.04 bash -s <<'EOF'
set -euo pipefail

cd /in
sha256sum -c "${PACK_NAME}.sha256"
echo "CHECKSUM_OK"

mkdir -p /workspace
tar -xzf "/in/${PACK_NAME}.tar.gz" -C /workspace
echo "EXTRACT_OK"

PY=/workspace/musubi-tuner/.venv/bin/python
"$PY" - <<'PY'
import sys
import cv2
import tensorboard
import torch
import transformers
import musubi_tuner
from torch.utils.tensorboard import SummaryWriter

print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("opencv", cv2.__version__)
print("tensorboard", tensorboard.__version__)
print("cuda_available", torch.cuda.is_available())
PY

"$PY" /workspace/musubi-tuner/src/musubi_tuner/qwen_image_train.py --help >/dev/null
echo "ENTRYPOINT_OK"
EOF
