#!/usr/bin/env bash
# Собирает переносимое окружение musubi-tuner и выкладывает архив в out_build.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Префикс PACK_ обязателен: NAME и IMAGE уже заняты в окружении WSL.
PACK_IMAGE="${PACK_IMAGE:-musubi-tuner-pack}"
PACK_NAME="${PACK_NAME:-musubi-tuner-cu124}"
PACK_OUT_DIR="${PACK_OUT_DIR:-$REPO_ROOT/../out_build}"

mkdir -p "$PACK_OUT_DIR"
PACK_OUT_DIR="$(cd "$PACK_OUT_DIR" && pwd)"

echo ">>> build image $PACK_IMAGE"
docker build -f packaging/Dockerfile -t "$PACK_IMAGE" .

# Архив создаётся внутри контейнера: так сохраняются симлинки и права,
# которые файловая система Windows не умеет представлять.
echo ">>> create archive inside container"
cid="$(docker run -d "$PACK_IMAGE" bash -c "
    tar -I 'pigz -p \$(nproc)' -cf /tmp/${PACK_NAME}.tar.gz -C /workspace musubi-tuner &&
    cd /tmp && sha256sum ${PACK_NAME}.tar.gz > ${PACK_NAME}.sha256
")"
trap 'docker rm -f "$cid" >/dev/null 2>&1 || true' EXIT

code="$(docker wait "$cid")"
if [ "$code" != "0" ]; then
    docker logs "$cid" >&2
    echo "archive step failed with exit code $code" >&2
    exit 1
fi

echo ">>> copy to $PACK_OUT_DIR"
docker cp "$cid:/tmp/${PACK_NAME}.tar.gz" "$PACK_OUT_DIR/"
docker cp "$cid:/tmp/${PACK_NAME}.sha256" "$PACK_OUT_DIR/"

ls -lh "$PACK_OUT_DIR/${PACK_NAME}.tar.gz"
cat "$PACK_OUT_DIR/${PACK_NAME}.sha256"
