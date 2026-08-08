# Упаковка musubi-tuner для закрытого контура

Собирает автономный архив с Python-окружением и всеми зависимостями. На целевом
сервере нет ни интернета, ни Docker, поэтому архив распаковывается и работает
как есть.

## Целевой сервер

| Параметр | Значение |
| --- | --- |
| GPU | 2x NVIDIA H200, `sm_90` |
| Драйвер | 535.247.01 (CUDA 12.2) |
| ОС | Ubuntu 24.04, glibc 2.39, x86_64 |
| Путь развёртывания | `/workspace/musubi-tuner` |

CUDA Toolkit на сервере не нужен: колёса torch несут свой CUDA runtime внутри,
требуется только драйвер.

## Как это работает

Два решения, на которых держится переносимость:

1. Окружение собирается внутри контейнера **по тому же абсолютному пути**,
   по которому будет лежать на сервере. Поэтому shebang'и в `.venv/bin/*`,
   `pyvenv.cfg` и пути editable-установки остаются валидными после распаковки.
2. В архив кладётся **собственный интерпретатор** (standalone CPython 3.10
   в `python/`), а не системный. От сервера требуются только ядро, glibc
   и драйвер NVIDIA.

Архив создаётся `tar` внутри контейнера, а не на хосте: файловая система Windows
не умеет представлять симлинки и биты прав, которые есть в venv.

## Сборка

```bash
bash packaging/pack.sh
```

Результат в `../out_build/`: `musubi-tuner-cu124.tar.gz` и файл с контрольной
суммой. Переменными можно переопределить `PACK_OUT_DIR`, `PACK_IMAGE`,
`PACK_NAME` - префикс `PACK_` нужен потому, что `NAME` и `IMAGE` уже заняты
в окружении WSL.

Проверка собранного архива:

```bash
bash packaging/verify.sh
```

Скрипт распаковывает архив в чистом контейнере без доустановленных пакетов
и запускает интерпретатор из архива. Так ловятся зависимости от системных
библиотек, которых на сервере может не быть. `cuda_available` там всегда
`False` - у контейнера нет GPU, это ожидаемо.

Требования к сборочной машине: Docker и прямой доступ к `pypi.org`,
`download.pytorch.org`, `ghcr.io`, `public.ecr.aws`. Docker Hub не нужен -
базовый образ берётся из зеркала AWS ECR, так как Hub блокирует часть регионов.

Ожидаемый размер архива - 6-9 ГБ. В Docker Desktop стоит заранее поднять лимит
дискового образа WSL2.

## Выбор версии CUDA

`cu124` выбран под драйвер 535. Благодаря CUDA minor version compatibility
колёса любого тулкита 12.x работают на драйвере от 525.60.13, и для `cu124` это
штатный сценарий. `cu128` тянет заметно более свежие cuDNN и NCCL, которые уже
упирались в старые драйверы, а выигрыша на `sm_90` не даёт. `cu130` требует
драйвер 580+ и неприменим.

Версии задаются аргументами сборки `TORCH_VERSION`, `TORCHVISION_VERSION`,
`TORCH_INDEX` в `Dockerfile`.

## Доставка на сервер

Архив публикуется в публичный dataset-репозиторий на Hugging Face, откуда
сервер забирает его через корпоративное зеркало.

```bash
pip install -U "huggingface_hub[cli,hf_transfer]"
hf auth login
HF_HUB_ENABLE_HF_TRANSFER=1 hf upload <user>/musubi-tuner-env \
    ../out_build/musubi-tuner-cu124.tar.gz --repo-type dataset
hf upload <user>/musubi-tuner-env \
    ../out_build/musubi-tuner-cu124.sha256 --repo-type dataset
```

На сервере, без установки каких-либо пакетов:

```bash
cd /workspace
curl -L -C - -O "$HF_MIRROR/datasets/<user>/musubi-tuner-env/resolve/main/musubi-tuner-cu124.tar.gz"
curl -L -O "$HF_MIRROR/datasets/<user>/musubi-tuner-env/resolve/main/musubi-tuner-cu124.sha256"
sha256sum -c musubi-tuner-cu124.sha256
tar -xzf musubi-tuner-cu124.tar.gz
```

`-C -` включает докачку с места обрыва.

## Проверка после распаковки

```bash
cd /workspace/musubi-tuner
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))"
.venv/bin/python src/musubi_tuner/qwen_image_train.py --help
```

Ожидаемый вывод первой команды: `2.6.0+cu124 True 2 NVIDIA H200`.

Если `torch.cuda.is_available()` вернёт `False` при живом `nvidia-smi`, значит
сборка torch несовместима с драйвером - пересобрать с `TORCH_INDEX` на `cu121`
и `TORCH_VERSION=2.5.1`.

## Запуск

Только через `.venv/bin/python`, без `activate` - чтобы не зависеть от того,
какое окружение активно в Jupyter:

```bash
cd /workspace/musubi-tuner
.venv/bin/python src/musubi_tuner/qwen_image_train.py --config_file 1_confyg_qwen_full_finetune/config.toml
```

Отдельное ядро для Jupyter, если нужно:

```bash
.venv/bin/python -m ipykernel install --user --name musubi-tuner
```

## Замена opencv-python на headless

`opencv-python` подгружает системные `libGL.so.1` и `libglib2.0`, наличие
которых на сервере не гарантировано. GUI-функции OpenCV в обучающих скриптах
не используются, поэтому после установки зависимостей пакет заменяется на
`opencv-python-headless` той же версии. Импорт `cv2` и API остаются прежними.

## Что не входит в сборку

`sageattention` и `flash-attn` требуют компиляции под конкретную архитектуру GPU
и в базовую сборку не включены. При необходимости добавляются отдельным слоем
в `Dockerfile` с `TORCH_CUDA_ARCH_LIST=9.0`.

Веса моделей в архив не кладутся - они качаются на сервер отдельно.
