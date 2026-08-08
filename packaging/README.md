# musubi-tuner: упаковка, доставка и запуск в закрытом контуре

Полный путь от сборки на рабочей машине до обучения на сервере DES. Собирается
автономный архив с Python-окружением и всеми зависимостями: на целевом сервере
нет ни интернета, ни Docker, поэтому архив распаковывается и работает как есть.

## Целевой сервер

| Параметр | Значение |
| --- | --- |
| GPU | 2x NVIDIA H200, `sm_90` |
| Драйвер | 535.247.01 (CUDA 12.2) |
| ОС | Ubuntu 24.04, glibc 2.39, x86_64 |
| Путь развёртывания | `/workspace/musubi-tuner` |

CUDA Toolkit на сервере не нужен: колёса torch несут свой CUDA runtime внутри,
требуется только драйвер.

## Как устроена переносимость

Два решения, на которых всё держится:

1. Окружение собирается внутри контейнера **по тому же абсолютному пути**,
   по которому будет лежать на сервере. Поэтому shebang'и в `.venv/bin/*`,
   `pyvenv.cfg` и пути editable-установки остаются валидными после распаковки.
2. В архив кладётся **собственный интерпретатор** (standalone CPython 3.10
   в `python/`), а не системный. От сервера требуются только ядро, glibc
   и драйвер NVIDIA.

Архив создаётся `tar` внутри контейнера, а не на хосте: файловая система Windows
не умеет представлять симлинки и биты прав, которые есть в venv.

## Шаг 1. Сборка

Запускается из PowerShell, из корня репозитория инструмента:

```powershell
cd musubi-tuner
bash packaging/pack.sh
```

`bash` здесь - это bash из WSL, который ставится вместе с Docker Desktop
на WSL2-бэкенде. Он сам транслирует текущую папку в `/mnt/c/...`, поэтому
относительные пути внутри скриптов работают, и `docker` виден через
WSL-интеграцию. Ничего доустанавливать не нужно.

Результат в `../out_build/`: `musubi-tuner-cu124.tar.gz` и файл с контрольной
суммой. Переменными можно переопределить `PACK_OUT_DIR`, `PACK_IMAGE`,
`PACK_NAME` - префикс `PACK_` нужен потому, что `NAME` и `IMAGE` уже заняты
в окружении WSL.

Требования к сборочной машине: Docker и прямой доступ к `pypi.org`,
`download.pytorch.org`, `ghcr.io`, `public.ecr.aws`. Docker Hub не нужен -
базовый образ берётся из зеркала AWS ECR, так как Hub блокирует часть регионов.

Сборка занимает около 15 минут, архив выходит примерно 3.4 ГБ. В Docker Desktop
стоит заранее поднять лимит дискового образа WSL2.

## Шаг 2. Проверка архива

```powershell
bash packaging/verify.sh
```

Скрипт распаковывает архив в чистом контейнере без доустановленных пакетов
и запускает интерпретатор из архива. Так ловятся зависимости от системных
библиотек, которых на сервере может не быть. `cuda_available` там всегда
`False` - у контейнера нет GPU, это ожидаемо.

## Шаг 3. Заливка на Hugging Face

Репозиторий `motionmaksim/environment`, тип - model (обычный, без
`--repo-type`): именно такой путь понимает зеркало Artifactory.

```powershell
cd ..\out_build
$env:HF_HUB_DISABLE_XET = "1"
hf upload motionmaksim/environment musubi-tuner-cu124.tar.gz
```

`HF_HUB_DISABLE_XET=1` обязателен. Без него включается Xet-бэкенд, который
льёт файл в много параллельных соединений и намертво встаёт на середине -
прогресс-бар стоит, ошибки нет. Переменная действует только в текущем окне
PowerShell, при новом окне её нужно задать снова.

Заливка идёт из папки с архивом, поэтому имя файла указывается без пути -
под этим же именем файл появится в репозитории.

Если всё же встанет, прервать по `Ctrl+C` и запустить ту же команду снова:
уже загруженные части повторно не отправляются.

## Шаг 4. Скачивание на сервер

Зеркало Artifactory проксирует Hugging Face, токен не нужен.

```bash
cd /workspace
BASE=https://binary.alfabank.ru/artifactory/api/huggingfaceml/huggingface/motionmaksim/environment/resolve/main
curl -L -C - -O "$BASE/musubi-tuner-cu124.tar.gz"
```

`-C -` продолжает закачку с места обрыва, поэтому при разрыве достаточно
повторить ту же команду. `-O` сохраняет файл под именем из URL.

Файл `.sha256` через зеркало тянуть бесполезно: маленькие файлы Artifactory
отдаёт не содержимым, а служебным указателем, и `sha256sum -c` на нём падает
с `no properly formatted checksum lines found`.

## Шаг 5. Проверка и распаковка

Сверяем размер и хеш с тем, что напечатал `pack.sh` в конце сборки:

```bash
cd /workspace
ls -l musubi-tuner-cu124.tar.gz
sha256sum musubi-tuner-cu124.tar.gz
tar -xzf musubi-tuner-cu124.tar.gz
```

Размер в байтах должен совпасть точно - это уже отсекает оборванную закачку.
Хеш считается несколько минут и даёт полную гарантию. Если не совпало -
удалить архив и скачать заново, распаковывать бессмысленно.

Распаковка создаёт `/workspace/musubi-tuner`. Путь менять нельзя: окружение
собрано под него, при переносе в другое место сломаются пути внутри `.venv`.

## Шаг 6. Проверка окружения

```bash
cd /workspace/musubi-tuner
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

Ожидаемый вывод: `2.6.0+cu124 True 2 NVIDIA H200`.

Если `torch.cuda.is_available()` вернёт `False` при живом `nvidia-smi`, значит
сборка torch несовместима с драйвером - пересобрать с `TORCH_INDEX` на `cu121`
и `TORCH_VERSION=2.5.1`.

Активировать окружение не нужно нигде: интерпретатор вызывается по полному
пути `/workspace/musubi-tuner/.venv/bin/python`, поэтому результат не зависит
от того, какое окружение активно в Jupyter. `source .venv/bin/activate` тоже
работает, но ничего не упрощает.

Отдельное ядро для Jupyter, если нужно запускать из ноутбуков:

```bash
.venv/bin/python -m ipykernel install --user --name musubi-tuner
```

## Шаг 7. Обучение

Полный файнтюн Qwen-Image 2512 20B. Одна команда на весь пайплайн: кеш
латентов -> кеш Text Encoder -> обучение. Всё пишется в один лог-файл,
обучение - в один event-файл TensorBoard.

Что должно быть на месте до запуска:

- Модели по путям из `config.toml` (`dit`, `vae`, `text_encoder`) -
  в архив они не входят и качаются на сервер отдельно.
- Картинки с подписями по путям `image_directory` из `dataset.toml`
  и `val_dataset.toml`. Папки `cache_directory` создавать не нужно, скрипты
  кеширования создают их сами.

```bash
mkdir -p /workspace/logs
LOG="/workspace/logs/qwen_train_$(date +%Y%m%d_%H%M%S).log"

VENV="/workspace/musubi-tuner/.venv/bin"
CFG="/workspace/musubi-tuner/1_confyg_qwen_full_finetune"   # папка с конфигами
GPUS="0"                                                   # "0", "1" или "0,1"

{
  echo "=== START $(date -Is) ===";
  echo "LOG=$LOG";

  cd /workspace/musubi-tuner || exit 1;

  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;
  export CUDA_VISIBLE_DEVICES="$GPUS";
  NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l);
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NPROC=$NPROC";

  VAE=$("$VENV/python" -c 'import sys, toml; print(toml.load(sys.argv[1])["vae"])' "$CFG/config.toml") || exit 1;
  TE=$("$VENV/python" -c 'import sys, toml; print(toml.load(sys.argv[1])["text_encoder"])' "$CFG/config.toml") || exit 1;
  echo "VAE=$VAE";
  echo "TEXT_ENCODER=$TE";

  echo "=== CACHE LATENTS (train + val) ===";
  for DS in dataset val_dataset; do
    stdbuf -oL -eL "$VENV/python" src/musubi_tuner/qwen_image_cache_latents.py \
      --dataset_config "$CFG/$DS.toml" \
      --vae "$VAE" \
      --model_version original \
      --batch_size 8 \
      --skip_existing || exit $?;
  done;

  echo "=== CACHE TEXT ENCODER (train + val) ===";
  for DS in dataset val_dataset; do
    stdbuf -oL -eL "$VENV/python" src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py \
      --dataset_config "$CFG/$DS.toml" \
      --text_encoder "$TE" \
      --model_version original \
      --batch_size 8 \
      --skip_existing || exit $?;
  done;

  echo "=== TRAINING ===";
  if [ "$NPROC" -gt 1 ]; then MULTI="--multi_gpu"; else MULTI=""; fi;

  stdbuf -oL -eL "$VENV/accelerate" launch $MULTI --num_processes "$NPROC" --mixed_precision bf16 \
    --num_cpu_threads_per_process 1 \
    src/musubi_tuner/qwen_image_train.py \
      --config_file "$CFG/config.toml" \
      --dataset_config "$CFG/dataset.toml" \
      --validation_dataset_config "$CFG/val_dataset.toml" \
      --sample_prompts "$CFG/sample_prompts.txt";

  EXIT=$?;
  echo "=== END $(date -Is) exit=$EXIT ===";
  exit $EXIT;
} 2>&1 | tee -a "$LOG"
```

### Выбор видеокарты

Меняется одна строка `GPUS`, количество процессов и флаг `--multi_gpu`
подставляются сами:

| `GPUS`  | что запустится                           |
| ------- | ---------------------------------------- |
| `"0"`   | одна GPU 0, `--num_processes 1`          |
| `"1"`   | одна GPU 1, `--num_processes 1`          |
| `"0,1"` | обе GPU, `--multi_gpu --num_processes 2` |

Кеширование всегда однопроцессное и идёт на первой GPU из списка.

### Повторные запуски

Кеширование вызывается с `--skip_existing`: файлы, для которых кеш уже есть,
повторно не считаются, поэтому команду можно перезапускать как есть. Кеши,
которых больше нет в датасете, удаляются автоматически (добавьте
`--keep_cache`, если это не нужно). Полный пересчёт - удалить папки
`cache_directory` из `dataset.toml` / `val_dataset.toml`.

### Зачем пути к датасетам дублируются в команде

В `config.toml` ключи `dataset_config`, `validation_dataset_config` заданы
относительными путями, а разрешаются они относительно текущей директории
(`/workspace/musubi-tuner`), а не относительно самого конфига. Явная передача
`--dataset_config`, `--validation_dataset_config` и `--sample_prompts` из `$CFG`
перекрывает значения конфига и убирает эту неоднозначность.

### Куда смотреть

- Лог запуска: `/workspace/logs/qwen_train_<дата>_<время>.log`
- Чекпоинты: `output_dir` из `config.toml`
- Сэмплы и кривые TensorBoard: `logging_dir` из `config.toml`

```bash
/workspace/musubi-tuner/.venv/bin/tensorboard \
    --logdir "/workspace/musubi-tuner/1_confyg_qwen_full_finetune/logs" \
    --host 0.0.0.0 --port 6006
```

Пакет `tensorboard` ставится в окружение намеренно, хотя в `pyproject.toml`
он лежит в dev-группе. Он нужен не для просмотра, а для **записи**: accelerate
поднимает трекер через `SummaryWriter`, и без пакета молча выкидывает его -
обучение идёт, папка `logs` не создаётся, графиков нет. Системный tensorboard
на сервере тут не помогает, он только просмотрщик.

Кривые: `loss/current`, `loss/average`, `loss/epoch`, `lr/unet` и
`loss/validation/<домен>` - по одной на каждый блок `[[datasets]]`
из `val_dataset.toml`.

### Если обучение падает сразу

`ValueError: No training items found in the dataset` означает, что датасет
пуст с точки зрения загрузчика. Почти всегда причина - `image_directory`
в `dataset.toml` или `val_dataset.toml` указывает на несуществующий путь:
ошибки об отсутствующей папке не будет, просто найдётся ноль картинок.

```bash
ls "$(grep image_directory "$CFG/dataset.toml" | cut -d'"' -f2)" | head
```

## Справка

### Выбор версии CUDA

`cu124` выбран под драйвер 535. Благодаря CUDA minor version compatibility
колёса любого тулкита 12.x работают на драйвере от 525.60.13, и для `cu124` это
штатный сценарий. `cu128` тянет заметно более свежие cuDNN и NCCL, которые уже
упирались в старые драйверы, а выигрыша на `sm_90` не даёт. `cu130` требует
драйвер 580+ и неприменим.

Версии задаются аргументами сборки `TORCH_VERSION`, `TORCHVISION_VERSION`,
`TORCH_INDEX` в `Dockerfile`.

### Замена opencv-python на headless

`opencv-python` подгружает системные `libGL.so.1` и `libglib2.0`, наличие
которых на сервере не гарантировано. GUI-функции OpenCV в обучающих скриптах
не используются, поэтому после установки зависимостей пакет заменяется на
`opencv-python-headless` той же версии. Импорт `cv2` и API остаются прежними.

### Что не входит в сборку

`sageattention` и `flash-attn` требуют компиляции под конкретную архитектуру GPU
и в базовую сборку не включены. При необходимости добавляются отдельным слоем
в `Dockerfile` с `TORCH_CUDA_ARCH_LIST=9.0`.

Веса моделей в архив не кладутся - они качаются на сервер отдельно.
