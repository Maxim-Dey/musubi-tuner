# Полный файнтюн Qwen-Image 2512 20B

Одна команда на весь пайплайн: кеш латентов -> кеш Text Encoder -> обучение.
Всё пишется в один лог-файл, обучение — в один event-файл TensorBoard.

## Предпосылки

- Окружение уже активировано, запуск из корня `/workspace/musubi-tuner`.
- Модели лежат по путям из `config.toml` (`dit`, `vae`, `text_encoder`).
- Конфиги (`config.toml`, `dataset.toml`, `val_dataset.toml`, `sample_prompts.txt`) лежат
в одной папке — путь задаётся переменной `CFG` в команде ниже.
- Картинки с подписями разложены по путям `image_directory` из `dataset.toml` и `val_dataset.toml`.
Папки `cache_directory` создавать не нужно — скрипты кеширования создают их сами.



## Запуск

```bash
mkdir -p /home/jovyan/logs
LOG="/home/jovyan/logs/qwen_train_$(date +%Y%m%d_%H%M%S).log"

CFG="/workspace/musubi-tuner/1_confyg_qwen_full_finetune"   # папка с конфигами
GPUS="0,1"                                           # какие GPU использовать: "0", "1" или "0,1"

{
  echo "=== START $(date -Is) ===";
  echo "LOG=$LOG";
  echo "USER=$(whoami)";

  cd /workspace/musubi-tuner || exit 1;
  echo "PWD=$(pwd)";

  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;
  export CUDA_VISIBLE_DEVICES="$GPUS";
  NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l);
  echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF";
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES";
  echo "NPROC=$NPROC";

  VAE=$(python -c 'import sys, toml; print(toml.load(sys.argv[1])["vae"])' "$CFG/config.toml") || exit 1;
  TE=$(python -c 'import sys, toml; print(toml.load(sys.argv[1])["text_encoder"])' "$CFG/config.toml") || exit 1;
  echo "VAE=$VAE";
  echo "TEXT_ENCODER=$TE";

  echo "=== CACHE LATENTS (train + val) ===";
  for DS in dataset val_dataset; do
    stdbuf -oL -eL python src/musubi_tuner/qwen_image_cache_latents.py \
      --dataset_config "$CFG/$DS.toml" \
      --vae "$VAE" \
      --model_version original \
      --batch_size 8 \
      --skip_existing || exit $?;
  done;

  echo "=== CACHE TEXT ENCODER (train + val) ===";
  for DS in dataset val_dataset; do
    stdbuf -oL -eL python src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py \
      --dataset_config "$CFG/$DS.toml" \
      --text_encoder "$TE" \
      --model_version original \
      --batch_size 8 \
      --skip_existing || exit $?;
  done;

  echo "=== TRAINING ===";
  if [ "$NPROC" -gt 1 ]; then MULTI="--multi_gpu"; else MULTI=""; fi;

  stdbuf -oL -eL accelerate launch $MULTI --num_processes "$NPROC" --mixed_precision bf16 \
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



## Выбор видеокарты

Меняется одна строка `GPUS`, количество процессов и флаг `--multi_gpu` подставляются сами:


| `GPUS`  | что запустится                           |
| ------- | ---------------------------------------- |
| `"0"`   | одна GPU 0, `--num_processes 1`          |
| `"1"`   | одна GPU 1, `--num_processes 1`          |
| `"0,1"` | обе GPU, `--multi_gpu --num_processes 2` |


Кеширование всегда однопроцессное и идёт на первой GPU из списка.

## Повторные запуски

Кеширование вызывается с `--skip_existing`: файлы, для которых кеш уже есть, повторно не
считаются, поэтому команду можно перезапускать как есть. Кеши, которых больше нет в датасете,
удаляются автоматически (добавьте `--keep_cache`, если это не нужно). Полный пересчёт —
удалить папки `cache_directory` из `dataset.toml` / `val_dataset.toml`.

## Зачем пути к датасетам дублируются в команде

В `config.toml` ключи `dataset_config`, `validation_dataset_config` заданы относительными
путями, а разрешаются они относительно текущей директории (`/workspace/musubi-tuner`), а не
относительно самого конфига. Явная передача `--dataset_config`, `--validation_dataset_config`
и `--sample_prompts` из `$CFG` перекрывает значения конфига и убирает эту неоднозначность.

## Куда смотреть

- Лог запуска: `/home/jovyan/logs/qwen_train_<дата>_<время>.log`
- Чекпоинты: `output_dir` из `config.toml`
- Сэмплы и кривые TensorBoard: `logging_dir` из `config.toml`

```bash
tensorboard --logdir "/workspace/musubi-tuner/1_confyg_qwen_full_finetune/logs" --host 0.0.0.0 --port 6006
```

Кривые: `loss/current`, `loss/average`, `loss/epoch`, `lr/unet` и
`loss/validation/<домен>` — по одной на каждый блок `[[datasets]]` из `val_dataset.toml`.