# Инструкция: перенос validation loss в другую версию musubi-tuner

## Цель

Скрипт полного файнтюна Qwen-Image (`src/musubi_tuner/qwen_image_train.py`) должен принимать
три ключа конфига, которых нет в апстриме:

```toml
validation_dataset_config = "val_dataset.toml"  # отдельный TOML с отложенным датасетом
validation_every_n_steps = 50                   # считать validation loss каждые N шагов
validation_seed = 1000                          # фиксированный seed валидации
```

Все остальные ключи `config.toml` (dit/vae/text_encoder, full_bf16, adafactor,
fused_backward_pass, save_*, sample_*, logging_* и т.д.) — стандартные, менять ничего не нужно.
Проверь только, что целевая версия поддерживает `--config_file` (функция `read_config_from_file`).

## Принцип работы

Validation loss — это тот же flow-matching MSE loss, что и train loss, но:

- считается на отдельном датасете без обновления весов (`eval()` + `torch.no_grad()`);
- шум и timestep'ы сэмплируются с фиксированным seed, поэтому значение сравнимо между шагами;
- каждый блок `[[datasets]]` валидационного TOML — отдельный «домен» со своей кривой
  `loss/validation/<имя>`;
- RNG-состояние PyTorch сохраняется до валидации и восстанавливается после, чтобы не влиять
  на воспроизводимость обучения.

## Логирование в TensorBoard

Все скаляры должны попадать в один event-файл (один запуск TensorBoard). Для этого валидация
логируется только через `accelerator.log(...)` — тем же трекером, которым апстрим пишет
train-метрики. Отдельные `SummaryWriter` и поддиректории создавать нельзя.

Итоговый набор кривых в одном файле:

- `loss/current`, `loss/average`, `lr/unet`, `loss/epoch` — уже есть в апстриме, не трогать;
- `loss/validation/<домен>` — добавляется этой инструкцией, по одной кривой на каждый
  `[[datasets]]`-блок валидационного TOML.

Общая усреднённая кривая `loss/validation` по всем доменам не нужна и не логируется
(обучение ведётся по одному домену за раз).

Кэши latent/Text Encoder для валидационного датасета создаются заранее теми же скриптами,
что и для обучающего (`qwen_image_cache_latents.py`, `qwen_image_cache_text_encoder_outputs.py`).

## Шаги реализации

Все правки — в одном файле `src/musubi_tuner/qwen_image_train.py`. Имена функций/импортов
(`collator_class`, `compute_loss_weighting_for_sd3`, `call_dit`, `get_noisy_model_input_and_timesteps`)
могут отличаться между версиями — используй те, что уже применяются в train-цикле целевой версии.

### 1. Аргументы CLI

В функцию, добавляющую finetune-специфичные аргументы (`qwen_image_finetune_setup_parser`):

```python
parser.add_argument("--validation_dataset_config", type=str, default=None,
                    help="path to validation dataset config TOML. Validation loss is logged during training")
parser.add_argument("--validation_every_n_steps", type=int, default=None,
                    help="compute validation loss every N steps. Validation is disabled if not specified")
parser.add_argument("--validation_seed", type=int, default=23,
                    help="fixed seed for validation noise/timestep sampling")
```

### 2. Загрузка валидационного датасета

Сразу после создания обучающей группы датасетов (`train_dataset_group` + `collator`),
тем же механизмом (`BlueprintGenerator` -> `generate_dataset_group_by_blueprint`):

```python
val_dataset_group = None
val_collator = None
if args.validation_dataset_config is not None:
    val_user_config = config_utils.load_user_config(args.validation_dataset_config)
    val_blueprint = blueprint_generator.generate(val_user_config, args, architecture=self.architecture)
    val_dataset_group = config_utils.generate_dataset_group_by_blueprint(
        val_blueprint.dataset_group, training=True,
        num_timestep_buckets=self.num_timestep_buckets, shared_epoch=current_epoch,
    )
    if val_dataset_group.num_train_items == 0:
        raise ValueError("No validation items found. Create latent/TE cache for the validation dataset first.")
    val_ds_for_collator = val_dataset_group if args.max_data_loader_n_workers == 0 else None
    val_collator = collator_class(current_epoch, val_ds_for_collator)
```

Важно: `training=True` (иначе не отдаст батчи из кэша), `shared_epoch` — тот же `current_epoch`,
что у train-группы.

### 3. DataLoader'ы по доменам

После создания `train_dataloader`, один DataLoader на каждый `[[datasets]]`-блок:

```python
val_dataloaders = []  # list of (domain_name, dataloader)
if val_dataset_group is not None:
    val_dataset_group.set_max_train_steps(args.max_train_steps)
    used_names = {}
    for val_dataset in val_dataset_group.datasets:
        domain_name = self.get_validation_domain_name(val_dataset)
        if domain_name in used_names:
            used_names[domain_name] += 1
            domain_name = f"{domain_name}_{used_names[domain_name]}"
        else:
            used_names[domain_name] = 0
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=False, collate_fn=val_collator,
            num_workers=n_workers, persistent_workers=args.persistent_data_loader_workers,
        )
        val_dataloaders.append((domain_name, val_dataloader))
```

`batch_size=1` здесь — параметр DataLoader; реальный батчинг делает сам датасет
(ключ `batch_size` в TOML), как и у train.

После `accelerator.prepare(optimizer, train_dataloader, lr_scheduler)`:

```python
if val_dataloaders:
    val_dataloaders = [(name, accelerator.prepare(dl)) for name, dl in val_dataloaders]
```

### 4. Имя домена

Имя кривой берётся из имени папки `image_directory`; если лист generic («val», «data» и т.п.) —
берётся родительская папка (`.../a1fa_ac1ub_1arge/val` -> `a1fa_ac1ub_1arge`):

```python
def get_validation_domain_name(self, dataset):
    generic_leaves = {"val", "validation", "train", "data", "image", "images"}
    for attr in ("image_directory", "image_jsonl_file", "video_directory", "cache_directory"):
        value = getattr(dataset, attr, None)
        if value:
            path = os.path.normpath(value)
            name = os.path.splitext(os.path.basename(path))[0]
            if name.lower() in generic_leaves:
                parent = os.path.basename(os.path.dirname(path))
                if parent:
                    name = parent
            return name
    return "val"
```

### 5. Вызов валидации в train-цикле

Внутри `if accelerator.sync_gradients:` (там же, где обрабатываются sampling и сохранение
чекпоинтов), после блока save/sample:

```python
should_validating = (
    args.validation_every_n_steps is not None and global_step % args.validation_every_n_steps == 0
)
if should_validating:
    optimizer_eval_fn()
    self.validate(accelerator, args, transformer, val_dataloaders, noise_scheduler, dit_dtype, global_step)
    optimizer_train_fn()
```

Обёртка `optimizer_eval_fn()` / `optimizer_train_fn()` обязательна — так же, как это уже
сделано для sampling/saving (важно для schedule-free и подобных оптимизаторов).

### 6. Метод validate

Прямой forward-проход по тому же пути, что и train-шаг. Использовать те же вызовы, что в
train-цикле целевой версии (`scale_shift_latents`, `get_noisy_model_input_and_timesteps`,
`compute_loss_weighting_for_sd3`, `call_dit`):

```python
def validate(self, accelerator, args, transformer, val_dataloaders, noise_scheduler, dit_dtype, global_step):
    if not val_dataloaders:
        return

    accelerator.wait_for_everyone()
    transformer.eval()

    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    logs = {}

    for domain_name, val_dataloader in val_dataloaders:
        # сброс seed на каждый домен: шум/timestep'ы воспроизводимы между шагами и независимы между доменами
        torch.manual_seed(args.validation_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.validation_seed)

        domain_loss = torch.zeros((), device=accelerator.device)
        domain_count = torch.zeros((), device=accelerator.device)
        with torch.no_grad():
            for batch in val_dataloader:
                latents = self.scale_shift_latents(batch["latents"])
                noise = torch.randn_like(latents)
                noisy_model_input, timesteps = self.get_noisy_model_input_and_timesteps(
                    args, noise, latents, batch["timesteps"], noise_scheduler, accelerator.device, dit_dtype
                )
                weighting = compute_loss_weighting_for_sd3(
                    args.weighting_scheme, noise_scheduler, timesteps, accelerator.device, dit_dtype
                )
                output = self.call_dit(
                    args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, dit_dtype
                )
                model_pred, target = self._unpack_dit_output(output)
                loss = torch.nn.functional.mse_loss(model_pred.to(dit_dtype), target, reduction="none")
                if weighting is not None:
                    loss = loss * weighting
                domain_loss += loss.mean().detach()
                domain_count += 1

        domain_loss = accelerator.reduce(domain_loss, reduction="sum")
        domain_count = accelerator.reduce(domain_count, reduction="sum")

        avg_domain = (domain_loss / domain_count).item() if domain_count.item() > 0 else 0.0
        logs[f"loss/validation/{domain_name}"] = avg_domain
        accelerator.print(f"\nvalidation loss [{domain_name}] at step {global_step}: {avg_domain:.6f}")

    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state_all(cuda_rng_state)

    transformer.train()

    if accelerator.is_main_process and len(accelerator.trackers) > 0:
        accelerator.log(logs, step=global_step)
```

### 7. Совместимость с возвращаемым значением call_dit

В разных версиях `call_dit` возвращает либо объект с `.pred`/`.target`, либо кортеж.
Хелпер и замена в train-цикле:

```python
@staticmethod
def _unpack_dit_output(output):
    if isinstance(output, tuple):
        return output[0], output[1]
    return output.pred, output.target
```

В train-цикле строку вида `loss = mse_loss(output.pred..., output.target, ...)` заменить на:

```python
model_pred, target = self._unpack_dit_output(output)
loss = torch.nn.functional.mse_loss(model_pred.to(dit_dtype), target, reduction="none")
```

## Инварианты (проверить после переноса)

1. Валидация не меняет веса и состояние оптимизатора (только forward, `no_grad`, без `lr_scheduler.step()`).
2. RNG-состояние после валидации идентично состоянию до неё (иначе ломается воспроизводимость train).
3. При `validation_dataset_config = None` поведение скрипта байт-в-байт совпадает с апстримом.
4. Loss считается тем же путём, что и train loss (тот же timestep_sampling, weighting_scheme,
   discrete_flow_shift из args) — иначе кривые train и validation несравнимы.
5. В мультиGPU суммы редуцируются через `accelerator.reduce`, логирует только main process.
6. Все кривые (`loss/current`, `loss/average`, `lr/unet`, `loss/epoch`, `loss/validation/<домен>`)
   пишутся в один event-файл TensorBoard — только через `accelerator.log`, без отдельных
   `SummaryWriter` и поддиректорий.

## Формат валидационного TOML

Тот же, что у обучающего датасета. Один `[[datasets]]` = один домен = одна кривая:

```toml
[general]
resolution = [1024, 1024]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "/path/datasets/image/<domain>/val"
cache_directory = "/path/datasets/cache/<domain>/val"
num_repeats = 1
```
