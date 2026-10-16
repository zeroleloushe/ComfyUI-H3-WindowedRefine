# ComfyUI-H3-WindowedRefine

Нода ComfyUI **MiniMax H3 · Windowed Latent Refine** (`H3WindowedLatentRefine`, категория `MiniMax H3/Refine`). Она дорабатывает готовый AV-латент MiniMax H3 (Ref2VA) для длинных видео: делит его на перекрывающиеся окна и прогоняет за один проход сэмплера.

> **English summary.** A ComfyUI node for MiniMax H3 (Ref2VA) that refines a long audio+video latent with overlapping temporal windows in a single sampler pass. Window order alternates (1→N / N→1), x0 is blended across overlaps, and each window can be guided by its neighbour's overlap (temporal guide via H3 keyframes) and optionally by an anchor latent. A `sequential` window mode runs window after window (all steps each), conditioning every next window on the finished overlap of the previous one (latent keyframes + locked x0), re-encoding the prompt for every next window with the main refs plus the previous window's last decoded frame as an extra `<Picture N>`; window 1 can use a ready `positive` conditioning. It takes character/location reference images, and a canvas preview on the node shows the window layout. Install: `git clone` into `ComfyUI/custom_nodes` and restart ComfyUI. No extra pip dependencies.

![Нода в ComfyUI](docs/node.png)

## Что умеет
- **Оконный рефайн.** Видеолатент `[B,24,T,H/16,W/16]` (T = 5k+2, то есть 17k+5 кадров) режется на окна по `5m+2` латент-кадров с перекрытием `5o+2`. Окна выровнены по блокам VAE.
- **Один проход сэмплера.** Используется обычный сэмплер ComfyUI с любым `sampler_name`/`scheduler`. Обёртка модели на каждом шаге обходит все окна.
- **Чередование направлений.** Порядок окон меняется от шага к шагу: 1→N, потом N→1. В зонах перекрытия x0 смешивается.
- **Temporal guide.** Окно получает перекрытие от уже обработанного соседа как keyframe H3. Начало задаёт `guide_from_step`.
- **Якорь.** Для части шагов (`anchor_steps`) x0 подмешивается к якорю с силой `anchor_strength`. Якорь — это `anchor_latent` (или исходный латент).
- **Референсы.** Входы `character_ref`, `location_ref`, `location_ref_2`, `location_ref_3`, `ref_4`–`ref_6` (до 9 картинок, батчи раскрываются). В промте на них ссылаются как `<Picture 1>`, `<Picture 2>`, …, нумерация идёт в порядке входов. Таблица соответствия выводится в лог и на ноду.
- **Последовательный режим** (`window_mode = последовательный`): окна идут по очереди, каждое проходит все шаги целиком, и следующее окно продолжает уже готовый результат. Подробности ниже.
- **Превью раскладки окон** прямо на ноде, с предупреждениями: число кадров не 17k+5; окно короче T=37 (124 кадра).
- **Вход латента:** AV NestedTensor H3, словарь/кортеж `(video, audio)` или только видео (аудио тогда берётся из `anchor_latent` или создаётся пустое).

## Установка
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/zeroleloushe/ComfyUI-H3-WindowedRefine.git
```
Перезапусти ComfyUI. Дополнительные pip-зависимости не нужны: нода использует torch и модули самого ComfyUI. Нужна сборка ComfyUI с поддержкой MiniMax H3 (Ref2VA).

## Подключение
1. `model` — модель MiniMax H3 Ref2VA (можно с turbo LoRA), `clip` — текстовый энкодер H3 (Qwen3-VL), `vae` — VAE H3.
2. `latent` — готовый латент H3, например результат первого прохода генерации.
3. По желанию подключи картинки к `character_ref` / `location_ref` / `ref_*` и упомяни их в промте как `<Picture N>`.
4. Выход `latent` (AV-латент) отправь в декодер H3. Можно включить `decode` и взять `images` прямо с ноды.

## Параметры
| Параметр | По умолчанию | Описание |
|---|---|---|
| steps | 20 | Число шагов (при denoise<1 выполняется только их часть). Turbo LoRA: 4 |
| cfg | 5.0 | CFG. Turbo LoRA: 1.0 |
| sampler_name | res_multistep (или euler) | Сэмплер |
| scheduler | simple | Расписание шагов |
| shift_video | 12.0 | Flow shift для видео; 0 — не трогать модель |
| shift_audio | 3.0 | Flow shift для аудио |
| denoise | 0.3 | Сила перерисовки |
| num_windows | 2 | Число окон; 0 — размер окна задаёт `window_blocks` |
| window_blocks (m) | 0 | Длина окна 5·m+2 латент-кадров (когда num_windows = 0) |
| overlap_blocks (o) | 2 | Перекрытие 5·o+2 латент-кадров (1 → 7, 2 → 12) |
| window_mode | по шагам | `по шагам` — все окна на каждом шаге (как раньше); `последовательный` — окно за окном |
| seq_lock_overlap | вкл | Посл. режим: зафиксировать перекрытие на готовом предыдущем окне |
| seq_ref_phrase | (см. ниже) | Посл. режим: фраза для окон 2+, `{n}` — номер кадра-референса |
| alternate_direction | вкл | Чередовать порядок окон 1→N / N→1 |
| temporal_guide | вкл | Подавать соседнее перекрытие как keyframe H3 |
| guide_from_step | 1 | С какого шага (от 0) включается guide |
| anchor_strength | 0.0 | Сила подмешивания якоря |
| anchor_steps | 0.5 | Доля шагов с якорем: round(доля × steps) |
| decode | выкл | Декодировать в IMAGE через VAE (иначе на выходе заглушка 64×64) |
| verbose | вкл | Подробный лог `[H3WR]` в консоли |
| preview_frames | 0 | Кадры для превью окон (только UI); 0 — взять из последнего запуска или 124 |

Выходы: `latent` (AV-латент H3) и `images`.

## Последовательный режим
Включается параметром `window_mode = последовательный`. Режим по умолчанию (`по шагам`) работает как раньше.

Как работает:
1. Окна идут по очереди: окно 1 проходит все шаги сэмплера целиком, потом окно 2 и т.д. Шум и сид те же, что в обычном режиме (срез общего шума).
2. **Окно 1** берёт готовый `positive` прошлой генерации, если он подключён к входу `positive` (CONDITIONING), как есть. Если вход пустой — кодируется промпт ноды с основными референсами.
3. **Каждое следующее окно кодирует кондишн заново**: промпт ноды + основные референсы + **последний кадр готового предыдущего окна** как новый `<Picture N>` (N = число основных картинок + 1). Кадр декодируется через VAE (декодируется только хвост окна, 2 латент-кадра). К промпту новой строкой дописывается фраза из `seq_ref_phrase`, `{n}` заменяется на N. По умолчанию:
   `Continuity: <Picture {n}> is the last frame of the previous segment of this video; continue from it and keep characters, faces, clothing, lighting and details consistent with <Picture {n}>.`
   Пустая фраза — дописывается просто `<Picture N>`. Негатив тоже кодируется с этим же набором референсов.
4. Готовые латент-кадры из перекрытия передаются следующему окну как keyframes H3 (отключается `temporal_guide`).
5. `seq_lock_overlap` (вкл по умолчанию): перекрытие зафиксировано на готовом предыдущем окне и не перерисовывается. Выкл — перекрытие перерисовывается, в итог идёт новая версия.

Подключение: `positive` — выход positive от ноды кодирования прошлой генерации (тот, что шёл в сэмплер первого прохода); `latent` — апскейленный латент; `anchor_latent` — исходный, как и раньше. Основных референсов не больше 8 (девятый слот занимает кадр предыдущего окна).

Что меняется в параметрах:
- `alternate_direction` и `guide_from_step` в этом режиме не используются.
- Якорь работает в каждом окне отдельно (срез `anchor_latent` под окно).
- Превью латентов в сэмплере не показывается; прогресс и ETA — на ноде и в логе. Текущее `<Picture N>` для окна видно в таблице референсов.

Когда использовать: если между окнами плывут или перерисовываются детали, лица, одежда. Минусы: дольше (кодирование промпта и декод кадра на каждое окно), ошибки могут копиться к концу ролика.

## Советы
- Окна короче **T=37 (124 кадра)** выходят за обучающий диапазон H3, и превью предупредит об этом. Лучше брать меньше окон, но длиннее.
- Длина клипа должна быть 17k+5 кадров (22, 39, …, 124, 141, …). Иначе нода выдаст ошибку с подсказкой.
- Для начала подойдут denoise 0.3–0.5 и overlap_blocks 1–2.

## Тесты
`pytest` из папки ноды (`tests/`). Скрипт `tools/seam_check.py` проверяет швы между окнами.

## Лицензия
Файла лицензии пока нет.
