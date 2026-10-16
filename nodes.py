import math

import torch

import comfy.model_management
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.utils
import node_helpers

from .h3_report import Reporter
from .h3_windows import WindowedH3Wrapper, plan_windows, FRAME_PER_TOKEN, audio_len_for_frames

CANVAS_MULTIPLE = 32


def _resize(image, width, height):
    s = image[..., :3].movedim(-1, 1)
    return comfy.utils.common_upscale(s, width, height, "lanczos", "disabled").movedim(1, -1)


MAX_REF_IMAGES = 9  # comfy_extras MiniMaxH3ReferenceToVideo: ref_images Autogrow max=9
REF_INPUTS = ["character_ref", "location_ref", "location_ref_2", "location_ref_3", "ref_4", "ref_5", "ref_6"]


def collect_refs(named):
    """named: {input_name: IMAGE batch or None}. Every batch image becomes its own <Picture N>,
    numbered 1.. in REF_INPUTS order then batch order (same 1-based per-type ordinal the H3 tokenizer uses)."""
    images, mapping = [], []
    for name in REF_INPUTS:
        img = named.get(name)
        if img is None:
            continue
        if not torch.is_tensor(img) or img.ndim not in (3, 4):
            raise ValueError(f"{name}: expected IMAGE [B,H,W,C], got {describe(img)}")
        if img.ndim == 3:
            img = img.unsqueeze(0)
        for b in range(img.shape[0]):
            images.append(img[b:b + 1])
            mapping.append((len(images), name, b, img.shape[0]))
    if len(images) > MAX_REF_IMAGES:
        detail = ", ".join(f"{n}×{c}" for _, n, b, c in mapping if b == 0)
        raise ValueError(f"too many reference images: {len(images)} (max {MAX_REF_IMAGES}, like MiniMaxH3ReferenceToVideo). Got: {detail}")
    return images, mapping


def mapping_lines(mapping):
    return [f"<Picture {n}> = {name}" + (f" [{b}]" if c > 1 else "") for n, name, b, c in mapping]


def encode_ref2va(clip, vae, prompt, ref_images, width, height):
    """Same presentation as comfy_extras MiniMaxH3ReferenceToVideo (images only, 'match' sizing)."""
    ref_items, ref_blocks = [], []
    for img in ref_images:
        if img is None:
            continue
        h, w = img.shape[1], img.shape[2]
        scale = min(1.0, math.sqrt((width * height) / (w * h)))
        tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        resized = _resize(img[:1], tw, th)
        ref_items.append({"type": "image", "data": resized})
        ref_blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": vae.encode(resized)})
    tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
    cond = clip.encode_from_tokens_scheduled(tokens)
    if ref_blocks:
        cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})
    return cond


def apply_shift(model, shift_video, shift_audio):
    """Same as comfy_extras MiniMaxH3SigmaShift."""
    import comfy.model_sampling
    m = model.clone()

    class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
        pass

    original = m.get_model_object("model_sampling")
    ms = ModelSamplingAdvanced(model.model.model_config)
    ms.set_parameters(shift=shift_video, audio_shift=shift_audio)
    if hasattr(original, "noise_scale"):
        ms.set_noise_scale(original.noise_scale)
    m.add_object_patch("model_sampling", ms)
    to = m.model_options["transformer_options"] = m.model_options.get("transformer_options", {}).copy()
    to["minimax_h3_sigma_shift_video"] = shift_video
    to["minimax_h3_sigma_shift_audio"] = shift_audio
    return m


def describe(x):
    """Human readable type/shape of whatever arrived in a LATENT."""
    if isinstance(x, dict):
        return "dict(keys=%s, samples=%s)" % (list(x.keys()), describe(x.get("samples")))
    if getattr(x, "is_nested", False):
        return "NestedTensor[%s]" % ", ".join(describe(t) for t in x.unbind())
    if isinstance(x, (list, tuple)):
        return "%s[%s]" % (type(x).__name__, ", ".join(describe(t) for t in x))
    if torch.is_tensor(x):
        return "Tensor%s(%s)" % (list(x.shape), str(x.dtype).replace("torch.", ""))
    return type(x).__name__


def _check_video(v, what):
    if not torch.is_tensor(v) or v.ndim != 5 or v.shape[1] != 24:
        hint = ""
        if torch.is_tensor(v) and v.ndim == 4:
            hint = " A 4D tensor is an image latent (no time axis): encode the clip with the MiniMax H3 video VAE instead."
        elif torch.is_tensor(v) and v.ndim == 5:
            hint = " Channel count must be 24 (MiniMax H3 video VAE); was the latent made by another model's VAE?"
        raise ValueError("%s: expected MiniMax H3 video latent [B,24,T,H/16,W/16], got %s.%s" % (what, describe(v), hint))
    t = v.shape[2]
    if t < 2 or (t - 2) % 5 != 0:
        raise ValueError("%s: H3 latent length must be 5k+2 (17k+5 pixel frames), got T=%d (%s). "
                         "Trim/pad the source clip to 22, 39, ..., 124, 141... frames before encoding." % (what, t, describe(v)))


def split_av(samples, fallback_audio=None):
    """Accept AV NestedTensor, (video, audio) list/tuple, or a plain video tensor.
    Returns (video, audio, audio_source)."""
    if isinstance(samples, dict):
        samples = samples.get("samples")
    if getattr(samples, "is_nested", False):
        parts = list(samples.unbind())
    elif isinstance(samples, (list, tuple)):
        parts = list(samples)
    elif torch.is_tensor(samples):
        parts = [samples]
    else:
        raise ValueError("latent: unsupported type %s" % describe(samples))
    if len(parts) == 0 or len(parts) > 2:
        raise ValueError("latent: expected 1 (video) or 2 (video, audio) tensors, got %s" % describe(samples))
    video = parts[0]
    _check_video(video, "latent")
    if len(parts) == 2:
        audio = parts[1]
        if not torch.is_tensor(audio) or audio.ndim != 4 or audio.shape[1] != 32:
            raise ValueError("latent: expected H3 audio latent [B,32,2,A], got %s" % describe(audio))
        return video, audio, "input"
    frames = sum(FRAME_PER_TOKEN[k % 5] for k in range(video.shape[2]))
    a_len = audio_len_for_frames(frames)
    if fallback_audio is not None and fallback_audio.shape[-1] == a_len and fallback_audio.shape[0] == video.shape[0]:
        return video, fallback_audio.to(video.device, video.dtype), "anchor_latent"
    # same as EmptyMiniMaxH3LatentAV / MiniMaxH3AV.fix_empty_latent: zero audio latent
    return video, torch.zeros((video.shape[0], 32, 2, a_len), dtype=video.dtype, device=video.device), "zeros"


def anchor_audio(anchor_latent):
    if anchor_latent is None:
        return None
    try:
        _, a, src = split_av(anchor_latent)
        return a if src == "input" else None
    except ValueError:
        return None


def to_av_latent(latent, video, audio):
    out = dict(latent)
    out["samples"] = comfy.nested_tensor.NestedTensor((video, audio))
    out.pop("noise_mask", None)
    return out


def make_anchor(video, anchor_latent):
    if anchor_latent is None:
        return video.clone()
    lo, _, _ = split_av(anchor_latent)
    if lo.shape[2] != video.shape[2]:
        raise ValueError(f"anchor latent has {lo.shape[2]} latent frames, upscaled latent has {video.shape[2]}")
    return torch.nn.functional.interpolate(lo.float(), size=video.shape[2:], mode="trilinear", align_corners=False).to(video.dtype)


def make_notifier(node_id):
    """send_sync to the frontend; silently no-op without a running PromptServer."""
    try:
        from server import PromptServer
        inst = getattr(PromptServer, "instance", None)
    except Exception:
        inst = None
    if inst is None:
        return None

    def notify(event, data):
        try:
            d = dict(data)
            d["node"] = str(node_id) if node_id is not None else None
            d["event"] = event
            inst.send_sync("h3wr.progress", d)
        except Exception:
            pass
    return notify


def run_windowed(model, positive, negative, latent, seed, steps, cfg, sampler_name, scheduler, denoise,
                 num_windows, window_blocks, overlap_blocks, alternate, temporal_guide, guide_from_step,
                 anchor_latent=None, anchor_strength=0.0, anchor_steps_frac=0.0,
                 shift_video=0.0, shift_audio=3.0, callback=None, log=None, verbose=True, audio_src="input", refs_count=0, emit=None, notify=None):
    video, audio, _ = split_av(latent, anchor_audio(anchor_latent))
    latent = to_av_latent(latent, video, audio)
    samples = latent["samples"]
    wins = plan_windows(video.shape[2], audio.shape[-1], num_windows, window_blocks, overlap_blocks)  # validate early
    if shift_video > 0:
        model = apply_shift(model, shift_video, shift_audio)
    model = model.clone()
    real_steps = steps  # KSampler with denoise<1 still runs `steps` steps
    wrapper = WindowedH3Wrapper(num_windows, window_blocks, overlap_blocks, alternate, temporal_guide, 1, guide_from_step,
                                anchor_video=make_anchor(video, anchor_latent) if anchor_strength > 0 else None,
                                anchor_strength=anchor_strength, anchor_steps=int(round(anchor_steps_frac * real_steps)), log=log)
    pbar = None
    try:
        pbar = comfy.utils.ProgressBar(steps * len(wins))
    except Exception:
        pass
    cuda_peak = None
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
            cuda_peak = torch.cuda.max_memory_allocated
        except Exception:
            pass
    reporter = Reporter(verbose, steps, len(wins), pbar, comfy.model_management.throw_exception_if_processing_interrupted, cuda_peak, emit, notify)
    wrapper.reporter = reporter
    try:
        ks = comfy.samplers.KSampler(model, steps=steps, device="cpu", sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)
        sig = ks.sigmas.tolist()
    except Exception as e:
        sig = []
    reporter.summary(video, audio, audio_src, wins, refs_count, steps, denoise, sig, cfg, sampler_name, scheduler,
                     f"{shift_video}/{shift_audio}" if shift_video > 0 else "model default",
                     f"{'on' if temporal_guide else 'off'}, from step {guide_from_step + 1} (0-based {guide_from_step})",
                     f"strength {anchor_strength}, {wrapper.anchor_steps} step(s) ({anchor_steps_frac} of {steps}), source {'anchor_latent' if anchor_latent is not None else 'input latent'}")
    model.set_model_unet_function_wrapper(wrapper)
    noise = comfy.sample.prepare_noise(samples, seed)
    out = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, samples,
                              denoise=denoise, callback=callback, disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)
    reporter.finish()
    res = latent.copy()
    res["samples"] = out
    return res


class H3WindowedLatentRefine:
    """Group order matters: the web extension draws section headers from WIDGET_GROUPS."""
    WIDGET_GROUPS = [
        ("Вход", ["prompt", "negative_prompt", "seed", "steps", "cfg", "sampler_name", "scheduler",
                  "shift_video", "shift_audio", "denoise"]),
        ("Окна", ["num_windows", "window_blocks", "overlap_blocks"]),
        ("Согласованность", ["alternate_direction", "temporal_guide", "guide_from_step"]),
        ("Якорь", ["anchor_strength", "anchor_steps"]),
        ("Выход", ["decode", "verbose"]),
        ("Превью", ["preview_frames"]),
    ]

    @classmethod
    def INPUT_TYPES(cls):
        S, K = comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS
        return {
            "required": {
                # --- входы
                "model": ("MODEL", {"tooltip": "Диффузионная модель MiniMax H3 Ref2VA (можно с turbo LoRA)."}),
                "clip": ("CLIP", {"tooltip": "Текстовый энкодер H3 (Qwen3-VL); кодирует промпт вместе с референсами."}),
                "vae": ("VAE", {"tooltip": "Видео-VAE H3: кодирует референсы и (опционально) декодирует результат."}),
                "latent": ("LATENT", {"tooltip": "Апскейленный латент H3: AV NestedTensor или только видео [B,24,T,H,W], T=5k+2."}),
                "prompt": ("STRING", {"multiline": True, "dynamic_prompts": True,
                                      "tooltip": "Промпт. Референсы — <Picture N>: нумерация с 1 по порядку подключённых входов (персонаж, локации, ref_4–6), внутри батча по порядку. Итоговая таблица печатается в лог и показывается на панели."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Негатив. При cfg = 1 не используется."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "Сид шума."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200, "tooltip": "Число реально выполняемых шагов (при denoise<1 тоже). Turbo LoRA Ref2VA: 4."}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": "CFG. Turbo LoRA: 1.0 (без негатива, вдвое быстрее)."}),
                "sampler_name": (S, {"default": "res_multistep" if "res_multistep" in S else "euler", "tooltip": "Сэмплер. Официально для H3: res_multistep."}),
                "scheduler": (K, {"default": "simple", "tooltip": "Расписание сигм. Официально для H3: simple."}),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Flow shift видео (H3: 12). 0 — не трогать модель."}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.1, "tooltip": "Flow shift аудио (H3: 3)."}),
                "denoise": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Доля перешума. При shift 12: 0.2 → старт с σ≈0.75, 0.4 → σ≈0.89 (почти перегенерация)."}),
                # --- окна
                "num_windows": ("INT", {"default": 2, "min": 0, "max": 64, "tooltip": "Число временных окон. 0 — длину задаёт «Длина окна»."}),
                "window_blocks": ("INT", {"default": 0, "min": 0, "max": 200, "tooltip": "m: длина окна 5·m+2 латент-кадров (используется, если окон = 0)."}),
                "overlap_blocks": ("INT", {"default": 2, "min": 0, "max": 100, "tooltip": "o: перекрытие не меньше 5·o+2 латент-кадров (1 → 7, 2 → 12)."}),
                # --- согласованность
                "alternate_direction": ("BOOLEAN", {"default": True, "tooltip": "Чередовать порядок окон по шагам: 1→N, затем N→1."}),
                "temporal_guide": ("BOOLEAN", {"default": True, "tooltip": "Передавать окну кадр соседа из зоны перекрытия как keyframe H3."}),
                "guide_from_step": ("INT", {"default": 1, "min": 0, "max": 200, "tooltip": "С какого шага (с 0) включать guide. 1 — со второго шага."}),
                # --- якорь
                "anchor_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Сила подмешивания x0 к якорю на первых шагах (затухает)."}),
                "anchor_steps": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Доля шагов с якорем: round(доля × steps)."}),
                # --- выход
                "decode": ("BOOLEAN", {"default": False, "tooltip": "Декодировать в IMAGE через VAE (иначе IMAGE — заглушка 64×64)."}),
                "verbose": ("BOOLEAN", {"default": True, "tooltip": "Подробный лог [H3WR] в консоли."}),
                # --- только для UI, на сэмплирование не влияет
                "preview_frames": ("INT", {"default": 0, "min": 0, "max": 3600, "step": 1,
                                           "tooltip": "Только для превью раскладки окон в UI (на генерацию не влияет). 0 — по последнему запуску; иначе число кадров ролика (ближайшее 17k+5)."}),
            },
            "optional": {
                "character_ref": ("IMAGE", {"tooltip": "Референс персонажа. Батч → несколько <Picture N>. Нумерация сквозная по порядку входов, итог — в логе и на панели."}),
                "location_ref": ("IMAGE", {"tooltip": "Референс локации (батч допустим)."}),
                "location_ref_2": ("IMAGE", {"tooltip": "Доп. референс локации (батч допустим)."}),
                "location_ref_3": ("IMAGE", {"tooltip": "Доп. референс локации (батч допустим)."}),
                "ref_4": ("IMAGE", {"tooltip": "Любой доп. референс (предмет, стиль…), батч допустим."}),
                "ref_5": ("IMAGE", {"tooltip": "Любой доп. референс, батч допустим."}),
                "ref_6": ("IMAGE", {"tooltip": "Любой доп. референс, батч допустим. Всего не больше 9 картинок."}),
                "anchor_latent": ("LATENT", {"tooltip": "Исходный низкоразрешённый латент H3: якорь и источник аудио."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_NAMES = ("latent", "images")
    OUTPUT_TOOLTIPS = ("AV-латент H3 (видео + аудио).", "Декодированные кадры (если decode включён).")
    DESCRIPTION = "Апскейл-рефайн латента MiniMax H3 перекрывающимися временными окнами, которые идут через сэмплер синхронно, шаг за шагом."
    RETURN_TYPES = ("LATENT", "IMAGE")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Refine"

    def run(self, model, clip, vae, prompt, negative_prompt, latent, seed, steps, cfg, sampler_name, scheduler,
            shift_video, shift_audio, denoise, num_windows, window_blocks, overlap_blocks, alternate_direction,
            temporal_guide, guide_from_step, anchor_strength, anchor_steps, decode, verbose=True,
            character_ref=None, location_ref=None, anchor_latent=None, unique_id=None,
            location_ref_2=None, location_ref_3=None, ref_4=None, ref_5=None, ref_6=None, preview_frames=0):
        del preview_frames  # UI-only
        import latent_preview
        video, audio, src = split_av(latent, anchor_audio(anchor_latent))
        if verbose:
            print("[H3WR] received latent=%s anchor=%s -> video %s, audio %s (audio from %s)" % (
            describe(latent), describe(anchor_latent) if anchor_latent is not None else None,
            list(video.shape), list(audio.shape), src), flush=True)
        latent = to_av_latent(latent, video, audio)
        width, height = video.shape[4] * 16, video.shape[3] * 16
        refs, mapping = collect_refs({"character_ref": character_ref, "location_ref": location_ref,
                                      "location_ref_2": location_ref_2, "location_ref_3": location_ref_3,
                                      "ref_4": ref_4, "ref_5": ref_5, "ref_6": ref_6})
        notify = make_notifier(unique_id)
        lines = mapping_lines(mapping)
        if verbose:
            print("[H3WR] references: " + ("none" if not lines else ""), flush=True)
            for l in lines:
                print("[H3WR]   " + l, flush=True)
        if notify:
            notify("refs", {"map": lines})
        positive = encode_ref2va(clip, vae, prompt, refs, width, height)
        negative = encode_ref2va(clip, vae, negative_prompt, refs, width, height)
        cb = latent_preview.prepare_callback(model, steps)
        out = run_windowed(model, positive, negative, latent, seed, steps, cfg, sampler_name, scheduler, denoise,
                           num_windows, window_blocks, overlap_blocks, alternate_direction, temporal_guide, guide_from_step,
                           anchor_latent, anchor_strength, anchor_steps, shift_video, shift_audio, callback=cb,
                           verbose=verbose, audio_src=src, refs_count=len(refs),
                           notify=notify)
        if decode:
            v = out["samples"].tensors[0] if getattr(out["samples"], "is_nested", False) else out["samples"]
            img = vae.decode(v)
            img = img.reshape((-1,) + tuple(img.shape[-3:]))
        else:
            img = torch.zeros((1, 64, 64, 3))
        return (out, img)


NODE_CLASS_MAPPINGS = {"H3WindowedLatentRefine": H3WindowedLatentRefine}
NODE_DISPLAY_NAME_MAPPINGS = {"H3WindowedLatentRefine": "MiniMax H3 · Windowed Latent Refine"}
