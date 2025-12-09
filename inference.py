"""Z-Image PyTorch Native Inference."""

import os
import time
import warnings

import torch
import numpy as np

warnings.filterwarnings("ignore")
from utils import (
    AttentionBackend,
    ensure_model_weights,
    load_from_local_dir,
    set_attention_backend,
)
from zimage import generate


def _banner(msg: str) -> None:
    print(f"\n========== {msg} ==========")


def main():
    model_path = ensure_model_weights(
        "ckpts/Z-Image-Turbo", verify=False
    )  # True to verify with md5
    dtype = torch.bfloat16
    compile = False  # default False for compatibility
    output_path = "example.png"
    height = 1024
    width = 1024
    num_inference_steps = 8
    guidance_scale = 0.0
    seed = np.random.randint(0, 10000)

    # Pick attention backend (env wins; default flash on CUDA, math elsewhere)
    attn_backend = os.environ.get("ZIMAGE_ATTENTION")
    if attn_backend is None:
        attn_backend = "_native_flash" if torch.cuda.is_available() else "_native_math"
    prompt = (
        "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. "
        "Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. "
        "Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, "
        "silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
    )

    # Device selection priority: cuda -> tpu -> mps -> cpu
    if torch.cuda.is_available():
        device = "cuda"
        _banner("Chosen device: cuda")
    else:
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm

            device = xm.xla_device()
            _banner("Chosen device: tpu")
        except (ImportError, RuntimeError):
            if torch.backends.mps.is_available():
                device = "mps"
                _banner("Chosen device: mps")
            else:
                device = "cpu"
                _banner("Chosen device: cpu")
    # Load models
    components = load_from_local_dir(
        model_path, device=device, dtype=dtype, compile=compile
    )
    AttentionBackend.print_available_backends()
    set_attention_backend(attn_backend)
    _banner(f"Attention backend: {attn_backend}")

    # Gen an image; if a kernel is unavailable, fall back to math backend automatically
    try:
        start_time = time.time()
        images = generate(
            prompt=prompt,
            **components,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device).manual_seed(seed),
        )
        end_time = time.time()
    except RuntimeError as e:
        if "No available kernel" in str(e) or "Triton" in str(e):
            fallback_backend = "_native_math"
            print(
                f"Attention backend {attn_backend} failed ({e}); retrying with {fallback_backend}."
            )
            set_attention_backend(fallback_backend)
            start_time = time.time()
            images = generate(
                prompt=prompt,
                **components,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device).manual_seed(seed),
            )
            end_time = time.time()
        else:
            raise

    _banner(f"Image saved to {output_path} | Time: {end_time - start_time:.2f} s")
    images[0].save(output_path)

    ### !! For best speed performance, recommend to use `_flash_3` backend and set `compile=True`
    ### This would give you sub-second generation speed on Hopper GPU (H100/H200/H800) after warm-up


if __name__ == "__main__":
    main()
