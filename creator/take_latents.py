"""Persistent sampler latents for generated takes.

These are project assets, not entries in :mod:`creator.latents`' disposable
reference cache.  A take sidecar sits beside its movie under output/, has the
same stem, and is named in the take report so a saved Timeline can find it
after a restart.

Only H3 joint audio/video latents are stored.  Safetensors cannot represent a
``NestedTensor`` directly, so the two streams are flattened into named tensors
and rebuilt on load.  Writes are staged and atomically renamed: losing a
sidecar must make a bridge fall back to frames, never hand sampling a partial
tensor.
"""

import json
import os
import uuid

SCHEMA = 1
SUFFIX = ".latent.safetensors"


def sidecar_name(video_filename):
    """A take movie filename -> its sidecar filename in the same directory."""
    stem, _extension = os.path.splitext(video_filename)
    return stem + SUFFIX


def _streams(latent):
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if samples is None:
        raise ValueError("the take has no sampler samples")
    if getattr(samples, "is_nested", False):
        streams = tuple(samples.unbind())
    else:
        streams = (samples,)
    if not streams or any(getattr(stream, "ndim", 0) < 3 for stream in streams):
        raise ValueError("the take's sampler output is not an H3 latent")
    return streams


def save(path, latent, *, frames, width, height, fps):
    """Write ``latent`` atomically to ``path``.  Raises on an unusable take."""
    from safetensors.torch import save_file

    streams = _streams(latent)
    tensors = {
        ("video" if index == 0 else "audio" if index == 1 else f"stream_{index}"):
        stream.detach().to("cpu").contiguous()
        for index, stream in enumerate(streams)
    }
    meta = {
        "schema": SCHEMA,
        "family": "h3",
        "frames": int(frames),
        "width": int(width),
        "height": int(height),
        "fps": float(fps),
        "streams": list(tensors),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    staging = os.path.join(os.path.dirname(path), f".{uuid.uuid4().hex}.tmp")
    try:
        save_file(tensors, staging, metadata={"continuity_take": json.dumps(meta, sort_keys=True)})
        os.replace(staging, path)
    finally:
        try:
            os.remove(staging)
        except FileNotFoundError:
            pass
    return meta


def load(path, vae=None):
    """Load and validate one H3 take sidecar -> ``(LATENT, metadata)``."""
    from safetensors.torch import load_file

    meta = header(path)
    tensors = load_file(path, device="cpu")
    names = list(meta.get("streams") or ())
    if not names or names[0] != "video" or any(name not in tensors for name in names):
        raise ValueError("the take latent is missing its video stream")

    device = getattr(vae, "output_device", None)
    dtype = vae.vae_output_dtype() if vae is not None and hasattr(vae, "vae_output_dtype") else None
    streams = [tensors[name].to(device=device or tensors[name].device,
                                dtype=dtype or tensors[name].dtype)
               for name in names]
    if len(streams) == 1:
        samples = streams[0]
    else:
        import comfy.nested_tensor

        samples = comfy.nested_tensor.NestedTensor(tuple(streams))
    return {"samples": samples}, meta


def header(path):
    """Read and validate only a sidecar's small safetensors header."""
    from safetensors import safe_open

    with safe_open(path, framework="pt") as handle:
        raw = (handle.metadata() or {}).get("continuity_take")
    meta = json.loads(raw) if raw else {}
    if meta.get("schema") != SCHEMA or meta.get("family") != "h3":
        raise ValueError("this is not a compatible Continuity H3 take latent")
    for key in ("frames", "width", "height"):
        if int(meta.get(key) or 0) <= 0:
            raise ValueError(f"the take latent has no valid {key}")
    if float(meta.get("fps") or 0) <= 0:
        raise ValueError("the take latent has no valid fps")
    return meta


def resolve(filename):
    """Resolve an annotated ComfyUI filename, requiring a real sidecar."""
    import folder_paths

    path = folder_paths.get_annotated_filepath(filename)
    if not path or not os.path.isfile(path):
        raise ValueError(f"the take latent is missing: {filename}")
    return path
