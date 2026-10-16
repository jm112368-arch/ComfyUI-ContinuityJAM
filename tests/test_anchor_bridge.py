"""Persistent H3 take latents and exact two-ended bridge overlaps.

    COMFYUI_PATH=~/ComfyUI <comfy-venv>/bin/python3 tests/test_anchor_bridge.py

CPU tensors only: no checkpoint, VAE or sampler is loaded.
"""

import os
from pathlib import Path
import tempfile
import sys

import layout
from harness import check, skip

comfy = Path(os.environ.get("COMFYUI_PATH", os.path.expanduser("~/ComfyUI")))
if not (comfy / "folder_paths.py").is_file():
    skip("set COMFYUI_PATH to a ComfyUI installation")
sys.path.insert(0, str(comfy))
sys.argv = ["test_anchor_bridge", "--cpu"]
import comfy.cli_args  # noqa: E402
comfy.cli_args.args.cpu = True
import torch  # noqa: E402
import comfy.nested_tensor  # noqa: E402

package = layout.load("take_latents", "timeline", package="anchor_bridge_contract")
take_latents = package.take_latents
timeline = package.timeline
encoder = layout.load("encode", package="anchor_bridge_encoder").encode

# The file is self-describing and bit-exact across a CPU round trip.
video = torch.randn(1, 24, 12, 4, 6, generator=torch.Generator().manual_seed(7))
with tempfile.TemporaryDirectory() as folder:
    path = os.path.join(folder, "shot.latent.safetensors")
    meta = take_latents.save(path, {"samples": video},
                             frames=39, width=96, height=64, fps=24)
    restored, header = take_latents.load(path)
    check("a take latent round-trips exactly",
          torch.equal(restored["samples"], video), True)
    check("a take latent records its alignment",
          [header[key] for key in ("family", "frames", "width", "height", "fps")],
          ["h3", 39, 96, 64, 24.0])
    check("the writer reports the same metadata it stored", meta, header)

# A following approved take contributes its opening run, never a VAE encode.
check("22 frames are the next take's first seven latent steps",
      torch.equal(encoder._ending_run(video, 22, 96, 64), video[:, :, :7]), True)
check("a next take at another canvas falls back to frames",
      encoder._ending_run(video, 22, 64, 64), None)

# Exact writeback changes only the two video overlaps. Audio stays on the
# existing seam path because its 40 Hz clock does not end on the video feather.
middle_video = torch.zeros(1, 24, 20, 4, 6)
middle_audio = torch.randn(1, 32, 2, 90)
left_video = torch.ones(1, 24, 12, 4, 6)
right_video = torch.full((1, 24, 12, 4, 6), 2.0)
audio_left = torch.randn_like(middle_audio)
audio_right = torch.randn_like(middle_audio)

def av(v, a):
    return {"samples": comfy.nested_tensor.NestedTensor((v, a))}

middle = av(middle_video, middle_audio)
out = timeline.ContinuityH3AnchorBridge.execute(
    middle, head_frames=22, tail_frames=22,
    prev_latent=av(left_video, audio_left),
    next_latent=av(right_video, audio_right)).result[0]
bridged_video, bridged_audio = out["samples"].unbind()
check("the previous take is exact at the bridge head",
      torch.equal(bridged_video[:, :, :7], left_video[:, :, -7:]), True)
check("the following take is exact at the bridge tail",
      torch.equal(bridged_video[:, :, -7:], right_video[:, :, :7]), True)
check("sampling owns the middle of the bridge",
      float(bridged_video[:, :, 7:-7].abs().sum()), 0.0)
check("anchor writeback leaves bridge audio untouched",
      torch.equal(bridged_audio, middle_audio), True)
check("exact writeback does not mutate the sampler output",
      float(middle_video.abs().sum()), 0.0)

try:
    timeline.ContinuityH3AnchorBridge.execute(
        av(torch.zeros(1, 24, 12, 4, 6), middle_audio),
        head_frames=22, tail_frames=22,
        prev_latent=av(left_video, audio_left),
        next_latent=av(right_video, audio_right))
    refused = False
except ValueError:
    refused = True
check("overlaps that leave no bridge are refused", refused, True)
