"""
ComfyUI node for Fish Audio S2 Pro TTS via Fish Speech API server.
Calls POST /v1/tts; does not edit any third-party code.
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any

import numpy as np
import torch

try:
    import requests
    import soundfile as sf
    import librosa
except ImportError as e:
    raise ImportError(
        "comfyui_fish_audio_s2 requires requests, soundfile, and librosa. "
        "Install with: pip install -r requirements.txt"
    ) from e

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

HF_REPO_ID = "fishaudio/s2-pro"

S2_SAMPLE_RATE = 44100
DEFAULT_API_URL = "http://127.0.0.1:8080"
REQUEST_TIMEOUT = 300

# Tier 1 and Tier 2 + a few; for COMBO. API may accept more.
LANGUAGE_OPTIONS = [
    ("auto", "Auto"),
    ("ja", "Japanese"),
    ("en", "English"),
    ("zh", "Chinese"),
    ("ko", "Korean"),
    ("es", "Spanish"),
    ("pt", "Portuguese"),
    ("ar", "Arabic"),
    ("ru", "Russian"),
    ("fr", "French"),
    ("de", "German"),
]
LANGUAGE_COMBO_ITEMS = [name for _, name in LANGUAGE_OPTIONS]
LANGUAGE_DISPLAY_TO_CODE = {name: code for code, name in LANGUAGE_OPTIONS}


def _audio_to_wav_bytes(
    waveform: torch.Tensor,
    sample_rate: int,
    target_sr: int = S2_SAMPLE_RATE,
) -> bytes:
    """Convert ComfyUI AUDIO (waveform [B,C,T], sample_rate) to WAV bytes at target_sr."""
    if waveform.dim() == 3:
        waveform = waveform.squeeze(0)
    wav_np = waveform.detach().cpu().float().numpy()
    if wav_np.shape[0] > 1:
        wav_np = wav_np.mean(axis=0, keepdims=True)
    wav_np = np.squeeze(wav_np)
    if sample_rate != target_sr:
        wav_np = librosa.resample(
            wav_np, orig_sr=sample_rate, target_sr=target_sr, res_type="kaiser_fast"
        )
    buf = io.BytesIO()
    sf.write(buf, wav_np, target_sr, format="WAV")
    return buf.getvalue()


def _call_tts_api(
    api_url: str,
    text: str,
    references: list[dict[str, str]],
    format: str = "wav",
    streaming: bool = False,
    timeout: int = REQUEST_TIMEOUT,
) -> bytes:
    """POST /v1/tts and return raw response body (WAV bytes)."""
    url = api_url.rstrip("/") + "/v1/tts"
    payload: dict[str, Any] = {
        "text": text,
        "format": format,
        "streaming": streaming,
        "references": references,
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def _wav_bytes_to_audio(wav_bytes: bytes) -> tuple[torch.Tensor, int]:
    """Decode WAV bytes to (waveform [1,1,T], sample_rate)."""
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=1)
    waveform = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
    return waveform, int(sr)


def _default_model_dir() -> str:
    """Default directory for S2 Pro model (under ComfyUI checkpoints if available)."""
    try:
        import folder_paths as fp
        dirs = fp.get_folder_paths("checkpoints")
        if dirs:
            return os.path.join(dirs[0], "s2-pro")
    except Exception:
        pass
    return os.path.join(os.path.expanduser("~"), "checkpoints", "s2-pro")


class FishAudioS2DownloadModel:
    """Download Fish Audio S2 Pro model from Hugging Face when needed."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "local_dir": ("STRING", {"default": "", "placeholder": "Leave empty for default (checkpoints/s2-pro)"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "run"
    CATEGORY = "audio/tts"

    def run(self, local_dir: str = ""):
        if snapshot_download is None:
            raise RuntimeError(
                "FishAudioS2DownloadModel: huggingface_hub is required. "
                "Install with: pip install huggingface_hub"
            )
        target = (local_dir or "").strip()
        if not target:
            target = _default_model_dir()
        target = os.path.abspath(os.path.expanduser(target))
        os.makedirs(target, exist_ok=True)
        snapshot_download(repo_id=HF_REPO_ID, local_dir=target, local_dir_use_symlinks=False)
        return (target,)


class FishAudioS2TTSNode:
    """TTS node that calls Fish Speech API server (e.g. S2 Pro)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "api_url": ("STRING", {"default": DEFAULT_API_URL}),
                "reference_audio": ("AUDIO",),
                "reference_text": ("STRING", {"default": "", "multiline": True}),
                "language": (LANGUAGE_COMBO_ITEMS, {"default": "Auto"}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"
    CATEGORY = "audio/tts"

    def run(
        self,
        text: str,
        api_url: str = DEFAULT_API_URL,
        reference_audio: dict | None = None,
        reference_text: str = "",
        language: str = "Auto",  # Reserved for future API support
    ):
        if not text.strip():
            raise ValueError("FishAudioS2TTS: text must not be empty.")

        references: list[dict[str, str]] = []
        if reference_audio is not None and reference_audio.get("waveform") is not None:
            waveform = reference_audio["waveform"]
            sr = reference_audio["sample_rate"]
            wav_bytes = _audio_to_wav_bytes(waveform, sr)
            ref_b64 = base64.b64encode(wav_bytes).decode("ascii")
            ref_text = (reference_text or "").strip() or " "
            references.append({"audio": ref_b64, "text": ref_text})

        try:
            wav_bytes = _call_tts_api(
                api_url=api_url.strip() or DEFAULT_API_URL,
                text=text,
                references=references,
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"FishAudioS2TTS: API request failed. Is the Fish Speech server running at {api_url}? {e}"
            ) from e

        waveform_t, out_sr = _wav_bytes_to_audio(wav_bytes)
        return ({"waveform": waveform_t, "sample_rate": out_sr},)
