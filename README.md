# ComfyUI Fish Audio S2 Pro TTS

ComfyUI custom node for [Fish Audio S2 Pro](https://huggingface.co/fishaudio/s2-pro) (fishaudio/s2-pro). This node acts as a **client** to the Fish Speech API server: inference runs on the server; the node sends requests and returns the generated audio as ComfyUI AUDIO.

**Prerequisite:** The [Fish Speech](https://github.com/fishaudio/fish-speech) API server must be running with the S2 Pro model before using this node.

## Prerequisite: Fish Speech API server

1. **Download the model**
   - **From ComfyUI:** Add node **audio → tts → Download Fish Audio S2 Pro Model**, leave `local_dir` empty (uses `checkpoints/s2-pro` under your ComfyUI checkpoints folder), then run the workflow. The node outputs the path where the model was saved; use that path when starting the API server below.
   - **From CLI:**
   ```bash
   pip install huggingface_hub[cli]
   hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro
   ```

2. **Set up Fish Speech**
   - Clone [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) and install (see [Local Model Setup](https://docs.fish.audio/developer-guide/self-hosting/local-setup)).
   - Example: `pip install -e .[cu121]` (use the CUDA variant that matches your environment).

3. **Start the API server**
   ```bash
   cd fish-speech
   python -m tools.api_server --listen 0.0.0.0:8080 --llama-checkpoint-path checkpoints/s2-pro --decoder-checkpoint-path checkpoints/s2-pro/codec.pth
   ```
   For S2 Pro, confirm the correct `--decoder-config-name` in the [Fish Speech README](https://github.com/fishaudio/fish-speech) or docs if needed. Optional: add `--compile` for faster inference, or `--half` for fp16 on GPUs without bf16.

4. **Check health**
   ```bash
   curl -X GET http://127.0.0.1:8080/v1/health
   ```

## Installation

1. **Add the node to ComfyUI's custom_nodes**
   Clone this repo into your ComfyUI `custom_nodes` directory **as** `comfyui_fish_audio_s2` (so ComfyUI loads it correctly):
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/msandroid/comfyui-fish-audio-s2.git comfyui_fish_audio_s2
   ```
   Example (Stability Matrix): use `C:\StabilityMatrix-win-x64\Data\Packages\ComfyUI\custom_nodes\` as the target.

2. **Install Python dependencies**
   In the ComfyUI environment (ComfyUI venv or Stability Matrix package venv):
   ```bash
   pip install -r comfyui_fish_audio_s2/requirements.txt
   ```
   Example (Stability Matrix, PowerShell):
   ```powershell
   & "C:\StabilityMatrix-win-x64\Data\Packages\ComfyUI\venv\Scripts\pip.exe" install -r "C:\StabilityMatrix-win-x64\Data\Packages\ComfyUI\custom_nodes\comfyui_fish_audio_s2\requirements.txt"
   ```

3. **Restart ComfyUI**
   The nodes appear under **Add Node → audio → tts**: **Fish Audio S2 Pro TTS** and **Download Fish Audio S2 Pro Model**.

## Nodes

### Download Fish Audio S2 Pro Model

- **local_dir** (optional): Directory where the model will be downloaded. If empty, uses `checkpoints/s2-pro` under ComfyUI's checkpoints folder (or `~/checkpoints/s2-pro` if that is not available).
- **Output:** `model_path` (STRING) — the absolute path to the downloaded model. Use this path for `--llama-checkpoint-path` and `--decoder-checkpoint-path` when starting the Fish Speech API server.

The model is downloaded from [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro) on first run; existing files are reused on subsequent runs.

### Fish Audio S2 Pro TTS

- **text** (required): Text to synthesize. Multiline supported. S2 Pro supports inline control with `[tag]` (e.g. `[whisper]`, `[pause]`, `[laughing]`). See the [model card](https://huggingface.co/fishaudio/s2-pro) for supported tags.
- **api_url** (optional): Base URL of the Fish Speech API server. Default: `http://127.0.0.1:8080`.
- **reference_audio** (optional): ComfyUI AUDIO. When connected, used as the reference voice for voice cloning. When unconnected, the server uses its default (e.g. random voice).
- **reference_text** (optional): Transcription (or description) of the reference audio. Used together with **reference_audio** for voice cloning.
- **language** (optional): Language hint (Auto, Japanese, English, etc.). Kept for future API support; the server may auto-detect language.

**Output:** One **AUDIO** output (typically 44.1 kHz mono from S2 Pro), compatible with other ComfyUI audio nodes (e.g. Save Audio, or downstream video/audio workflows).

You can use the same **Load Reference Audio** node as for Chatterbox TTS (or any node that outputs AUDIO) and connect it to **reference_audio**.

## Model and license

- Model: [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro) (Fish Audio S2 Pro).
- License: Fish Audio Research License. Research and non-commercial use are permitted free of charge. Commercial use requires a separate license from Fish Audio; contact business@fish.audio.
