# Servers — turn-based mode backends

Two remote servers back `main.py`, depending on whether audio is requested. Both run on the lab GPU host (`ssh ucsc_lab_sv11`, 2× NVIDIA RTX 6000 Ada, 48 GB each). Duplex-mode backends (PersonaPlex, MoshiVis, GPT Realtime, Gemini Live) live separately — see [`../duplex/docs/backends.md`](../duplex/docs/backends.md).

## vLLM (text / text+image)

**Used when:** `--audio` is NOT set on `main.py` (text-only or `--image` without `--audio`).

- **URL from Mac:** `http://100.116.140.1:8901/v1`
- **Working folder on remote:** `~/repo/vllm_deployment/`
- **Start command (on remote):**
  ```bash
  cd ~/repo/vllm_deployment
  ~/.local/bin/uv run vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
      --port 8901 --dtype bfloat16 --max-model-len 65536 \
      --allowed-local-media-path / -tp 2
  ```

If the service isn't running when needed, SSH in and start it.

## Omni server (text+audio)

**Used when:** `--audio` is set on `main.py` (regardless of whether `--image` is also set).

- **URL from Mac:** `http://100.116.140.1:8902`
- **Working folder on remote:** `~/repo/mock_aitutor_server/`
- **Hardware:** needs both ~48 GB GPUs on the host — the model loads via `device_map="auto"` across available devices.
- **Server script:** `omni_server.py` (FastAPI, Qwen3-Omni text+audio via `transformers==4.57.0` — pinned to the version the model was built for).
- **Start command (on remote):**
  ```bash
  cd ~/repo/mock_aitutor_server
  uv sync                                    # first time only
  uv run python omni_server.py --port 8902
  ```
- **Endpoint:** `POST /v1/chat/completions` — accepts `{"messages": [...], "speaker": "Chelsie"}`, returns `{"text": "...", "audio_base64": "...", "sample_rate": 24000}`.
- **Health check:** `GET /health`.
- **Available voices:** `Ethan`, `Chelsie`, `Aiden`.

If the service isn't running when needed, SSH in and start it.

**Reference-copy warning:** the local `server/omni_server.py` is a reference copy only. The remote `~/repo/mock_aitutor_server/omni_server.py` is the source of truth — always SSH and check there before making changes.

## Cleanup

Kill remote server processes after testing or running tasks so other people can use the GPUs:

```bash
# vLLM
pkill -f 'vllm serve'

# Omni server
pkill -f 'omni_server.py'
```
