"""FastAPI server for Qwen3-Omni text+audio generation via transformers.

Runs on the remote GPU machine. Loads the full Qwen3-Omni model with talker
support, returning both text and base64-encoded WAV audio for each request.

Usage:
    python omni_server.py --port 8902

Note: This file is a local copy for reference. The deployed version lives on
the remote server at ~/repo/mock_aitutor_server/omni_server.py.

Required patches (transformers 4.57.0 bugs):
    All patches target the same file in the installed transformers package:
        .venv/.../transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py

    Patch 1 — Talker None hidden states (line ~3188):
        The talker's code predictor returns None hidden states on longer
        sequences, crashing audio generation.

        Before:
            mid_residual_hiddens = [hid[0].to(last_id_hidden.device) for hid in predictor_result.hidden_states[1:]]
        After:
            mid_residual_hiddens = [hid[0].to(last_id_hidden.device) for hid in predictor_result.hidden_states[1:] if hid is not None]

    Patch 2 — get_rope_index audio token count mismatch (line ~338 and ~382):
        Two bugs cause get_rope_index to compute the wrong number of audio
        position IDs, crashing with "RuntimeError: shape mismatch":

        (a) bfloat16 precision loss: when omni_server.py does
            inputs.to(model.dtype), feature_attention_mask is converted to
            bfloat16. Later, feature_attention_mask.sum() loses precision for
            lengths >256 (bfloat16 can only represent integers exactly up to
            2^8). This feeds wrong values into _get_feat_extract_output_lengths,
            producing wrong audio token counts. Fixed in omni_server.py by
            preserving feature_attention_mask as integer (see code below).

        (b) Talker segment misalignment: the model.generate() method passes
            audio_feature_lengths (computed from ALL audio segments) to the
            talker, but the talker skips system and non-last assistant segments.
            The talker's get_rope_index then uses sequential indices into
            audio_feature_lengths that don't match its actual audio segments.

        Both are fixed by replacing _get_feat_extract_output_lengths with direct
        token counting. Add this helper function before the
        Qwen3OmniMoePreTrainedModelForConditionalGeneration class:

            def _count_audio_tokens(input_tokens, start_pos, audio_token_id):
                count = 0
                pos = start_pos
                while pos < len(input_tokens) and input_tokens[pos] == audio_token_id:
                    count += 1
                    pos += 1
                return count

        Then replace both occurrences (in "Audio Only" and "Audio in Video"
        branches) of:
            audio_len = _get_feat_extract_output_lengths(audio_seqlens[audio_idx])
        With:
            audio_len = _count_audio_tokens(input_tokens, st + int(text_len + bos_len), audio_token_id)
"""

import argparse
import base64
import io
import logging
import re

import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Qwen3-Omni Server")

AVAILABLE_SPEAKERS = ["Ethan", "Chelsie", "Aiden"]
SAMPLE_RATE = 24000

# Populated at startup
model = None
processor = None


class ChatRequest(BaseModel):
    messages: list[dict]
    speaker: str = "Chelsie"


class ChatResponse(BaseModel):
    text: str
    audio_base64: str
    sample_rate: int = SAMPLE_RATE


def _openai_to_qwen_messages(messages: list[dict]) -> list[dict]:
    """Convert OpenAI image_url content arrays to Qwen image format."""
    converted = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            new_parts = []
            for part in content:
                if part.get("type") == "image_url" and "image_url" in part:
                    new_parts.append(
                        {"type": "image", "image": part["image_url"]["url"]}
                    )
                else:
                    new_parts.append(part)
            converted.append({**msg, "content": new_parts})
        else:
            converted.append(msg)
    return converted


def _load_image_from_data_uri(data_uri: str) -> Image.Image:
    """Decode a base64 data URI to a PIL Image."""
    # Strip the data:image/...;base64, prefix
    match = re.match(r"data:[^;]+;base64,(.*)", data_uri, re.DOTALL)
    if match:
        image_data = base64.b64decode(match.group(1))
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    raise ValueError(f"Invalid data URI: {data_uri[:50]}...")


def _extract_images(messages: list[dict]) -> list[Image.Image]:
    """Extract PIL Images from Qwen-format messages."""
    images = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "image":
                    images.append(_load_image_from_data_uri(part["image"]))
    return images


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions", response_model=ChatResponse)
def chat_completions(request: ChatRequest):
    if request.speaker not in AVAILABLE_SPEAKERS:
        return {"error": f"Unknown speaker. Choose from {AVAILABLE_SPEAKERS}"}

    qwen_messages = _openai_to_qwen_messages(request.messages)
    text_input = processor.apply_chat_template(
        qwen_messages, tokenize=False, add_generation_prompt=True
    )
    audios, images, _videos = process_mm_info(qwen_messages, use_audio_in_video=False)
    inputs = processor(
        text=text_input,
        images=images or None,
        audio=audios or None,
        return_tensors="pt",
    )
    # Preserve feature_attention_mask as integer before dtype conversion —
    # bfloat16 loses precision on mask sums >256, causing get_rope_index
    # to compute wrong audio position IDs and crash with a shape mismatch.
    feature_mask = inputs.get("feature_attention_mask")
    inputs = inputs.to(model.device).to(model.dtype)
    if feature_mask is not None:
        inputs["feature_attention_mask"] = feature_mask.to(model.device)
    input_len = inputs["input_ids"].shape[1]

    text_ids, audio = model.generate(
        **inputs,
        speaker=request.speaker,
        thinker_return_dict_in_generate=True,
        use_audio_in_video=False,
        return_audio=True,
    )

    # Decode only the generated tokens (skip the input prompt tokens)
    response_text = processor.batch_decode(
        text_ids.sequences[:, input_len:], skip_special_tokens=True
    )[0]

    # Convert audio tensor to base64-encoded WAV
    audio_base64 = ""
    if audio is not None:
        audio_np = audio.reshape(-1).detach().cpu().float().numpy()
        buf = io.BytesIO()
        sf.write(buf, audio_np, SAMPLE_RATE, format="WAV")
        audio_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return ChatResponse(
        text=response_text,
        audio_base64=audio_base64,
        sample_rate=SAMPLE_RATE,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-Omni text+audio server")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Model name or path",
    )
    parser.add_argument("--port", type=int, default=8902, help="Server port")
    parser.add_argument(
        "--device", default="auto", help="Device for model (default: auto)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logger.info("Loading model %s ...", args.model)
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    logger.info("Model loaded on %s", model.device)

    uvicorn.run(app, host="0.0.0.0", port=args.port)
