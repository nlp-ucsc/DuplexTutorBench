"""CLI entry point for generating simulated student–tutor conversations."""

import argparse
import base64
import io
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

from src.conversation import simulate_conversation
from src.data import load_mathvista
from src.models import OpenAIChatModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate simulated student–tutor conversations from MathVista."
    )
    parser.add_argument(
        "--split", default="testmini", help="MathVista split (default: testmini)"
    )
    parser.add_argument(
        "--n", type=int, default=None, help="Number of questions to process"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Model name (default: Qwen/Qwen3-Omni-30B-A3B-Instruct)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="API base URL (default: http://100.116.140.1:8901/v1 for Qwen, omitted for OpenAI)",
    )
    parser.add_argument(
        "--max-turns", type=int, default=10, help="Max turns per conversation"
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Name for this run (creates output/{run_name}/ with conversations.jsonl and audio/)",
    )
    parser.add_argument(
        "--tutor-prompt",
        default="prompts/tutor_system.txt",
        help="Path to tutor system prompt template",
    )
    parser.add_argument(
        "--student-prompt",
        default="prompts/student_system.txt",
        help="Path to student system prompt template",
    )
    parser.add_argument(
        "--image",
        action="store_true",
        help="Include question image in conversation",
    )
    parser.add_argument(
        "--audio",
        action="store_true",
        help="Enable audio generation (uses omni server instead of vLLM)",
    )
    parser.add_argument(
        "--omni-url",
        default="http://100.116.140.1:8902",
        help="Omni server URL (default: http://100.116.140.1:8902)",
    )
    parser.add_argument(
        "--speaker",
        default="Chelsie",
        choices=["Ethan", "Chelsie", "Aiden"],
        help="Voice for audio generation (default: Chelsie)",
    )
    parser.add_argument(
        "--audio-history",
        action="store_true",
        help="Pass generated audio as conversation history instead of text (requires --audio)",
    )
    parser.add_argument(
        "--tutor-speaker",
        default="Ethan",
        choices=["Ethan", "Chelsie", "Aiden"],
        help="Tutor voice when --audio-history is set (default: Ethan)",
    )
    parser.add_argument(
        "--student-speaker",
        default="Chelsie",
        choices=["Ethan", "Chelsie", "Aiden"],
        help="Student voice when --audio-history is set (default: Chelsie)",
    )
    return parser.parse_args()


def encode_image_base64(image: Image.Image) -> str:
    """Encode a PIL Image as a base64 data URI string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    load_dotenv()
    args = parse_args()

    if args.audio_history and not args.audio:
        raise SystemExit("Error: --audio-history requires --audio")

    # Load prompt templates
    tutor_prompt = Path(args.tutor_prompt).read_text()
    student_prompt = Path(args.student_prompt).read_text()

    # Set up output directory: output/{run_name}/
    run_dir = Path("output") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "conversations.jsonl"

    # Set up image directory
    image_dir = None
    if args.image:
        image_dir = run_dir / "image"
        image_dir.mkdir(parents=True, exist_ok=True)

    # Build model(s) and determine audio directory
    audio_dir = None
    if args.audio:
        from src.audio import OmniChatModel

        audio_dir = run_dir / "audio"
        if args.audio_history:
            # Separate models with different speaker voices
            tutor_model = OmniChatModel(
                base_url=args.omni_url,
                speaker=args.tutor_speaker,
                audio_dir=audio_dir,
            )
            student_model = OmniChatModel(
                base_url=args.omni_url,
                speaker=args.student_speaker,
                audio_dir=audio_dir,
            )
        else:
            tutor_model = student_model = OmniChatModel(
                base_url=args.omni_url,
                speaker=args.speaker,
                audio_dir=audio_dir,
            )
    else:
        # Infer base_url for the default Qwen model when not explicitly provided
        base_url = args.base_url
        is_default_qwen = "Qwen" in args.model and base_url is None
        if is_default_qwen:
            base_url = "http://100.116.140.1:8901/v1"

        # Local/remote servers don't need an API key; OpenAI does.
        api_key = "no-key-needed" if base_url else os.getenv("OPENAI_API_KEY")

        tutor_model = student_model = OpenAIChatModel(
            model=args.model,
            api_key=api_key,
            base_url=base_url,
        )

    # Load questions
    questions = load_mathvista(split=args.split, n=args.n)

    logger.info(
        "Generating conversations for %d questions → %s", len(questions), output_path
    )

    with open(output_path, "w") as f:
        for i, question in enumerate(questions):
            try:
                # Encode and save image if requested
                image_base64 = None
                image_path = ""
                if image_dir and question.image is not None:
                    question.image.save(image_dir / f"{question.pid}.png")
                    image_base64 = encode_image_base64(question.image)
                    image_path = f"image/{question.pid}.png"

                conversation = simulate_conversation(
                    question=question,
                    tutor_model=tutor_model,
                    student_model=student_model,
                    tutor_prompt_template=tutor_prompt,
                    student_prompt_template=student_prompt,
                    max_turns=args.max_turns,
                    audio_dir=audio_dir,
                    image_base64=image_base64,
                    image_path=image_path,
                    audio_history=args.audio_history,
                )
                f.write(json.dumps(conversation.to_dict()) + "\n")
                f.flush()
                logger.info(
                    "[%d/%d] pid=%s — %d turns",
                    i + 1,
                    len(questions),
                    question.pid,
                    conversation.num_turns,
                )
            except Exception:
                logger.exception(
                    "[%d/%d] pid=%s — failed, skipping.",
                    i + 1,
                    len(questions),
                    question.pid,
                )

    logger.info("Done. Output written to %s", output_path)


if __name__ == "__main__":
    main()
