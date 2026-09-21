"""Turn-by-turn conversation simulation between student and tutor."""

import logging
from pathlib import Path

from src.models import ChatModel
from src.schemas import Conversation, MathVistaQuestion, Utterance

logger = logging.getLogger(__name__)

END_TOKEN = "[END]"


def _format_prompt(template: str, question: MathVistaQuestion) -> str:
    """Fill in template variables for a system prompt."""
    choices_block = ""
    if question.choices:
        choices_block = "Choices:\n" + "\n".join(
            f"  {chr(65 + i)}. {c}" for i, c in enumerate(question.choices)
        )

    return template.format(
        question=question.question,
        answer=question.answer,
        choices_block=choices_block,
    )


def _maybe_save_audio(
    model: ChatModel, audio_dir: Path | None, pid: str, turn: int
) -> str:
    """If model supports audio, save it and return a relative path."""
    if audio_dir and hasattr(model, "save_last_audio"):
        path = audio_dir / pid / f"turn_{turn}.wav"
        if model.save_last_audio(path):
            return f"audio/{pid}/turn_{turn}.wav"
    return ""


def _make_image_content(image_base64: str, text: str) -> list[dict]:
    """Build an OpenAI-style content array with an image and text."""
    return [
        {"type": "image_url", "image_url": {"url": image_base64}},
        {"type": "text", "text": text},
    ]


def _make_audio_content(audio_data_uri: str) -> list[dict]:
    """Build a content array with an audio block."""
    return [{"type": "audio", "audio": audio_data_uri}]


def _get_audio_uri(model: ChatModel) -> str:
    """Get audio data URI from model if it supports audio history."""
    if hasattr(model, "get_last_audio_data_uri"):
        return model.get_last_audio_data_uri()
    return ""


def simulate_conversation(
    question: MathVistaQuestion,
    tutor_model: ChatModel,
    student_model: ChatModel,
    tutor_prompt_template: str,
    student_prompt_template: str,
    max_turns: int = 10,
    audio_dir: Path | None = None,
    image_base64: str | None = None,
    image_path: str = "",
    audio_history: bool = False,
) -> Conversation:
    """Simulate a multi-turn student–tutor conversation about a question.

    The student speaks first. Each agent sees the other's messages as
    role="user" (natural ChatML framing). The conversation ends when
    either agent emits [END] or max_turns is reached.
    """
    tutor_system = _format_prompt(tutor_prompt_template, question)
    student_system = _format_prompt(student_prompt_template, question)

    # Message histories from each agent's perspective
    tutor_messages: list[dict] = [{"role": "system", "content": tutor_system}]
    student_messages: list[dict] = [{"role": "system", "content": student_system}]

    utterances: list[Utterance] = []
    turn = 0

    # If image is provided, inject it as the first user message for the student
    if image_base64:
        student_messages.append(
            {
                "role": "user",
                "content": _make_image_content(
                    image_base64,
                    "Here is the math problem. Look at the image and begin.",
                ),
            }
        )

    # Student opens the conversation
    student_reply = student_model.generate(student_messages)
    audio_path = _maybe_save_audio(student_model, audio_dir, question.pid, turn)
    utterances.append(
        Utterance(
            role="student", content=student_reply, turn=turn, audio_path=audio_path
        )
    )
    # Student sees own message as assistant; tutor sees it as user
    if audio_history:
        audio_uri = _get_audio_uri(student_model)
        student_messages.append(
            {"role": "assistant", "content": _make_audio_content(audio_uri)}
        )
        # Tutor sees audio, optionally with image
        if image_base64:
            tutor_messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_base64}},
                        {"type": "audio", "audio": audio_uri},
                    ],
                }
            )
        else:
            tutor_messages.append(
                {"role": "user", "content": _make_audio_content(audio_uri)}
            )
    else:
        student_messages.append({"role": "assistant", "content": student_reply})
        # Tutor sees the student's first reply — include image if available
        if image_base64:
            tutor_messages.append(
                {
                    "role": "user",
                    "content": _make_image_content(image_base64, student_reply),
                }
            )
        else:
            tutor_messages.append({"role": "user", "content": student_reply})
    turn += 1

    if END_TOKEN in student_reply:
        logger.info("pid=%s: student ended on first message.", question.pid)
        return _build_conversation(
            question, utterances, tutor_model, student_model, image_path
        )

    while turn < max_turns:
        # Tutor responds
        tutor_reply = tutor_model.generate(tutor_messages)
        audio_path = _maybe_save_audio(tutor_model, audio_dir, question.pid, turn)
        utterances.append(
            Utterance(
                role="tutor", content=tutor_reply, turn=turn, audio_path=audio_path
            )
        )
        if audio_history:
            audio_uri = _get_audio_uri(tutor_model)
            tutor_messages.append(
                {"role": "assistant", "content": _make_audio_content(audio_uri)}
            )
            student_messages.append(
                {"role": "user", "content": _make_audio_content(audio_uri)}
            )
        else:
            tutor_messages.append({"role": "assistant", "content": tutor_reply})
            student_messages.append({"role": "user", "content": tutor_reply})
        turn += 1

        if END_TOKEN in tutor_reply:
            logger.info("pid=%s: tutor ended at turn %d.", question.pid, turn)
            break

        if turn >= max_turns:
            break

        # Student responds
        student_reply = student_model.generate(student_messages)
        audio_path = _maybe_save_audio(student_model, audio_dir, question.pid, turn)
        utterances.append(
            Utterance(
                role="student", content=student_reply, turn=turn, audio_path=audio_path
            )
        )
        if audio_history:
            audio_uri = _get_audio_uri(student_model)
            student_messages.append(
                {"role": "assistant", "content": _make_audio_content(audio_uri)}
            )
            tutor_messages.append(
                {"role": "user", "content": _make_audio_content(audio_uri)}
            )
        else:
            student_messages.append({"role": "assistant", "content": student_reply})
            tutor_messages.append({"role": "user", "content": student_reply})
        turn += 1

        if END_TOKEN in student_reply:
            logger.info("pid=%s: student ended at turn %d.", question.pid, turn)
            break

    return _build_conversation(
        question, utterances, tutor_model, student_model, image_path
    )


def _build_conversation(
    question: MathVistaQuestion,
    utterances: list[Utterance],
    tutor_model: ChatModel,
    student_model: ChatModel,
    image_path: str = "",
) -> Conversation:
    return Conversation(
        pid=question.pid,
        question=question.question,
        answer=question.answer,
        tutor_model=getattr(tutor_model, "model", "unknown"),
        student_model=getattr(student_model, "model", "unknown"),
        image_path=image_path,
        utterances=utterances,
        metadata=question.metadata,
    )
