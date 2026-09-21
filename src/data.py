"""MathVista dataset loading utilities."""

import logging

from datasets import load_dataset

from src.schemas import MathVistaQuestion

logger = logging.getLogger(__name__)


def load_mathvista(
    split: str = "testmini", n: int | None = None
) -> list[MathVistaQuestion]:
    """Load questions from the MathVista dataset.

    Args:
        split: Dataset split to load ("testmini" or "test").
        n: If set, only return the first n questions (useful for dev/testing).
    """
    logger.info("Loading MathVista split=%s ...", split)
    ds = load_dataset("AI4Math/MathVista", split=split)

    questions = []
    for row in ds:
        q = MathVistaQuestion(
            pid=str(row["pid"]),
            question=row["question"],
            answer=str(row["answer"]),
            question_type=row.get("question_type", ""),
            answer_type=row.get("answer_type", ""),
            choices=row.get("choices", []),
            image=row.get("decoded_image") or row.get("image"),
            query=row.get("query", row["question"]),
            metadata={
                k: v
                for k, v in row.items()
                if k
                not in {
                    "pid",
                    "question",
                    "answer",
                    "question_type",
                    "answer_type",
                    "choices",
                    "image",
                    "decoded_image",
                    "query",
                }
            },
        )
        questions.append(q)
        if n is not None and len(questions) >= n:
            break

    logger.info("Loaded %d questions.", len(questions))
    return questions
