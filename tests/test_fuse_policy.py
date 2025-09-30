from robimb.extraction.fuse import Candidate, Fuser, FusePolicy


def _accept(candidate: Candidate):
    return True, []


def test_fuser_prefers_highest_confidence() -> None:
    fuser = Fuser(policy=FusePolicy.VALIDATE_THEN_MAX_CONF)
    candidates: list[Candidate] = [
        {
            "value": 42,
            "source": "parser",
            "raw": "42",
            "span": (0, 2),
            "confidence": 0.88,
            "errors": [],
        },
        {
            "value": 42,
            "source": "qa_llm",
            "raw": "42",
            "span": (0, 2),
            "confidence": 0.92,
            "errors": [],
        },
    ]
    winner = fuser.fuse(candidates, _accept)
    assert winner["source"] == "qa_llm"


def test_fuser_uses_source_priority_on_ties() -> None:
    fuser = Fuser(policy=FusePolicy.VALIDATE_THEN_MAX_CONF)
    candidates: list[Candidate] = [
        {
            "value": "A",
            "source": "parser",
            "raw": "A",
            "span": (0, 1),
            "confidence": 0.9,
            "errors": [],
        },
        {
            "value": "A",
            "source": "qa_llm",
            "raw": "A",
            "span": (0, 1),
            "confidence": 0.9,
            "errors": [],
        },
    ]
    winner = fuser.fuse(candidates, _accept)
    assert winner["source"] == "parser"
