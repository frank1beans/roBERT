from robimb.extraction.fuse import Candidate, CandidateSource, Fuser, FusePolicy


def _accept(candidate: Candidate):
    return True, []


def test_fuser_prefers_highest_confidence() -> None:
    fuser = Fuser(policy=FusePolicy.VALIDATE_THEN_MAX_CONF)
    candidates: list[Candidate] = [
        {
            "value": 42,
            "source": CandidateSource.PARSER,
            "raw": "42",
            "span": (0, 2),
            "confidence": 0.88,
            "errors": [],
        },
        {
            "value": 42,
            "source": CandidateSource.QA_LLM,
            "raw": "42",
            "span": (0, 2),
            "confidence": 0.92,
            "errors": [],
        },
    ]
    winner = fuser.fuse(candidates, _accept)
    assert winner["source"] == CandidateSource.QA_LLM


def test_fuser_uses_source_priority_on_ties() -> None:
    fuser = Fuser(policy=FusePolicy.VALIDATE_THEN_MAX_CONF)
    candidates: list[Candidate] = [
        {
            "value": "A",
            "source": CandidateSource.PARSER,
            "raw": "A",
            "span": (0, 1),
            "confidence": 0.9,
            "errors": [],
        },
        {
            "value": "A",
            "source": CandidateSource.QA_LLM,
            "raw": "A",
            "span": (0, 1),
            "confidence": 0.9,
            "errors": [],
        },
    ]
    winner = fuser.fuse(candidates, _accept)
    assert winner["source"] == CandidateSource.PARSER


def test_fuser_handles_matcher_fallback_source() -> None:
    fuser = Fuser(policy=FusePolicy.VALIDATE_THEN_MAX_CONF)
    candidates: list[Candidate] = [
        {
            "value": "Brand",
            "source": CandidateSource.MATCHER_FALLBACK,
            "raw": None,
            "span": None,
            "confidence": 0.4,
            "errors": [],
        },
        {
            "value": "Manual Brand",
            "source": CandidateSource.MANUAL,
            "raw": None,
            "span": None,
            "confidence": 0.4,
            "errors": [],
        },
    ]
    winner = fuser.fuse(candidates, _accept)
    assert winner["source"] == CandidateSource.MANUAL
