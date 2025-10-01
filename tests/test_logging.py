import json
from pathlib import Path

from robimb.utils.logging import configure_json_logger, flush_handlers, log_event


def test_structured_logger_emits_jsonl(tmp_path: Path) -> None:
    log_file = tmp_path / "events.jsonl"
    logger = configure_json_logger(log_file)

    trace_id = log_event(logger, "test.start", input="in.jsonl")
    log_event(logger, "test.completed", trace_id=trace_id, documents=2)
    flush_handlers(logger)

    lines = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(lines) == 2
    assert all(line["trace_id"] == trace_id for line in lines)
    assert {line["event"] for line in lines} == {"test.start", "test.completed"}
    assert lines[0]["input"] == "in.jsonl"
    assert lines[1]["documents"] == 2
