"""Smoke tests for the Typer CLI router."""
import json

from typer.testing import CliRunner

from robimb.cli.main import app

runner = CliRunner()


def test_help_shows_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("convert", "extract", "evaluate", "pack"):
        assert command in result.stdout


def test_version_option() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "robimb" in result.stdout


def test_convert_extract_evaluate_pack_help() -> None:
    for cmd in ("convert", "extract", "evaluate", "pack"):
        result = runner.invoke(app, [cmd, "--help"])
        assert result.exit_code == 0, f"help for {cmd} should be accessible"


def test_sample_categories_command(tmp_path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    rows = [
        {"text": "Parete A", "super": "Strutture", "cat": "Pareti"},
        {"text": "Parete B", "super": "Strutture", "cat": "Pareti"},
        {"text": "Solaio", "super": "Strutture", "cat": "Solai"},
    ]
    with dataset.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    output = tmp_path / "sampled.jsonl"
    result = runner.invoke(
        app,
        [
            "sample-categories",
            "--dataset",
            str(dataset),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    content = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 2

    parsed = [json.loads(line) for line in content]
    categories = {entry["cat"] for entry in parsed}
    assert categories == {"Pareti", "Solai"}
    assert parsed[0]["text"] == "Parete A"
