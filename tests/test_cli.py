"""Smoke tests for the consolidated Typer CLI."""
from typer.testing import CliRunner

from robimb.cli.main import app

runner = CliRunner()


def test_help_shows_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "convert" in result.stdout
    assert "train" in result.stdout
    assert "validate" in result.stdout
    assert "tapt" in result.stdout


def test_version_option() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "robimb" in result.stdout
