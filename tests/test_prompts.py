from pathlib import Path

import pytest

from robimb.extraction.prompts import PromptLibrary, load_prompt_library


def test_default_library_loads(tmp_path: Path) -> None:
    library = load_prompt_library()
    prompt = library.render("property_question", text="ABC", question="Domanda?", schema="{}")
    assert "ABC" in prompt
    assert "Domanda?" in prompt


def test_custom_library(tmp_path: Path) -> None:
    payload = '{"custom": {"description": "", "template": "Value={{value}}"}}'
    path = tmp_path / "prompts.json"
    path.write_text(payload, encoding="utf-8")

    library = PromptLibrary.from_path(path)
    rendered = library.render("custom", value="ok")
    assert rendered == "Value=ok"

    with pytest.raises(KeyError):
        library.render("missing")
