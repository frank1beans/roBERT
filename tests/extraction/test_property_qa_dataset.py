from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from robimb.extraction.property_qa import PropertyQADataset, QAExample


def _build_test_tokenizer(tmp_path: Path) -> PreTrainedTokenizerFast:
    vocab = {
        "[PAD]": 0,
        "[CLS]": 1,
        "[SEP]": 2,
        "[UNK]": 3,
        "parete": 4,
        "in": 5,
        "cartongesso": 6,
        "spessore": 7,
        "12,5": 8,
        "mm": 9,
    }
    tokenizer_backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer_json = tmp_path / "tokenizer.json"
    tokenizer_backend.save(str(tokenizer_json))

    config = {
        "model_max_length": 256,
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
    }
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json),
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        model_max_length=256,
    )
    return tokenizer


def test_property_qa_dataset_handles_offsets(tmp_path) -> None:
    tokenizer = _build_test_tokenizer(tmp_path)
    examples = [
        QAExample(
            qid="ex1",
            context="parete in cartongesso spessore 12,5 mm",
            question="Nel capitolato qual è lo spessore?",
            answers=["12,5 mm"],
            answer_starts=[30],
            property_id="spessore_mm",
        )
    ]

    dataset = PropertyQADataset(examples, tokenizer, max_length=32, doc_stride=8)

    assert len(dataset) == 1
    feature = dataset.features[0]
    assert feature["example_id"] == 0
    assert feature["offset_mapping"]

    item = dataset[0]
    assert "input_ids" in item
    assert "start_positions" in item and item["start_positions"] >= 0
    assert "end_positions" in item and item["end_positions"] >= item["start_positions"]
