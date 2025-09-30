from pathlib import Path

from robimb.extraction.pack_testing import run_pack_dataset_evaluation


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_run_pack_dataset_evaluation(tmp_path) -> None:
    dataset = _repo_root() / "data" / "train" / "classification" / "raw" / "dataset.jsonl"
    pack_path = _repo_root() / "pack" / "v1_limited"

    summary = run_pack_dataset_evaluation(
        dataset_path=dataset,
        output_dir=tmp_path,
        pack_path=pack_path,
        limit=40,
        sample_size=5,
    )

    assert summary["processed_records"] > 0
    assert summary["artifacts"]["summary"]
    assert (tmp_path / "summary.json").exists()
    assert summary["artifacts"]["matched_examples"].endswith("matched_examples.jsonl")
    for generated in tmp_path.iterdir():
        assert generated.suffix != ".npy"
