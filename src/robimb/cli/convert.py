"""CLI command to prepare datasets, label maps and ontology masks."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

import numpy as np

from ..reporting import generate_dataset_reports
from ..utils.data_utils import (
    build_mask_and_report,
    create_or_load_label_maps,
    prepare_classification_dataset,
    prepare_mlm_corpus,
    save_datasets,
)

__all__ = [
    "ConversionConfig",
    "ConversionArtifacts",
    "run_conversion",
    "build_arg_parser",
    "main",
]


@dataclass(frozen=True)
class ConversionConfig:
    """Configuration for dataset conversion."""

    train_file: Path
    val_file: Optional[Path]
    ontology: Optional[Path]
    label_maps: Path
    out_dir: Path
    done_uids: Optional[Path] = None
    val_split: float = 0.2
    random_state: int = 42
    make_mlm_corpus: bool = False
    mlm_output: Optional[Path] = None
    extra_mlm: Sequence[Path] = ()
    reports_dir: Optional[Path] = None

    def iter_mlm_sources(self) -> Iterable[Path]:
        if not self.make_mlm_corpus:
            return []
        sources = [self.train_file]
        if self.val_file is not None:
            sources.append(self.val_file)
        sources.extend(self.extra_mlm)
        return sources


@dataclass(frozen=True)
class ConversionArtifacts:
    """Paths produced by :func:`run_conversion`."""

    train_dataset: Path
    val_dataset: Path
    label_maps: Path
    mask_matrix: Path
    mask_report: Path
    mlm_corpus: Optional[Path] = None
    reports_dir: Optional[Path] = None

    def as_dict(self) -> Mapping[str, str]:
        payload = {
            "train_dataset": str(self.train_dataset),
            "val_dataset": str(self.val_dataset),
            "label_maps": str(self.label_maps),
            "mask_matrix": str(self.mask_matrix),
            "mask_report": str(self.mask_report),
        }
        if self.mlm_corpus is not None:
            payload["mlm_corpus"] = str(self.mlm_corpus)
        if self.reports_dir is not None:
            payload["reports_dir"] = str(self.reports_dir)
        return payload


def run_conversion(config: ConversionConfig) -> ConversionArtifacts:
    """Execute the conversion pipeline using the provided configuration."""

    config.out_dir.mkdir(parents=True, exist_ok=True)

    label_maps_path = config.label_maps
    ontology_path = config.ontology
    if not label_maps_path.exists() and ontology_path is None:
        raise ValueError("An ontology must be provided to build label maps from scratch")

    super_name_to_id, cat_name_to_id, super_id_to_name, cat_id_to_name = create_or_load_label_maps(
        label_maps_path, ontology_path=ontology_path
    )

    train_df, val_df, _, _ = prepare_classification_dataset(
        config.train_file,
        config.val_file,
        label_maps_path=label_maps_path,
        ontology_path=ontology_path,
        done_uids_path=config.done_uids,
        val_split=config.val_split,
        random_state=config.random_state,
    )

    save_datasets(train_df, val_df, config.out_dir)

    mask_matrix, mask_report = build_mask_and_report(ontology_path, super_name_to_id, cat_name_to_id)
    mask_matrix_path = config.out_dir / "mask_matrix.npy"
    mask_report_path = config.out_dir / "mask_report.json"
    np.save(mask_matrix_path, mask_matrix)
    with open(mask_report_path, "w", encoding="utf-8") as handle:
        json.dump(mask_report, handle, indent=2)

    label_maps_dump = config.out_dir / "label_maps.json"
    with open(label_maps_dump, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "super2id": super_name_to_id,
                "cat2id": cat_name_to_id,
                "id2super": super_id_to_name,
                "id2cat": cat_id_to_name,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    reports_dir = config.reports_dir or (config.out_dir / "reports")
    generate_dataset_reports(
        train_df,
        val_df,
        super_id_to_name=super_id_to_name,
        cat_id_to_name=cat_id_to_name,
        output_dir=reports_dir,
    )

    mlm_corpus_path: Optional[Path] = None
    if config.make_mlm_corpus:
        if config.mlm_output is None:
            raise SystemExit("--mlm-output is required when --make-mlm-corpus is provided")
        mlm_sources = [str(path) for path in config.iter_mlm_sources()]
        count = prepare_mlm_corpus(mlm_sources, config.mlm_output)
        print(f"[INFO] Salvate {count} frasi nel corpus MLM {config.mlm_output}")
        mlm_corpus_path = config.mlm_output

    return ConversionArtifacts(
        train_dataset=config.out_dir / "train_processed.jsonl",
        val_dataset=config.out_dir / "val_processed.jsonl",
        label_maps=label_maps_dump,
        mask_matrix=mask_matrix_path,
        mask_report=mask_report_path,
        mlm_corpus=mlm_corpus_path,
        reports_dir=reports_dir,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert raw BIM data to training-ready artefacts")
    parser.add_argument("--train-file", required=True, help="Path to the raw training jsonl file")
    parser.add_argument("--val-file", default=None, help="Optional validation jsonl file")
    parser.add_argument("--ontology", default=None, help="Ontology json file")
    parser.add_argument("--label-maps", required=True, help="Path where label maps should be stored")
    parser.add_argument("--out-dir", required=True, help="Directory that will receive processed datasets")
    parser.add_argument("--done-uids", default=None, help="Optional text file listing UIDs to skip")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio if no val file is provided")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--make-mlm-corpus",
        action="store_true",
        help="Create a concatenated text corpus for MLM/TAPT pre-training",
    )
    parser.add_argument(
        "--mlm-output",
        default=None,
        help="Target path for the MLM corpus (required when --make-mlm-corpus is used)",
    )
    parser.add_argument(
        "--extra-mlm",
        nargs="*",
        default=[],
        help="Additional jsonl files whose text field should be added to the MLM corpus",
    )
    parser.add_argument(
        "--reports-dir",
        default=None,
        help="Directory where dataset reports and visualizations will be stored",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = ConversionConfig(
        train_file=Path(args.train_file),
        val_file=Path(args.val_file) if args.val_file else None,
        ontology=Path(args.ontology) if args.ontology else None,
        label_maps=Path(args.label_maps),
        out_dir=Path(args.out_dir),
        done_uids=Path(args.done_uids) if args.done_uids else None,
        val_split=args.val_split,
        random_state=args.random_state,
        make_mlm_corpus=args.make_mlm_corpus,
        mlm_output=Path(args.mlm_output) if args.mlm_output else None,
        extra_mlm=[Path(p) for p in args.extra_mlm],
        reports_dir=Path(args.reports_dir) if args.reports_dir else None,
    )

    artifacts = run_conversion(config)
    summary = json.dumps(artifacts.as_dict(), indent=2)
    print(summary)


if __name__ == "__main__":
    main()
