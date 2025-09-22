"""CLI command to prepare datasets, label maps and ontology masks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

from ..utils.data_utils import (
    build_mask_and_report,
    create_or_load_label_maps,
    prepare_classification_dataset,
    prepare_mlm_corpus,
    save_datasets,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert raw BIM data to training-ready artefacts")
    parser.add_argument("--train-file", required=True, help="Path to the raw training jsonl file")
    parser.add_argument("--val-file", default=None, help="Optional validation jsonl file")
    parser.add_argument("--ontology", required=True, help="Ontology json file")
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
        "--extra-mlm", nargs="*", default=[], help="Additional jsonl files whose text field should be added to the MLM corpus"
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    label_maps_path = Path(args.label_maps)
    ontology_path = Path(args.ontology)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    super_name_to_id, cat_name_to_id, super_id_to_name, cat_id_to_name = create_or_load_label_maps(
        label_maps_path, ontology_path=ontology_path
    )

    train_df, val_df, _, _ = prepare_classification_dataset(
        args.train_file,
        args.val_file,
        label_maps_path=label_maps_path,
        ontology_path=ontology_path,
        done_uids_path=args.done_uids,
        val_split=args.val_split,
        random_state=args.random_state,
    )

    save_datasets(train_df, val_df, out_dir)

    mask_matrix, mask_report = build_mask_and_report(ontology_path, super_name_to_id, cat_name_to_id)
    np.save(out_dir / "mask_matrix.npy", mask_matrix)
    with open(out_dir / "mask_report.json", "w", encoding="utf-8") as handle:
        json.dump(mask_report, handle, indent=2)

    with open(out_dir / "label_maps.json", "w", encoding="utf-8") as handle:
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

    if args.make_mlm_corpus:
        if not args.mlm_output:
            raise SystemExit("--mlm-output is required when --make-mlm-corpus is provided")
        mlm_sources = [args.train_file]
        if args.val_file:
            mlm_sources.append(args.val_file)
        mlm_sources.extend(args.extra_mlm)
        count = prepare_mlm_corpus(mlm_sources, args.mlm_output)
        print(f"[INFO] Salvate {count} frasi nel corpus MLM {args.mlm_output}")


if __name__ == "__main__":
    main()
