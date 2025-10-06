"""CLI command to prepare datasets, label maps and ontology masks using bundled packs."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import typer

from ..reporting import generate_dataset_reports
from ..utils.dataset_prep import (
    build_mask_and_report,
    create_or_load_label_maps,
    prepare_classification_dataset,
    prepare_dataset_simple,
    prepare_mlm_corpus,
    save_datasets,
)
from ..config import get_settings

__all__ = [
    "ConversionConfig",
    "ConversionArtifacts",
    "run_conversion",
    "convert_command",
    "build_arg_parser",
    "main",
]

# ---------------------------------------------------------------------
# Defaults: prefer the versioned pack/ bundle (with fallback to the configured data directory)
# ---------------------------------------------------------------------

_SETTINGS = get_settings()
_PACK_ROOT = _SETTINGS.pack_dir
_DATA_PROPERTIES_DIR = (_SETTINGS.data_dir / "properties").resolve()

_REQUIRED_REGISTRY_CANDIDATES = (
    "properties_registry_extended.json",
    "properties_registry.json",
    "registry.json",
)
_REQUIRED_EXTRACTORS_CANDIDATES = (
    "extractors_extended.json",
    "extractors.json",
)


def _find_first_existing(base: Path, candidates: Sequence[str]) -> Optional[Path]:
    for name in candidates:
        p = base / name
        if p.exists():
            return p
    return None


def _iter_pack_bases() -> Iterable[Path]:
    env_current = os.getenv("ROBIMB_PACK_CURRENT")
    if env_current:
        yield Path(env_current)
    current = _PACK_ROOT / "current"
    if current.exists():
        yield current
    if _PACK_ROOT.exists():
        for version_dir in sorted(_PACK_ROOT.glob("v*"), reverse=True):
            yield version_dir
    yield _DATA_PROPERTIES_DIR


def _resolve_registry_path() -> Path:
    for base in _iter_pack_bases():
        if base.is_file() and base.name.endswith(".json"):
            if base.name == "pack.json":
                # Inline bundle, let RegistryLoader handle it later.
                return base
            if base.name in _REQUIRED_REGISTRY_CANDIDATES:
                return base
            if base.name == "registry.json":
                return base
        if base.is_dir():
            candidate = _find_first_existing(base, _REQUIRED_REGISTRY_CANDIDATES)
            if candidate is not None:
                return candidate
            generic = base / "registry.json"
            if generic.exists():
                return generic
    raise FileNotFoundError(
        "Impossibile individuare un registry JSON nel pack distribuito o nella directory dati configurata."
    )


def _resolve_extractors_path() -> Path:
    for base in _iter_pack_bases():
        if base.is_file() and base.name in ("extractors.json", "extractors_extended.json", "pack.json"):
            return base
        if base.is_dir():
            candidate = _find_first_existing(base, _REQUIRED_EXTRACTORS_CANDIDATES)
            if candidate is not None:
                return candidate
            pack_json = base / "pack.json"
            if pack_json.exists():
                return pack_json
    raise FileNotFoundError(
        "Impossibile individuare un extractors JSON nel pack distribuito o nella directory dati configurata."
    )

DEFAULT_PROPERTIES_REGISTRY: Path = _resolve_registry_path()
DEFAULT_EXTRACTORS_PACK: Path = _resolve_extractors_path()


# ---------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ConversionConfig:
    """Configuration for dataset conversion leveraging the distributed pack."""

    train_file: Path
    val_file: Optional[Path]
    ontology: Optional[Path]
    label_maps: Path
    out_dir: Path
    done_uids: Optional[Path] = None
    val_split: float = 0.2
    random_state: int = 42

    # MLM/TAPT
    make_mlm_corpus: bool = False
    mlm_output: Optional[Path] = None
    extra_mlm: Sequence[Path] = ()
    reports_dir: Optional[Path] = None

    # Properties / Extractors: resolved from the bundled pack (with legacy fallback)
    properties_registry: Path = DEFAULT_PROPERTIES_REGISTRY
    extractors_pack: Path = DEFAULT_EXTRACTORS_PACK

    # Text column
    text_field: str = "text"

    # Property extraction (deprecated - use `extract properties` instead)
    extract_properties: bool = False

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


# ---------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------

def _validate_inputs_exist(config: ConversionConfig) -> None:
    missing: list[Tuple[str, Path]] = []
    for label, p in (
        ("train_file", config.train_file),
        ("label_maps (dir parent)", config.label_maps.parent),
        ("out_dir", config.out_dir),
        ("properties_registry", config.properties_registry),
        ("extractors_pack", config.extractors_pack),
    ):
        if label in ("label_maps (dir parent)", "out_dir"):
            # For directories we only ensure parent exists / can be created later
            continue
        if p is not None and not p.exists():
            missing.append((label, p))
    if missing:
        lines = "\n".join(f"- {k}: {v}" for k, v in missing)
        raise FileNotFoundError(f"I seguenti path non esistono:\n{lines}")

def run_conversion(config: ConversionConfig) -> ConversionArtifacts:
    """Execute the conversion pipeline using the bundled pack for extraction."""

    _validate_inputs_exist(config)
    config.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Label maps
    label_maps_path = config.label_maps
    ontology_path = config.ontology
    if not label_maps_path.exists() and ontology_path is None:
        raise ValueError("Manca l'ontologia: è necessaria per costruire le label maps ex novo.")

    label_maps = create_or_load_label_maps(
        label_maps_path, ontology_path=ontology_path
    )

    # 2) Dataset prep (classification only, property extraction moved to `extract` command)
    if config.extract_properties:
        # Legacy mode with property extraction
        train_df, val_df, label_maps = prepare_classification_dataset(
            config.train_file,
            config.val_file,
            label_maps_path=label_maps_path,
            ontology_path=ontology_path,
            done_uids_path=config.done_uids,
            val_split=config.val_split,
            random_state=config.random_state,
            properties_registry_path=config.properties_registry,
            extractors_pack_path=config.extractors_pack,
            text_field=config.text_field,
        )
    else:
        # New mode: just normalize dataset without extraction
        train_df, val_df, label_maps = prepare_dataset_simple(
            config.train_file,
            config.val_file,
            label_maps_path=label_maps_path,
            ontology_path=ontology_path,
            done_uids_path=config.done_uids,
            val_split=config.val_split,
            random_state=config.random_state,
            text_field=config.text_field,
        )

    save_datasets(train_df, val_df, config.out_dir)

    # 3) Ontology masks
    mask_matrix, mask_report = build_mask_and_report(ontology_path, label_maps)
    mask_matrix_path = config.out_dir / "mask_matrix.npy"
    mask_report_path = config.out_dir / "mask_report.json"
    np.save(mask_matrix_path, mask_matrix)
    with open(mask_report_path, "w", encoding="utf-8") as handle:
        json.dump(mask_report, handle, indent=2, ensure_ascii=False)

    # 4) Dump label maps (freeze what we used)
    label_maps_dump = config.out_dir / "label_maps.json"
    with open(label_maps_dump, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "super2id": label_maps.super_name_to_id,
                "cat2id": label_maps.cat_name_to_id,
                "id2super": label_maps.super_id_to_name,
                "id2cat": label_maps.cat_id_to_name,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    # 5) Reports
    reports_dir = config.reports_dir or (config.out_dir / "reports")
    generate_dataset_reports(
        train_df,
        val_df,
        super_id_to_name=label_maps.super_id_to_name,
        cat_id_to_name=label_maps.cat_id_to_name,
        output_dir=reports_dir,
    )

    # 6) (Optional) MLM/TAPT corpus
    mlm_corpus_path: Optional[Path] = None
    if config.make_mlm_corpus:
        if config.mlm_output is None:
            raise SystemExit("--mlm-output è obbligatorio quando usi --make-mlm-corpus")
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

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert raw AEC/BIM data relying on the bundled knowledge pack for property extraction."
    )
    parser.add_argument("--train-file", required=True, help="Path to the raw training jsonl file")
    parser.add_argument("--val-file", default=None, help="Optional validation jsonl file")
    parser.add_argument("--ontology", default=None, help="Ontology json file")
    parser.add_argument("--label-maps", required=True, help="Path where label maps should be stored")
    parser.add_argument("--out-dir", required=True, help="Directory that will receive processed datasets")
    parser.add_argument("--done-uids", default=None, help="Optional text file listing UIDs to skip")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio if no val file is provided")
    parser.add_argument("--random-state", type=int, default=42)

    # MLM/TAPT
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

    # Hard-wired defaults (still overridable, resolved from the bundled pack)
    parser.add_argument(
        "--properties-registry",
        default=str(DEFAULT_PROPERTIES_REGISTRY),
        help="Registry JSON resolved from the distributed pack (auto-resolved)",
    )
    parser.add_argument(
        "--extractors-pack",
        default=str(DEFAULT_EXTRACTORS_PACK),
        help="Extractors JSON resolved from the distributed pack (auto-resolved)",
    )

    parser.add_argument("--text-field", default="text", help="Column name for the textual description")
    return parser


def convert_command(
    train_file: Path = typer.Option(..., "--train-file", exists=True, dir_okay=False, help="Path to raw training JSONL"),
    val_file: Optional[Path] = typer.Option(None, "--val-file", exists=True, dir_okay=False, help="Optional validation JSONL"),
    ontology: Optional[Path] = typer.Option(None, "--ontology", exists=True, dir_okay=False, help="Ontology JSON mapping"),
    label_maps: Path = typer.Option(..., "--label-maps", dir_okay=False, help="Output path for generated label maps"),
    out_dir: Path = typer.Option(..., "--out-dir", help="Directory that will receive processed data"),
    done_uids: Optional[Path] = typer.Option(None, "--done-uids", exists=True, dir_okay=False, help="Text file listing UIDs to skip"),
    val_split: float = typer.Option(0.2, "--val-split", min=0.0, max=0.5, help="Validation ratio when --val-file is missing"),
    random_state: int = typer.Option(42, "--random-state", help="Random seed used for deterministic splits"),
    make_mlm_corpus: bool = typer.Option(False, "--make-mlm-corpus", help="Produce MLM/TAPT corpus"),
    mlm_output: Optional[Path] = typer.Option(None, "--mlm-output", help="Corpus output path when --make-mlm-corpus is set"),
    extra_mlm: Optional[List[Path]] = typer.Option(
        None,
        "--extra-mlm",
        help="Additional JSONL files contributing text to the MLM corpus",
        metavar="PATH",
        show_default=False,
    ),
    reports_dir: Optional[Path] = typer.Option(
        None,
        "--reports-dir",
        help="Directory where dataset plots and summary files will be saved",
    ),
    properties_registry: Optional[Path] = typer.Option(
        DEFAULT_PROPERTIES_REGISTRY,
        "--properties-registry",
        exists=True,
        dir_okay=False,
        help="Optional registry JSON or knowledge pack containing property schemas",
    ),
    extractors_pack: Optional[Path] = typer.Option(
        DEFAULT_EXTRACTORS_PACK,
        "--extractors-pack",
        exists=True,
        dir_okay=False,
        help="Knowledge pack or extractors JSON used to auto-populate property values",
    ),
    text_field: str = typer.Option(
        "text",
        "--text-field",
        help="Column containing the textual description analysed for property extraction",
    ),
    extract_properties: bool = typer.Option(
        False,
        "--extract-properties/--no-extract-properties",
        help="Extract properties using legacy system (deprecated - use `robimb extract properties` instead)",
    ),
) -> None:
    """Typer entrypoint that proxies to :func:`run_conversion`."""

    config = ConversionConfig(
        train_file=train_file,
        val_file=val_file,
        ontology=ontology,
        label_maps=label_maps,
        out_dir=out_dir,
        done_uids=done_uids,
        val_split=val_split,
        random_state=random_state,
        make_mlm_corpus=make_mlm_corpus,
        mlm_output=mlm_output,
        extra_mlm=tuple(extra_mlm or ()),
        reports_dir=reports_dir,
        properties_registry=properties_registry or DEFAULT_PROPERTIES_REGISTRY,
        extractors_pack=extractors_pack or DEFAULT_EXTRACTORS_PACK,
        text_field=text_field,
        extract_properties=extract_properties,
    )
    artifacts = run_conversion(config)
    typer.echo(json.dumps(artifacts.as_dict(), indent=2, ensure_ascii=False))


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
        properties_registry=Path(args.properties_registry) if args.properties_registry else DEFAULT_PROPERTIES_REGISTRY,
        extractors_pack=Path(args.extractors_pack) if args.extractors_pack else DEFAULT_EXTRACTORS_PACK,
        text_field=args.text_field,
    )

    artifacts = run_conversion(config)
    print(json.dumps(artifacts.as_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
