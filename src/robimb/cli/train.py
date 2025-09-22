"""Main entry point for model training commands."""
from __future__ import annotations

import argparse
import sys

from ..training.hier_trainer import (
    HierTrainingArgs,
    build_arg_parser as build_hier_parser,
    train_hier_model,
)
from ..training.label_trainer import (
    LabelTrainingArgs,
    build_arg_parser as build_label_parser,
    train_label_model,
)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train BIM NLP models")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("label", help="Train the label embedding model")
    subparsers.add_parser("hier", help="Train the hierarchical masked model")

    args, remaining = parser.parse_known_args(argv)
    if args.command == "label":
        label_ns = build_label_parser().parse_args(remaining)
        train_label_model(LabelTrainingArgs(**vars(label_ns)))
    elif args.command == "hier":
        hier_ns = build_hier_parser().parse_args(remaining)
        train_hier_model(HierTrainingArgs(**vars(hier_ns)))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
