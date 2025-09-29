import argparse
from pathlib import Path

from ..cli.extract import extract_command
from ..cli.pack import pack_command


def main() -> None:
    parser = argparse.ArgumentParser(prog="robimb", description="Registry/Extractors converter")
    sub = parser.add_subparsers(dest="cmd", required=True)

    cmd_extract = sub.add_parser("convert", help="Monolith (registry+extractors) -> properties folders")
    cmd_extract.add_argument("--in-registry", required=True, type=Path)
    cmd_extract.add_argument("--in-extractors", required=True, type=Path)
    cmd_extract.add_argument("--out-dir", required=True, type=Path)

    cmd_pack = sub.add_parser("pack", help="properties folders -> monolith")
    cmd_pack.add_argument("--in-dir", required=True, type=Path)
    cmd_pack.add_argument("--out-registry", required=True, type=Path)
    cmd_pack.add_argument("--out-extractors", required=True, type=Path)

    args = parser.parse_args()
    if args.cmd == "convert":
        extract_command(
            in_registry=args.in_registry,
            in_extractors=args.in_extractors,
            out_dir=args.out_dir,
        )
    else:
        pack_command(
            properties_root=args.in_dir,
            out_registry=args.out_registry,
            out_extractors=args.out_extractors,
        )