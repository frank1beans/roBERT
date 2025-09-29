import argparse
from pathlib import Path
from .unpack import convert_monolith_to_folders
from ..registry import pack_folders_to_monolith

def main():
    p = argparse.ArgumentParser(prog="robimb", description="Registry/Extractors converter")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert", help="Monolith (registry+extractors) -> properties folders")
    c.add_argument("--in-registry", required=True, type=Path)
    c.add_argument("--in-extractors", required=True, type=Path)
    c.add_argument("--out-dir", required=True, type=Path)

    k = sub.add_parser("pack", help="properties folders -> monolith")
    k.add_argument("--in-dir", required=True, type=Path)
    k.add_argument("--out-registry", required=True, type=Path)
    k.add_argument("--out-extractors", required=True, type=Path)

    a = p.parse_args()
    if a.cmd == "convert":
        convert_monolith_to_folders(a.in_registry, a.in_extractors, a.out_dir)
        print(f"OK: properties written to {a.out_dir}")
    else:
        pack_folders_to_monolith(a.in_dir, a.out_registry, a.out_extractors)
        print(f"OK: registry -> {a.out_registry}")
        print(f"OK: extractors -> {a.out_extractors}")