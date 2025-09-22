
from __future__ import annotations
import json, random, re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from .io import read_jsonl, write_jsonl

@dataclass
class ConvertStats:
    total: int = 0
    kept: int = 0
    dropped_empty: int = 0
    dropped_missing_labels: int = 0
    deduped: int = 0
    num_super: int = 0
    num_cat: int = 0

def _clean_text(s: str, lowercase: bool=False) -> str:
    s = s.replace("\n"," ").replace("\r"," ")
    s = re.sub(r"\s+"," ", s).strip()
    return s.lower() if lowercase else s

def _build_label_index(rows: List[dict]) -> Tuple[Dict[str,int], Dict[str,int]]:
    supers, cats = [], []
    for r in rows:
        s = r.get("super","").strip()
        c = r.get("cat","").strip()
        if s and s not in supers: supers.append(s)
        if c and c not in cats: cats.append(c)
    supers.sort(key=lambda x: x.lower())
    cats.sort(key=lambda x: x.lower())
    return {s:i for i,s in enumerate(supers)}, {c:i for i,c in enumerate(cats)}

def _stratified_split(rows: List[dict], label_key: str, seed: int=42, ratios=(0.8,0.1,0.1)) -> Tuple[List[int],List[int],List[int]]:
    random.seed(seed)
    by_label: Dict[int, List[int]] = {}
    for i, r in enumerate(rows):
        by_label.setdefault(r[label_key], []).append(i)
    tr, va, te = [], [], []
    for lab, idxs in by_label.items():
        random.shuffle(idxs)
        n = len(idxs)
        n_tr = int(n*ratios[0]); n_val = int(n*ratios[1])
        tr += idxs[:n_tr]
        va += idxs[n_tr:n_tr+n_val]
        te += idxs[n_tr+n_val:]
    return tr, va, te

def convert_jsonl_to_training(src_jsonl: str, out_root: str, seed: int=42, lowercase: bool=False, dedupe: bool=True, min_len: int=2) -> dict:
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    stats = ConvertStats()

    # Read and clean
    raw_rows = list(read_jsonl(src_jsonl))
    stats.total = len(raw_rows)

    rows: List[dict] = []
    seen = set()
    for r in raw_rows:
        txt = _clean_text(str(r.get("text","")), lowercase=lowercase)
        sup = str(r.get("super","")).strip()
        cat = str(r.get("cat","")).strip()
        if not txt or len(txt.split()) < min_len:
            stats.dropped_empty += 1; continue
        if not sup or not cat:
            stats.dropped_missing_labels += 1; continue
        key = (txt, sup, cat)
        if dedupe and key in seen:
            stats.deduped += 1; continue
        seen.add(key)
        rows.append({"text": txt, "super": sup, "cat": cat})

    stats.kept = len(rows)

    if not rows:
        raise RuntimeError("Nessuna riga valida dopo pulizia.")

    # Label index
    super2id, cat2id = _build_label_index(rows)
    stats.num_super, stats.num_cat = len(super2id), len(cat2id)

    # Enrich with ids
    enriched = []
    for r in rows:
        enriched.append({
            "text": r["text"],
            "super": r["super"], "cat": r["cat"],
            "super_id": super2id[r["super"]],
            "cat_id": cat2id[r["cat"]]
        })

    # Splits stratificati su cat_id
    tr, va, te = _stratified_split(enriched, "cat_id", seed=seed)

    # Output dirs
    (out/"processed/classification/super").mkdir(parents=True, exist_ok=True)
    (out/"processed/classification/cat").mkdir(parents=True, exist_ok=True)
    (out/"processed/hier").mkdir(parents=True, exist_ok=True)
    (out/"processed/mlm").mkdir(parents=True, exist_ok=True)
    (out/"metadata/label_index").mkdir(parents=True, exist_ok=True)
    (out/"metadata/splits").mkdir(parents=True, exist_ok=True)

    # Label index files
    with open(out/"metadata/label_index/super.json","w",encoding="utf-8") as f:
        json.dump({"label2id": super2id, "id2label": {str(v):k for k,v in super2id.items()}}, f, ensure_ascii=False, indent=2)
    with open(out/"metadata/label_index/cat.json","w",encoding="utf-8") as f:
        json.dump({"label2id": cat2id, "id2label": {str(v):k for k,v in cat2id.items()}}, f, ensure_ascii=False, indent=2)

    # Splits file
    with open(out/"metadata/splits/split_seed{}.json".format(seed),"w",encoding="utf-8") as f:
        json.dump({"seed": seed, "train": tr, "val": va, "test": te}, f, ensure_ascii=False, indent=2)

    def _sel(idxs): return [enriched[i] for i in idxs]

    # SUPER classification
    for name, idxs in [("train",tr),("val",va),("test",te)]:
        data = [{"text":r["text"], "label":r["super_id"], "label_text":r["super"]} for r in _sel(idxs)]
        write_jsonl(str(out/f"processed/classification/super/{name}.jsonl"), data)

    # CAT classification
    for name, idxs in [("train",tr),("val",va),("test",te)]:
        data = [{"text":r["text"], "label":r["cat_id"], "label_text":r["cat"]} for r in _sel(idxs)]
        write_jsonl(str(out/f"processed/classification/cat/{name}.jsonl"), data)

    # HIER
    for name, idxs in [("train",tr),("val",va),("test",te)]:
        write_jsonl(str(out/f"processed/hier/{name}.jsonl"), _sel(idxs))

    # MLM corpus (solo train)
    with open(out/"processed/mlm/train.txt","w",encoding="utf-8") as f:
        for r in _sel(tr):
            f.write(r["text"] + "\n")

    # Stats
    with open(out/"metadata/stats.json","w",encoding="utf-8") as f:
        json.dump({
            "generated_at": __import__("datetime").datetime.utcnow().isoformat()+"Z",
            "total": stats.total,
            "kept": stats.kept,
            "dropped_empty": stats.dropped_empty,
            "dropped_missing_labels": stats.dropped_missing_labels,
            "deduped": stats.deduped,
            "num_super": stats.num_super,
            "num_cat": stats.num_cat
        }, f, ensure_ascii=False, indent=2)

    return {"stats": stats.__dict__, "super2id": super2id, "cat2id": cat2id}
