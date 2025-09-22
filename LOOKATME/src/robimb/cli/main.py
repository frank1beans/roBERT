
import argparse, json, os, sys
from ..data.convert import convert_jsonl_to_training

def app():
    ap = argparse.ArgumentParser(prog="robimb", description="roBIMB CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    cv = sub.add_parser("convert", help="Converti jsonl (text, super, cat) in dataset per training")
    cv.add_argument("--src", required=True, help="Percorso file sorgente .jsonl")
    cv.add_argument("--out", required=True, help="Cartella di output (root dataset)")
    cv.add_argument("--seed", type=int, default=42)
    cv.add_argument("--lowercase", action="store_true")
    cv.add_argument("--no-dedupe", action="store_true")
    cv.add_argument("--min-len", type=int, default=2)

    args = ap.parse_args()
    if args.cmd == "convert":
        res = convert_jsonl_to_training(
            src_jsonl=args.src, out_root=args.out, seed=args.seed,
            lowercase=args.lowercase, dedupe=not args.no_dedupe, min_len=args.min_len
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))


from ..core.pack_validate import validate_pack

def _cmd_pack_validate(ap):
    pv = ap.add_parser("pack-validate", help="Valida il Knowledge Pack contro gli schemi JSON")
    pv.add_argument("--pack", default="pack/current/pack.json", help="Percorso al pack.json")
    return pv

def app():
    import argparse, json
    ap = argparse.ArgumentParser(prog="robimb", description="roBIMB CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # convert
    cv = sub.add_parser("convert", help="Converti jsonl (text, super, cat) in dataset per training")
    cv.add_argument("--src", required=True, help="Percorso file sorgente .jsonl")
    cv.add_argument("--out", required=True, help="Cartella di output (root dataset)")
    cv.add_argument("--seed", type=int, default=42)
    cv.add_argument("--lowercase", action="store_true")
    cv.add_argument("--no-dedupe", action="store_true")
    cv.add_argument("--min-len", type=int, default=2)

    # pack-validate
    _cmd_pack_validate(sub)

    args = ap.parse_args()
    if args.cmd == "convert":
        from ..data.convert import convert_jsonl_to_training
        res = convert_jsonl_to_training(
            src_jsonl=args.src, out_root=args.out, seed=args.seed,
            lowercase=args.lowercase, dedupe=not args.no_dedupe, min_len=args.min_len
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))

    elif args.cmd == "pack-validate":
        rep = validate_pack(args.pack)
        out = {
            "ok": rep.ok,
            "files": [fr.__dict__ for fr in rep.files]
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))


from ..core.pack_loader import load_pack
from ..features.extractors import dry_run as extract_dry_run

def _cmd_extract(ap):
    ex = ap.add_parser("extract", help="Dry-run: estrai proprietà da un testo usando il pack extractors")
    ex.add_argument("--pack", default="pack/current/pack.json", help="Percorso al pack.json")
    ex.add_argument("--text", required=True, help="Testo da analizzare")
    return ex

def app():
    import argparse, json
    ap = argparse.ArgumentParser(prog="robimb", description="roBIMB CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # convert
    cv = sub.add_parser("convert", help="Converti jsonl (text, super, cat) in dataset per training")
    cv.add_argument("--src", required=True, help="Percorso file sorgente .jsonl")
    cv.add_argument("--out", required=True, help="Cartella di output (root dataset)")
    cv.add_argument("--seed", type=int, default=42)
    cv.add_argument("--lowercase", action="store_true")
    cv.add_argument("--no-dedupe", action="store_true")
    cv.add_argument("--min-len", type=int, default=2)

    # pack-validate
    from ..core.pack_validate import validate_pack
    pv = sub.add_parser("pack-validate", help="Valida il Knowledge Pack contro gli schemi JSON")
    pv.add_argument("--pack", default="pack/current/pack.json", help="Percorso al pack.json")

    # extract
    _cmd_extract(sub)

    args = ap.parse_args()
    if args.cmd == "convert":
        from ..data.convert import convert_jsonl_to_training
        res = convert_jsonl_to_training(
            src_jsonl=args.src, out_root=args.out, seed=args.seed,
            lowercase=args.lowercase, dedupe=not args.no_dedupe, min_len=args.min_len
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))

    elif args.cmd == "pack-validate":
        rep = validate_pack(args.pack)
        out = {"ok": rep.ok, "files": [fr.__dict__ for fr in rep.files]}
        print(json.dumps(out, ensure_ascii=False, indent=2))

    elif args.cmd == "extract":
        pack = load_pack(args.pack)
        out = extract_dry_run(args.text, pack.extractors)
        print(json.dumps(out, ensure_ascii=False, indent=2))


from ..features.extractors import extract_properties
from ..core.pack_loader import load_pack
from ..validators.engine import validate as validate_props

def _cmd_validate(ap):
    vl = ap.add_parser("validate", help="Valida proprietà (da JSON o estratte dal testo) con le regole del pack")
    vl.add_argument("--pack", default="pack/current/pack.json", help="Percorso al pack.json")
    src = vl.add_mutually_exclusive_group(required=True)
    src.add_argument("--props", help="JSON inline con proprietà, es: '{"geo.spessore_elemento":250}'")
    src.add_argument("--text", help="Testo da cui estrarre proprietà con extractors pack")
    vl.add_argument("--category", default="", help="Etichetta categoria (se nota)")
    return vl

def app():
    import argparse, json
    ap = argparse.ArgumentParser(prog="robimb", description="roBIMB CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # convert
    cv = sub.add_parser("convert", help="Converti jsonl (text, super, cat) in dataset per training")
    cv.add_argument("--src", required=True, help="Percorso file sorgente .jsonl")
    cv.add_argument("--out", required=True, help="Cartella di output (root dataset)")
    cv.add_argument("--seed", type=int, default=42)
    cv.add_argument("--lowercase", action="store_true")
    cv.add_argument("--no-dedupe", action="store_true")
    cv.add_argument("--min-len", type=int, default=2)

    # pack-validate
    from ..core.pack_validate import validate_pack
    pv = sub.add_parser("pack-validate", help="Valida il Knowledge Pack contro gli schemi JSON")
    pv.add_argument("--pack", default="pack/current/pack.json", help="Percorso al pack.json")

    # extract
    ex = sub.add_parser("extract", help="Dry-run: estrai proprietà da un testo usando il pack extractors")
    ex.add_argument("--pack", default="pack/current/pack.json", help="Percorso al pack.json")
    ex.add_argument("--text", required=True, help="Testo da analizzare")

    # validate
    _cmd_validate(sub)

    args = ap.parse_args()
    if args.cmd == "convert":
        from ..data.convert import convert_jsonl_to_training
        res = convert_jsonl_to_training(
            src_jsonl=args.src, out_root=args.out, seed=args.seed,
            lowercase=args.lowercase, dedupe=not args.no_dedupe, min_len=args.min_len
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))

    elif args.cmd == "pack-validate":
        rep = validate_pack(args.pack)
        out = {"ok": rep.ok, "files": [fr.__dict__ for fr in rep.files]}
        print(json.dumps(out, ensure_ascii=False, indent=2))

    elif args.cmd == "extract":
        pack = load_pack(args.pack)
        out = extract_dry_run(args.text, pack.extractors)
        print(json.dumps(out, ensure_ascii=False, indent=2))

    elif args.cmd == "validate":
        pack = load_pack(args.pack)
        if args.props:
            props = json.loads(args.props)
        else:
            props = extract_properties(args.text, pack.extractors)
        issues = validate_props(args.category, props, context={}, rules_pack=pack.validators, cat_entry=None)
        print(json.dumps({"issues": issues, "props": props}, ensure_ascii=False, indent=2))


def _cmd_train(ap):
    tr = ap.add_parser("train", help="Trainer: tapt/mlm/label/hier")
    tr.add_argument("--task", required=True, choices=["tapt","mlm","label","hier"])
    tr.add_argument("--config", required=True, help="Path al file di config JSON")
    return tr

def app():
    import argparse, json
    ap = argparse.ArgumentParser(prog="robimb", description="roBIMB CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # convert
    cv = sub.add_parser("convert", help="Converti jsonl (text, super, cat) in dataset per training")
    cv.add_argument("--src", required=True, help="Percorso file sorgente .jsonl")
    cv.add_argument("--out", required=True, help="Cartella di output (root dataset)")
    cv.add_argument("--seed", type=int, default=42)
    cv.add_argument("--lowercase", action="store_true")
    cv.add_argument("--no-dedupe", action="store_true")
    cv.add_argument("--min-len", type=int, default=2)

    # pack-validate
    from ..core.pack_validate import validate_pack
    pv = sub.add_parser("pack-validate", help="Valida il Knowledge Pack contro gli schemi JSON")
    pv.add_argument("--pack", default="pack/current/pack.json", help="Percorso al pack.json")

    # extract
    ex = sub.add_parser("extract", help="Dry-run: estrai proprietà da un testo usando il pack extractors")
    ex.add_argument("--pack", default="pack/current/pack.json", help="Percorso al pack.json")
    ex.add_argument("--text", required=True, help="Testo da analizzare")

    # validate
    vl = sub.add_parser("validate", help="Valida proprietà (da JSON o estratte dal testo) con le regole del pack")
    vl.add_argument("--pack", default="pack/current/pack.json", help="Percorso al pack.json")
    src = vl.add_mutually_exclusive_group(required=True)
    src.add_argument("--props", help="JSON inline con proprietà")
    src.add_argument("--text", help="Testo da cui estrarre proprietà")
    vl.add_argument("--category", default="", help="Etichetta categoria (se nota)")

    # train
    _cmd_train(sub)

    args = ap.parse_args()
    if args.cmd == "convert":
        from ..data.convert import convert_jsonl_to_training
        res = convert_jsonl_to_training(
            src_jsonl=args.src, out_root=args.out, seed=args.seed,
            lowercase=args.lowercase, dedupe=not args.no_dedupe, min_len=args.min_len
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))

    elif args.cmd == "pack-validate":
        rep = validate_pack(args.pack)
        out = {"ok": rep.ok, "files": [fr.__dict__ for fr in rep.files]}
        print(json.dumps(out, ensure_ascii=False, indent=2))

    elif args.cmd == "extract":
        from ..core.pack_loader import load_pack
        from ..features.extractors import dry_run as extract_dry_run
        pack = load_pack(args.pack)
        out = extract_dry_run(args.text, pack.extractors)
        print(json.dumps(out, ensure_ascii=False, indent=2))

    elif args.cmd == "validate":
        from ..core.pack_loader import load_pack
        from ..features.extractors import extract_properties
        from ..validators.engine import validate as validate_props
        pack = load_pack(args.pack)
        if args.props:
            props = json.loads(args.props)
        else:
            props = extract_properties(args.text, pack.extractors)
        issues = validate_props(args.category, props, context={}, rules_pack=pack.validators, cat_entry=None)
        print(json.dumps({"issues": issues, "props": props}, ensure_ascii=False, indent=2))

    elif args.cmd == "train":
        if args.task in ["tapt","mlm"]:
            from ..training.tapt import main as tapt_main
            tapt_main(args.config)
        elif args.task == "label":
            from ..training.label import main as label_main
            label_main(args.config)
        elif args.task == "hier":
            from ..training.hier import main as hier_main
            hier_main(args.config)


def _cmd_predict(ap):
    pd = ap.add_parser("predict", help="Pipeline end-to-end: categoria + proprietà + validazione + descrizione")
    pd.add_argument("--pack", default="pack/current/pack.json")
    pd.add_argument("--text", required=True)
    pd.add_argument("--model", required=True, help="Path o nome HF del classificatore (CAT)")
    pd.add_argument("--label-index", required=True, help="File label_index (cat.json)")
    pd.add_argument("--topk", type=int, default=5)
    pd.add_argument("--calibrator", help="File calibrator.json (temperature scaling)", default=None)
    return pd

def _cmd_calibrate(ap):
    cb = ap.add_parser("calibrate", help="Calibrazione temperature scaling sul validation set (label)")
    cb.add_argument("--model", required=True, help="Path o nome HF del classificatore")
    cb.add_argument("--label-index", required=True, help="File label_index (cat.json)")
    cb.add_argument("--val", required=True, help="Percorso val.jsonl (text, label)")
    cb.add_argument("--out", required=True, help="File output calibrator.json")
    cb.add_argument("--max-len", type=int, default=256)
    return cb

def app():
    import argparse, json
    ap = argparse.ArgumentParser(prog="robimb", description="roBIMB CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # convert
    cv = sub.add_parser("convert", help="Converti jsonl (text, super, cat) in dataset per training")
    cv.add_argument("--src", required=True, help="Percorso file sorgente .jsonl")
    cv.add_argument("--out", required=True, help="Cartella di output (root dataset)")
    cv.add_argument("--seed", type=int, default=42)
    cv.add_argument("--lowercase", action="store_true")
    cv.add_argument("--no-dedupe", action="store_true")
    cv.add_argument("--min-len", type=int, default=2)

    # pack-validate
    from ..core.pack_validate import validate_pack
    pv = sub.add_parser("pack-validate", help="Valida il Knowledge Pack contro gli schemi JSON")
    pv.add_argument("--pack", default="pack/current/pack.json", help="Percorso al pack.json")

    # extract
    ex = sub.add_parser("extract", help="Dry-run: estrai proprietà da un testo usando il pack extractors")
    ex.add_argument("--pack", default="pack/current/pack.json", help="Percorso al pack.json")
    ex.add_argument("--text", required=True, help="Testo da analizzare")

    # validate
    vl = sub.add_parser("validate", help="Valida proprietà (da JSON o estratte dal testo) con le regole del pack")
    vl.add_argument("--pack", default="pack/current/pack.json", help="Percorso al pack.json")
    src = vl.add_mutually_exclusive_group(required=True)
    src.add_argument("--props", help="JSON inline con proprietà")
    src.add_argument("--text", help="Testo da cui estrarre proprietà")
    vl.add_argument("--category", default="", help="Etichetta categoria (se nota)")

    # train
    tr = sub.add_parser("train", help="Trainer: tapt/mlm/label/hier")
    tr.add_argument("--task", required=True, choices=["tapt","mlm","label","hier"])
    tr.add_argument("--config", required=True, help="Path al file di config JSON")

    # predict & calibrate
    _cmd_predict(sub)
    _cmd_calibrate(sub)

    args = ap.parse_args()
    if args.cmd == "convert":
        from ..data.convert import convert_jsonl_to_training
        res = convert_jsonl_to_training(
            src_jsonl=args.src, out_root=args.out, seed=args.seed,
            lowercase=args.lowercase, dedupe=not args.no_dedupe, min_len=args.min_len
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))

    elif args.cmd == "pack-validate":
        rep = validate_pack(args.pack)
        out = {"ok": rep.ok, "files": [fr.__dict__ for fr in rep.files]}
        print(json.dumps(out, ensure_ascii=False, indent=2))

    elif args.cmd == "extract":
        from ..core.pack_loader import load_pack
        from ..features.extractors import dry_run as extract_dry_run
        pack = load_pack(args.pack)
        out = extract_dry_run(args.text, pack.extractors)
        print(json.dumps(out, ensure_ascii=False, indent=2))

    elif args.cmd == "validate":
        from ..core.pack_loader import load_pack
        from ..features.extractors import extract_properties
        from ..validators.engine import validate as validate_props
        pack = load_pack(args.pack)
        if args.props:
            props = json.loads(args.props)
        else:
            props = extract_properties(args.text, pack.extractors)
        issues = validate_props(args.category, props, context={}, rules_pack=pack.validators, cat_entry=None)
        print(json.dumps({"issues": issues, "props": props}, ensure_ascii=False, indent=2))

    elif args.cmd == "train":
        if args.task in ["tapt","mlm"]:
            from ..training.tapt import main as tapt_main
            tapt_main(args.config)
        elif args.task == "label":
            from ..training.label import main as label_main
            label_main(args.config)
        elif args.task == "hier":
            from ..training.hier import main as hier_main
            hier_main(args.config)

    elif args.cmd == "predict":
        from ..core.pack_loader import load_pack
        from ..inference.pipeline import run_pipeline
        pack = load_pack(args.pack)
        out = run_pipeline(args.text, pack, model_name_or_path=args.model, label_index_path=args.label_index, topk=args.topk, calibrator_path=args.calibrator)
        print(json.dumps(out, ensure_ascii=False, indent=2))

    elif args.cmd == "calibrate":
        # run model on validation set, collect logits and labels, fit T
        import torch, json
        from datasets import load_dataset
        from ..inference.predict_category import load_classifier, _load_id2label
        from ..inference.calibration import TemperatureCalibrator
        tok, mdl = load_classifier(args.model)
        id2label = _load_id2label(args.label_index)
        # invert id2label to ensure label_text->id mapping if present
        label2id = {v:k for k,v in id2label.items()}

        ds = load_dataset("json", data_files={"val": args.val})["val"]
        logits_list, labels_list = [], []
        for ex in ds:
            text = ex["text"]; lab = ex.get("label", None)
            if lab is None and "label_text" in ex:
                lab = label2id.get(ex["label_text"])
            if lab is None:
                continue
            with torch.no_grad():
                inputs = tok(text, return_tensors="pt", truncation=True, max_length=args.max_len)
                logits = mdl(**inputs).logits[0]  # [C]
            logits_list.append(logits)
            labels_list.append(int(lab))
        if not logits_list:
            raise SystemExit("Nessun esempio valido in --val per calibrazione.")
        logits = torch.stack(logits_list, dim=0)
        labels = torch.tensor(labels_list, dtype=torch.long)
        calib = TemperatureCalibrator.fit_from_logits(logits, labels)
        with open(args.out,"w",encoding="utf-8") as f:
            json.dump(calib.state_dict(), f, indent=2)
        print(json.dumps({"calibrator": calib.state_dict()}, indent=2))


def _cmd_eval(ap):
    ev = ap.add_parser("eval", help="Valutazione classificatore su split (metriche + grafici + report HTML)")
    ev.add_argument("--model", required=True, help="Path o nome HF del classificatore")
    ev.add_argument("--label-index", required=True, help="File label_index (cat.json o super.json)")
    ev.add_argument("--data", required=True, help="Percorso jsonl con {text,label} (test o val)")
    ev.add_argument("--out", required=True, help="Cartella di output per risultati/figure/report")
    ev.add_argument("--max-len", type=int, default=256)
    ev.add_argument("--calibrator", help="calibrator.json (opzionale)", default=None)
    return ev

def app():
    import argparse, json, os
    ap = argparse.ArgumentParser(prog="robimb", description="roBIMB CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # convert
    cv = sub.add_parser("convert", help="Converti jsonl (text, super, cat) in dataset per training")
    cv.add_argument("--src", required=True, help="Percorso file sorgente .jsonl")
    cv.add_argument("--out", required=True, help="Cartella di output (root dataset)")
    cv.add_argument("--seed", type=int, default=42)
    cv.add_argument("--lowercase", action="store_true")
    cv.add_argument("--no-dedupe", action="store_true")
    cv.add_argument("--min-len", type=int, default=2)

    # pack-validate
    from ..core.pack_validate import validate_pack
    pv = sub.add_parser("pack-validate", help="Valida il Knowledge Pack contro gli schemi JSON")
    pv.add_argument("--pack", default="pack/current/pack.json", help="Percorso al pack.json")

    # extract
    ex = sub.add_parser("extract", help="Dry-run: estrai proprietà da un testo usando il pack extractors")
    ex.add_argument("--pack", default="pack/current/pack.json", help="Percorso al pack.json")
    ex.add_argument("--text", required=True, help="Testo da analizzare")

    # validate
    vl = sub.add_parser("validate", help="Valida proprietà (da JSON o estratte dal testo) con le regole del pack")
    vl.add_argument("--pack", default="pack/current/pack.json", help="Percorso al pack.json")
    src = vl.add_mutually_exclusive_group(required=True)
    src.add_argument("--props", help="JSON inline con proprietà")
    src.add_argument("--text", help="Testo da cui estrarre proprietà")
    vl.add_argument("--category", default="", help="Etichetta categoria (se nota)")

    # train
    tr = sub.add_parser("train", help="Trainer: tapt/mlm/label/hier")
    tr.add_argument("--task", required=True, choices=["tapt","mlm","label","hier"])
    tr.add_argument("--config", required=True, help="Path al file di config JSON")

    # predict & calibrate
    pd = sub.add_parser("predict", help="Pipeline end-to-end: categoria + proprietà + validazione + descrizione")
    pd.add_argument("--pack", default="pack/current/pack.json")
    pd.add_argument("--text", required=True)
    pd.add_argument("--model", required=True, help="Path o nome HF del classificatore (CAT)")
    pd.add_argument("--label-index", required=True, help="File label_index (cat.json)")
    pd.add_argument("--topk", type=int, default=5)
    pd.add_argument("--calibrator", help="File calibrator.json (temperature scaling)", default=None)

    cb = sub.add_parser("calibrate", help="Calibrazione temperature scaling sul validation set (label)")
    cb.add_argument("--model", required=True, help="Path o nome HF del classificatore")
    cb.add_argument("--label-index", required=True, help="File label_index (cat.json)")
    cb.add_argument("--val", required=True, help="Percorso val.jsonl (text, label)")
    cb.add_argument("--out", required=True, help="File output calibrator.json")
    cb.add_argument("--max-len", type=int, default=256)

    # eval
    _cmd_eval(sub)

    args = ap.parse_args()
    if args.cmd == "convert":
        from ..data.convert import convert_jsonl_to_training
        res = convert_jsonl_to_training(
            src_jsonl=args.src, out_root=args.out, seed=args.seed,
            lowercase=args.lowercase, dedupe=not args.no_dedupe, min_len=args.min_len
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))

    elif args.cmd == "pack-validate":
        rep = validate_pack(args.pack)
        out = {"ok": rep.ok, "files": [fr.__dict__ for fr in rep.files]}
        print(json.dumps(out, ensure_ascii=False, indent=2))

    elif args.cmd == "extract":
        from ..core.pack_loader import load_pack
        from ..features.extractors import dry_run as extract_dry_run
        pack = load_pack(args.pack)
        out = extract_dry_run(args.text, pack.extractors)
        print(json.dumps(out, ensure_ascii=False, indent=2))

    elif args.cmd == "validate":
        from ..core.pack_loader import load_pack
        from ..features.extractors import extract_properties
        from ..validators.engine import validate as validate_props
        pack = load_pack(args.pack)
        if args.props:
            props = json.loads(args.props)
        else:
            props = extract_properties(args.text, pack.extractors)
        issues = validate_props(args.category, props, context={}, rules_pack=pack.validators, cat_entry=None)
        print(json.dumps({"issues": issues, "props": props}, ensure_ascii=False, indent=2))

    elif args.cmd == "train":
        if args.task in ["tapt","mlm"]:
            from ..training.tapt import main as tapt_main
            tapt_main(args.config)
        elif args.task == "label":
            from ..training.label import main as label_main
            label_main(args.config)
        elif args.task == "hier":
            from ..training.hier import main as hier_main
            hier_main(args.config)

    elif args.cmd == "predict":
        from ..core.pack_loader import load_pack
        from ..inference.pipeline import run_pipeline
        pack = load_pack(args.pack)
        out = run_pipeline(args.text, pack, model_name_or_path=args.model, label_index_path=args.label_index, topk=args.topk, calibrator_path=args.calibrator)
        print(json.dumps(out, ensure_ascii=False, indent=2))

    elif args.cmd == "calibrate":
        import torch, json
        from datasets import load_dataset
        from ..inference.predict_category import load_classifier, _load_id2label
        from ..inference.calibration import TemperatureCalibrator
        tok, mdl = load_classifier(args.model)
        id2label = _load_id2label(args.label_index)
        label2id = {v:k for k,v in id2label.items()}
        ds = load_dataset("json", data_files={"val": args.val})["val"]
        logits_list, labels_list = [], []
        for ex in ds:
            text = ex["text"]; lab = ex.get("label", None)
            if lab is None and "label_text" in ex:
                lab = label2id.get(ex["label_text"])
            if lab is None: continue
            with torch.no_grad():
                inputs = tok(text, return_tensors="pt", truncation=True, max_length=args.max_len)
                logits = mdl(**inputs).logits[0]
            logits_list.append(logits); labels_list.append(int(lab))
        if not logits_list: raise SystemExit("Nessun esempio valido in --val per calibrazione.")
        logits = torch.stack(logits_list, dim=0)
        labels = torch.tensor(labels_list, dtype=torch.long)
        calib = TemperatureCalibrator.fit_from_logits(logits, labels)
        with open(args.out,"w",encoding="utf-8") as f: json.dump(calib.state_dict(), f, indent=2)
        print(json.dumps({"calibrator": calib.state_dict()}, indent=2))

    elif args.cmd == "eval":
        import torch, numpy as np
        from datasets import load_dataset
        from ..inference.predict_category import load_classifier, _load_id2label
        from ..inference.calibration import TemperatureCalibrator
        from ..eval.metrics import evaluate_all, calibration_errors
        from ..eval.plots import plot_confusion, plot_reliability, plot_confidence_hist, plot_topk_curve, plot_risk_coverage, plot_pr_macro, plot_roc_macro
        from ..eval.report import write_report

        os.makedirs(args.out, exist_ok=True)
        tok, mdl = load_classifier(args.model)
        id2label = _load_id2label(args.label_index)
        labels_text = [id2label[i] for i in range(len(id2label))]
        label2id = {v:k for k,v in id2label.items()}
        calib = None
        if args.calibrator and os.path.exists(args.calibrator):
            with open(args.calibrator,"r",encoding="utf-8") as f: sd=json.load(f)
            from ..inference.calibration import TemperatureCalibrator
            calib = TemperatureCalibrator.from_state_dict(sd)

        ds = load_dataset("json", data_files={"data": args.data})["data"]
        probs_list = []; labels_list = []; preds_list = []
        rows_out = []
        for ex in ds:
            text = ex["text"]; lab = ex.get("label", None)
            if lab is None and "label_text" in ex:
                lab = label2id.get(ex["label_text"])
            if lab is None: continue
            with torch.no_grad():
                inputs = tok(text, return_tensors="pt", truncation=True, max_length=args.max_len)
                logits = mdl(**inputs).logits  # [1, C]
                if calib is not None:
                    logits = calib.apply(logits)
                p = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            probs_list.append(p); labels_list.append(int(lab))
            pred_id = int(p.argmax())
            preds_list.append(pred_id)
            topk_idx = p.argsort()[-5:][::-1].tolist()
            rows_out.append({
                "text": text,
                "label": int(lab),
                "label_text": id2label[int(lab)],
                "pred": pred_id,
                "pred_text": id2label[pred_id],
                "score": float(p[pred_id]),
                "top5": [{"id": int(i), "label": id2label[int(i)], "score": float(p[i])} for i in topk_idx]
            })
        if not probs_list:
            raise SystemExit("Nessun record valutabile in --data")
        probs = np.stack(probs_list, axis=0)
        labels = np.array(labels_list, dtype=int)

        # metrics
        res = evaluate_all(probs, labels)
        ece, mce, det = calibration_errors(probs, labels, n_bins=10)

        # save metrics and predictions
        with open(os.path.join(args.out, "metrics.json"),"w",encoding="utf-8") as f:
            json.dump({
                "accuracy": res.accuracy,
                "f1_macro": res.f1_macro,
                "f1_micro": res.f1_micro,
                "topk_accuracy": res.topk_accuracy,
                "nll": res.nll,
                "brier": res.brier,
                "ece": res.ece,
                "mce": res.mce,
                "pr_auc_macro": res.pr_auc_macro,
                "roc_auc_macro": res.roc_auc_macro
            }, f, indent=2)
        with open(os.path.join(args.out, "predictions.jsonl"),"w",encoding="utf-8") as f:
            for r in rows_out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # figures
        figures = {}
        cm_path = os.path.join(args.out, "confusion_matrix.png")
        plot_confusion(res.confusion, labels_text, cm_path); figures["Confusion Matrix"] = cm_path
        rel_path = os.path.join(args.out, "reliability.png")
        plot_reliability(det, rel_path); figures["Reliability Diagram"] = rel_path
        hist_path = os.path.join(args.out, "confidence_hist.png")
        plot_confidence_hist(probs.max(axis=1), hist_path); figures["Confidence Histogram"] = hist_path
        tk_path = os.path.join(args.out, "topk_accuracy.png")
        plot_topk_curve(res.topk_accuracy, tk_path); figures["Top-k Accuracy"] = tk_path
        rc_path = os.path.join(args.out, "risk_coverage.png")
        plot_risk_coverage(probs, labels, rc_path); figures["Risk-Coverage"] = rc_path
        pr_path = os.path.join(args.out, "pr_macro.png")
        plot_pr_macro(res.pr_auc_macro, pr_path); figures["PR AUC (macro)"] = pr_path
        roc_path = os.path.join(args.out, "roc_macro.png")
        plot_roc_macro(res.roc_auc_macro, roc_path); figures["ROC AUC (macro)"] = roc_path

        # report
        summary = {
            "model": args.model,
            "data": args.data,
            "label_index": args.label_index,
            "calibrator": args.calibrator,
            "metrics": {
                "accuracy": res.accuracy,
                "f1_macro": res.f1_macro,
                "f1_micro": res.f1_micro,
                "topk_accuracy": res.topk_accuracy,
                "nll": res.nll,
                "brier": res.brier,
                "ece": res.ece,
                "mce": res.mce,
                "pr_auc_macro": res.pr_auc_macro,
                "roc_auc_macro": res.roc_auc_macro
            }
        }
        from ..eval.report import write_report
        report_path = write_report(args.out, summary, figures)
        print(json.dumps({"report": report_path, "out_dir": args.out}, indent=2))


def _cmd_pack_build(ap):
    pb = ap.add_parser("pack-build", help="Costruisci un pack versione (copia files, genera manifest, aggiorna current)")
    pb.add_argument("--src", default="pack/v0", help="Directory sorgente (contiene i json del pack)")
    pb.add_argument("--out", required=True, help="Directory di output (es. pack/v1)")
    pb.add_argument("--set-current", action="store_true", help="Aggiorna pack/current/pack.json a questa versione")
    return pb

def app():
    import argparse, json, os
    ap = argparse.ArgumentParser(prog="robimb", description="roBIMB CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # convert
    cv = sub.add_parser("convert", help="Converti jsonl (text, super, cat) in dataset per training")
    cv.add_argument("--src", required=True, help="Percorso file sorgente .jsonl")
    cv.add_argument("--out", required=True, help="Cartella di output (root dataset)")
    cv.add_argument("--seed", type=int, default=42)
    cv.add_argument("--lowercase", action="store_true")
    cv.add_argument("--no-dedupe", action="store_true")
    cv.add_argument("--min-len", type=int, default=2)

    # pack-validate
    from ..core.pack_validate import validate_pack
    pv = sub.add_parser("pack-validate", help="Valida il Knowledge Pack contro gli schemi JSON")
    pv.add_argument("--pack", default="pack/current/pack.json", help="Percorso al pack.json")

    # pack-build
    _cmd_pack_build(sub)

    # extract
    ex = sub.add_parser("extract", help="Dry-run: estrai proprietà da un testo usando il pack extractors")
    ex.add_argument("--pack", default="pack/current/pack.json", help="Percorso al pack.json")
    ex.add_argument("--text", required=True, help="Testo da analizzare")

    # validate
    vl = sub.add_parser("validate", help="Valida proprietà (da JSON o estratte dal testo) con le regole del pack")
    vl.add_argument("--pack", default="pack/current/pack.json", help="Percorso al pack.json")
    src = vl.add_mutually_exclusive_group(required=True)
    src.add_argument("--props", help="JSON inline con proprietà")
    src.add_argument("--text", help="Testo da cui estrarre proprietà")
    vl.add_argument("--category", default="", help="Etichetta categoria (se nota)")

    # train
    tr = sub.add_parser("train", help="Trainer: tapt/mlm/label/hier")
    tr.add_argument("--task", required=True, choices=["tapt","mlm","label","hier"])
    tr.add_argument("--config", required=True, help="Path al file di config JSON")

    # predict & calibrate
    pd = sub.add_parser("predict", help="Pipeline end-to-end: categoria + proprietà + validazione + descrizione")
    pd.add_argument("--pack", default="pack/current/pack.json")
    pd.add_argument("--text", required=True)
    pd.add_argument("--model", required=True, help="Path o nome HF del classificatore (CAT)")
    pd.add_argument("--label-index", required=True, help="File label_index (cat.json)")
    pd.add_argument("--topk", type=int, default=5)
    pd.add_argument("--calibrator", help="File calibrator.json (temperature scaling)", default=None)

    cb = sub.add_parser("calibrate", help="Calibrazione temperature scaling sul validation set (label)")
    cb.add_argument("--model", required=True, help="Path o nome HF del classificatore")
    cb.add_argument("--label-index", required=True, help="File label_index (cat.json)")
    cb.add_argument("--val", required=True, help="Percorso val.jsonl (text, label)")
    cb.add_argument("--out", required=True, help="File output calibrator.json")
    cb.add_argument("--max-len", type=int, default=256)

    # eval
    ev = sub.add_parser("eval", help="Valutazione classificatore su split (metriche + grafici + report HTML)")
    ev.add_argument("--model", required=True, help="Path o nome HF del classificatore")
    ev.add_argument("--label-index", required=True, help="File label_index (cat.json o super.json)")
    ev.add_argument("--data", required=True, help="Percorso jsonl con {text,label} (test o val)")
    ev.add_argument("--out", required=True, help="Cartella di output per risultati/figure/report")
    ev.add_argument("--max-len", type=int, default=256)
    ev.add_argument("--calibrator", help="calibrator.json (opzionale)", default=None)

    # al-export (active learning)
    al = sub.add_parser("al-export", help="Selezione campioni (active learning) da predictions.jsonl → CSV")
    al.add_argument("--preds", required=True, help="File predictions.jsonl prodotto da 'robimb eval'")
    al.add_argument("--out", required=True, help="File CSV di output")
    al.add_argument("--k", type=int, default=200, help="Numero massimo di esempi")
    al.add_argument("--strategy", choices=["margin","entropy","leastconf"], default="margin")
    al.add_argument("--per-class", type=int, default=0, help="Limite per classe (0=nessun limite)")

    args = ap.parse_args()
    if args.cmd == "convert":
        from ..data.convert import convert_jsonl_to_training
        res = convert_jsonl_to_training(
            src_jsonl=args.src, out_root=args.out, seed=args.seed,
            lowercase=args.lowercase, dedupe=not args.no_dedupe, min_len=args.min_len
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))

    elif args.cmd == "pack-validate":
        rep = validate_pack(args.pack)
        out = {"ok": rep.ok, "files": [fr.__dict__ for fr in rep.files]}
        print(json.dumps(out, ensure_ascii=False, indent=2))

    elif args.cmd == "pack-build":
        from ..core.pack_tools import build_manifest, update_current
        res = build_manifest(args.src, args.out)
        if args.set_current:
            update_current("pack", args.out)
        # validate new pack
        new_pack = "pack/current/pack.json" if args.set_current else None
        if new_pack:
            rep = validate_pack(new_pack)
            res["validated"] = rep.ok
        print(json.dumps(res, ensure_ascii=False, indent=2))

    elif args.cmd == "extract":
        from ..core.pack_loader import load_pack
        from ..features.extractors import dry_run as extract_dry_run
        pack = load_pack(args.pack)
        out = extract_dry_run(args.text, pack.extractors)
        print(json.dumps(out, ensure_ascii=False, indent=2))

    elif args.cmd == "validate":
        from ..core.pack_loader import load_pack
        from ..features.extractors import extract_properties
        from ..validators.engine import validate as validate_props
        pack = load_pack(args.pack)
        if args.props:
            props = json.loads(args.props)
        else:
            props = extract_properties(args.text, pack.extractors)
        issues = validate_props(args.category, props, context={}, rules_pack=pack.validators, cat_entry=None)
        print(json.dumps({"issues": issues, "props": props}, ensure_ascii=False, indent=2))

    elif args.cmd == "train":
        if args.task in ["tapt","mlm"]:
            from ..training.tapt import main as tapt_main
            tapt_main(args.config)
        elif args.task == "label":
            from ..training.label import main as label_main
            label_main(args.config)
        elif args.cmd == "hier":
            from ..training.hier import main as hier_main
            hier_main(args.config)

    elif args.cmd == "predict":
        from ..core.pack_loader import load_pack
        from ..inference.pipeline import run_pipeline
        pack = load_pack(args.pack)
        out = run_pipeline(args.text, pack, model_name_or_path=args.model, label_index_path=args.label_index, topk=args.topk, calibrator_path=args.calibrator)
        print(json.dumps(out, ensure_ascii=False, indent=2))

    elif args.cmd == "calibrate":
        import torch, json
        from datasets import load_dataset
        from ..inference.predict_category import load_classifier, _load_id2label
        from ..inference.calibration import TemperatureCalibrator
        tok, mdl = load_classifier(args.model)
        id2label = _load_id2label(args.label_index)
        label2id = {v:k for k,v in id2label.items()}
        ds = load_dataset("json", data_files={"val": args.val})["val"]
        logits_list, labels_list = [], []
        for ex in ds:
            text = ex["text"]; lab = ex.get("label", None)
            if lab is None and "label_text" in ex:
                lab = label2id.get(ex["label_text"])
            if lab is None: continue
            with torch.no_grad():
                inputs = tok(text, return_tensors="pt", truncation=True, max_length=args.max_len)
                logits = mdl(**inputs).logits[0]
            logits_list.append(logits); labels_list.append(int(lab))
        if not logits_list: raise SystemExit("Nessun esempio valido in --val per calibrazione.")
        logits = __import__("torch").stack(logits_list, dim=0)
        labels = __import__("torch").tensor(labels_list, dtype=__import__("torch").long)
        calib = TemperatureCalibrator.fit_from_logits(logits, labels)
        with open(args.out,"w",encoding="utf-8") as f: json.dump(calib.state_dict(), f, indent=2)
        print(json.dumps({"calibrator": calib.state_dict()}, indent=2))

    elif args.cmd == "eval":
        # delegate to already implemented eval block (see step 9) – here we reuse it.
        from ..cli.main import app as _app  # fallback (avoid duplicate logic)
        _app()

    elif args.cmd == "al-export":
        import csv, math
        def entropy(scores):
            return -sum(p*math.log(p+1e-12) for p in scores)
        # read predictions.jsonl
        rows = []
        with open(args.preds,"r",encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                r = json.loads(line)
                # compute uncertainty
                top5 = r.get("top5", [])
                s1 = top5[0]["score"] if top5 else r.get("score", 0.0)
                s2 = top5[1]["score"] if len(top5) > 1 else 0.0
                if args.strategy == "margin":
                    unc = 1.0 - (s1 - s2)  # lower margin ⇒ higher uncertainty
                elif args.strategy == "entropy":
                    probs = [t["score"] for t in top5] if top5 else [s1, 1.0-s1]
                    unc = entropy(probs)
                else:  # leastconf
                    unc = 1.0 - s1
                r["_uncertainty"] = float(unc)
                rows.append(r)
        # sort by uncertainty desc
        rows.sort(key=lambda x: x["_uncertainty"], reverse=True)

        # per-class limit
        if args.per_class > 0:
            out = []
            counts = {}
            for r in rows:
                lbl = r.get("label_text") or str(r.get("label"))
                if counts.get(lbl, 0) < args.per_class:
                    out.append(r); counts[lbl] = counts.get(lbl, 0) + 1
                if len(out) >= args.k: break
            rows = out
        else:
            rows = rows[:args.k]

        # write CSV
        with open(args.out,"w",newline="",encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["text","true_label","pred_label","score","uncertainty","top5_json","misclassified"])
            for r in rows:
                tl = r.get("label_text","")
                pl = r.get("pred_text","")
                mis = (tl != "" and pl != "" and tl != pl)
                w.writerow([r.get("text",""), tl, pl, "{:.4f}".format(r.get("score",0.0)), "{:.4f}".format(r.get("_uncertainty",0.0)), json.dumps(r.get("top5",[]), ensure_ascii=False), mis])
        print(json.dumps({"exported": len(rows), "out": args.out}, indent=2))
