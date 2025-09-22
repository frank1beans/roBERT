
from __future__ import annotations
import os, json, datetime
from typing import Dict

def write_report(out_dir: str, summary: Dict, figures: Dict[str,str]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    html = ["<html><head><meta charset='utf-8'><title>robimb Evaluation Report</title></head><body>"]
    html.append(f"<h1>robimb Evaluation Report</h1>")
    html.append(f"<p>Generated at: {datetime.datetime.utcnow().isoformat()}Z</p>")
    html.append("<h2>Summary</h2><pre>")
    html.append(json.dumps(summary, indent=2))
    html.append("</pre>")
    html.append("<h2>Figures</h2>")
    for title, path in figures.items():
        rel = os.path.relpath(path, out_dir)
        html.append(f"<h3>{title}</h3><img src='{rel}' style='max-width:100%;'/>")
    html.append("</body></html>")
    out_path = os.path.join(out_dir, "report.html")
    with open(out_path,"w",encoding="utf-8") as f:
        f.write("\n".join(html))
    return out_path
