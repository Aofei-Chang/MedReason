"""Read-only metadata census and deterministic, bounded image audit.

No source data is changed. Candidate flags are lexical screens, not medical labels.
Run: python Plans/audit_radiopaedia.py --root /data/lingwei/radiopaedia_0826
"""
import argparse
import collections
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time


PATTERNS = {
    "confirmation_mention": r"histolog|histopath|biopsy|pathologically|pathology (?:confirmed|revealed|showed)|confirmed (?:on|by|with)|culture (?:grew|positive)|PCR",
    "uncertainty_mention": r"differential|non.?specific|indeterminate|cannot (?:exclude|distinguish)|not specific|overlap|mimic",
    "followup_mention": r"follow.?up|subsequent|interval|repeat (?:CT|MRI|scan|imaging)|prior (?:CT|MRI|scan|imaging)",
}


def read_record(item):
    sub, p = item
    try:
        raw = (p / "data.json").read_bytes()
        d = json.loads(raw)
        ci = d.get("citation_info", {})
        text = str(d.get("discussion", "")) if sub.startswith("cases") else str(d.get("text", ""))
        row = dict(folder=sub, rid=p.name, title=d.get("title", ""), certainty=d.get("certainty", ""),
                   systems=ci.get("Systems", ""), sections=ci.get("Sections", ""),
                   text_chars=len(text), presentation_chars=len(d.get("presentation", "")),
                   patient_chars=len(d.get("patient_data", "")), qa_count=len(d.get("qa_pairs", [])),
                   related_count=len(d.get("related-article", [])), references=len(d.get("reference", [])),
                   figure_count=len(d.get("case_and_figures", [])),
                   differential_count=len(d.get("imaging_differential_diagnosis", [])),
                   sha256=hashlib.sha256(raw).hexdigest(),
                   text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                   metadata_bytes=len(raw), error="")
        row.update({k: bool(re.search(v, text, re.I)) for k, v in PATTERNS.items()})
        row["radiographic_heading"] = bool(re.search(r"radiographic features|imaging features", text, re.I))
        row["differential_heading"] = bool(re.search(r"differential diagnosis", text, re.I))
        row["keys"] = sorted(d)
        if sub.startswith("cases"):
            row["related_titles"] = [x[0] for x in d.get("related-article", []) if isinstance(x, list) and len(x)>1]
        return row
    except Exception as exc:
        return dict(folder=sub, rid=p.name, error=str(exc))


def image_audit(item):
    sub, p = item
    row = dict(folder=sub, rid=p.name, image_files=0, image_bytes=0, viewers=0,
               modalities=[], viewer_texts=0, titles=0, extensions={}, largest_series_files=0,
               hash_checked=0, hash_unique=0, decoded=0, decode_errors=0, dimensions=[], errors=[])
    files=[]
    for base, dirs, names in os.walk(p / "images"):
        series_n=0
        for name in names:
            f=Path(base)/name
            if name == "viewer_info.json":
                try:
                    d=json.loads(f.read_text()); row["viewers"]+=1
                    row["modalities"].append(d.get("modality", ""))
                    row["viewer_texts"]+=bool(d.get("text", ""))
                    row["titles"]+=len(d.get("titles", []))
                except Exception as e: row["errors"].append(str(e))
            elif f.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".dcm", ".mp4"}:
                files.append(f); series_n+=1
                row["image_bytes"]+=f.stat().st_size
                ext=f.suffix.lower(); row["extensions"][ext]=row["extensions"].get(ext,0)+1
        row["largest_series_files"] = max(row["largest_series_files"], series_n)
    row["image_files"]=len(files)
    # A bounded filename-ordered subset: NOT an unbiased image duplicate estimate.
    hashes=[]
    for f in sorted(files)[:12]:
        try: hashes.append(hashlib.sha256(f.read_bytes()).hexdigest())
        except Exception as e: row["errors"].append(str(e))
    row["hash_checked"]=len(hashes); row["hash_unique"]=len(set(hashes))
    try:
        from PIL import Image
        for f in sorted(files)[:2]:
            try:
                with Image.open(f) as im:
                    row["dimensions"].append(list(im.size)); im.verify(); row["decoded"]+=1
            except Exception: row["decode_errors"]+=1
    except ImportError: row["errors"].append("Pillow unavailable")
    return row


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path(__file__).parent / "radiopaedia_audit")
    ap.add_argument("--workers", type=int, default=8); args=ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    folders=["cases", "cases_new", "articles", "articles_new"]
    items={s:[(s,Path(e.path)) for e in sorted(os.scandir(args.root/s), key=lambda e:e.name) if e.is_dir()] for s in folders}
    summary={"root":str(args.root), "seed":20261015,"directory_counts":{s:len(v) for s,v in items.items()}, "folders":{}}
    rows=[]
    for sub in folders:
        start=time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            rr=list(ex.map(read_record,items[sub]))
        rows.extend(rr); good=[r for r in rr if not r["error"]]
        stats={"valid_json":len(good),"errors":len(rr)-len(good), "metadata_bytes":sum(r["metadata_bytes"] for r in good)}
        for k in ["certainty", "systems", "sections"]: stats[k]=dict(collections.Counter(r[k] for r in good).most_common())
        for k in ["text_chars","presentation_chars","patient_chars","qa_count","related_count","references","figure_count","differential_count",*PATTERNS,"radiographic_heading","differential_heading"]:
            stats[k+"_nonzero"]=sum(bool(r[k]) for r in good)
        stats["keys"]=dict(collections.Counter(k for r in good for k in r["keys"]))
        summary["folders"][sub]=stats
        print(sub,len(rr),'records',round(time.time()-start,1),'seconds',flush=True)
    by={(r["folder"],r["rid"]):r for r in rows}
    for base in ["cases","articles"]:
        overlap=set(r["rid"] for r in rows if r["folder"]==base)&set(r["rid"] for r in rows if r["folder"]==base+"_new")
        comparable=[i for i in overlap if not by[base,i]["error"] and not by[base+"_new",i]["error"]]
        summary[base+"_overlap"]={"ids":len(overlap),"byte_identical_json":sum(by[base,i]["sha256"]==by[base+"_new",i]["sha256"] for i in comparable)}
    with (args.output/"metadata.jsonl").open('w') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    sample=[]
    rng=random.Random(summary["seed"])
    for sub,n in [("cases",512),("cases_new",128)]: sample.extend(rng.sample(items[sub],min(n,len(items[sub]))))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex: images=list(ex.map(image_audit,sample))
    (args.output/"image_sample.json").write_text(json.dumps(images,indent=2))
    summary["image_sample"]={}
    for sub in ["cases","cases_new"]:
        im=[r for r in images if r["folder"]==sub]
        summary["image_sample"][sub]={"n":len(im),"with_images":sum(r["image_files"]>0 for r in im),
          "multiple_viewers":sum(r["viewers"]>1 for r in im),"with_viewer_text":sum(r["viewer_texts"]>0 for r in im),
          "cases_with_exact_duplicate_in_bounded_subset":sum(r["hash_checked"]>r["hash_unique"] for r in im),
          **{k:sum(r[k] for r in im) for k in ["image_files","image_bytes","viewers","hash_checked","hash_unique","decoded","decode_errors"]},
          "modalities_case_counts":dict(collections.Counter(m for r in im for m in set(r["modalities"]))),
          "extensions":dict(sum((collections.Counter(r["extensions"]) for r in im),collections.Counter()))}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2))
    print('Wrote',args.output,flush=True)


if __name__ == "__main__": main()
