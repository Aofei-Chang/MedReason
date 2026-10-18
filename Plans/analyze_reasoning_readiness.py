"""Summarize existing audit metadata; does not infer clinical difficulty or quality.

Run from any directory: python Plans/analyze_reasoning_readiness.py
"""
import collections
import hashlib
import json
from pathlib import Path
import statistics


def main():
    root = Path(__file__).resolve().parent / "radiopaedia_audit"
    source = root / "metadata.jsonl"
    digest = hashlib.sha256()
    rows = []
    with source.open("rb") as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            if row.get("folder") == "cases" and not row.get("error"):
                rows.append(row)
    lengths = sorted(r["text_chars"] for r in rows if r["text_chars"])
    result = {
        "source": source.name,
        "source_sha256": digest.hexdigest(),
        "scope": "Base-case records in existing 2026-10-15 audit; not a fresh source census",
        "case_count": len(rows),
        "nonempty_discussions": len(lengths),
        "nonempty_discussion_characters": {
            "q1": statistics.quantiles(lengths, n=4)[0],
            "median": statistics.median(lengths),
            "q3": statistics.quantiles(lengths, n=4)[2],
            "definition": "Length of saved discussion string, including any heading; not tokens",
        },
        "cases_with_qa": sum(r["qa_count"] > 0 for r in rows),
        "total_qa_pairs": sum(r["qa_count"] for r in rows),
        "top_level_key_counts": dict(collections.Counter(k for r in rows for k in r["keys"])),
        "limitations": [
            "No measured human or model difficulty labels.",
            "No clinician validation of reasoning or per-author credentials.",
            "Top-level schema does not establish the absence of dialogue embedded in prose.",
            "Discussion length does not measure clinical quality or reasoning depth.",
        ],
    }
    output = root / "reasoning_readiness.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
