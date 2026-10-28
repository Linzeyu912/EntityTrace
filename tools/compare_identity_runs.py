"""比较同一批原文提及的两个归组版本；变化数量不等于正确率或纠错量。"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path


def load(directory, filename):
    return json.loads((directory / filename).read_text(encoding="utf-8"))


def compare(before: Path, after: Path) -> dict:
    old, new = load(before, "result.json"), load(after, "result.json")
    if old["source"]["id"] != new["source"]["id"]:
        raise ValueError("只能比较同一个原文版本")
    def discovery(result):
        return {(m["id"], m["start"], m["end"], m["text"]) for m in result["mentions"]}
    if discovery(old) != discovery(new):
        raise ValueError("两次发现集合不同；先单独评估发现变化，不能将归组变化作同输入对比")
    def owner(result):
        mapping = {mid: entity["id"] for entity in result["entities"] for mid in entity["mention_ids"]}
        if len(mapping) != len(result["mentions"]) or sum(len(e["mention_ids"]) for e in result["entities"]) != len(mapping):
            raise ValueError("成员未恰好归组")
        return mapping
    old_owner, new_owner = owner(old), owner(new)
    contingency = collections.Counter((old_owner[mid], new_owner[mid]) for mid in old_owner)
    choose_two = lambda n: n * (n - 1) // 2
    shared = sum(choose_two(n) for n in contingency.values())
    linked = [sum(choose_two(len(g["mention_ids"])) for g in result["entities"]) for result in (old, new)]
    summaries = []
    for folder, result in [(before, old), (after, new)]:
        manifest, reviews = load(folder, "manifest.json"), load(folder, "review.json")
        latest_id = manifest["state"].get("last_invocation")
        latest = next((i for i in manifest["invocations"] if i["id"] == latest_id), None)
        summaries.append({"run_id": result["run_id"], "status": result["status"], "counts": manifest["counts"],
            "groups_by_status": dict(collections.Counter(e["identity_status"] for e in result["entities"])),
            "kinds": dict(collections.Counter(e.get("kind", "legacy") for e in result["entities"])),
            "reviews_by_stage": dict(collections.Counter(r["stage"] for r in reviews["items"])),
            "latest_invocation": latest, "total_model_metrics": manifest["total_model_metrics"],
            "identity_state": manifest["state"].get("identity"), "resolution_rounds": manifest.get("resolution_rounds", [])})
    return {"scope": "同一发现集合的归组变化；不是人工标注准确率，增删合并对不能直接称为纠错或新错误",
            "source_id": old["source"]["id"], "same_discovery": True, "mention_count": len(old_owner),
            "coassigned_mention_pairs": {"before": linked[0], "after": linked[1], "shared": shared,
                                          "added": linked[1] - shared, "removed": linked[0] - shared},
            "semantic_accuracy_measured": False, "before": summaries[0], "after": summaries[1]}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()
    report = compare(args.before, args.after)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({key: report[key] for key in ["scope", "same_discovery", "mention_count", "coassigned_mention_pairs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
