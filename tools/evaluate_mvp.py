"""只读评估已导出的结果；章节参考只在此处使用，绝不进入生产模型流程。"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(directory: Path, reference: dict) -> dict:
    result = read_json(directory / "result.json")
    manifest = read_json(directory / "manifest.json")
    review = read_json(directory / "review.json")
    source = result["source"]
    text = (directory / "source.txt").read_bytes().decode("utf-8")
    errors = []
    def check(condition, message):
        if not condition:
            errors.append(message)
    check(result["status"] != "incomplete", "计算流程未完成")
    check(hashlib.sha256((directory / "source.original").read_bytes()).hexdigest() == source["raw_sha256"], "原件指纹不符")
    check(hashlib.sha256(text.encode("utf-8")).hexdigest() == source["text_sha256"], "定位文本指纹不符")
    check(source["raw_sha256"] == reference["sha256"], "输入与独立参考的来源版本不符")
    snapshot = dict(result)
    snapshot.pop("result_revision")
    calculated = hashlib.sha256(json.dumps(snapshot, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    check(calculated == result["result_revision"] == manifest["result_revision"] == review["result_revision"], "结果版本不一致")
    chapters = sorted(result["chapters"], key=lambda c: c["start"])
    check(chapters[0]["start"] == 0 and chapters[-1]["end"] == len(text), "章节首尾不完整")
    check(all(a["end"] == b["start"] for a, b in zip(chapters, chapters[1:])), "章节范围有空隙或重叠")
    predicted = [c for c in chapters if c["kind"] == "chapter"]
    expected = reference["chapters"]
    match_count = sum((a["start"], a["title"]) == (b["section_start"], b["title"]) for a, b in zip(predicted, expected))
    check(len(predicted) == len(expected) == match_count, "章题或起点与独立参考不符")
    for chapter, answer in zip(predicted, expected):
        check(chapter["body_start"] <= answer["body_start"] <= answer["body_end"] <= chapter["end"], "章节没有包含参考正文")
        check(not text[chapter["body_start"]:answer["body_start"]].strip(), "标题至参考正文间出现非空白内容")
    mentions = {m["id"]: m for m in result["mentions"]}
    observations = {o["id"]: o for o in result["observations"]}
    chapter_ids = {c["id"]: c for c in chapters}
    check(len(mentions) == len(result["mentions"]), "提及 ID 重复")
    for mention in mentions.values():
        check(text[mention["start"]:mention["end"]] == mention["text"], "提及摘录不等于原文")
        chapter = chapter_ids.get(mention["chapter_id"])
        check(chapter is not None and chapter["start"] <= mention["start"] < mention["end"] <= chapter["end"], "提及不属于所标章节")
        if chapter:
            check(mention["chapter_start"] + chapter["start"] == mention["start"] and
                  mention["chapter_end"] + chapter["start"] == mention["end"], "章内地址换算错误")
        check(all(oid in observations for oid in mention["observation_ids"]), "提及引用不存在的发现记录")
    memberships = [mid for entity in result["entities"] for mid in entity["mention_ids"]]
    check(sorted(memberships) == sorted(mentions), "分组没有恰好覆盖所有提及")
    group = {mid: e["id"] for e in result["entities"] for mid in e["mention_ids"]}
    relations = {p["id"]: p for p in result["identity_relations"]}
    for pair in relations.values():
        check(pair["left"] in mentions and pair["right"] in mentions, "身份关系引用不存在的提及")
        if pair["choice"] == "different":
            check(group[pair["left"]] != group[pair["right"]], "已接受的不同关系出现在同一分组")
    requests = {}
    with (directory / "requests.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            request = json.loads(line)
            requests[request["id"]] = request
    decision_ids = set()
    span_count = 0
    def inspect_addresses(value):
        nonlocal span_count
        if isinstance(value, dict):
            if "text_version_id" in value and "offset_unit" in value:
                span_count += 1
                check(value["text_version_id"] == source["id"] and value["offset_unit"] == "unicode_code_point" and
                      type(value["start"]) is int and type(value["end"]) is int and
                      0 <= value["start"] < value["end"] <= len(text), "证据地址越界或版本错误")
            for child in value.values():
                inspect_addresses(child)
        elif isinstance(value, list):
            for child in value:
                inspect_addresses(child)
    inspect_addresses(result)
    inspect_addresses(review)
    with (directory / "decisions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            decision = json.loads(line)
            decision_ids.add(decision["id"])
            request = requests.get(decision["request_id"])
            check(request is not None, "决策的请求记录缺失")
            if request:
                key = decision["question_id"]
                check(key in request["request"]["questions"], "决策的问题不在对应请求内")
                check(request["response"]["answers"].get(key) == decision["answer"], "保存的决策与服务响应不一致")
            inspect_addresses(decision)
    for pair in relations.values():
        check(all(did in decision_ids for did in pair["decision_ids"]), "身份关系引用不存在的决策")
    for observation in observations.values():
        check(observation["request_id"] in requests, "发现记录的模型请求缺失")
    for entity in result["entities"]:
        check(all(pid in relations and relations[pid]["choice"] == "same" for pid in entity["support_pair_ids"]), "分组依据不是已接受的同一关系")
    return {"title": reference["title"], "run_id": result["run_id"], "result_revision": result["result_revision"],
            "status": result["status"], "implementation": manifest["config"]["implementation"],
            "input_sha256": source["raw_sha256"], "code_points": len(text), "counts": manifest["counts"],
            "chapter_reference_matches": match_count, "chapter_reference_total": len(expected),
            "checked_address_occurrences": span_count, "checks_passed": not errors,
            "errors": sorted(set(errors)), "semantic_accuracy_measured": False,
            "relations_by_choice": dict(collections.Counter(p["choice"] for p in relations.values())),
            "groups_by_status": dict(collections.Counter(e["identity_status"] for e in result["entities"])),
            "reviews_by_stage": dict(collections.Counter(r["stage"] for r in review["items"])),
            "total_model_metrics": manifest["total_model_metrics"], "invocations": manifest["invocations"],
            "largest_groups": [{"name": e["name"], "names": e["names"], "mentions": len(e["mention_ids"])}
                               for e in sorted(result["entities"], key=lambda e: -len(e["mention_ids"]))[:10]]}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="评估现有小说输出，不调用模型或修改运行数据库")
    parser.add_argument("outputs", nargs="+", type=Path)
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()
    references = read_json(ROOT / "test/text-library/reference/source-manifest.json")["books"]
    reports = []
    for directory in args.outputs:
        filename = read_json(directory / "manifest.json")["source"]["filename"]
        reference = next(b for b in references if Path(b["filename"]).name == filename)
        reports.append(evaluate(directory, reference))
    report = {"scope": "章节参考、原文地址及数据引用一致性；没有完整实体／身份人工金标准", "books": reports}
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{k: book[k] for k in ["title", "status", "counts", "checks_passed", "errors"]} for book in reports], ensure_ascii=False, indent=2))
    return 0 if all(book["checks_passed"] for book in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
