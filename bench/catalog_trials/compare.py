"""Сравнивает офлайн и Gemma через обработчик веб-подбора."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from web.serve import ROOT, Api, Catalog

HERE = Path(__file__).resolve().parent


def check(turn: dict, answer: dict, cards: list[dict]) -> bool:
    options = answer.get("options", [])
    questions = sorted(question["field"] for question in answer.get("questions", []))
    prices = {card["id"]: card["price_rub"] for card in cards}
    for option in options:
        ids = option["listing_ids"]
        if not all(item_id in prices for item_id in ids) or sum(prices[item_id] for item_id in ids) != option["total_rub"]:
            return False

    ids = [item_id for option in options for item_id in option["listing_ids"]]
    if "questions" in turn:
        return answer.get("status") == "clarify" and questions == sorted(turn["questions"])
    if "no_match" in turn:
        message = " ".join(answer.get("message", "").casefold().split())
        return answer.get("status") == "no_match" and turn["no_match"] in message
    if answer.get("status") != "ok" or not options:
        return False
    if "ids" in turn:
        return ids == turn["ids"]
    if "allowed_ids" in turn:
        return bool(ids) and set(ids) <= set(turn["allowed_ids"])
    categories = {card["id"]: card["category"] for card in cards}
    return all(sorted(categories[item_id] for item_id in option["listing_ids"]) == sorted(turn["categories"])
               for option in options)


def assistant_reply(answer: dict) -> str:
    if answer.get("status") == "clarify":
        return "\n".join(question["text"] for question in answer["questions"])
    if answer.get("status") == "ok":
        return f"Подобрал вариантов: {len(answer['options'])} — смотрите справа."
    return answer.get("message") or "Подобрать не удалось."


def run_case(api: Api, case: dict) -> list[dict]:
    history: list[dict[str, str]] = []
    token = ""
    results = []
    for number, turn in enumerate(case["turns"], start=1):
        query = turn["text"].replace("{zeros}", "0" * 2400)
        history.append({"role": "user", "text": query})
        payload = json.dumps({"messages": history}, ensure_ascii=False).encode("utf-8")
        start = perf_counter()
        code, document, metrics = api.answer(payload, include_cards=True, dialogue_token=token)
        elapsed = round((perf_counter() - start) * 1000, 1)
        answer = document.get("answer", {})
        cards = document.get("items", [])
        results.append({
            "turn": number,
            "pass": code == 200 and check(turn, answer, cards),
            "http_status": code,
            "wall_ms": elapsed,
            "backend_ms": metrics.get("elapsed_ms") or None,
            "model_calls": metrics.get("model_calls", 0),
            "model_elapsed_ms": metrics.get("model_elapsed_ms"),
            "tokens": metrics.get("tokens"),
            "model_reused": bool(metrics.get("model_reused")),
            "model_unavailable": bool(metrics.get("model_unavailable")),
            "answer": answer if code == 200 else document,
            "cards": [{key: card[key] for key in ("id", "title", "category", "city_name", "price_rub")} for card in cards],
        })
        if code != 200:
            break
        history.append({"role": "assistant", "text": assistant_reply(answer)})
        token = metrics.get("dialogue_token", "")
    return results


def summarize(rows: list[dict]) -> dict:
    turns = [turn for row in rows for turn in row["results"]]
    calls = [turn for turn in turns if turn["model_calls"]]
    measured_calls = [turn["tokens"] for turn in calls if turn["tokens"] is not None]
    local = [turn for turn in turns if not turn["model_calls"] and turn["http_status"] == 200]
    return {
        "cases_passed": sum(row["pass"] for row in rows),
        "cases_total": len(rows),
        "turns_passed": sum(turn["pass"] for turn in turns),
        "turns_total": len(turns),
        "errors": sum(turn["http_status"] != 200 for turn in turns),
        "model_calls": sum(turn["model_calls"] for turn in calls),
        "model_reuses": sum(turn["model_reused"] for turn in turns),
        "tokens_reported": sum(measured_calls),
        "called_turns_without_usage": len(calls) - len(measured_calls),
        "tokens_per_call_median": median(measured_calls) if measured_calls else None,
        "wall_ms_median_all": median(turn["wall_ms"] for turn in turns),
        "wall_ms_median_local": median(turn["wall_ms"] for turn in local) if local else None,
        "wall_ms_median_called": median(turn["wall_ms"] for turn in calls) if calls else None,
        "wall_ms_mean_called": round(mean(turn["wall_ms"] for turn in calls), 1) if calls else None,
        "called_turns_over_10s": sum(turn["wall_ms"] > 10_000 for turn in calls),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="добавить реальные вызовы Gemma")
    parser.add_argument("--extra-repeat", type=int, default=0, help="сколько раз повторить opt-birds с чистой историей")
    parser.add_argument("--output", type=Path, required=True, help="путь к отчёту JSON")
    args = parser.parse_args()

    cases = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))
    repeat_case = next(case for case in cases if case["id"] == "opt-birds")
    cases += [{**repeat_case, "id": f"opt-birds-repeat-{number}"} for number in range(1, args.extra_repeat + 1)]
    modes = ("offline", "on_demand") if args.live else ("offline",)
    rows: dict[str, list[dict]] = {mode: [] for mode in modes}
    apis: dict[tuple[str, str], Api] = {}
    for case in cases:
        name = case["catalog"]
        source = ROOT / "data" / "catalog.json" if name == "stock" else HERE / f"{name}.json"
        for mode in modes:
            key = (mode, name)
            if key not in apis:
                apis[key] = Api(Catalog(source), source, mode == "offline", text_search=name != "stock")
            results = run_case(apis[key], case)
            passed = len(results) == len(case["turns"]) and all(turn["pass"] for turn in results)
            rows[mode].append({"id": case["id"], "catalog": name, "pass": passed, "results": results})
            tags = [f"{turn['answer'].get('status', turn['http_status'])}/{turn['wall_ms']}ms/{turn['tokens']}tok"
                    for turn in results]
            print(mode, case["id"], "pass" if passed else "miss", " ".join(tags), file=sys.stderr, flush=True)

    summaries = {mode: summarize([row for row in results if "-repeat-" not in row["id"]])
                 for mode, results in rows.items()}
    report = {"summary": summaries, "runs": rows}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
