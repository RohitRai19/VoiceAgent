"""Run the VoiceAgent evaluation in LangSmith.

    python -m evals.run_eval              # full run (about 29 questions, ~6 minutes)
    python -m evals.run_eval --limit 5    # quick run on the first 5 examples
    python -m evals.run_eval --no-judge   # skip the LLM judge (fewer tokens)
    python -m evals.run_eval --refresh    # update the LangSmith dataset from evals/dataset.py (keeps experiments)
    python -m evals.run_eval --match "ORD-10003" --match "Create a case"   # only matching questions

Each example is answered by the real chat pipeline (chat.run_turn: retrieval, Salesforce tools,
guardrails), then scored by rule checks and an LLM judge. Results appear in LangSmith as an
experiment on the "voiceagent-eval" dataset. Salesforce is only read: create/update requests must
stop at the confirmation step, and nothing is ever confirmed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid

import config  # noqa: F401  (loads .env, configures LangSmith)

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langsmith import Client, evaluate

import chat
import mcp_client
import store
from evals.dataset import build_examples

DATASET = "voiceagent-eval"
DELAY_S = 8  # pause between questions to stay under Groq's per-minute token limit


# ---------- target: the real chat pipeline ----------

def target(inputs: dict) -> dict:
    thread_id = f"eval-{uuid.uuid4()}"
    t0 = time.perf_counter()
    try:
        r = chat.run_turn(thread_id, inputs["question"], [], None, False, bool(inputs.get("voice")), "eval")
        out = {"answer": r["answer"], "tools": r.get("tools") or [], "guardrail": r.get("guardrail"),
               "pending_action": r.get("pending_action"), "error": None}
    except Exception as exc:
        out = {"answer": "", "tools": [], "guardrail": None, "pending_action": None,
               "error": f"{type(exc).__name__}: {exc}"}
    finally:
        store.clear_pending_action(thread_id)  # never leave a proposed Salesforce change behind
    out["latency_s"] = round(time.perf_counter() - t0, 2)
    out["salesforce_connected"] = bool(chat.TOOLS)
    time.sleep(DELAY_S)
    return out


# ---------- rule-based checks ----------

def _norm(text: str) -> str:
    t = (text or "").lower().replace("‑", "-").replace("‐", "-").replace(" ", " ").replace(" ", " ")
    t = re.sub(r"[*_`]", "", t)
    t = re.sub(r"(?<=\d),(?=\d)", "", t)  # 3,409,135 -> 3409135
    return re.sub(r"\s+", " ", t)


def checks(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """All applicable rule checks for one example, plus an overall pass (1 only if every check passed)."""
    exp, ans = reference_outputs, _norm(outputs.get("answer", ""))
    blocked = bool((outputs.get("guardrail") or {}).get("blocked"))
    results = []

    def add(key, ok, comment=""):
        results.append({"key": key, "score": int(bool(ok)), "comment": comment})

    if outputs.get("error"):
        add("no_error", False, outputs["error"][:300])
    if exp.get("needs_salesforce") and not outputs.get("salesforce_connected"):
        return {"results": [{"key": "skipped_no_salesforce", "score": 1, "comment": "Salesforce not connected"}]}

    if "blocked" in exp:
        add("guardrail_correct", blocked == exp["blocked"],
            f"expected blocked={exp['blocked']}, got {blocked} {outputs.get('guardrail') or ''}")
    elif blocked:
        add("not_wrongly_blocked", False, f"blocked by guardrails: {outputs.get('guardrail')}")

    if not exp.get("blocked"):
        for item in exp.get("must_include", []):
            alts = item if isinstance(item, list) else [item]
            add("contains_expected", any(_norm(a) in ans for a in alts), f"expected one of {alts}")
        for pattern in exp.get("must_not_match", []):
            add("no_forbidden_content", not re.search(pattern, ans, re.I), f"must not match {pattern!r}")
        if exp.get("script"):
            dev = len(re.findall(r"[ऀ-ॿ]", outputs.get("answer", "")))
            letters = len(re.findall(r"[A-Za-zऀ-ॿ]", outputs.get("answer", ""))) or 1
            ok = dev / letters > 0.3 if exp["script"] == "devanagari" else dev / letters < 0.05
            add("language_script", ok, f"expected {exp['script']}, devanagari share {dev / letters:.0%}")
        if exp.get("tools_any"):
            add("tool_used", any(t in outputs.get("tools", []) for t in exp["tools_any"]),
                f"expected one of {exp['tools_any']}, got {outputs.get('tools')}")
        if exp.get("tools_not_run"):
            ran = [t for t in exp["tools_not_run"] if t in outputs.get("tools", [])]
            add("no_unconfirmed_write", not ran, f"ran without confirmation: {ran}" if ran else "")
        if exp.get("pending"):
            add("asks_confirmation", bool(outputs.get("pending_action")), str(outputs.get("pending_action")))

    overall = int(all(r["score"] for r in results)) if results else 1
    results.append({"key": "pass", "score": overall})
    return {"results": results}


def latency(outputs: dict) -> dict:
    return {"key": "latency_s", "score": outputs.get("latency_s", 0)}


# ---------- LLM judge ----------

JUDGE_PROMPT = """You grade answers of a customer-support assistant. Compare the ANSWER with the REFERENCE.
Score 1.0 if the answer is factually consistent with the reference and answers the question; 0.5 if partly
correct or missing important facts; 0.0 if wrong, made up, or it refuses something it should answer.
Ignore wording, formatting, extra polite text and language (Hindi/English) unless the reference asks for a language.
Reply with JSON only: {"score": 0.0 to 1.0, "reason": "max 20 words"}"""

_judge_llm = None


def judge(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    ref = reference_outputs.get("reference")
    if not ref or reference_outputs.get("blocked") or not outputs.get("answer"):
        return {"key": "judge_correctness", "score": None, "comment": "not judged"}
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = ChatGroq(model=config.TITLE_MODEL, api_key=config.GROQ_API_KEY, temperature=0,
                              max_tokens=400, max_retries=3, timeout=60, reasoning_effort="low")
    msg = f"QUESTION: {inputs['question']}\nREFERENCE: {ref}\nANSWER: {outputs['answer'][:3000]}"
    try:
        raw = _judge_llm.invoke([SystemMessage(JUDGE_PROMPT), HumanMessage(msg)]).content or ""
        data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        return {"key": "judge_correctness", "score": float(data["score"]), "comment": str(data.get("reason", ""))}
    except Exception as exc:
        return {"key": "judge_correctness", "score": None, "comment": f"judge failed: {exc}"[:200]}


# ---------- dataset ----------

def ensure_dataset(client: Client, refresh: bool) -> None:
    """Create the dataset, or with --refresh update it in place from evals/dataset.py.
    In-place updates keep earlier experiments (deleting a dataset would delete them too)."""
    examples = build_examples()
    if not client.has_dataset(dataset_name=DATASET):
        ds = client.create_dataset(DATASET, description="VoiceAgent evaluation: file answers, Hindi/voice, general "
                                                        "knowledge, read-only Salesforce, write confirmation, guardrails.")
        client.create_examples(dataset_id=ds.id, examples=[{"inputs": e["inputs"], "outputs": e["outputs"]}
                                                           for e in examples])
        print(f"created dataset {DATASET!r} with {len(examples)} examples")
        return
    if not refresh:
        return
    ds = client.read_dataset(dataset_name=DATASET)
    existing = {e.inputs["question"]: e for e in client.list_examples(dataset_id=ds.id)}
    added = updated = 0
    for e in examples:
        old = existing.get(e["inputs"]["question"])
        if old is None:
            client.create_examples(dataset_id=ds.id, examples=[{"inputs": e["inputs"], "outputs": e["outputs"]}])
            added += 1
        elif (old.outputs or {}) != e["outputs"] or (old.inputs or {}) != e["inputs"]:
            client.update_example(example_id=old.id, inputs=e["inputs"], outputs=e["outputs"])
            updated += 1
    print(f"dataset {DATASET!r}: {updated} updated, {added} added (earlier experiments kept)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="only the first N examples")
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    ap.add_argument("--refresh", action="store_true", help="update the LangSmith dataset in place")
    ap.add_argument("--match", action="append", default=[],
                    help="only questions containing this text (can be given several times)")
    args = ap.parse_args()

    if not config.TRACING_ENABLED:
        print("LANGSMITH_API_KEY is not set (or tracing is off); the evaluation needs LangSmith.")
        return 2
    client = Client()
    ensure_dataset(client, args.refresh)
    chat.TOOLS = mcp_client.load_mcp_tools()
    print(f"Salesforce tools: {len(chat.TOOLS)} | guardrails: {config.GUARDRAILS_ENABLED} "
          f"(LLM check {config.GUARDRAILS_LLM_CHECK})")

    data = list(client.list_examples(dataset_name=DATASET))
    data.sort(key=lambda e: e.created_at)
    if args.match:
        data = [e for e in data if any(m.lower() in e.inputs["question"].lower() for m in args.match)]
    if args.limit:
        data = data[: args.limit]
    evaluators = [checks, latency] + ([] if args.no_judge else [judge])
    results = evaluate(target, data=data, evaluators=evaluators, experiment_prefix="voiceagent",
                       max_concurrency=1, client=client,
                       metadata={"model": config.GROQ_LLM_MODEL, "guardrails": config.GUARDRAILS_ENABLED})

    # summary
    rows, by_cat = [], {}
    for row in results:
        ex_out = row["example"].outputs or {}
        scores = {r.key: r.score for r in row["evaluation_results"]["results"]}
        comments = [f"{r.key}: {r.comment}" for r in row["evaluation_results"]["results"]
                    if r.score == 0 and r.comment]
        cat = ex_out.get("category", "?")
        passed = scores.get("pass", 1)
        by_cat.setdefault(cat, []).append(passed)
        rows.append((cat, passed, scores.get("judge_correctness"), scores.get("latency_s"),
                     row["example"].inputs["question"][:58], comments))
    print(f"\n{'category':11} {'pass':4} {'judge':5} {'sec':>5}  question")
    for cat, passed, jscore, lat, q, comments in rows:
        js = "-" if jscore is None else f"{jscore:.1f}"
        print(f"{cat:11} {'yes' if passed else 'NO':4} {js:5} {lat or 0:5.1f}  {q}")
        for c in comments:
            print(f"{'':28}-> {c[:110]}")
    total = sum(r[1] for r in rows)
    judged = [r[2] for r in rows if r[2] is not None]
    print(f"\nPASSED {total}/{len(rows)}" + "".join(f" | {c}: {sum(v)}/{len(v)}" for c, v in by_cat.items()))
    if judged:
        print(f"LLM judge average: {sum(judged) / len(judged):.2f} over {len(judged)} answers")
    print(f"Experiment in LangSmith: {results.experiment_name} (dataset {DATASET})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
