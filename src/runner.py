"""Model runner: concurrent OpenAI calls with an on-disk cache.

The cache is keyed by (model, temperature, prompt), so a re-run costs nothing and
an interrupted run resumes where it stopped. Every raw completion is kept, which
is what lets the error typology be coded later without paying for the calls twice.
"""
from __future__ import annotations
import hashlib, json, os, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)
_lock = threading.Lock()


def load_env(path=None):
    p = Path(path or ROOT / ".env")
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _key(model, temp, prompt, cache=None):
    h = hashlib.sha256(f"{model}|{temp}|{prompt}".encode()).hexdigest()[:24]
    return (cache or CACHE) / f"{h}.json"


# USD per 1M tokens (input, output), OpenAI standard tier list prices for the
# GPT-4.1 family. Used ONLY by estimate_cost, to put a number in front of the
# person paying before a run starts. Check the provider's price page before a
# large run; a model missing here gets no estimate rather than a guessed one.
PRICES_PER_M = {"gpt-4.1-nano": (0.10, 0.40),
                "gpt-4.1-mini": (0.40, 1.60),
                "gpt-4.1": (2.00, 8.00)}


def read_cached(model, temperature, prompt, cache=None):
    """The cached record, or None when there is none or it does not parse.

    call() pays again for a file that does not parse, so an estimate that
    counted such a file as cached would come in low.
    """
    f = _key(model, temperature, prompt, cache)
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# CLadder prompts run 4.0-4.2 characters per GPT-4.1 token (measured on cached
# calls); 3.9 errs high. Used only when the run has no cached call to measure.
CHARS_PER_TOKEN = 3.9


def estimate_cost(jobs, model, temperature=0.0, ref_out=None):
    """What running `jobs` on `model` would cost. Nothing is sent.

    Counted as run_batch pays: once per DISTINCT prompt, 0 when cached.

    Output tokens are what decides the bill, and they vary far more by query
    type than by condition - backadj answers run about half the length of the
    causal types - so a plain mean over whatever happens to be cached can be
    badly skewed. In order of preference:
      1. `ref_out`, a Series item -> out_tok: the SAME item's real output under
         the closest condition already paid for (pilot.py --cost-ref)
      2. cached calls of this run with the same condition AND query type
      3. the same query type, then the same condition, then all cached calls
    With none of these the estimate is None rather than a guess.
    """
    seen, cached, fresh = set(), [], []
    for j in jobs:
        if j["prompt"] in seen:
            continue
        seen.add(j["prompt"])
        r = read_cached(model, temperature, j["prompt"])
        (fresh if r is None else cached).append((j, r))
    base = {"model": model, "calls": len(jobs), "distinct": len(seen),
            "cached": len(cached), "new": len(fresh)}
    if model not in PRICES_PER_M:
        return base | {"usd": None}

    tpc = (sum(r["in_tok"] for _, r in cached) / sum(len(j["prompt"]) for j, _ in cached)
           if cached else 1 / CHARS_PER_TOKEN)
    pools = {}
    for j, r in cached:
        for k in ((j["cond"], j.get("query_type")), ("qt", j.get("query_type")),
                  ("cond", j["cond"]), ("all",)):
            pools.setdefault(k, []).append(r["out_tok"])

    def out_for(j):
        if ref_out is not None and j["item"] in ref_out.index:
            return float(ref_out[j["item"]])
        for k in ((j["cond"], j.get("query_type")), ("qt", j.get("query_type")),
                  ("cond", j["cond"]), ("all",)):
            if pools.get(k):
                return sum(pools[k]) / len(pools[k])
        return None

    outs = [out_for(j) for j, _ in fresh]
    if any(o is None for o in outs):
        return base | {"usd": None}
    pi, po = PRICES_PER_M[model]
    tin = sum(tpc * len(j["prompt"]) for j, _ in fresh)
    tout = sum(outs)
    new_by = {}
    for j, _ in fresh:
        new_by[j["cond"]] = new_by.get(j["cond"], 0) + 1
    return base | {"new_by_cond": new_by, "in_tok": round(tin),
                   "out_tok": round(tout),
                   "usd": round((tin * pi + tout * po) / 1e6, 2)}


def call(client, model, prompt, temperature=0.0, max_tokens=700, retries=4,
         cache=None):
    """One completion, cached. Returns dict with text and usage.

    `cache` is another directory to cache in. scripts/check_drift.py uses one so
    that it can ask a question the main cache has already answered.
    """
    f = _key(model, temperature, prompt, cache)
    hit = read_cached(model, temperature, prompt, cache)
    if hit is not None:
        return hit | {"cached": True}

    last = None
    for attempt in range(retries):
        try:
            kw = {"model": model,
                  "messages": [{"role": "user", "content": prompt}],
                  "max_completion_tokens": max_tokens}
            if temperature != 1.0:
                kw["temperature"] = temperature
            r = client.chat.completions.create(**kw)
            rec = {"text": r.choices[0].message.content or "",
                   "in_tok": r.usage.prompt_tokens,
                   "out_tok": r.usage.completion_tokens,
                   "model": model, "cached": False, "ok": True}
            with _lock:
                f.write_text(json.dumps(rec), encoding="utf-8")
            return rec
        except Exception as e:                       # rate limits, transient 5xx
            last = e
            msg = str(e)
            if "temperature" in msg and "unsupported" in msg.lower():
                temperature = 1.0                    # some models fix temperature
                continue
            time.sleep(1.5 * (2 ** attempt))
    # A failure is NOT written to the cache, so a re-run retries it. It is also
    # flagged ok=False, because "" is indistinguishable from a model that said
    # nothing useful, and callers that score "" as a wrong answer would book an
    # infrastructure failure as a model deficit. Callers must check ok.
    return {"text": "", "in_tok": 0, "out_tok": 0, "model": model,
            "cached": False, "ok": False, "error": str(last)[:200]}


def run_batch(jobs, model, temperature=0.0, workers=16, max_tokens=700, on_tick=None,
              cache=None):
    """jobs: list of dicts each carrying a 'prompt'. Returns them with 'result'.

    Each DISTINCT prompt is sent once. Two jobs can carry the same prompt - on
    the complete 3-node families SCRAMBLE is the full reversal, which is DR_k3 -
    and with 16 threads both would miss the cache at the same moment and both
    be paid for.
    """
    from openai import OpenAI
    load_env()
    client = OpenAI()
    prompts = list(dict.fromkeys(j["prompt"] for j in jobs))
    done = [0]

    def one(p):
        r = call(client, model, p, temperature, max_tokens, cache=cache)
        with _lock:
            done[0] += 1
            if on_tick and done[0] % 25 == 0:
                on_tick(done[0], len(prompts))
        return p, r

    res = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, p) for p in prompts]
        for f in as_completed(futs):
            p, r = f.result()
            res[p] = r
    return [j | {"result": res[j["prompt"]]} for j in jobs]


def is_ok(result):
    """Did this call actually reach the model?

    Records cached before the ok flag existed have no such key. Those were all
    written on the success path - the failure path never writes to the cache -
    so a missing key means success.
    """
    return result.get("ok", not result.get("error"))


def usage_summary(records):
    ins = sum(r["result"]["in_tok"] for r in records)
    outs = sum(r["result"]["out_tok"] for r in records)
    cached = sum(1 for r in records if r["result"].get("cached"))
    errs = sum(1 for r in records if not is_ok(r["result"]))
    return {"calls": len(records), "cached": cached, "errors": errs,
            "err_rate": round(errs / len(records), 4) if records else 0.0,
            "in_tok": ins, "out_tok": outs}


def guard_errors(records, max_rate=0.01, label=""):
    """Stop the run rather than let infrastructure failures enter the results.

    Round 6 of the review panel found that a failed call returned {"text": ""},
    which parse_answer turns into None, which every caller scored as correct=0.
    An API outage would therefore have been published as a model deficit, and
    nothing in the pipeline would have said so. Measured over the 39,402 calls
    already on disk the real error rate is 0.000%, so this guard has never had
    anything to catch - which is exactly why it has to exist before the next
    batch rather than after.
    """
    bad = [r for r in records if not is_ok(r["result"])]
    if not bad:
        return 0
    rate = len(bad) / len(records)
    ex = bad[0]["result"].get("error", "")[:160]
    print(f"\n  !! {len(bad)}/{len(records)} calls FAILED ({100 * rate:.2f}%)"
          f"{' - ' + label if label else ''}")
    print(f"     vi du: {ex}")
    if rate > max_rate:
        raise SystemExit(
            f"\nSTOPPING: error rate {100 * rate:.2f}% exceeds the {100 * max_rate:.2f}% threshold.\n"
            "Continuing would contaminate the results - infrastructure failures would\n"
            "enter the tables as model mistakes. Fix the connection or the API key and\n"
            "run again; every successful call is already cached, so nothing is paid twice.")
    print(f"     below the {100 * max_rate:.2f}% threshold: excluded from the analysis, continuing")
    return len(bad)
