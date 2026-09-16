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


def _key(model, temp, prompt):
    h = hashlib.sha256(f"{model}|{temp}|{prompt}".encode()).hexdigest()[:24]
    return CACHE / f"{h}.json"


def call(client, model, prompt, temperature=0.0, max_tokens=700, retries=4):
    """One completion, cached. Returns dict with text and usage."""
    f = _key(model, temperature, prompt)
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8")) | {"cached": True}
        except json.JSONDecodeError:
            pass

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


def run_batch(jobs, model, temperature=0.0, workers=16, max_tokens=700, on_tick=None):
    """jobs: list of dicts each carrying a 'prompt'. Returns them with 'result'."""
    from openai import OpenAI
    load_env()
    client = OpenAI()
    done = [0]

    def one(j):
        r = call(client, model, j["prompt"], temperature, max_tokens)
        with _lock:
            done[0] += 1
            if on_tick and done[0] % 25 == 0:
                on_tick(done[0], len(jobs))
        return j | {"result": r}

    out = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for f in as_completed(futs):
            out.append(f.result())
    return out


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
    print(f"\n  !! {len(bad)}/{len(records)} luot goi THAT BAI ({100 * rate:.2f}%)"
          f"{' - ' + label if label else ''}")
    print(f"     vi du: {ex}")
    if rate > max_rate:
        raise SystemExit(
            f"\nDUNG: ty le loi {100 * rate:.2f}% vuot nguong {100 * max_rate:.2f}%.\n"
            "Ket qua se bi nhiem neu chay tiep - loi ha tang se vao bang nhu\n"
            "khuyet diem cua model. Sua ket noi hoac khoa API roi chay lai;\n"
            "cac luot da thanh cong deu nam trong cache nen khong mat tien.")
    print(f"     duoi nguong {100 * max_rate:.2f}%, loai khoi phan tich va chay tiep")
    return len(bad)
