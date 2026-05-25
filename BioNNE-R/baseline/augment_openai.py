"""
Data augmentation for BioNNE-R rare/weak classes using OpenAI API (synchronous).

Generator:  gpt-5-mini   — produces candidate (sentence, entity1, entity2, relation) tuples
Verifier:   gpt-5.4-mini — validates label correctness and biomedical plausibility

Usage:
    # English
    python augment_openai.py \\
        --train-tsv ../data/en/train/eng-train-rel.tsv \\
        --train-texts ../data/en/train/texts/ \\
        --out-jsonl data/eng_augmented.txt

    # Russian
    python augment_openai.py \\
        --train-tsv ../data/ru/train/rus-train-rel.tsv \\
        --train-texts ../data/ru/train/texts/ \\
        --out-jsonl data/rus_augmented.txt --lang ru
"""

import argparse, json, os, random, re, sys
from collections import Counter
from pathlib import Path
import pandas as pd

# ── Load .env ────────────────────────────────────────────────────────────────

def _load_dotenv():
    for p in [Path(__file__).parent / ".env", Path(".env")]:
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())
            return

_load_dotenv()
from openai import OpenAI

# ── Relation definitions (bilingual) ─────────────────────────────────────────

RELATION_DEFS = {
    "en": {
        "ALTERNATIVE_NAME": (
            "ALTERNATIVE_NAME: entity1 and entity2 refer to the SAME biomedical concept "
            "under different names (both always the SAME entity type). "
            "Patterns: full synonym pairs, brand vs generic drug names, historical vs current terminology. "
            "NOT short acronyms like EEG/PCR (those are ABBREVIATION). NOT IS-A (that is SUBCLASS_OF)."
        ),
        "APPLIED_TO": (
            "APPLIED_TO: a procedure (LABPROC) or drug/chemical (CHEM) is physically "
            "performed on or administered to an anatomical structure (ANATOMY) or physiological process (PHYS). "
            "The action is interventional — something is DONE TO the anatomy. "
            "Contrast: TO_DETECT_OR_STUDY is purely observational (gathering information, not intervening)."
        ),
        "TO_DETECT_OR_STUDY": (
            "TO_DETECT_OR_STUDY: a procedure (LABPROC) is used to DETECT, MEASURE, or CHARACTERIZE "
            "a biomedical entity (CHEM, DISO, ANATOMY, PHYS, FINDING). Purely observational — "
            "no intervention on the patient/tissue. "
            "Contrast: APPLIED_TO involves actual physical intervention or administration."
        ),
        "USED_IN": (
            "USED_IN: a chemical/drug (CHEM), laboratory procedure (LABPROC), or device (DEVICE) "
            "is employed as part of another procedure or context. "
            "More abstract than APPLIED_TO — describes a component-of or used-within relationship."
        ),
    },
    "ru": {
        "ALTERNATIVE_NAME": (
            "ALTERNATIVE_NAME: entity1 и entity2 обозначают ОДНУ И ТУ ЖЕ биомедицинскую концепцию "
            "под разными названиями (оба объекта всегда одного типа). "
            "Паттерны: синонимы, торговое vs генерическое название препарата, устаревшее vs современное название. "
            "НЕ короткие аббревиатуры вроде ЭЭГ/ПЦР (это ABBREVIATION). НЕ IS-A отношения (это SUBCLASS_OF)."
        ),
        "APPLIED_TO": (
            "APPLIED_TO: процедура (LABPROC) или препарат/химическое вещество (CHEM) физически "
            "выполняется или вводится в анатомическую структуру (ANATOMY) или физиологический процесс (PHYS). "
            "Действие интервенционное — что-то ДЕЛАЕТСЯ С объектом. "
            "Контраст: TO_DETECT_OR_STUDY — чисто наблюдательное (сбор информации, без вмешательства)."
        ),
        "TO_DETECT_OR_STUDY": (
            "TO_DETECT_OR_STUDY: процедура (LABPROC) используется для ОБНАРУЖЕНИЯ, ИЗМЕРЕНИЯ или "
            "ХАРАКТЕРИСТИКИ биомедицинского объекта (CHEM, DISO, ANATOMY, PHYS, FINDING). "
            "Чисто наблюдательное — без вмешательства в пациента/ткань. "
            "Контраст: APPLIED_TO — реальное физическое воздействие."
        ),
        "USED_IN": (
            "USED_IN: химическое вещество (CHEM), лабораторная процедура (LABPROC) или устройство (DEVICE) "
            "применяется в рамках другой процедуры или контекста. "
            "Более абстрактно, чем APPLIED_TO — описывает отношение компонент/используется-в."
        ),
    },
}

# ── Jobs per language: (relation, head_type, tail_type, target_n, contrast_rel) ──

JOBS = {
    "en": [
        ("ALTERNATIVE_NAME", "DISO",     "DISO",     120, None),
        ("ALTERNATIVE_NAME", "CHEM",     "CHEM",      90, None),
        ("ALTERNATIVE_NAME", "ANATOMY",  "ANATOMY",   80, None),
        ("ALTERNATIVE_NAME", "FINDING",  "FINDING",   60, None),
        ("ALTERNATIVE_NAME", "LABPROC",  "LABPROC",   30, None),
        ("ALTERNATIVE_NAME", "PHYS",     "PHYS",      20, None),
        ("APPLIED_TO",       "LABPROC",  "ANATOMY",  130, "TO_DETECT_OR_STUDY"),
        ("APPLIED_TO",       "CHEM",     "ANATOMY",   70, None),
        ("TO_DETECT_OR_STUDY", "LABPROC","DISO",       80, None),
        ("TO_DETECT_OR_STUDY", "LABPROC","CHEM",       70, None),
    ],
    "ru": [
        ("ALTERNATIVE_NAME", "ANATOMY",  "ANATOMY",  100, None),
        ("ALTERNATIVE_NAME", "DISO",     "DISO",      90, None),
        ("ALTERNATIVE_NAME", "CHEM",     "CHEM",      60, None),
        ("ALTERNATIVE_NAME", "PHYS",     "PHYS",      40, None),
        ("ALTERNATIVE_NAME", "FINDING",  "FINDING",   30, None),
        ("APPLIED_TO",       "LABPROC",  "ANATOMY",  120, "TO_DETECT_OR_STUDY"),
        ("APPLIED_TO",       "CHEM",     "ANATOMY",   50, None),
        ("APPLIED_TO",       "LABPROC",  "PHYS",      30, "TO_DETECT_OR_STUDY"),
        ("USED_IN",          "CHEM",     "CHEM",      70, None),
        ("USED_IN",          "LABPROC",  "LABPROC",   50, None),
        ("USED_IN",          "DEVICE",   "LABPROC",   40, None),
    ],
}

# ── Prompt builders ───────────────────────────────────────────────────────────

def _fmt_seeds(seeds: list[dict], lang: str) -> str:
    lines = []
    for i, s in enumerate(seeds, 1):
        lines.append(
            f"  {i}. \"{s['text']}\"\n"
            f"     head: \"{s['head']}\" ({s['head_type']}) at [{s['head_start']},{s['head_end']}]\n"
            f"     tail: \"{s['tail']}\" ({s['tail_type']}) at [{s['tail_start']},{s['tail_end']}]"
        )
    return "\n".join(lines)


def gen_prompt(relation, head_type, tail_type, seeds, lang, defs, contrast_rel=None, n=5):
    contrast = ""
    if contrast_rel:
        contrast = (
            f"\n\nADDITIONALLY generate {n} CONTRASTIVE examples labeled \"{contrast_rel}\" "
            f"with the same entity types ({head_type}, {tail_type}) to teach the distinction.\n"
            f"Definition of {contrast_rel}: {defs[contrast_rel]}"
        )
    lang_instruction = (
        "Generate sentences in Russian (biomedical journal style, similar to PubMed Russian publications)."
        if lang == "ru" else
        "Generate sentences in English (PubMed abstract style)."
    )
    return f"""You are a biomedical NLP expert generating training data for relation extraction.

TASK: Generate {n} new biomedical sentence examples for the relation {relation}.
Entity types: head={head_type}, tail={tail_type}.
{lang_instruction}

RELATION DEFINITION:
{defs[relation]}

SEED EXAMPLES (style reference only — do NOT copy verbatim):
{_fmt_seeds(seeds, lang)}

OUTPUT: JSON array of {n} objects:
{{
  "sentence": "<full sentence>",
  "head": "<exact head text as it appears in sentence>",
  "head_type": "{head_type}",
  "head_start": <int, char offset 0-indexed>,
  "head_end": <int, exclusive>,
  "tail": "<exact tail text>",
  "tail_type": "{tail_type}",
  "tail_start": <int>,
  "tail_end": <int>,
  "relation": "{relation}"
}}{contrast}

CRITICAL RULES:
1. sentence[head_start:head_end] must EXACTLY equal head text (case-sensitive)
2. sentence[tail_start:tail_end] must EXACTLY equal tail text (case-sensitive)
3. Vary biomedical domain and sentence structure across examples
4. {"Do NOT generate short acronyms for ALTERNATIVE_NAME — only full synonym pairs" if relation == "ALTERNATIVE_NAME" else "Ensure the relation label is unambiguous from sentence context alone"}

Output ONLY the JSON array."""


def ver_prompt(candidates: list[dict], lang: str) -> str:
    lang_note = "Sentences are in Russian." if lang == "ru" else "Sentences are in English."
    return f"""You are a strict biomedical NLP expert reviewing auto-generated training data. {lang_note}

For each example: ACCEPT only if ALL are true:
✓ Sentence is natural, sounds like a real publication
✓ Relation label is unambiguous from context alone
✓ sentence[start:end] == entity text (spans are correct)
✓ No other relation type would fit equally well

{json.dumps(candidates, indent=2, ensure_ascii=False)}

Return a JSON array adding to each object:
  "verdict": "ACCEPT" or "REJECT"
  "reason": "<one sentence>"

Output ONLY the JSON array."""

# ── Seed loading ──────────────────────────────────────────────────────────────

def load_seeds(tsv_path: str, texts_dir: str) -> dict[str, list[dict]]:
    df = pd.read_csv(tsv_path, sep="\t")
    texts = {p.stem: p.read_text(encoding="utf-8") for p in Path(texts_dir).glob("*.txt")}
    seeds: dict[str, list[dict]] = {}
    for _, row in df.iterrows():
        doc_id = str(row["document_id"])
        if doc_id not in texts:
            continue
        full = texts[doc_id]
        h_s, h_e = map(int, str(row["head_span"]).split("-"))
        t_s, t_e = map(int, str(row["tail_span"]).split("-"))
        ctx_s = max(0, min(h_s, t_s) - 150)
        ctx_e = min(len(full), max(h_e, t_e) + 150)
        sentence = full[ctx_s:ctx_e].strip()
        seeds.setdefault(row["relation"], []).append({
            "text":       sentence,
            "head":       row["head_text"],
            "head_type":  row["head_type"],
            "head_start": h_s - ctx_s,
            "head_end":   h_e - ctx_s,
            "tail":       row["tail_text"],
            "tail_type":  row["tail_type"],
            "tail_start": t_s - ctx_s,
            "tail_end":   t_e - ctx_s,
            "relation":   row["relation"],
        })
    return seeds

# ── Validation ────────────────────────────────────────────────────────────────

def valid_spans(obj: dict) -> bool:
    """Return True and fix obj in-place if spans are within ±3 chars of correct position."""
    s = obj.get("sentence", "")
    for role in ("head", "tail"):
        start, end = obj.get(f"{role}_start"), obj.get(f"{role}_end")
        name = obj.get(role, "")
        if not isinstance(name, str) or not name:
            return False
        if not (isinstance(start, int) and isinstance(end, int)):
            return False
        if s[start:end] == name:
            continue  # exact match, good
        # Near-miss correction: search ±3 chars around the given position
        WINDOW = 3
        found = False
        for delta in range(-WINDOW, WINDOW + 1):
            ns, ne = start + delta, end + delta
            if 0 <= ns < ne <= len(s) and s[ns:ne] == name:
                obj[f"{role}_start"] = ns
                obj[f"{role}_end"] = ne
                found = True
                break
        if not found:
            # Also try: find `name` near start position
            search_from = max(0, start - WINDOW)
            idx = s.find(name, search_from)
            if idx != -1 and abs(idx - start) <= WINDOW:
                obj[f"{role}_start"] = idx
                obj[f"{role}_end"] = idx + len(name)
            else:
                return False
    return True


def to_jsonl(obj: dict, aug_id: str, lang: str) -> dict:
    return {
        "text":      obj["sentence"],
        "h":         {"name": obj["head"], "pos": [obj["head_start"], obj["head_end"]]},
        "t":         {"name": obj["tail"], "pos": [obj["tail_start"], obj["tail_end"]]},
        "relation":  obj["relation"],
        "doc_id":    aug_id,
        "lang":      "russian" if lang == "ru" else "english",
        "head_type": obj["head_type"],
        "tail_type": obj["tail_type"],
        "head_span": f"{obj['head_start']}-{obj['head_end']}",
        "tail_span": f"{obj['tail_start']}-{obj['tail_end']}",
    }

# ── API call ──────────────────────────────────────────────────────────────────

def call_api(client: OpenAI, prompt: str, model: str, call_label: str) -> str:
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=8192,
            )
            content = r.choices[0].message.content or "[]"
            usage = r.usage
            tok_str = f"in={usage.prompt_tokens} out={usage.completion_tokens}" if usage else "?"
            finish = r.choices[0].finish_reason
            print(f"  ✓ {call_label} [{tok_str} finish={finish}]", flush=True)
            return content
        except Exception as e:
            msg = str(e)
            if "max_tokens" in msg or "output limit" in msg:
                # Model output limit exceeded — can't increase; skip this call
                print(f"  ✗ {call_label} output limit (attempt {attempt+1})", flush=True)
                return "[]"
            print(f"  ✗ {call_label} API error (attempt {attempt+1}): {e}", flush=True)
            if attempt < 2:
                import time; time.sleep(5)
    return "[]"


def parse_array(raw: str) -> list[dict]:
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        return [x for x in json.loads(m.group()) if isinstance(x, dict)]
    except json.JSONDecodeError:
        return []

# ── Core generation loop ──────────────────────────────────────────────────────

def run_job(client, relation, head_type, tail_type, target_n, contrast_rel,
            all_seeds, lang, defs, gen_model, ver_model, counter):
    label = f"{relation}({head_type},{tail_type})"
    class_seeds = [s for s in all_seeds.get(relation, [])
                   if s["head_type"] == head_type and s["tail_type"] == tail_type]
    if not class_seeds:
        class_seeds = all_seeds.get(relation, [])[:8]

    results, accepted, attempts = [], 0, 0
    max_attempts = target_n * 8

    print(f"\n{'─'*60}", flush=True)
    print(f"[{label}]  target={target_n}  seeds={len(class_seeds)}", flush=True)

    while accepted < target_n and attempts < max_attempts:
        sample = random.sample(class_seeds, min(5, len(class_seeds)))
        prompt = gen_prompt(relation, head_type, tail_type, sample, lang, defs,
                            contrast_rel=contrast_rel, n=5)
        raw = call_api(client, prompt, gen_model,
                       f"gen attempt {attempts+1}–{attempts+5} [{label}]")
        candidates = [c for c in parse_array(raw) if valid_spans(c)]
        attempts += 5
        print(f"  → {len(candidates)} valid-span candidates", flush=True)

        if not candidates:
            continue

        ver_raw = call_api(client, ver_prompt(candidates, lang), ver_model,
                           f"verify {len(candidates)} [{label}]")
        verified = parse_array(ver_raw)

        batch_accepted = 0
        for obj in verified:
            if not isinstance(obj, dict) or obj.get("verdict") != "ACCEPT":
                continue
            if not valid_spans(obj):
                continue
            rel = obj.get("relation", relation)
            if rel not in ([relation] + ([contrast_rel] if contrast_rel else [])):
                continue
            counter[0] += 1
            results.append(to_jsonl(obj, f"aug_{counter[0]:05d}", lang))
            if rel == relation:
                accepted += 1
                batch_accepted += 1

        print(f"  → accepted {batch_accepted} | total {accepted}/{target_n}", flush=True)

    status = "✓ done" if accepted >= target_n else f"⚠ only {accepted}/{target_n}"
    print(f"[{label}] {status}", flush=True)
    return results

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-tsv",   required=True)
    ap.add_argument("--train-texts", required=True)
    ap.add_argument("--out-jsonl",   required=True)
    ap.add_argument("--lang",        default="en", choices=["en", "ru"])
    ap.add_argument("--gen-model",   default="gpt-5-mini")
    ap.add_argument("--ver-model",   default="gpt-5.4-mini")
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--dry-run",     action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set (checked env + .env)")

    defs = RELATION_DEFS[args.lang]
    jobs = JOBS[args.lang]

    print(f"Language: {args.lang}  Generator: {args.gen_model}  Verifier: {args.ver_model}",
          flush=True)
    print(f"Jobs: {len(jobs)}  Total target: {sum(j[3] for j in jobs)} examples", flush=True)

    print(f"\nLoading seeds from {args.train_tsv} ...", flush=True)
    seeds = load_seeds(args.train_tsv, args.train_texts)
    for rel in sorted({j[0] for j in jobs}):
        print(f"  {rel}: {len(seeds.get(rel, []))} seeds", flush=True)

    if args.dry_run:
        job = jobs[0]
        s = [x for x in seeds.get(job[0], []) if x["head_type"] == job[1]][:5] or seeds.get(job[0], [])[:5]
        print("\n--- SAMPLE PROMPT ---")
        print(gen_prompt(job[0], job[1], job[2], s, args.lang, defs))
        return

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    counter = [0]
    all_results = []

    for relation, head_type, tail_type, target_n, contrast_rel in jobs:
        job_results = run_job(
            client, relation, head_type, tail_type, target_n, contrast_rel,
            seeds, args.lang, defs, args.gen_model, args.ver_model, counter,
        )
        all_results.extend(job_results)

    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for rec in all_results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    dist = Counter(r["relation"] for r in all_results)
    print(f"\n{'='*60}", flush=True)
    print(f"Wrote {len(all_results)} augmented examples → {out}", flush=True)
    for rel, n in sorted(dist.items()):
        print(f"  {rel}: {n}", flush=True)


if __name__ == "__main__":
    main()
