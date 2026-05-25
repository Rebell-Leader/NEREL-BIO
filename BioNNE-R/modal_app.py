import modal
import os
import sys
import time
from pathlib import Path

# 1. Standard Image (Our Framework & Extraction)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install("sentencepiece")  # required for mDeBERTa-v3-base tokenizer
    .run_commands(
        "git clone --depth 1 https://github.com/nerel-ds/NEREL-BIO.git /root/NEREL-BIO",
        "cd /root/NEREL-BIO/BioNNE-R && pip install -r baseline/requirements.txt || pip install transformers==4.40.0 torch pandas scikit-learn nltk tqdm pathlib",
        "python -m nltk.downloader punkt_tab"
    )
    .add_local_file(Path(__file__).parent / "baseline/bionne_dataset.py", remote_path="/root/NEREL-BIO/BioNNE-R/baseline/bionne_dataset.py")
    .add_local_file(Path(__file__).parent / "baseline/bionne_predict.py", remote_path="/root/NEREL-BIO/BioNNE-R/baseline/bionne_predict.py")
    .add_local_file(Path(__file__).parent / "baseline/bionne_train.py", remote_path="/root/NEREL-BIO/BioNNE-R/baseline/bionne_train.py")
    .add_local_file(Path(__file__).parent / "baseline/bionne_model.py", remote_path="/root/NEREL-BIO/BioNNE-R/baseline/bionne_model.py")
    .add_local_file(Path(__file__).parent / "baseline/prepare_data.py", remote_path="/root/NEREL-BIO/BioNNE-R/baseline/prepare_data.py")
    .add_local_file(Path(__file__).parent / "baseline/score.py", remote_path="/root/NEREL-BIO/BioNNE-R/baseline/score.py")
    .add_local_file(Path(__file__).parent / "baseline/data/valid_triplets.json", remote_path="/root/NEREL-BIO/BioNNE-R/baseline/data/valid_triplets.json")
    .add_local_file(Path(__file__).parent / "RECOVERY_PLAN.md", remote_path="/root/NEREL-BIO/BioNNE-R/RECOVERY_PLAN.md")
)

# 2. OpenNRE Image (Authors' Baseline)
opennre_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "curl")
    .pip_install("torch>=2.0.0", "transformers", "nltk", "pandas", "scikit-learn", "tqdm")
    .run_commands(
        "pip install git+https://github.com/thunlp/OpenNRE.git",
        "git clone --depth 1 https://github.com/nerel-ds/NEREL-BIO.git /root/NEREL-BIO",
        "python -m nltk.downloader punkt_tab"
    )
    .add_local_file(Path(__file__).parent / "baseline/baseline.py", remote_path="/root/NEREL-BIO/BioNNE-R/baseline/baseline.py", copy=True)
    .add_local_file(Path(__file__).parent / "baseline/patch_opennre.py", remote_path="/root/NEREL-BIO/BioNNE-R/baseline/patch_opennre.py", copy=True)
    .run_commands("cd /root/NEREL-BIO/BioNNE-R/baseline && python patch_opennre.py")
)

# 3. Volume
volume = modal.Volume.from_name("bionne-v2", create_if_missing=True)
MOUNT_PATH = "/vol"

app = modal.App("bionne-r-pipeline")

def wait_for_data_ready(data_name: str, timeout: int = 7200):
    vol_data = Path(MOUNT_PATH) / "data"
    ckpt_path = vol_data / f"{data_name}.ckpt"
    start = time.time()
    while True:
        volume.reload()  # required: Modal volumes need explicit reload to see writes from other containers
        if ckpt_path.exists():
            import json
            try:
                with open(ckpt_path) as f:
                    if json.load(f).get('last_doc_idx') == 9999999:
                        print(f"Data ready: {data_name}")
                        return True
            except: pass
        if time.time() - start > timeout:
            raise TimeoutError(f"Timeout waiting for {data_name} extraction.")
        print(f"Waiting for {data_name} extraction to finish...")
        time.sleep(60)

@app.function(image=image, volumes={MOUNT_PATH: volume}, timeout=7200, cpu=4, memory=32768)
def extract_single_split(config: dict):
    import sys, json, random, pandas as pd
    from pathlib import Path
    vol_data = Path(MOUNT_PATH) / "data"
    vol_data.mkdir(parents=True, exist_ok=True)
    out_path, ckpt_path = vol_data / config['out_name'], vol_data / f"{config['out_name']}.ckpt"
    if not config.get('force', False):
        if ckpt_path.exists():
            try:
                with open(ckpt_path, 'r') as f:
                    if json.load(f).get('last_doc_idx') == 9999999: return True
            except: pass
    last_doc_idx = -1
    if ckpt_path.exists():
        try:
            with open(ckpt_path, 'r') as f: last_doc_idx = json.load(f).get('last_doc_idx', -1)
        except: pass
    repo_root = Path("/root/NEREL-BIO")
    sys.path.insert(0, str(repo_root / "BioNNE-R/baseline"))
    import prepare_data, nltk
    texts = prepare_data.load_texts(Path(config['texts']))
    write_mode = 'a' if last_doc_idx >= 0 else 'w'
    # Parse annotation config and load NLTK tokenizer once per function call
    valid_type_pairs = prepare_data.parse_config(str(repo_root / "nerel-bio-v1.0/annotation.conf"))
    try:
        sent_tokenizer = nltk.data.load(f"tokenizers/punkt_tab/{config['lang']}.pickle")
    except LookupError:
        sent_tokenizer = nltk.data.load("tokenizers/punkt_tab/english.pickle")
    if config['type'] == 'labeled':
        df = pd.read_csv(Path(config['tsv']), sep="\t")
        pos_by_doc = {}
        for _, row in df.iterrows(): pos_by_doc.setdefault(str(row["document_id"]), []).append(row)
        ent_by_doc = prepare_data.load_entities(Path(config['entities'])) if config.get('entities') else None
        doc_ids = sorted(list(set(list(pos_by_doc.keys()) + (list(ent_by_doc.keys()) if ent_by_doc else []))))
        rng = random.Random(42)
        with open(out_path, write_mode, encoding="utf-8") as f:
            for idx, doc_id in enumerate(doc_ids):
                if idx <= last_doc_idx or doc_id not in texts: continue
                text, pos_list, pos_spans = texts[doc_id], pos_by_doc.get(doc_id, []), set()
                sentence_spans = list(sent_tokenizer.span_tokenize(text))
                for row in pos_list:
                    inst = prepare_data._make_instance(text, row["head_type"], row["head_text"], str(row["head_span"]), row["tail_type"], row["tail_text"], str(row["tail_span"]), row["relation"], doc_id, config['lang'], sentence_spans=sentence_spans)
                    if inst: f.write(json.dumps(inst, ensure_ascii=False) + "\n"); pos_spans.add((str(row["head_span"]), str(row["tail_span"])))
                neg_ratio = config.get('neg_ratio', 0)
                if ent_by_doc and doc_id in ent_by_doc and neg_ratio > 0:
                    cand = prepare_data.generate_pairs(ent_by_doc[doc_id], valid_type_pairs)
                    negs = [p for p in cand if (p[0][2], p[1][2]) not in pos_spans]
                    n_sample = min(neg_ratio * len(pos_spans), len(negs))
                    if n_sample > 0:
                        for p in rng.sample(negs, n_sample):
                            inst = prepare_data._make_instance(text, p[0][0], p[0][1], p[0][2], p[1][0], p[1][1], p[1][2], "no_relation", doc_id, config['lang'], sentence_spans=sentence_spans)
                            if inst: f.write(json.dumps(inst, ensure_ascii=False) + "\n")
                if (idx + 1) % 5 == 0:
                    with open(ckpt_path, 'w') as cf: json.dump({'last_doc_idx': idx}, cf)
                    volume.commit()
    else:
        ent_by_doc = prepare_data.load_entities(Path(config['tsv']))
        doc_ids = sorted(ent_by_doc.keys())
        with open(out_path, write_mode, encoding="utf-8") as f:
            for idx, doc_id in enumerate(doc_ids):
                if idx <= last_doc_idx or doc_id not in texts: continue
                sentence_spans = list(sent_tokenizer.span_tokenize(texts[doc_id]))
                pairs = prepare_data.generate_pairs(ent_by_doc[doc_id], valid_type_pairs)
                for p in pairs:
                    inst = prepare_data._make_instance(texts[doc_id], p[0][0], p[0][1], p[0][2], p[1][0], p[1][1], p[1][2], "no_relation", doc_id, config['lang'], sentence_spans=sentence_spans)
                    if inst: f.write(json.dumps(inst, ensure_ascii=False) + "\n")
                if (idx + 1) % 5 == 0:
                    with open(ckpt_path, 'w') as cf: json.dump({'last_doc_idx': idx}, cf)
                    volume.commit()
    with open(ckpt_path, 'w') as cf: json.dump({'last_doc_idx': 9999999}, cf)
    volume.commit()
    return True

@app.function(image=image, volumes={MOUNT_PATH: volume})
def concat_bilingual():
    import shutil
    vol_data = Path(MOUNT_PATH) / "data"
    for suffix in ["_train.txt", "_dev.txt", "_dev_blind.txt", "_test_blind.txt"]:
        out = vol_data / ("bilingual" + suffix)
        e, r = vol_data / ("eng" + suffix), vol_data / ("rus" + suffix)
        if e.exists() and r.exists():
            with open(out, "wb") as wfd:
                for s in [e, r]:
                    with open(s, "rb") as fd: shutil.copyfileobj(fd, wfd)
    volume.commit()

@app.function(image=image, gpu="A100", volumes={MOUNT_PATH: volume}, timeout=14400)
def train_model(
    train_data: str,
    val_data: str,
    ckpt_name: str,
    model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    epochs: int = 10,
    batch_size: int = 128,
    resume: bool = True,
    use_fp16: bool = False,
    gradient_checkpointing: bool = False,
):
    import sys, threading
    sys.path.insert(0, "/root/NEREL-BIO/BioNNE-R/baseline")
    import bionne_train
    vol_ckpt, vol_data = Path(MOUNT_PATH)/"checkpoints", Path(MOUNT_PATH)/"data"
    vol_ckpt.mkdir(parents=True, exist_ok=True)
    wait_for_data_ready(train_data)
    wait_for_data_ready(val_data)

    # Commit volume every 5 minutes so resume checkpoints survive Modal timeout.
    stop_event = threading.Event()
    def _periodic_commit():
        while not stop_event.wait(300):
            volume.commit()
    commit_thread = threading.Thread(target=_periodic_commit, daemon=True)
    commit_thread.start()
    try:
        bionne_train.train(
            str(vol_data/train_data), str(vol_data/val_data),
            str(vol_data/"rel2id.json"), str(vol_ckpt/ckpt_name),
            model_name, 512, batch_size, 2e-5, epochs, resume=resume,
            use_fp16=use_fp16, gradient_checkpointing=gradient_checkpointing,
        )
    finally:
        stop_event.set()
    volume.commit()

@app.function(image=image, gpu="L4", volumes={MOUNT_PATH: volume}, timeout=14400)
def predict_model(data_name: str, ckpt_name: str, out_name: str, threshold: float = 0.5, max_length: int = 512, batch_size: int = 256, existence_filter: bool = False, thresholds_json: str = ""):
    """Run bionne_predict on a trained checkpoint and write predictions to /vol/predictions/.

    thresholds_json: optional JSON object of per-class thresholds, e.g.
        '{"ALTERNATIVE_NAME": 0.5, "PHYSIOLOGY_OF": 0.95, ...}'
        When provided, overrides the scalar `threshold` parameter.
    """
    import sys, json as _json
    sys.path.insert(0, "/root/NEREL-BIO/BioNNE-R/baseline")
    import bionne_predict
    vol_data = Path(MOUNT_PATH) / "data"
    vol_ckpt = Path(MOUNT_PATH) / "checkpoints"
    vol_out  = Path(MOUNT_PATH) / "predictions"
    vol_out.mkdir(parents=True, exist_ok=True)
    wait_for_data_ready(data_name)
    effective_threshold = _json.loads(thresholds_json) if thresholds_json else threshold
    bionne_predict.predict(
        data_path=str(vol_data / data_name),
        ckpt_path=str(vol_ckpt / ckpt_name),
        output_path=str(vol_out / out_name),
        max_length=max_length,
        batch_size=batch_size,
        threshold=effective_threshold,
        existence_filter=existence_filter,
    )
    volume.commit()


@app.function(image=image, gpu="L4", volumes={MOUNT_PATH: volume}, timeout=7200)
def calibrate_threshold(track: str, max_length: int = 256, batch_size: int = 64, existence_filter: bool = False):
    """Run inference on dev_blind once and sweep thresholds to find optimal macro F1."""
    import sys
    sys.path.insert(0, "/root/NEREL-BIO/BioNNE-R/baseline")
    import bionne_predict
    vol_data = Path(MOUNT_PATH) / "data"
    vol_ckpt = Path(MOUNT_PATH) / "checkpoints"
    configs = {
        "en":  ("eng_dev_blind.txt",      "eng_dev_gold.tsv",      "mdeberta_en_v1.pt"),
        "ru":  ("rus_dev_blind.txt",       "rus_dev_gold.tsv",      "mdeberta_ru_v2.pt"),
        "bil": ("bilingual_dev_blind.txt", "bilingual_dev_gold.tsv","mdeberta_bilingual_v2.pt"),
    }
    data_file, gold_file, ckpt_file = configs[track]
    thresholds = [round(i/20, 2) for i in range(1, 21)]  # 0.05 to 1.00 in steps of 0.05
    best_t = bionne_predict.find_threshold(
        blind_dev_data_path=str(vol_data / data_file),
        ckpt_path=str(vol_ckpt / ckpt_file),
        gold_tsv_path=str(vol_data / gold_file),
        thresholds=thresholds,
        max_length=max_length,
        batch_size=batch_size,
        existence_filter=existence_filter,
    )
    return best_t


@app.function(image=opennre_image, gpu="L4", volumes={MOUNT_PATH: volume}, timeout=14400)
def author_baseline_predict(track: str, data_name: str, out_name: str, seed: int = 42, stochastic: bool = False, batch_size: int = 512):
    import sys, torch, random, numpy as np, json, pandas as pd, time
    from pathlib import Path
    vol_data, vol_ckpt, vol_out = Path(MOUNT_PATH)/"data", Path(MOUNT_PATH)/"author_checkpoints", Path(MOUNT_PATH)/"author_predictions"
    vol_ckpt.mkdir(parents=True, exist_ok=True); vol_out.mkdir(parents=True, exist_ok=True)
    wait_for_data_ready(data_name)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    ckpt_map = {'eng': 'eng_model.pth.tar', 'rus': 'rus_model.pth.tar', 'bil': 'bil_model.pth.tar'}
    ckpt_path = vol_ckpt / ckpt_map[track]
    if not ckpt_path.exists():
        import subprocess
        subprocess.run(["curl", "-L", f"https://github.com/nerel-ds/NEREL-BIO/releases/download/BioNNE-R/{ckpt_map[track]}", "-o", str(ckpt_path)], check=True)

    with open(vol_data / "rel2id.json") as f: rel2id = json.load(f)
    id2rel = {v: k for k, v in rel2id.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  track={track}  |  stochastic={stochastic}  |  batch_size={batch_size}", flush=True)

    import opennre
    opennre_enc = opennre.encoder.BERTEntityEncoder(max_length=256, pretrain_path="bert-base-multilingual-cased")
    model = opennre.model.SoftmaxNN(opennre_enc, len(rel2id), rel2id)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    if stochastic:
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout): m.train()
    print("Model loaded.", flush=True)

    # Load all instances
    instances = []
    with open(str(vol_data / data_name), encoding="utf-8") as f:
        for line in f:
            if line.strip(): instances.append(json.loads(line))
    print(f"Loaded {len(instances):,} instances.", flush=True)

    # ── Fast batch tokenization with correct entity marker insertion ───────────
    # BertTokenizerFast does NOT tokenize [unusedN] strings as single tokens in text
    # (they get split into subwords like `[`, `unu`, `##sed`, `##N`, `]`).
    # Correct approach: tokenize raw text → find entity token positions via
    # offset_mapping → manually insert marker token IDs (UNK=100 for [unused0],
    # ID1=1 for [unused1], ID2=2 for [unused2], ID3=3 for [unused3]).
    # This replicates what OpenNRE's slow-tokenizer enc.tokenize() produces.
    from transformers import BertTokenizerFast
    hf_tok = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")

    # mBERT marker token IDs (same as OpenNRE uses after word-level split)
    UNK_ID = 100   # [unused0] → maps to [UNK]
    ID1, ID2, ID3 = 1, 2, 3  # [unused1], [unused2], [unused3] are in mBERT vocab
    CLS_ID, SEP_ID, PAD_ID = 101, 102, 0
    MAX_LEN = 256

    def build_marked_sequence(raw_ids, offsets, h_start, h_end, t_start, t_end):
        """Return (token_ids_256, attn_mask_256, pos1, pos2) with entity markers inserted.

        Markers: UNK_ID=[unused0] before head, ID1=[unused1] after head end,
                 ID2=[unused2] before tail, ID3=[unused3] after tail end.
        pos1 = token index (1-based after CLS) of [unused0] marker.
        pos2 = token index (1-based after CLS) of [unused2] marker.
        """
        core_ids = raw_ids[1:-1]        # strip [CLS] and [SEP]
        core_off = offsets[1:-1]        # strip [CLS] and [SEP] offsets

        # Find first token starting at/after each entity boundary
        def first_tok_at(char_pos):
            for j, (cs, _) in enumerate(core_off):
                if cs >= char_pos:
                    return j
            return len(core_ids)

        h_tok_s = first_tok_at(h_start)
        h_tok_e = first_tok_at(h_end)
        t_tok_s = first_tok_at(t_start)
        t_tok_e = first_tok_at(t_end)

        # Build insertion map: position → list of (token_id, is_pos1/pos2)
        ins = {}
        for pos, tok_id, tag in [(h_tok_s, UNK_ID, 1), (h_tok_e, ID1, 0),
                                   (t_tok_s, ID2,    2), (t_tok_e, ID3, 0)]:
            ins.setdefault(pos, []).append((tok_id, tag))

        result = []
        pos1, pos2 = 1, 1  # fallback to position 1
        for j, tok_id in enumerate(core_ids):
            if j in ins:
                for m_id, tag in ins[j]:
                    if tag == 1: pos1 = len(result) + 1  # +1 for CLS
                    if tag == 2: pos2 = len(result) + 1
                    result.append(m_id)
            result.append(tok_id)
        if len(core_ids) in ins:
            for m_id, tag in ins[len(core_ids)]:
                if tag == 1: pos1 = len(result) + 1
                if tag == 2: pos2 = len(result) + 1
                result.append(m_id)

        # Truncate at MAX_LEN - 2 (leave room for CLS and SEP)
        core_trunc = result[:MAX_LEN - 2]
        seq = [CLS_ID] + core_trunc + [SEP_ID]
        attn_len = len(seq)
        seq  += [PAD_ID] * (MAX_LEN - attn_len)
        attn  = [1] * attn_len + [0] * (MAX_LEN - attn_len)
        return seq, attn, pos1, pos2

    CHUNK_SIZE = 5_000
    all_token_ids, all_attn_masks, all_pos1, all_pos2 = [], [], [], []
    t_tok = time.time()
    for chunk_start in range(0, len(instances), CHUNK_SIZE):
        chunk = instances[chunk_start:chunk_start + CHUNK_SIZE]
        texts = [inst['text'] for inst in chunk]

        # Batch tokenize raw texts (no markers); truncation at MAX_LEN-4 to leave headroom
        enc_out = hf_tok(texts, max_length=MAX_LEN - 4, truncation=True, padding=False,
                         return_offsets_mapping=True)

        for i, inst in enumerate(chunk):
            raw_ids  = enc_out['input_ids'][i]
            offsets  = enc_out['offset_mapping'][i]
            h_s, h_e = inst['h']['pos'][0], inst['h']['pos'][1]
            t_s, t_e = inst['t']['pos'][0], inst['t']['pos'][1]
            seq, attn, p1, p2 = build_marked_sequence(raw_ids, offsets, h_s, h_e, t_s, t_e)
            all_token_ids.append(seq)
            all_attn_masks.append(attn)
            all_pos1.append(p1)
            all_pos2.append(p2)

        done = chunk_start + len(chunk)
        elapsed = time.time() - t_tok
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(instances) - done) / rate if rate > 0 else 0
        print(f"  Tokenized {done:,}/{len(instances):,} ({rate:.0f} inst/s | ETA {eta/60:.1f} min)", flush=True)

    token_ids_t = torch.tensor(all_token_ids, dtype=torch.long)    # [N, MAX_LEN]
    attn_mask_t = torch.tensor(all_attn_masks, dtype=torch.long)   # [N, MAX_LEN]
    pos1_t      = torch.tensor(all_pos1, dtype=torch.long)          # [N]
    pos2_t      = torch.tensor(all_pos2, dtype=torch.long)          # [N]
    del all_token_ids, all_attn_masks, all_pos1, all_pos2
    print(f"Tokenization complete: {time.time()-t_tok:.1f}s | shape {tuple(token_ids_t.shape)}", flush=True)

    # GPU batched inference — FP32 (author model was trained without FP16)
    rows = []
    n_batches = (len(instances) + batch_size - 1) // batch_size
    t0 = time.time()
    with torch.no_grad():
        for bi in range(n_batches):
            s, e = bi * batch_size, min((bi + 1) * batch_size, len(instances))
            tokens = token_ids_t[s:e].to(device)
            masks  = attn_mask_t[s:e].to(device)
            p1     = pos1_t[s:e].to(device).unsqueeze(1)   # [B, 1] — scatter_ requires 2D
            p2     = pos2_t[s:e].to(device).unsqueeze(1)
            logits = model(tokens, masks, p1, p2)
            pred_ids = logits.argmax(dim=-1).cpu().tolist()

            for inst, pred_id in zip(instances[s:e], pred_ids):
                rows.append({
                    "document_id": inst["doc_id"],
                    "relation":    id2rel[pred_id],
                    "head_text":   inst["h"]["name"],
                    "head_span":   inst["head_span"],
                    "head_type":   inst["head_type"],
                    "tail_text":   inst["t"]["name"],
                    "tail_span":   inst["tail_span"],
                    "tail_type":   inst["tail_type"],
                })

            if (bi + 1) % 100 == 0 or (bi + 1) == n_batches:
                elapsed = time.time() - t0
                rate = e / elapsed
                eta = (len(instances) - e) / rate if rate > 0 else 0
                print(f"  batch {bi+1}/{n_batches} | {e:,}/{len(instances):,} inst | "
                      f"{rate:.0f} inst/s | ETA {eta/60:.1f} min", flush=True)

    result_df = pd.DataFrame(rows)
    result_df = result_df[result_df["relation"] != "no_relation"].reset_index(drop=True)
    result_df.to_csv(vol_out / out_name, sep="\t", index=False)
    print(f"Saved {len(result_df):,} positive predictions → {vol_out / out_name}", flush=True)
    volume.commit()


@app.function(image=image, volumes={MOUNT_PATH: volume}, timeout=300, cpu=4, memory=8192)
def ensemble_track(track: str):
    """Majority-vote ensemble of deterministic + 3 stochastic author predictions for a track key (eng/rus/bil)."""
    import pandas as pd
    vol_out = Path(MOUNT_PATH) / "author_predictions"
    volume.reload()
    input_names = [
        f"author_{track}_test_deterministic.tsv",
        f"author_{track}_test_stochastic_s1.tsv",
        f"author_{track}_test_stochastic_s2.tsv",
        f"author_{track}_test_stochastic_s3.tsv",
    ]
    out_name = f"author_{track}_test_ensemble.tsv"
    dfs = []
    for name in input_names:
        df = pd.read_csv(vol_out / name, sep="\t", dtype=str)
        dfs.append(df)
        print(f"  {name}: {len(df):,} rows", flush=True)
    combined = pd.concat(dfs, ignore_index=True)
    n_runs = len(input_names)
    KEY = "__key__"
    combined[KEY] = (combined["document_id"] + "|||" +
                     combined["head_span"]    + "|||" +
                     combined["tail_span"])
    n_unique = combined[KEY].nunique()
    print(f"Unique pairs: {n_unique:,}", flush=True)

    # For each key, count votes per relation (vectorized groupby)
    vote_counts = (combined.groupby([KEY, "relation"])
                            .size()
                            .reset_index(name="votes"))
    # winner = relation with most votes per key
    idx = vote_counts.groupby(KEY)["votes"].idxmax()
    winners = vote_counts.loc[idx].set_index(KEY)["relation"]

    # Keep rows where winner != no_relation; attach metadata from first occurrence
    meta = combined.drop_duplicates(KEY).set_index(KEY)
    result_rows = []
    for key, winner in winners.items():
        if winner == "no_relation":
            continue
        ref = meta.loc[key]
        result_rows.append({
            "document_id": ref["document_id"], "relation": winner,
            "head_text": ref["head_text"], "head_span": ref["head_span"],
            "head_type": ref["head_type"], "tail_text": ref["tail_text"],
            "tail_span": ref["tail_span"], "tail_type": ref["tail_type"],
        })
    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(vol_out / out_name, sep="\t", index=False)
    print(f"Ensemble → {out_name}: {len(result_df):,} positive predictions", flush=True)
    volume.commit()


@app.function(image=image, volumes={MOUNT_PATH: volume}, timeout=120, cpu=2)
def inspect_checkpoints(ckpt_names: list) -> dict:
    """Return {ckpt_name: {'epoch': N, 'best_f1': X}} for each name in ckpt_names."""
    import torch
    results = {}
    ckpt_dir = Path(MOUNT_PATH) / "checkpoints"
    volume.reload()
    for name in ckpt_names:
        resume_path = ckpt_dir / (name + ".resume")
        if resume_path.exists():
            state = torch.load(str(resume_path), map_location="cpu", weights_only=False)
            results[name] = {
                "epoch": state.get("epoch", 0),
                "best_f1": state.get("best_macro_f1", state.get("best_f1", 0.0)),
            }
        else:
            results[name] = {"epoch": 0, "best_f1": 0.0}
    return results


@app.local_entrypoint()
def main(neg_ratio: int = 15, force: bool = False):
    dr = "/root/NEREL-BIO/BioNNE-R/data"
    configs = [
        {'type':'labeled', 'tsv':f'{dr}/en/train/eng-train-rel.tsv', 'texts':f'{dr}/en/train/texts', 'entities':f'{dr}/en/train/eng-train-ent.tsv', 'out_name':'eng_train.txt', 'lang':'english', 'neg_ratio':neg_ratio, 'force':force},
        {'type':'labeled', 'tsv':f'{dr}/en/dev/eng-dev-rel.tsv', 'texts':f'{dr}/en/dev/texts', 'out_name':'eng_dev.txt', 'lang':'english', 'force':force},
        {'type':'labeled', 'tsv':f'{dr}/ru/train/rus-train-rel.tsv', 'texts':f'{dr}/ru/train/texts', 'entities':f'{dr}/ru/train/rus-train-ent.tsv', 'out_name':'rus_train.txt', 'lang':'russian', 'neg_ratio':neg_ratio, 'force':force},
        {'type':'labeled', 'tsv':f'{dr}/ru/dev/rus-dev-rel.tsv', 'texts':f'{dr}/ru/dev/texts', 'out_name':'rus_dev.txt', 'lang':'russian', 'force':force},
        {'type':'blind', 'tsv':f'{dr}/en/dev/eng-dev-ent.tsv', 'texts':f'{dr}/en/dev/texts', 'out_name':'eng_dev_blind.txt', 'lang':'english', 'force':force},
        {'type':'blind', 'tsv':f'{dr}/ru/dev/rus-dev-ent.tsv', 'texts':f'{dr}/ru/dev/texts', 'out_name':'rus_dev_blind.txt', 'lang':'russian', 'force':force},
        {'type':'blind', 'tsv':f'{dr}/en/test/eng-test-ent.tsv', 'texts':f'{dr}/en/test/texts', 'out_name':'eng_test_blind.txt', 'lang':'english', 'force':force},
        {'type':'blind', 'tsv':f'{dr}/ru/test/rus-test-ent.tsv', 'texts':f'{dr}/ru/test/texts', 'out_name':'rus_test_blind.txt', 'lang':'russian', 'force':force},
    ]
    for _ in extract_single_split.map(configs): pass
    concat_bilingual.remote()
