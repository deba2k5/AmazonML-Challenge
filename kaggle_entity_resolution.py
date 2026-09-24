# %% [markdown]
# # Amazon ML Challenge 2026: Business Entity Resolution (Kaggle notebook)
#
# **Pipeline:** normalise text, then block (char-trigram TF-IDF top-K on the GPU), then pair features (rapidfuzz), then LightGBM, then a threshold chosen on macro F0.5, then write the output files.
#
# **Before running:**
# 1. Upload the challenge zip (`student_resource`) as a **private** Kaggle Dataset, then *Add Input* to this notebook.
# 2. Settings: Accelerator **GPU P100**, Internet **On** (only for `pip install`), Persistence **Files only**.
# 3. First run: set `TRAIN_S1 = 50_000` to smoke-test. Then use the full value and *Save Version, Save & Run All* so it runs in the background.
#
# Facts checked on the training data that the design relies on:
# * matched pairs **always have the same country**, so every search is done per country (France becomes its own group automatically);
# * every S2/S3 record belongs to **at most one** S1 entity, so each candidate is kept only for its best S1 (a one-to-one rule);
# * 5.6% of S1 entities are singletons, and 26% of S2/S3 records match nothing (distractors).

# %%
!pip install -q rapidfuzz unidecode

# %% [markdown]
# ## 1. Imports, paths, config, GPU check

# %%
import os, re, gc, glob, json, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import lightgbm as lgb
from multiprocessing import Pool
from unidecode import unidecode
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as l2norm
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

LOCAL_TEST = os.environ.get("LOCAL_TEST") == "1"   # only for a quick CPU run outside Kaggle

# ---- paths: auto-detect the uploaded dataset ----
_root = os.environ.get("DATA_ROOT", "/kaggle/input")
_hit = [p for p in glob.glob(f"{_root}/**/train_source1.tsv", recursive=True) if "__MACOSX" not in p]
assert _hit, "Dataset not found: upload student_resource as a Kaggle Dataset and add it to the notebook"
DATA = os.path.dirname(os.path.dirname(_hit[0]))          # .../dataset
RES = os.path.dirname(DATA)                               # .../student_resource
WORK = os.environ.get("WORK_DIR", "/kaggle/working")
OUT, CACHE = f"{WORK}/output", f"{WORK}/cache"
os.makedirs(OUT, exist_ok=True); os.makedirs(CACHE, exist_ok=True)

CFG = dict(
    SEED=42,
    N_HASH=2**17,          # char trigrams over [a-z0-9 ] = 37^3 ~ 50k, so 2^17 buckets has few collisions
    K_NAME=10,             # top-K by name similarity, per source (S2 and S3 searched separately)
    K_ADDR=5,              # top-K by address similarity, per source
    K_COMBO=10,            # top-K by name+address similarity, per source
    K_SKEL=5,              # top-K by consonant-skeleton name similarity, per source
    MIN_COS=0.15,          # drop blocker hits below this cosine
    Q_BATCH=512,           # S1 rows per GPU matmul (lower to 256 on CUDA out-of-memory)
    Q_CHUNK=250_000,       # S1 rows per feature/predict chunk (bounds RAM)
    TRAIN_S1=300_000,      # train S1 entities used to build training pairs (50_000 for a smoke test)
    VAL_FRAC=0.2,          # share of those S1 entities held out for validation / threshold
    N_JOBS=os.cpu_count(),
    NROWS=None,            # read only the first N rows of every file (debug only)
    TEST_S1_LIMIT=None,    # only predict the first N test S1 rows per country (debug only)
)
if LOCAL_TEST:
    CFG.update(TRAIN_S1=6_000, NROWS=300_000, TEST_S1_LIMIT=1_500, N_JOBS=1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    print(torch.cuda.get_device_name(0), "| torch", torch.__version__, "| archs", torch.cuda.get_arch_list())
    try:
        torch.sparse.mm(torch.eye(3).to_sparse_csr().cuda(), torch.ones(3, 2, device="cuda")).sum().item()
    except Exception as e:
        raise RuntimeError("This PyTorch build cannot run on this GPU (P100 = sm_60). "
                           "Switch Accelerator to 'GPU T4 x2' and rerun.") from e
print("DATA:", DATA, "| device:", DEVICE, "| cpus:", CFG["N_JOBS"])

def tic():
    t = time.time()
    return lambda msg: print(f"  {msg}: {time.time() - t:.1f}s")

# %% [markdown]
# ## 2. Loading helpers and a first look at the data

# %%
READ_KW = dict(sep="\t", dtype=str, keep_default_na=False, na_filter=False, quoting=3)

def read_tsv(path):
    return pd.read_csv(path, nrows=CFG["NROWS"], **READ_KW)

def src_path(split, n):
    return f"{DATA}/{split}/{split}_source{n}.tsv"

gt = read_tsv(f"{DATA}/train/train_ground_truth.tsv")
gt_pairs = (gt[gt.matched_entity_ids != ""]
            .assign(cand_id=lambda d: d.matched_entity_ids.str.split(","))
            .explode("cand_id")[["source1_entity_id", "cand_id"]]
            .rename(columns={"source1_entity_id": "s1_id"})
            .reset_index(drop=True))
n_match = gt.matched_entity_ids.map(lambda s: len(s.split(",")) if s else 0)
print("train S1:", len(gt), "| true pairs:", len(gt_pairs))
print((n_match.clip(upper=6).value_counts(normalize=True).sort_index() * 100).round(1).rename("% of S1 by #matches"))
print(pd.read_csv(src_path("train", 2), nrows=8, **READ_KW).to_string())

# %% [markdown]
# ## 3. Normalisation (cached to parquet)
# * `name_norm`: lowercased, accents and Devanagari transliterated (unidecode), punctuation removed, abbreviations expanded, website parts (`www.`, `.com`) removed
# * `name_core`: `name_norm` without legal suffixes (inc, llc, pvt, limited, sarl, ...)
# * `name_key`: `name_core` without spaces, so `maurewilliamscolombier.com` equals `Maure Williams Colombier Inc`
# * `name_skel`: consonant skeleton, which helps with transliterations like `praaim sevn impeks` vs `prime seven impex`
# * `addr_norm`: address with abbreviations expanded and US/Indian region names mapped to short codes; also the digits: `nums`, `house`, `post`

# %%
LEGAL = {"inc", "incorporated", "llc", "ltd", "limited", "pvt", "private", "corp", "corporation", "co", "company",
         "llp", "lp", "plc", "pllc", "pc", "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "gmbh", "the",
         # transliterated Hindi forms seen in the data
         "praaivett", "limittedd", "praa", "li", "kmpnii"}
NAME_ABBR = {"pvt": "private", "ltd": "limited", "corp": "corporation", "co": "company", "intl": "international",
             "mfg": "manufacturing", "svc": "services", "svcs": "services", "assoc": "associates",
             "bros": "brothers", "mgmt": "management", "tech": "technologies", "grp": "group"}
ADDR_ABBR = {"rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
             "bd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court", "pl": "place", "sq": "square",
             "hwy": "highway", "pkwy": "parkway", "cir": "circle", "ter": "terrace", "trl": "trail",
             "apt": "apartment", "ste": "suite", "fl": "floor", "flr": "floor", "bldg": "building",
             "nr": "near", "opp": "opposite", "no": "number", "hno": "house number", "twp": "township",
             "mt": "mount", "ft": "fort", "n": "north", "s": "south", "e": "east", "w": "west"}
REGION = {  # full name -> short code (US states, Indian states/UTs)
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh",
    "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "ts", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "west bengal": "wb", "delhi": "dl", "new delhi": "dl",
    "jammu and kashmir": "jk", "puducherry": "py", "chandigarh": "ch",
}
_REGION_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, REGION), key=len, reverse=True)) + r")\b")
_NONALNUM = re.compile(r"[^a-z0-9]+")
_WEB = re.compile(r"^(https?://)?(www\.)?|\.(com|net|org|biz|info|co|in|fr|us)(\.[a-z]{2})?\b")
_NONLATIN = re.compile(r"[^\x00-ɏ]")
_REPEAT = re.compile(r"(.)\1+")
_VOWEL = re.compile(r"[aeiouyh]")

def _skel(key):
    s = key.replace("ph", "f").replace("c", "k").replace("q", "k").replace("z", "s")
    return _REPEAT.sub(r"\1", _VOWEL.sub("", _REPEAT.sub(r"\1", s)))

def name_fields(raw):
    s = _WEB.sub(" ", unidecode(raw).lower()).replace("&", " and ")
    toks = [NAME_ABBR.get(t, t) for t in _NONALNUM.sub(" ", s).split()]
    core = [t for t in toks if t not in LEGAL] or toks
    key = "".join(core)
    return " ".join(toks), " ".join(core), key, _skel(key)

def addr_fields(raw):
    s = " ".join(ADDR_ABBR.get(t, t) for t in _NONALNUM.sub(" ", unidecode(raw).lower()).split())
    s = _REGION_RE.sub(lambda m: REGION[m.group(1)], s)
    nums = re.findall(r"\d+", s)
    post = next((n for n in reversed(nums[1:]) if len(n) in (5, 6)), "")   # nums[0] is usually the house no.
    return s, " ".join(sorted(set(nums))), post, (nums[0] if nums else "")

def _norm_rows(args):
    names, addrs = args
    return [name_fields(x) for x in names], [addr_fields(x) for x in addrs]

def pmap(fn, items):
    """Map over chunks with a process pool (or serially when N_JOBS == 1)."""
    if CFG["N_JOBS"] == 1 or len(items) == 1:
        return [fn(x) for x in items]
    with Pool(CFG["N_JOBS"]) as pool:
        return pool.map(fn, items)

def normalize_df(df):
    step = 100_000
    chunks = [(df.business_name.values[i:i + step], df.business_address.values[i:i + step])
              for i in range(0, len(df), step)]
    res = pmap(_norm_rows, chunks)
    out = pd.DataFrame([r for a, _ in res for r in a], columns=["name_norm", "name_core", "name_key", "name_skel"])
    out[["addr_norm", "nums", "post", "house"]] = pd.DataFrame([r for _, b in res for r in b], index=out.index)
    out.insert(0, "entity_id", df.entity_id.values)
    out["country"] = df.country.values
    out["nonlatin"] = df.business_name.str.contains(_NONLATIN).values.astype(np.int8)
    return out

def norm_path(split, n):
    return f"{CACHE}/{split}_s{n}.parquet"

for split in ("train", "test"):
    for n in (1, 2, 3):
        if not os.path.exists(norm_path(split, n)):
            t = tic()
            normalize_df(read_tsv(src_path(split, n))).to_parquet(norm_path(split, n), index=False)
            t(f"normalised {split} source{n}")
        gc.collect()

def read_country(split, n, country):
    return pd.read_parquet(norm_path(split, n), filters=[("country", "==", country)]).reset_index(drop=True)

pd.read_parquet(norm_path("train", 3)).head(8)

# %% [markdown]
# ## 4. Blocking: char-trigram TF-IDF with exact top-K cosine on the GPU
# For each S1 record, per source (S2, S3): top `K_NAME` neighbours on `name_key` plus top `K_ADDR` neighbours on `addr_norm`.
# The address blocker finds matches whose names are completely different (trade names, Hindi-script names).
# Similarities are an exact sparse x dense matmul on the GPU (cuSPARSE) followed by `topk`.

# %%
def _hash_chunk(args):
    texts, analyzer = args
    hv = HashingVectorizer(analyzer=analyzer, ngram_range=(3, 3), n_features=CFG["N_HASH"],
                           alternate_sign=False, norm=None, lowercase=False, dtype=np.float32)
    return hv.transform(texts)

def tfidf_family(text_arrays, analyzer):
    """Hash char-3grams for several text arrays, share one IDF across them, L2-normalise rows."""
    step = 200_000
    mats = []
    for texts in text_arrays:
        parts = pmap(_hash_chunk, [(texts[i:i + step], analyzer) for i in range(0, len(texts), step)])
        mats.append(sp.vstack(parts).tocsr() if parts else sp.csr_matrix((0, CFG["N_HASH"]), dtype=np.float32))
    n_docs = sum(m.shape[0] for m in mats)
    dfreq = sum(np.bincount(m.indices, minlength=CFG["N_HASH"]) for m in mats)
    idf = sp.diags((np.log((1 + n_docs) / (1 + dfreq)) + 1).astype(np.float32))
    out = []
    for m in mats:
        m.data = np.log1p(m.data)                      # sublinear tf
        out.append(l2norm(m @ idf).astype(np.float32).tocsr())
    return out

def _to_torch_csr(m):
    return torch.sparse_csr_tensor(torch.from_numpy(m.indptr.astype(np.int64)),
                                   torch.from_numpy(m.indices.astype(np.int64)),
                                   torch.from_numpy(m.data), size=m.shape).to(DEVICE)

@torch.no_grad()
def knn(Qm, Cm, k):
    """Exact top-k cosine neighbours in Cm for every row of Qm (both L2-normalised CSR). Returns (idx, score)."""
    nq, nc = Qm.shape[0], Cm.shape[0]
    k = min(k, nc)
    if nq == 0 or k == 0:
        return np.zeros((nq, 0), np.int64), np.zeros((nq, 0), np.float32)
    C = _to_torch_csr(Cm)
    idx, sc = np.empty((nq, k), np.int64), np.empty((nq, k), np.float32)
    B = CFG["Q_BATCH"]
    for s in range(0, nq, B):
        q = Qm[s:s + B].tocoo()
        qd = torch.zeros((CFG["N_HASH"], q.shape[0]), device=DEVICE)
        qd[torch.from_numpy(q.col.astype(np.int64)).to(DEVICE),
           torch.from_numpy(q.row.astype(np.int64)).to(DEVICE)] = torch.from_numpy(q.data).to(DEVICE)
        v, i = torch.topk(torch.sparse.mm(C, qd), k, dim=0)          # (k, b)
        idx[s:s + B], sc[s:s + B] = i.T.cpu().numpy(), v.T.cpu().numpy()
    del C, qd
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return idx, sc

def rowdot(A, B, ia, ib, step=2_000_000):
    """cosine(A[ia[j]], B[ib[j]]) for every j (rows are L2-normalised)."""
    out = np.empty(len(ia), np.float32)
    for s in range(0, len(ia), step):
        out[s:s + step] = np.asarray(A[ia[s:s + step]].multiply(B[ib[s:s + step]]).sum(axis=1)).ravel()
    return out

# Each blocker: (text builder, analyzer, K per source).
#   name  - spaceless core name: typos, word order, website-style names
#   addr  - address alone: trade names / DBA with the same address
#   combo - name + address together: common names ("global infotech") are disambiguated by address
#   skel  - consonant skeleton of the name: transliterations ("yuunaaittedd phuudds" ~ "united foods")
BLOCKERS = {
    "name": (lambda d: (" " + d.name_key + " ").values, "char", "K_NAME"),
    "addr": (lambda d: d.addr_norm.values, "char_wb", "K_ADDR"),
    "combo": (lambda d: (d.name_key + " " + d.addr_norm).values, "char_wb", "K_COMBO"),
    "skel": (lambda d: (" " + d.name_skel + " ").values, "char", "K_SKEL"),
}

def block_country(Q, C2, C3):
    """Candidates for one country. Q = S1 rows; C = concat(C2, C3). Returns pairs (q, c) indexing Q and C,
    with each blocker's rank (99 = not found by it) and cosine for every pair."""
    t = tic()
    mats, parts = {}, []
    for field, (text, analyzer, k_key) in BLOCKERS.items():
        qm, m2, m3 = tfidf_family([text(Q), text(C2), text(C3)], analyzer)
        mats[field] = (qm, sp.vstack([m2, m3]).tocsr())
        off = 0
        for cm in (m2, m3):
            idx, sc = knn(qm, cm, CFG[k_key])
            k = idx.shape[1]
            df = pd.DataFrame({"q": np.repeat(np.arange(len(Q), dtype=np.int64), k), "c": idx.ravel() + off,
                               f"{field}_rank": np.tile(np.arange(k, dtype=np.float32), len(Q)), "s": sc.ravel()})
            parts.append(df[df.s >= CFG["MIN_COS"]].drop(columns="s"))
            off += cm.shape[0]
        t(f"blocker {field}")
    ranks = [f"{f}_rank" for f in BLOCKERS]
    P = (pd.concat(parts, ignore_index=True).groupby(["q", "c"], as_index=False)[ranks].min()
         .fillna({r: 99 for r in ranks}))
    for field, (qm, cm) in mats.items():
        P[f"{field}_cos"] = rowdot(qm, cm, P.q.values, P.c.values)
    P["is_s3"] = (P.c.values >= len(C2)).astype(np.int8)
    t(f"{len(P):,} pairs ({len(P) / max(len(Q), 1):.1f} per S1)")
    return P

# %% [markdown]
# ## 5. Pair features
# Name/address string similarities (rapidfuzz, multithreaded), digit agreement, missing-field flags, and
# context features comparing each candidate against the best candidate of the same S1 (the `_gap` columns).
# Nothing depends on the country value, so the model transfers to France.

# %%
def _num_jacc(a, b):
    if not a or not b:
        return -1.0
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / len(sa | sb)

def _eq_or_missing(a, b):
    return np.where((a == "") | (b == ""), -1, (a == b).astype(np.int8)).astype(np.int8)

def pair_features(P, Q, C):
    q, c = P.q.values, P.c.values
    col = lambda df, name, ix: df[name].values[ix]
    qn, cn = col(Q, "name_norm", q), col(C, "name_norm", c)
    qc, cc = col(Q, "name_core", q), col(C, "name_core", c)
    qk, ck = col(Q, "name_key", q), col(C, "name_key", c)
    qs, cs = col(Q, "name_skel", q), col(C, "name_skel", c)
    qa, ca = col(Q, "addr_norm", q), col(C, "addr_norm", c)
    W = dict(workers=-1, dtype=np.float32)
    sim = lambda a, b, scorer: cpdist(a.tolist(), b.tolist(), scorer=scorer, **W)

    F = {k: P[k].values.astype(np.float32) for k in P.columns if k.endswith(("_cos", "_rank")) or k == "is_s3"}
    F["n_ratio"] = sim(qn, cn, fuzz.ratio)
    F["n_tsort"] = sim(qn, cn, fuzz.token_sort_ratio)
    F["n_tset"] = sim(qn, cn, fuzz.token_set_ratio)
    F["core_tset"] = sim(qc, cc, fuzz.token_set_ratio)
    F["key_partial"] = sim(qk, ck, fuzz.partial_ratio)
    F["key_jw"] = sim(qk, ck, JaroWinkler.normalized_similarity)
    F["skel_ratio"] = sim(qs, cs, fuzz.ratio)
    q_ae, c_ae = (qa == ""), (ca == "")
    any_ae = q_ae | c_ae
    F["a_ratio"] = np.where(any_ae, -1, sim(qa, ca, fuzz.ratio)).astype(np.float32)
    F["a_tset"] = np.where(any_ae, -1, sim(qa, ca, fuzz.token_set_ratio)).astype(np.float32)
    F["a_partial"] = np.where(any_ae, -1, sim(qa, ca, fuzz.partial_ratio)).astype(np.float32)
    F["house_eq"] = _eq_or_missing(col(Q, "house", q), col(C, "house", c))
    F["post_eq"] = _eq_or_missing(col(Q, "post", q), col(C, "post", c))
    F["num_jacc"] = np.fromiter((_num_jacc(a, b) for a, b in zip(col(Q, "nums", q), col(C, "nums", c))),
                                np.float32, len(q))
    F["key_contain"] = np.fromiter(((a in b or b in a) if a and b else False for a, b in zip(qk, ck)),
                                   np.int8, len(q))
    lq, lc = np.fromiter(map(len, qk), np.int32, len(q)), np.fromiter(map(len, ck), np.int32, len(q))
    F["key_len_ratio"] = (np.minimum(lq, lc) / np.maximum(np.maximum(lq, lc), 1)).astype(np.float32)
    F["q_addr_empty"], F["c_addr_empty"] = q_ae.astype(np.int8), c_ae.astype(np.int8)
    F["c_name_empty"] = (ck == "").astype(np.int8)
    F["q_nonlatin"], F["c_nonlatin"] = col(Q, "nonlatin", q), col(C, "nonlatin", c)

    X = pd.DataFrame(F)
    g = X.groupby(q)
    for k in ("name_cos", "addr_cos", "combo_cos", "skel_cos", "n_tset", "key_jw", "a_tset"):
        X[k + "_gap"] = (g[k].transform("max") - X[k]).astype(np.float32)
    X["n_cands"] = g["name_cos"].transform("size").astype(np.float32)
    X["n_tset_rank"] = g["n_tset"].rank(ascending=False, method="min").astype(np.float32)
    return X

def run_country(Q, C2, C3):
    """Block + featurise one country, chunked over S1 rows. Yields (X, s1_ids, cand_ids)."""
    C = pd.concat([C2, C3], ignore_index=True)
    if len(Q) == 0 or len(C) == 0:
        return
    P = block_country(Q, C2, C3)
    bounds = np.searchsorted(P.q.values, np.arange(0, len(Q) + CFG["Q_CHUNK"], CFG["Q_CHUNK"]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b > a:
            p = P.iloc[a:b]
            yield pair_features(p, Q, C), Q.entity_id.values[p.q.values], C.entity_id.values[p.c.values]

# %% [markdown]
# ## 6. Build training pairs from train (sampled S1 against the FULL S2/S3 pool of each country)

# %%
t_all = tic()
tr1 = pd.read_parquet(norm_path("train", 1))
s1_sample = tr1.sample(n=min(CFG["TRAIN_S1"], len(tr1)), random_state=CFG["SEED"]).reset_index(drop=True)
del tr1; gc.collect()

frames = []
for country, Q in s1_sample.groupby("country", sort=False):
    print(f"[train] {country}: {len(Q):,} S1")
    Q = Q.reset_index(drop=True)
    C2, C3 = read_country("train", 2, country), read_country("train", 3, country)
    for X, s1_ids, cand_ids in run_country(Q, C2, C3):
        X["s1_id"], X["cand_id"] = s1_ids, cand_ids
        frames.append(X)
    del C2, C3; gc.collect()

train_pairs = pd.concat(frames, ignore_index=True); del frames
truth = gt_pairs[gt_pairs.s1_id.isin(s1_sample.entity_id)]
train_pairs = train_pairs.merge(truth.assign(label=1), on=["s1_id", "cand_id"], how="left")
train_pairs["label"] = train_pairs.label.fillna(0).astype(np.int8)
print(f"pairs: {len(train_pairs):,} | positives: {train_pairs.label.sum():,}")
print(f"BLOCKING RECALL: {train_pairs.label.sum() / max(len(truth), 1):.4f}  (share of true pairs kept by blocking)")
train_pairs.to_parquet(f"{CACHE}/train_pairs.parquet", index=False)
t_all("train pairs built")

# %% [markdown]
# ## 7. Metric, decision rule, LightGBM
# Validation is split by **S1 entity** (never by row). The decision rule is one-to-one (each S2/S3 record goes only to its
# best S1), and then keeps pairs whose probability is at least `t`. `t` is chosen to maximise **macro F0.5**, the official metric.

# %%
def macro_f05(pred, truth, s1_ids):
    """Official metric. pred/truth: DataFrames (s1_id, cand_id); s1_ids: every evaluated S1 id (incl. singletons)."""
    idx = pd.Index(s1_ids)
    npred = pred.groupby("s1_id").size().reindex(idx, fill_value=0).values
    ntrue = truth.groupby("s1_id").size().reindex(idx, fill_value=0).values
    tp = (pred.merge(truth, on=["s1_id", "cand_id"]).groupby("s1_id").size()
          .reindex(idx, fill_value=0).values)
    with np.errstate(divide="ignore", invalid="ignore"):
        p, r = tp / np.maximum(npred, 1), tp / np.maximum(ntrue, 1)
        f = np.where(tp > 0, 1.25 * p * r / (0.25 * p + r), 0.0)
    return float(np.where((npred == 0) & (ntrue == 0), 1.0, f).mean())

def decide(scored, t):
    """One-to-one: keep each candidate only for its highest-probability S1, then threshold."""
    d = scored.sort_values("prob", ascending=False).drop_duplicates("cand_id")
    return d[d.prob >= t]

FEATS = [c for c in train_pairs.columns if c not in ("s1_id", "cand_id", "label")]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=CFG["N_JOBS"], verbose=-1, seed=CFG["SEED"])

rng = np.random.default_rng(CFG["SEED"])
all_ids = s1_sample.entity_id.values
val_ids = set(rng.choice(all_ids, int(len(all_ids) * CFG["VAL_FRAC"]), replace=False))
is_val = train_pairs.s1_id.isin(val_ids).values

t = tic()
dtr = lgb.Dataset(train_pairs.loc[~is_val, FEATS], train_pairs.loc[~is_val, "label"])
dva = lgb.Dataset(train_pairs.loc[is_val, FEATS], train_pairs.loc[is_val, "label"], reference=dtr)
model = lgb.train(PARAMS, dtr, num_boost_round=5000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
t(f"lightgbm (best iter {model.best_iteration})")

val = train_pairs.loc[is_val, ["s1_id", "cand_id", "label"]].copy()
val["prob"] = model.predict(train_pairs.loc[is_val, FEATS], num_iteration=model.best_iteration)
val_truth = truth[truth.s1_id.isin(val_ids)]
val_s1 = list(val_ids)

oracle = macro_f05(val[val.label == 1], val_truth, val_s1)
print(f"oracle F0.5 (perfect model on these candidates) = {oracle:.4f}")
grid = np.round(np.arange(0.20, 0.96, 0.025), 3)
scores = {t_: macro_f05(decide(val, t_), val_truth, val_s1) for t_ in grid}
BEST_T = max(scores, key=scores.get)
print(pd.Series(scores).round(4).to_string())
print(f"==> best threshold {BEST_T}  validation macro F0.5 = {scores[BEST_T]:.4f}")

imp = pd.Series(model.feature_importance("gain"), index=FEATS).sort_values(ascending=False)
print((imp / imp.sum()).round(3).head(20).to_string())

# %% [markdown]
# ## 8. Error analysis (paste examples into the documentation)

# %%
pred_val = decide(val, BEST_T)
fp = pred_val[pred_val.label == 0].head(15)
fn = val_truth.merge(pred_val, on=["s1_id", "cand_id"], how="left", indicator=True)
fn = fn[fn._merge == "left_only"].head(15)

def show_pairs(df):
    if df.empty:
        print("  (none)"); return
    ids = set(df.s1_id) | set(df.cand_id)
    look = pd.concat([pd.read_parquet(norm_path("train", n), columns=["entity_id", "name_norm", "addr_norm"],
                                      filters=[("entity_id", "in", list(ids))]) for n in (1, 2, 3)]).set_index("entity_id")
    for r in df.itertuples():
        a, b = look.loc[r.s1_id], look.loc[r.cand_id] if r.cand_id in look.index else None
        print(f"{r.s1_id}: {a.name_norm} | {a.addr_norm}")
        print(f"   {r.cand_id}: {'' if b is None else b.name_norm + ' | ' + b.addr_norm}"
              f"   p={getattr(r, 'prob', float('nan')):.3f}")

print("=== FALSE POSITIVES (wrong merges) ==="); show_pairs(fp)
print("\n=== FALSE NEGATIVES (missed) ==="); show_pairs(fn)

# %% [markdown]
# ## 9. Retrain on all training pairs, then save the model

# %%
n_iter = int(model.best_iteration * 1.1) + 1
final_model = lgb.train(PARAMS, lgb.Dataset(train_pairs[FEATS], train_pairs.label), num_boost_round=n_iter)
final_model.save_model(f"{WORK}/model.txt")
json.dump({"threshold": float(BEST_T), "features": FEATS, "cfg": CFG, "val_f05": scores[BEST_T]},
          open(f"{WORK}/model_meta.json", "w"), indent=2, default=str)
del train_pairs, dtr, dva, val; gc.collect()

# %% [markdown]
# ## 10. Inference on test (scores cached per country, so re-thresholding is instant)

# %%
final_model = lgb.Booster(model_file=f"{WORK}/model.txt")
meta = json.load(open(f"{WORK}/model_meta.json"))
FEATS, BEST_T = meta["features"], meta["threshold"]

test_countries = pd.read_parquet(norm_path("test", 1), columns=["country"]).country.unique()
for country in test_countries:
    path = f"{CACHE}/test_scores_{re.sub(r'[^A-Za-z0-9]+', '_', country) or 'EMPTY'}.parquet"
    if os.path.exists(path):
        continue
    t = tic()
    Q = read_country("test", 1, country)
    if CFG["TEST_S1_LIMIT"]:
        Q = Q.head(CFG["TEST_S1_LIMIT"])
    print(f"[test] {country}: {len(Q):,} S1")
    C2, C3 = read_country("test", 2, country), read_country("test", 3, country)
    parts = []
    for X, s1_ids, cand_ids in run_country(Q, C2, C3):
        parts.append(pd.DataFrame({"s1_id": s1_ids, "cand_id": cand_ids,
                                   "prob": final_model.predict(X[FEATS]).astype(np.float32)}))
    scored = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["s1_id", "cand_id", "prob"])
    scored.to_parquet(path, index=False)
    del Q, C2, C3, parts, scored; gc.collect()
    t(f"{country} done")

# %% [markdown]
# ## 11. Write `matching_results.tsv` and `candidate_pairs.tsv`, then validate
# To try another threshold, change `T` and rerun **only this cell**.

# %%
T = BEST_T
scored = pd.concat([pd.read_parquet(p) for p in glob.glob(f"{CACHE}/test_scores_*.parquet")], ignore_index=True)
matches = decide(scored, T).sort_values(["s1_id", "prob"], ascending=[True, False])

all_s1 = pd.read_parquet(norm_path("test", 1), columns=["entity_id"]).entity_id.values
cand_str = scored.groupby("s1_id").cand_id.agg(",".join)
match_str = matches.groupby("s1_id").cand_id.agg(",".join)

def write_tsv(series, col, path):
    s = series.reindex(all_s1, fill_value="")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        f.writelines(f"{a}\t{b}\n" for a, b in zip(s.index, s.values))

write_tsv(match_str, "matched_entity_ids", f"{OUT}/matching_results.tsv")
write_tsv(cand_str, "candidate_entity_ids", f"{OUT}/candidate_pairs.tsv")
print(f"threshold {T}: S1 rows {len(all_s1):,} | with matches {len(match_str):,} "
      f"| avg matches {len(matches) / len(all_s1):.2f} | avg candidates {len(scored) / len(all_s1):.1f}")

validator = [p for p in glob.glob(f"{RES}/**/validate_submission.py", recursive=True) if "__MACOSX" not in p]
if validator:
    import subprocess, sys
    r = subprocess.run([sys.executable, validator[0], "--matching", f"{OUT}/matching_results.tsv",
                        "--candidate", f"{OUT}/candidate_pairs.tsv", "--test-dir", f"{DATA}/test"],
                       capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-2000:])
