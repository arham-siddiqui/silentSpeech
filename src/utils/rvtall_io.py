# src/utils/rvtall_io.py
import os, re, json
from pathlib import Path
from glob import glob
import numpy as np
import pandas as pd
from scipy.io import wavfile

# ---------------- Project / base discovery ----------------

def find_project_root(start: Path | None = None) -> Path:
    p = Path(start or Path.cwd()).resolve()
    for parent in (p, *p.parents):
        if (parent / "src").is_dir():
            return parent
    raise FileNotFoundError("Could not find project root (folder containing 'src').")

def find_rvtall_base(start: Path | None = None) -> Path:
    root = find_project_root(start)
    cand = root / "src" / "data" / "RVTALL" / "Processed_cut_data"
    if cand.is_dir():
        return cand
    # fallback search
    for p in root.rglob("Processed_cut_data"):
        if p.parent.name == "RVTALL":
            return p
    raise FileNotFoundError("RVTALL/Processed_cut_data not found under project.")

# ---------------- Directory introspection ----------------

def list_sets(BASE: Path) -> list[str]:
    p = BASE / "kinect_processed"
    if not p.is_dir(): return []
    return sorted([d.name for d in p.iterdir() if d.is_dir() and d.name.isdigit()])

def list_corpora_dirs(BASE: Path, set_id: str) -> list[Path]:
    """Return actual corpus directories (Path objects) under kinect_processed/set_id/*"""
    p = BASE / "kinect_processed" / str(set_id)
    if not p.is_dir(): return []
    return sorted([d for d in p.iterdir() if d.is_dir()])

def corpus_type(name: str) -> str:
    n = name.lower()
    if n.startswith("vowel"): return "vowel"
    if n.startswith("word"): return "word"
    if n.startswith("sentence"): return "sentence"
    return "other"

_num_pat = re.compile(r"(\d+)$")        # matches trailing digits: word5 -> 5
_num_und_pat = re.compile(r"_(\d+)$")   # matches trailing _digits: word_5 -> 5

def parse_corpus_name(raw: str) -> tuple[str, int | None]:
    """Return (type, index) from dir name like 'word5' or 'word_5' or 'sentences1'."""
    t = corpus_type(raw)
    n = raw.lower()
    m = _num_pat.search(n) or _num_und_pat.search(n)
    idx = int(m.group(1)) if m else None
    return t, idx

def resolve_corpus_dir(BASE: Path, set_id: str, desired_name: str) -> Path | None:
    """
    Flexible resolver:
    - exact match first,
    - else try normalization (e.g., 'word5' <-> 'word_5', 'sentences1' <-> 'sentence1'),
    - else fuzzy: match by (type, index).
    """
    root = BASE / "kinect_processed" / str(set_id)
    if not root.is_dir(): return None

    wanted = desired_name.strip().lower()
    exact = root / desired_name
    if exact.is_dir(): 
        return exact

    # try simple variants: underscore/no underscore, sentence/sentences
    variants = {wanted}
    variants.add(wanted.replace("_", ""))
    variants.add(wanted.replace("_", "-"))
    if wanted.startswith("sentences"):
        variants.add(wanted.replace("sentences", "sentence", 1))
    if wanted.startswith("sentence"):
        variants.add(wanted.replace("sentence", "sentences", 1))
    for v in variants:
        pv = root / v
        if pv.is_dir():
            return pv

    # last resort: match by (type, index)
    wt, wi = parse_corpus_name(wanted)
    candidates = list_corpora_dirs(BASE, set_id)
    for d in candidates:
        dt, di = parse_corpus_name(d.name)
        if dt == wt and (wi is None or di == wi):
            return d
    return None

# ---------------- Kinect: enumerate paired (audio, landmark) ----------------

_aud_pat = re.compile(r"audio_proc_(\d+)\.wav$", re.IGNORECASE)
_land_pat = re.compile(r"land_proc_(\d+)\.csv$", re.IGNORECASE)

def _index_from_name(name: str, pat: re.Pattern) -> int | None:
    m = pat.search(name)
    return int(m.group(1)) if m else None

def kinect_pair_index_paths(corpus_dir: Path) -> list[tuple[int, Path, Path]]:
    """
    Return sorted list of (idx, audio_path, landmark_path) pairs with matching numeric index.
    Corpus layout expected:
      corpus_dir/
        audios/audio_proc_<i>.wav
        landmarkers/land_proc_<i>.csv
        videos/video_<i>/...  (not needed here)
    """
    aud_dir = corpus_dir / "audios"
    lan_dir = corpus_dir / "landmarkers"
    if not aud_dir.is_dir() or not lan_dir.is_dir():
        # Some drops are flat; support flat too
        aud_dir = corpus_dir
        lan_dir = corpus_dir

    aud = { _index_from_name(p.name, _aud_pat): p for p in sorted(aud_dir.glob("audio_proc_*.wav")) }
    lan = { _index_from_name(p.name, _land_pat): p for p in sorted(lan_dir.glob("land_proc_*.csv")) }

    common = sorted(set(k for k in aud.keys() if k is not None) & set(k for k in lan.keys() if k is not None))
    pairs = [(i, aud[i], lan[i]) for i in common]
    return pairs

def kinect_first_pair(BASE: Path, set_id: str, corpus: str | Path):
    """Convenience: resolve corpus dir, return first (idx, audio_path, landmark_path)."""
    cdir = corpus if isinstance(corpus, Path) else resolve_corpus_dir(BASE, set_id, corpus)
    if not cdir: return None
    pairs = kinect_pair_index_paths(cdir)
    return pairs[0] if pairs else None

# --------------- Loaders / features ----------------

def load_audio_env(path: Path, rms_ms=20):
    fs, y = wavfile.read(str(path))
    y = y.astype(float)
    y /= (np.max(np.abs(y)) + 1e-9)
    win = max(1, int(fs * rms_ms/1000.0))
    k = np.ones(win)/win
    env = np.sqrt(np.convolve(y**2, k, mode="same"))
    t = np.arange(len(env)) / fs
    return env, t, fs

def load_lip_aperture_csv(csv_path: Path, upper_idx=51, lower_idx=57):
    df = pd.read_csv(csv_path)
    up = df.iloc[:, upper_idx*3 + 2].to_numpy()
    lo = df.iloc[:, lower_idx*3 + 2].to_numpy()
    return lo - up

def zscore(x: np.ndarray):
    x = np.asarray(x, float)
    s = x.std() if x.std() > 1e-8 else 1.0
    return (x - x.mean())/s

# --------------- Other modalities ----------------

def first_npy(BASE: Path, modality: str, set_id: str, corpus: str | Path):
    """Return the first sample_*.npy under <modality>_processed/set_id/corpus."""
    cdir = corpus if isinstance(corpus, Path) else resolve_corpus_dir(BASE, set_id, corpus)
    if not cdir: return None
    mdir = BASE / f"{modality}_processed" / str(set_id) / cdir.name
    if not mdir.is_dir(): return None
    hits = sorted(mdir.glob("sample_*.npy"))
    return hits[0] if hits else None

def load_npy(path: Path):
    return np.load(str(path), allow_pickle=True)
