import os, glob, argparse, subprocess
from pathlib import Path
from src.utils.paths import resolve_path

def try_extract_7z(folder: Path):
    """Extracts 7z archives only if target folder missing."""
    for f in folder.glob("*.7z"):
        target = folder / f.stem
        if target.exists() and any(target.iterdir()):
            print(f"[OK] Skipping {f.name} (already extracted)")
            continue
        print(f"[i] Extracting {f.name} ...")
        try:
            subprocess.run(["7z", "x", str(f), f"-o{target}"], check=True)
        except Exception as e:
            print(f"[!] Could not extract {f.name}: {e}")

def count_files(pattern: str) -> int:
    return len(glob.glob(pattern, recursive=True))

def list_unexpected_files(proc: Path):
    allowed_exts = {".wav", ".png", ".npy", ".csv", ".txt"}
    unexpected = []
    for root, _, files in os.walk(proc):
        for f in files:
            ext = Path(f).suffix.lower()
            if ext and ext not in allowed_exts:
                unexpected.append(str(Path(root) / f))
    if unexpected:
        print(f"[!] {len(unexpected)} unexpected file(s) found (first 10 shown):")
        for u in unexpected[:10]:
            print("   -", u)
    else:
        print("[OK] No unexpected file types detected")

def summarize(proc: Path):
    counts = {
        "audio_wavs": count_files(str(proc / "kinect_processed" / "*" / "*" / "audios" / "*.wav")),
        "mouth_pngs": count_files(str(proc / "kinect_processed" / "*" / "*" / "videos" / "video_*" / "mouths" / "*.png")),
        "radar_npys": count_files(str(proc / "radar_processed" / "**" / "*.npy")),
        "uwb_npys": count_files(str(proc / "uwb_processed" / "**" / "*.npy")),
        "laser_npys": count_files(str(proc / "laser_processed" / "**" / "*.npy")),
    }
    total = sum(counts.values())
    print(f"audio wavs : {counts['audio_wavs']}")
    print(f"mouth pngs : {counts['mouth_pngs']}")
    print(f"radar npy  : {counts['radar_npys']}")
    print(f"uwb npy    : {counts['uwb_npys']}")
    print(f"laser npy  : {counts['laser_npys']}")
    print(f"[OK] Total files indexed: {total}")
    return counts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="src/data/RVTALL")
    args = ap.parse_args()
    base = Path(resolve_path(args.base))
    proc = None
    for candidate in ["Processed_cut_data", "Processed_sliced_data"]:
        c = base / candidate
        if c.exists():
            proc = c
            break
    if not proc:
        print(f"[!] No Processed_* folder found under {base}")
        return
    print(f"[OK] Found: {proc}")
    try_extract_7z(proc)
    summarize(proc)
    list_unexpected_files(proc)

if __name__ == "__main__":
    main()


'''
import os
import argparse
import subprocess
from pathlib import Path
import glob
from src.utils.paths import resolve_path


def try_extract_7z(folder: Path):
    """Automatically extract any .7z archives in this folder (if not already unpacked)."""
    for f in folder.glob("*.7z"):
        target = folder / f.stem
        if not target.exists():
            print(f"[i] Extracting {f.name} ...")
            try:
                subprocess.run(["7z", "x", str(f), f"-o{target}"], check=True)
            except Exception as e:
                print(f"[!] Could not extract {f.name}: {e}")


def count_files(pattern: str) -> int:
    return len(glob.glob(pattern, recursive=True))


def summarize_set(base_dir: Path):
    """Counts per modality."""
    counts = {}
    counts["audio_wavs"] = count_files(str(base_dir / "kinect_processed" / "*" / "*" / "audios" / "*.wav"))
    counts["mouth_pngs"] = count_files(str(base_dir / "kinect_processed" / "*" / "*" / "videos" / "video_*" / "mouths" / "*.png"))
    counts["radar_npys"] = count_files(str(base_dir / "radar_processed" / "**" / "*.npy"))
    counts["uwb_npys"] = count_files(str(base_dir / "uwb_processed" / "**" / "*.npy"))
    counts["laser_npys"] = count_files(str(base_dir / "laser_processed" / "**" / "*.npy"))
    return counts


def print_summary(counts):
    print(f"audio wavs : {counts['audio_wavs']}")
    print(f"mouth pngs : {counts['mouth_pngs']}")
    print(f"radar npy  : {counts['radar_npys']}")
    print(f"uwb npy    : {counts['uwb_npys']}")
    print(f"laser npy  : {counts['laser_npys']}")
    total = sum(counts.values())
    print(f"[W] Total files indexed: {total}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="src/data/RVTALL")
    args = ap.parse_args()

    base = Path(resolve_path(args.base))
    proc = None
    for candidate in ["Processed_cut_data", "Processed_sliced_data"]:
        c = base / candidate
        if c.exists():
            proc = c
            break
    if not proc:
        print(f"[!] No Processed_* folder found under {base}")
        return

    print(f"[W] Found: {proc}")
    try_extract_7z(proc)

    counts = summarize_set(proc)
    print_summary(counts)

    # Optional deeper structure listing
    sets = sorted([d for d in (proc / "kinect_processed").glob("*") if d.is_dir()])
    print(f"\nDetected {len(sets)} set(s): {[d.name for d in sets]}")
    for s in sets:
        sents = [x.name for x in s.glob("*/") if x.is_dir()]
        print(f"  {s.name}: {len(sents)} sessions → {sents[:5]}{'...' if len(sents) > 5 else ''}")


if __name__ == "__main__":
    main()

'''

'''
import os, argparse, subprocess
from pathlib import Path
from src.utils.paths import resolve_path


def count_files(root, pattern):
    try:
        import glob
        return len(glob.glob(os.path.join(root, pattern), recursive=True))
    except Exception:
        return 0

def try_extract_7z(folder):
    for f in Path(folder).glob("*.7z"):
        name = f.stem
        target = Path(folder) / name
        if not target.exists():
            print(f"[i] Extracting {f.name} ...")
            subprocess.run(["7z", "x", str(f), f"-o{target}"], check=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="src/data/RVTALL")
    args = ap.parse_args()
    base = Path(resolve_path(args.base))
    proc = base / "Processed_cut_data"
    if not proc.exists():
        print("[!] Missing Processed_cut_data under", base)
        return
    print("[W] Found:", proc)
    #try_extract_7z(proc)
    # quick stats
    aud = count_files(proc, "kinect_processed/*/*/*/audios/*.wav")
    mouth = count_files(proc, "kinect_processed/*/*/*/videos/*/mouth/*.png")
    radar = count_files(proc, "radar_processed/*/*/sample_*.npy")
    uwb   = count_files(proc, "uwb_processed/*/*/sample_*.npy")
    laser = count_files(proc, "laser_processed/*/*/sample_*.npy")
    print(f"audio wavs : {aud}")
    print(f"mouth pngs : {mouth}")
    print(f"radar npy  : {radar}")
    print(f"uwb npy    : {uwb}")
    print(f"laser npy  : {laser}")

if __name__ == '__main__':
    main()
'''