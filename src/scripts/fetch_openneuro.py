import argparse, sys
from src.utils.paths import data_path, ensure_dir
try:
    import openneuro as on
except Exception:
    on = None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ds004196")
    p.add_argument("--subjects", nargs="*", default=[])
    args = p.parse_args()
    if on is None:
        print("[!] openneuro-py not installed. pip install openneuro-py")
        sys.exit(1)
    dest = data_path(args.dataset)
    ensure_dir(dest)
    include = args.subjects if args.subjects else None
    print(f"[i] downloading {args.dataset} → {dest}")
    try:
        on.download(dataset=args.dataset, target_dir=dest, include=include, snapshot="latest")
    except Exception as e:
        print("[!] download failed:", e)
        sys.exit(2)

if __name__ == "__main__":
    main()
