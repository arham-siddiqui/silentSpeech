import yaml, argparse
from src.utils.paths import resolve_path, ensure_dir

def main(cfg):
    c = yaml.safe_load(open(cfg))
    root = resolve_path(c['data']['root'])
    out = ensure_dir(resolve_path(c['eval']['save_dir']))
    print("ROOT:", root)
    print("OUT:", out)

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True)
    a = p.parse_args()
    main(a.config)
