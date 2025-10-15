import os
from pathlib import Path

def _infer_repo_root() -> str:
    here = Path(__file__).resolve()
    repo = here.parent.parent.parent
    if (repo / "src").exists():
        return str(repo)
    return str(Path.cwd())

BASE_DIR = os.environ.get("SSI_BASE", _infer_repo_root())

def resolve_path(p: str) -> str:
    if not p:
        return BASE_DIR
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str(Path(BASE_DIR) / p)

def data_path(*parts) -> str:
    return str(Path(BASE_DIR, "data", *parts))

def output_path(*parts) -> str:
    return str(Path(BASE_DIR, "outputs", *parts))

def ensure_dir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return p
