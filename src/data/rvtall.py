from typing import Dict, Any, List
from src.data.rvtall_auto import RVTALLAutoDataset

def RVTALLDataset(root: str, split: str='train', tiny: bool=False, modalities: List[str]=None, cfg: Dict[str, Any]=None):
    if modalities is None:
        modalities = ['video','mmwave','uwb','laser','audio']
    if cfg is None:
        cfg = {}
    return RVTALLAutoDataset(root=root, modalities=modalities, cfg=cfg, tiny=tiny)
