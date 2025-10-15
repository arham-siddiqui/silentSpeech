import os, yaml, argparse, torch
from torch.utils.data import DataLoader
from src.data.rvtall import RVTALLDataset
from src.models.encoders import VideoEncoder, RadarEncoder
from src.models.fusion import AttentionFusion, info_nce
from src.utils.paths import resolve_path, ensure_dir

def get_dataloader(cfg):
    root = resolve_path(cfg['data']['root'])
    modalities = cfg['data']['modalities']
    tiny = bool(cfg['data'].get('tiny', False))
    ds = RVTALLDataset(root, split='train', tiny=tiny, modalities=modalities, cfg=cfg['data'])
    return DataLoader(ds, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=cfg['data']['num_workers'])

def build_models(cfg, device):
    latent = cfg['model']['latent_dim']
    encs = {}
    mods = cfg['data']['modalities']
    if 'video' in mods:
        encs['video'] = VideoEncoder(latent).to(device)
    if 'mmwave' in mods:
        encs['mmwave'] = RadarEncoder(1, latent).to(device)
    if 'uwb' in mods:
        encs['uwb'] = RadarEncoder(1, latent).to(device)
    if 'laser' in mods:
        encs['laser'] = RadarEncoder(1, latent).to(device)
    if 'audio' in mods:
        encs['audio'] = RadarEncoder(1, latent).to(device)
    fusion = AttentionFusion(latent).to(device) if cfg['model'].get('fusion','attention')=='attention' else None
    return encs, fusion

def forward_batch(batch, encs, device):
    zs = []
    for k, enc in encs.items():
        if k in batch:
            zs.append(enc(batch[k].to(device)))
    return zs

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = ensure_dir(resolve_path(cfg['eval']['save_dir']))
    loader = get_dataloader(cfg)
    encs, fusion = build_models(cfg, device)
    opt = torch.optim.Adam([p for m in encs.values() for p in m.parameters()],
                           lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    for epoch in range(cfg['train']['max_epochs']):
        tot = 0.0
        for batch in loader:
            opt.zero_grad()
            zs = forward_batch(batch, encs, device)
            loss = info_nce(zs, cfg['model']['temperature'])
            loss.backward(); opt.step()
            tot += float(loss.item())
        print(f"Epoch {epoch+1}: loss={tot/max(1,len(loader)):.4f}")
    torch.save({k: v.state_dict() for k,v in encs.items()}, os.path.join(save_dir, 'encoders.pt'))
    print('Saved encoders:', os.path.join(save_dir, 'encoders.pt'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
