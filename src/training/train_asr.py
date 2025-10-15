import os, yaml, argparse, torch
from torch.utils.data import DataLoader
from src.data.rvtall import RVTALLDataset
from src.models.encoders import VideoEncoder
from src.models.decoders import CTCHead
from src.utils.paths import resolve_path, ensure_dir

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = RVTALLDataset(resolve_path(cfg['data']['root']), split='train', tiny=bool(cfg['data'].get('tiny', False)),
                       modalities=cfg['data']['modalities'], cfg=cfg['data'])
    dl = DataLoader(ds, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=cfg['data']['num_workers'])
    enc = VideoEncoder(cfg['model']['latent_dim']).to(device)
    head = CTCHead(cfg['model']['latent_dim'], vocab_size=96).to(device)
    opt = torch.optim.Adam(list(enc.parameters())+list(head.parameters()), lr=cfg['train']['lr'])
    for epoch in range(cfg['train']['max_epochs']):
        for batch in dl:
            opt.zero_grad()
            z = enc(batch['video'].to(device))
            logits = head(z)
            loss = z.pow(2).mean()
            loss.backward(); opt.step()
        print("epoch", epoch+1, "ok")
    out = os.path.join(ensure_dir(resolve_path(cfg['eval']['save_dir'])), 'ctc_baseline.pt')
    torch.save({'enc': enc.state_dict(), 'head': head.state_dict()}, out)
    print("saved", out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
