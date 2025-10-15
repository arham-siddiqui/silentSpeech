import os, re, glob
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

from PIL import Image
from scipy.io import wavfile
from scipy.signal import resample

import torch
from torch.utils.data import Dataset

from src.utils.paths import resolve_path

def _first(patts: List[str]) -> Optional[str]:
    for pat in patts:
        m = glob.glob(pat)
        if m:
            return m[0]
    return None

def _sorted_glob(p: str) -> List[str]:
    xs = glob.glob(p)
    xs.sort()
    return xs

def _ensure_size_2d(arr: np.ndarray, size: int) -> np.ndarray:
    # resize 2D array to (size, size) via PIL
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[None, :]  # (1, N)
    # normalize to 0..255 for PIL
    a = arr.astype(np.float32)
    a = a - a.min()
    denom = (a.max() + 1e-8)
    a = a / denom * 255.0
    im = Image.fromarray(a.astype(np.uint8))
    im = im.resize((size, size), Image.BILINEAR)
    out = np.asarray(im).astype(np.float32) / 255.0
    return out  # (H, W)

class RVTALLAutoDataset(Dataset):
    """Universal RVTALL loader.
    - If `root` points to Processed_sliced_data/, load processed, synchronized slices.
    - If only raw folders exist, we index paths but do not parse binary sensors.
    """
    def __init__(self, root: str, modalities: List[str], cfg: Dict[str, Any], tiny: bool=False):
        super().__init__()
        self.root = Path(resolve_path(root))
        self.modalities = modalities
        self.cfg = cfg
        self.mode = 'processed' if (self.root.name.lower().startswith('processed') or (self.root / 'kinect_processed').exists()) else 'auto'
        if self.mode == 'auto':
            # try to find the processed folder beneath the given root
            cand = self.root / 'Processed_sliced_data'
            if cand.exists():
                self.root = cand
                self.mode = 'processed'
        self.index: List[Dict[str, Any]] = []
        if self.mode == 'processed':
            self._build_index_processed()
        else:
            self._build_index_raw()
        if tiny:
            self.index = self.index[: min(64, len(self.index))]

    # ---------------- Index builders ----------------
    def _build_index_processed(self):
        base = self.root
        kinect_root = base / 'kinect_processed'
        radar_root  = base / 'radar_processed'
        uwb_root    = base / 'uwb_processed'
        laser_root  = base / 'laser_processed'
        # iterate sets
        for set_dir in sorted([d for d in kinect_root.glob('*') if d.is_dir()]):
            set_id = set_dir.name
            for corpus_dir in sorted([d for d in set_dir.glob('*') if d.is_dir()]):
                corpus_type = corpus_dir.name
                # corpus_index dirs (1..N)
                for topic_dir in sorted([d for d in corpus_dir.glob('*') if d.is_dir()]):
                    # audio files carry the <idx>
                    audios_dir = topic_dir / 'audios'
                    if not audios_dir.exists():
                        continue
                    for wav in sorted(audios_dir.glob('audio_proc_*.wav')):
                        # extract rep idx
                        m = re.search(r'audio_proc_(\d+)', wav.name)
                        if not m: 
                            continue
                        rep = m.group(1)
                        rec: Dict[str, Any] = {
                            'set_id': set_id,
                            'corpus': corpus_type,
                            'topic': topic_dir.name,
                            'rep': rep,
                            'paths': {}
                        }
                        rec['paths']['audio'] = str(wav)

                        # join radar/uwb/laser by rep index
                        rec['paths']['mmwave'] = str(radar_root / set_id / corpus_type / f'sample_{rep}.npy')
                        rec['paths']['uwb']    = str(uwb_root   / set_id / corpus_type / f'sample_{rep}.npy')
                        rec['paths']['laser']  = str(laser_root / set_id / corpus_type / f'sample_{rep}.npy')

                        # mouth/video frames and landmarks (best-effort)
                        vid_dir = _first([
                            str(topic_dir / 'videos' / f'video_proc_{rep}'),
                            str(topic_dir / 'videos' / f'video_proc_{rep}_cv'),
                            str(topic_dir / 'videos' / f'{rep}'),
                        ])
                        if vid_dir:
                            mouth_dir = Path(vid_dir) / 'mouth'
                            if mouth_dir.exists():
                                rec['paths']['video_mouth'] = str(mouth_dir)
                            # frames index
                            fidx = _first([str(Path(vid_dir) / '*frames_index*.npy')])
                            if fidx:
                                rec['paths']['video_frames_index'] = fidx
                            # CV landmarks
                            lcv = Path(vid_dir) / 'landmarkers_cv'
                            if lcv.exists():
                                rec['paths']['video_landmarks_cv'] = str(lcv)

                        # kinect landmarks
                        land = _first([
                            str(topic_dir / 'landmarkers' / f'land_proc_{rep}.csv'),
                            str(topic_dir / 'landmarkers' / f'land_proc_{rep}.npy'),
                        ])
                        if land:
                            rec['paths']['kinect_landmarks'] = land

                        self.index.append(rec)

    def _build_index_raw(self):
        # Only index, don’t parse. Provide helpful message at __getitem__.
        base = self.root
        raw_glob = list(base.glob('Raw data file*')) + list(base.glob('raw*'))
        if raw_glob:
            raw_root = raw_glob[0]
        else:
            raw_root = base
        for set_dir in sorted([d for d in raw_root.glob('*') if d.is_dir()]):
            # look for audio and timestamps as anchors
            audios = list(set_dir.rglob('audio_*[0-9].wav'))
            for wav in sorted(audios):
                self.index.append({'raw': True, 'paths': {'audio': str(wav)}, 'set_id': set_dir.name})

    # ---------------- Loaders ----------------
    def __len__(self):
        return len(self.index)

    def _load_audio(self, path: str, sr: int) -> torch.Tensor:
        try:
            fs, x = wavfile.read(path)
            if x.dtype != np.float32:
                # normalize to [-1,1]
                maxv = np.iinfo(x.dtype).max if np.issubdtype(x.dtype, np.integer) else 1.0
                x = x.astype(np.float32) / float(maxv if maxv != 0 else 1.0)
            if fs != sr:
                n = int(len(x) * sr / fs)
                x = resample(x, n).astype(np.float32)
        except Exception:
            x = np.zeros((sr*1,), dtype=np.float32)
        return torch.from_numpy(x[None, :])  # (1, T)

    def _load_mouth_clip(self, mouth_dir: str, frames: int, size: int) -> torch.Tensor:
        imgs = _sorted_glob(str(Path(mouth_dir) / '*.png'))
        if len(imgs) == 0:
            return torch.zeros((frames, 1, size, size), dtype=torch.float32)
        # sample evenly
        idxs = np.linspace(0, len(imgs)-1, frames).astype(int)
        arrs = []
        for i in idxs:
            try:
                im = Image.open(imgs[i]).convert('L').resize((size, size), Image.BILINEAR)
                arr = np.asarray(im).astype(np.float32)/255.0
            except Exception:
                arr = np.zeros((size, size), dtype=np.float32)
            arrs.append(arr[None, ...])  # (1,H,W)
        clip = np.stack(arrs, axis=0)  # (T,1,H,W)
        return torch.from_numpy(clip)

    def _load_npy_2d(self, path: str, size: int) -> torch.Tensor:
        if not os.path.exists(path):
            return torch.zeros((1, size, size), dtype=torch.float32)
        try:
            a = np.load(path, allow_pickle=True)
            if a.ndim == 3:
                # pick first channel/frame
                a = a[0]
        except Exception:
            a = np.zeros((size, size), dtype=np.float32)
        a2 = _ensure_size_2d(a, size)
        return torch.from_numpy(a2[None, ...])  # (1,H,W)

    def _load_vector(self, path: str, length: int) -> torch.Tensor:
        if not os.path.exists(path):
            return torch.zeros((length,), dtype=torch.float32)
        try:
            a = np.load(path, allow_pickle=True).astype(np.float32).ravel()
        except Exception:
            a = np.zeros((length,), dtype=np.float32)
        if len(a) != length:
            # simple center crop/pad
            out = np.zeros((length,), dtype=np.float32)
            m = min(length, len(a))
            out[:m] = a[:m]
            a = out
        return torch.from_numpy(a)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.index[idx]
        if rec.get('raw', False):
            raise RuntimeError('Raw RVTALL decoding is not implemented yet; please use Processed_sliced_data.')
        out: Dict[str, Any] = {}
        mcfg = self.cfg or {}
        # VIDEO
        if 'video' in self.modalities:
            clip = torch.zeros((mcfg.get('video',{}).get('frames',8),1,mcfg.get('video',{}).get('size',112),mcfg.get('video',{}).get('size',112)))
            if 'video_mouth' in rec['paths']:
                clip = self._load_mouth_clip(rec['paths']['video_mouth'],
                                             frames=mcfg.get('video',{}).get('frames',8),
                                             size=mcfg.get('video',{}).get('size',112))
            out['video'] = clip
        # RADAR (FMCW/mmWave)
        if 'mmwave' in self.modalities:
            out['mmwave'] = self._load_npy_2d(rec['paths'].get('mmwave',''), mcfg.get('radar',{}).get('size',64))
        # UWB
        if 'uwb' in self.modalities:
            out['uwb'] = self._load_npy_2d(rec['paths'].get('uwb',''), mcfg.get('uwb',{}).get('size',64))
        # LASER
        if 'laser' in self.modalities:
            out['laser'] = self._load_vector(rec['paths'].get('laser',''), mcfg.get('laser',{}).get('length', mcfg.get('laser',{}).get('size',256)))
        # AUDIO
        if 'audio' in self.modalities:
            out['audio'] = self._load_audio(rec['paths'].get('audio',''), mcfg.get('audio',{}).get('sr',16000))
        out['meta'] = {'set_id': rec['set_id'], 'corpus': rec['corpus'], 'topic': rec['topic'], 'rep': rec['rep']}
        return out
