# ssi_multimodal_alpha (universal, portable)

**Purpose:** A portable alpha codebase for silent speech / multimodal experimentation with:
- A **universal RVTALL loader** that understands the *Processed sliced data* layout from the Figshare release and
  gracefully detects raw folders when present (index only).
- A **portable path layer** (`src/utils/paths.py`) so configs can use *relative* paths and team members only set `SSI_BASE` if desired.
- Minimal training scripts (contrastive / ASR placeholder) that operate on the loader.
- Small validation utilities to confirm your local data layout.

## 0) Portability convention
- Project root is auto-inferred from `src/utils/paths.py`.
- Optionally set an environment variable: `SSI_BASE=/absolute/path/to/ssi_multimodal_alpha`.
- **Configs must use relative paths** like `data/RVTALL/Processed_sliced_data` and `outputs/...`.

## 1) Place and unzip the RVTALL zips (Windows Git Bash)
You downloaded:
- `Processed_sliced_data.zip`
- `Code.zip` (contains `RVTALL-Preprocess-main` and MATLAB scripts)

From your shell in the repo root:
```bash
# ensure base
pwd  # should be .../ssi_multimodal_alpha

# make dataset area
mkdir -p data/RVTALL

# move zips from Downloads (Git Bash path ~ maps to C:\Users\<you>)
mv ~/Downloads/Processed_sliced_data.zip data/RVTALL/ || true
mv ~/Downloads/Code.zip                 data/RVTALL/ || true

# unzip (use one that exists on your system; Git Bash often has `tar`)
cd data/RVTALL
# Option A: tar (works on many Git Bash installs)
tar -xf Processed_sliced_data.zip || true
tar -xf Code.zip || true

# Option B: unzip (if installed)
# unzip Processed_sliced_data.zip
# unzip Code.zip

# You should now see:
#   data/RVTALL/Processed_sliced_data/
#   data/RVTALL/RVTALL-Preprocess-main/
```

Validation:
```bash
cd ../../../
python -m src.scripts.validate_rvtall --base data/RVTALL
```

## 2) Configs
- **RVTALL contrastive**: `src/configs/rvtall_contrastive.yaml`
  - `data.root: data/RVTALL/Processed_sliced_data`
  - `data.modalities: [video, mmwave, uwb, laser, audio]`
- **Inner speech (EEG+fMRI)** is included as a placeholder config.

## 3) Quick run
```bash
# (optional) set explicit base for portability
export SSI_BASE=$(pwd)

# sanity check
python -m src.training.print_paths --config src/configs/rvtall_contrastive.yaml

# small contrastive loop (on a tiny subset)
python -m src.training.train_contrastive --config src/configs/rvtall_contrastive.yaml
```

## 4) Colab notes
In Colab:
```python
!git clone https://github.com/arham-siddiqui/silentSpeech.git
%cd silentSpeech
!pip -q install -r requirements.txt

import os
os.environ["SSI_BASE"] = os.getcwd()

!python -m src.scripts.validate_rvtall --base data/RVTALL
!python -m src.training.train_contrastive --config src/configs/rvtall_contrastive.yaml
```

## 5) Universal loader
- Implemented in `src/data/rvtall_auto.py`.
- Detects `Processed_sliced_data/` and enumerates samples by `<SetID>/<CorpusType>/*/audios/audio_proc_<idx>.wav`.
- For each `<idx>` it joins across modalities:
  - `radar_processed/<SetID>/<CorpusType>/sample_<idx>.npy`
  - `uwb_processed/<SetID>/<CorpusType>/sample_<idx>.npy`
  - `laser_processed/<SetID>/<CorpusType>/sample_<idx>.npy`
  - `kinect_processed/.../videos/video_proc_<idx>/mouth/*.png` (frames)
  - `kinect_processed/.../landmarkers/land_proc_<idx>.*` (if present)
- Returns fixed-size tensors per modality (resize/sampling) to keep DataLoader collate happy.

> If only raw folders exist, the loader will **index paths** and raise a clear message that raw signal decoding is not implemented yet.

## 6) Git repair (your earlier error)
If `~/ntab/software/mri` has its own `.git` that conflicts with `speechdecoding/ssi_multimodal_alpha`, run:
```bash
cd ~/ntab/software/mri
rm -rf .git                   # remove parent .git

cd ~/ntab/software/mri/speechdecoding/ssi_multimodal_alpha
git init
git add .
git commit -m "portable universal alpha"
git branch -M main
git remote add origin https://github.com/arham-siddiqui/silentSpeech.git
git push -u origin main
```
(See `scripts/git_fix_parent_repo.sh` for a copy.)

---

**License**: research use only.  Contributions welcome.
