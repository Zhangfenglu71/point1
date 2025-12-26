# Clean E6 baseline

This directory contains a minimal E6 baseline for video-conditioned radar spectrogram generation using Rectified Flow, FiLM conditioning, CFG-training, and CFG-sampling (w=3).

## Directory layout
```
e6_clean/
  core/
    config.py           # TrainConfig
    datasets/
      video_cond_radar.py
    train_utils.py      # eval_one_epoch
    utils/seed.py       # set_seed
  models/
    unet_video_cond_film.py
  scripts/
    train_video_cond_flow_film_cfg.py
    sample_baseline_e6_w3.py
```

## Installation (Rectified Flow)
```bash
pip install -e ./rectified-flow-clean
```

## Training
```bash
python -m scripts.train_video_cond_flow_film_cfg
```

## Sampling (CFG w=3)
```bash
python -m scripts.sample_baseline_e6_w3
```
