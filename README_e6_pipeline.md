# E6 one-click pipeline

This pipeline runs the **self-consistent E6 baseline**:
- Train: `scripts.train_video_cond_flow_film_cfg` (velocity RF, FiLM, CFG-training)
- Sample: `scripts.sample_baseline_e6_w3` (Velocity-CFG w=3, SDESampler)
- Eval: `scripts.eval_gen_with_cls` (ResNet18 classifier ACC)

## Usage

```bash
bash run_e6_pipeline.sh
```

Override knobs via env vars:

```bash
NUM_STEPS=40 SEED=123 ALPHA=0.25 bash run_e6_pipeline.sh
```

Logs go to `runs/e6_pipeline_YYYYMMDD_HHMMSS/`.
