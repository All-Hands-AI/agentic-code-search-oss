# Training

## Build Dataset

```
uv run src/build_dataset.py --output ../data/
```

## Train Model

```
bash scripts run_training.sh -m Qwen/Qwen3-0.6B -d <Absolute Path to Data>
```