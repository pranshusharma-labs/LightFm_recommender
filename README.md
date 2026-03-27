# LightFM Hybrid Recommender

Minimal LightFM recommender project with a small reusable package and thin app/script entrypoints.

## Structure
```
lightfm_project/
├── requirements.txt
├── recommender/
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── features.py
│   └── inference.py
├── scripts/
│   ├── train.py
│   └── generate_test_inputs.py
├── app/
│   ├── api.py
│   └── ui.py
├── notebooks/
│   └── eda.ipynb
├── data/
│   ├── products.csv
│   └── events.csv
└── model/            ← saved after training
```

## Setup
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run
```bash
# 1. Train
python scripts/train.py

# 2. API
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

# 3. UI
streamlit run app/ui.py
```

Existing files in `model/` do not need to be retrained just because the code layout changed.

## Data Split Strategy
Session-boundary temporal split — zero leakage:
- Sessions detected via 30-min inactivity gaps
- Global 80/20 time cutoff applied at session level
- Train = sessions that ended completely before cutoff
- Test  = sessions that started completely after cutoff
- Sessions straddling the cutoff are DROPPED from both

## Modes
| Mode    | item_alpha | Features | ID embeddings |
|---------|-----------|----------|---------------|
| collab  | default   | No       | Yes           |
| content | 1e-3      | Yes      | Suppressed    |
| hybrid  | 1e-6      | Yes      | Yes           |

## Swap models
Edit the training flow in [scripts/train.py](scripts/train.py) and the reusable helpers in `recommender/`.
