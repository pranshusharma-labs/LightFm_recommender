# LightFM Hybrid Recommender

4 files. Train → API → UI.

## Structure
```
lightfm_project/
├── data_loader.py    ← load, clean, split data
├── train.py          ← train all 3 modes + evaluate + save
├── api.py            ← FastAPI serving recommendations
├── ui.py             ← Streamlit UI
├── requirements.txt
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
# 1. Train (prints eval table for all 3 modes)
python train.py

# 2. API
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# 3. UI
streamlit run ui.py
```

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
Edit the training sections in train.py marked with comments.
The data loading and evaluation code is model-agnostic.
