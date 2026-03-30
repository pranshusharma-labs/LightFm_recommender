# LightFM Hybrid Recommender

Minimal LightFM recommender project with a small reusable package and thin app/script entrypoints.

## Structure
```
lightfm_project/
├── Dockerfile.api
├── Dockerfile.ui
├── docker-compose.yml
├── .dockerignore
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

## Live App
- UI: https://lightfm-ui.onrender.com
- API: https://lightfm-api.onrender.com
- API docs: https://lightfm-api.onrender.com/docs

## Docker
This repo uses a two-container deployment setup:
- `api` for FastAPI
- `ui` for Streamlit

The current Docker setup bakes the existing `model/` artifacts into both images to keep first deployment simple.

### Notes
- The UI reads `API_URL` from the environment.
- The API and UI both read `MODEL_DIR` from the environment; in Docker it is set to `/app/model/`.
- The API image installs LightFM from the official GitHub repository, with a small build-time patch applied to work around the upstream `__LIGHTFM_SETUP__` packaging issue.
- For the first deployment path, the model is baked into the image. Rebuild the images whenever `model/` changes.
- Both Dockerfiles support a dynamic `PORT` environment variable for container platforms like Render.

## Render Deployment
The current recommended cloud deployment path is Render with two separate Web Services:
- `lightfm-api` using `Dockerfile.api`
- `lightfm-ui` using `Dockerfile.ui`

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


