"""Generate sample API inputs from the trained artifacts."""

import json
import pickle
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


MODEL_DIR = "model/"
N_USERS = 10
MODES = ["hybrid", "collab", "content"]
SEED = 42


def main():
    random.seed(SEED)

    with open(f"{MODEL_DIR}/artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)

    all_users = list(artifacts["user_id_map"].keys())
    sample_users = random.sample(all_users, min(N_USERS, len(all_users)))

    print("=" * 60)
    print("  Sample API Inputs — POST /recommend")
    print("=" * 60)

    samples = []
    for i, user_id in enumerate(sample_users):
        mode = MODES[i % len(MODES)]
        sample = {"user_id": user_id, "top_k": 10, "mode": mode}
        samples.append(sample)
        print(f"\n── Sample {i + 1} ({mode}) ──────────────────────────────")
        print(json.dumps(sample, indent=2))

    with open("sample_inputs.json", "w") as f:
        json.dump(samples, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  Saved {len(samples)} samples to sample_inputs.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
