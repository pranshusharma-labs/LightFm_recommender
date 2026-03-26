# generate_test_inputs.py
# ────────────────────────
# Generates sample API test inputs from the trained model artifacts.
# Usage: python generate_test_inputs.py

import pickle
import json
import random

MODEL_DIR  = 'model/'
N_USERS    = 10
MODES      = ['hybrid', 'collab', 'content']
SEED       = 42

random.seed(SEED)

with open(f'{MODEL_DIR}/artifacts.pkl', 'rb') as f:
    artifacts = pickle.load(f)

all_users = list(artifacts['user_id_map'].keys())
sample_users = random.sample(all_users, min(N_USERS, len(all_users)))

print("=" * 60)
print("  Sample API Inputs — POST /recommend")
print("=" * 60)

samples = []
for i, user_id in enumerate(sample_users):
    mode   = MODES[i % len(MODES)]   # rotate through modes
    sample = {
        "user_id": user_id,
        "top_k":   10,
        "mode":    mode
    }
    samples.append(sample)
    print(f"\n── Sample {i+1} ({mode}) ──────────────────────────────")
    print(json.dumps(sample, indent=2))

# Also save to a JSON file for easy copy-paste
with open('sample_inputs.json', 'w') as f:
    json.dump(samples, f, indent=2)

print("\n" + "=" * 60)
print(f"  Saved {len(samples)} samples to sample_inputs.json")
print("=" * 60)
