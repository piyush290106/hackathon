---
title: Supply Chain OpenEnv
emoji: 📦
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# Supply Chain OpenEnv 🏭

A **real-world supply chain management environment** for AI agents, built to the [OpenEnv](https://openenv.ai) specification.

## Tasks

| ID | Difficulty | Description |
|----|-----------|-------------|
| `inventory_reorder` | 🟢 Easy | Decide which SKUs to restock and in what quantities |
| `route_optimization` | 🟡 Medium | Sequence delivery stops across multiple vehicles |
| `demand_forecast_allocation` | 🔴 Hard | Forecast demand + allocate inventory across warehouses |

## API

```bash
# Reset to a new episode
curl -X POST /reset -d '{"task_id": "inventory_reorder", "seed": 42}'

# Submit an action
curl -X POST /step -d '{"task_id": "inventory_reorder", "action": {"orders": {"SKU-000": 40}}}'

# Check state
curl /state
```

## Baseline Scores (seed=42)

| Task | Score |
|------|-------|
| inventory_reorder | ~0.82 |
| route_optimization | ~0.71 |
| demand_forecast_allocation | ~0.65 |

## Running Locally

```bash
docker build -t supply-chain-env .
docker run -p 8000:8000 -e PORT=8000 supply-chain-env

# In another terminal:
python inference.py --no-offline
```

## Running Inference (offline, no server needed)

```bash
pip install -r requirements.txt
python inference.py --offline
```
