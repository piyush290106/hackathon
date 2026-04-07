"""
inference.py — Baseline agent inference script for Supply Chain OpenEnv.

This script runs a heuristic baseline agent against all three tasks,
prints per-step rewards, and reports a reproducible final score.

Usage:
    python inference.py                     # runs against live server
    python inference.py --offline           # uses embedded env directly (no server needed)
    python inference.py --task inventory_reorder
    python inference.py --task route_optimization
    python inference.py --task demand_forecast_allocation

The offline mode is what the OpenEnv validator uses to verify scores.
"""

from __future__ import annotations
import argparse
import json
import math
import random
import sys
from typing import Dict, Optional

# ── Try to import the env directly (offline mode) ─────────────────────────────
try:
    from env import SupplyChainEnv
    from models import TaskId
    OFFLINE_AVAILABLE = True
except ImportError:
    OFFLINE_AVAILABLE = False

# ── Try requests for online mode ──────────────────────────────────────────────
try:
    import requests as _requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

BASE_URL = "http://localhost:8000"
SEED = 42  # fixed seed → reproducible scores


# ══════════════════════════════════════════════════════════════════════════════
# Heuristic Baseline Agents
# ══════════════════════════════════════════════════════════════════════════════

class InventoryAgent:
    """Simple reorder-point agent: order up-to-max when below reorder point."""

    def act(self, obs: dict) -> dict:
        orders: Dict[str, int] = {}
        for item in obs["items"]:
            if item["current_stock"] < item["reorder_point"]:
                qty = item["max_stock"] - item["current_stock"] - item["pending_order_qty"]
                orders[item["sku_id"]] = max(0, qty)
        return {"orders": orders}


class RouteAgent:
    """
    Nearest-neighbour greedy routing agent.
    Assigns stops to vehicles in round-robin by demand, then sorts each
    vehicle's stops by proximity to the previous stop.
    """

    @staticmethod
    def _dist(lat1, lon1, lat2, lon2) -> float:
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
        return R * 2 * math.asin(math.sqrt(a))

    def act(self, obs: dict) -> dict:
        stops = obs["stops"]
        vehicles = obs["vehicles"]
        depot_lat, depot_lon = obs["depot_lat"], obs["depot_lon"]

        # sort stops by distance from depot
        stops_sorted = sorted(
            stops,
            key=lambda s: self._dist(depot_lat, depot_lon, s["lat"], s["lon"])
        )

        # assign to vehicles respecting capacity
        vehicle_stops: Dict[str, list] = {v["vehicle_id"]: [] for v in vehicles}
        vehicle_load: Dict[str, int] = {v["vehicle_id"]: 0 for v in vehicles}
        caps: Dict[str, int] = {v["vehicle_id"]: v["capacity_units"] for v in vehicles}

        for stop in stops_sorted:
            # find vehicle with most remaining capacity
            best_veh = max(
                vehicles,
                key=lambda v: caps[v["vehicle_id"]] - vehicle_load[v["vehicle_id"]]
            )
            vid = best_veh["vehicle_id"]
            if vehicle_load[vid] + stop["demand_units"] <= caps[vid]:
                vehicle_stops[vid].append(stop)
                vehicle_load[vid] += stop["demand_units"]

        # nearest-neighbour sequence per vehicle
        routes = []
        for v in vehicles:
            vid = v["vehicle_id"]
            assigned = vehicle_stops[vid][:]
            if not assigned:
                routes.append({"vehicle_id": vid, "stop_sequence": []})
                continue
            seq = []
            cur_lat, cur_lon = depot_lat, depot_lon
            remaining = assigned[:]
            while remaining:
                nearest = min(remaining, key=lambda s: self._dist(cur_lat, cur_lon, s["lat"], s["lon"]))
                seq.append(nearest["stop_id"])
                cur_lat, cur_lon = nearest["lat"], nearest["lon"]
                remaining.remove(nearest)
            routes.append({"vehicle_id": vid, "stop_sequence": seq})

        return {"routes": routes}


class ForecastAgent:
    """
    7-day moving average forecaster + proportional allocation agent.
    """

    def act(self, obs: dict) -> dict:
        history_map: Dict[tuple, list] = {}
        for h in obs["history"]:
            history_map[(h["sku_id"], h["warehouse_id"])] = h["sales_last_28d"]

        central_map: Dict[str, int] = {c["sku_id"]: c["available_units"] for c in obs["central_inventory"]}

        forecasts: Dict[str, Dict[str, int]] = {}
        allocations = []

        for sku in obs["skus"]:
            sid = sku["sku_id"]
            forecasts[sid] = {}
            wh_forecasts: Dict[str, int] = {}
            for wh in obs["warehouses"]:
                wid = wh["warehouse_id"]
                sales = history_map.get((sid, wid), [10] * 28)
                # 7-day moving average (last week)
                pred = max(1, int(sum(sales[-7:]) / 7))
                wh_forecasts[wid] = pred
            forecasts[sid] = wh_forecasts

            # proportional allocation from central inventory
            total_forecast = sum(wh_forecasts.values())
            available = central_map.get(sid, 0)
            for wh in obs["warehouses"]:
                wid = wh["warehouse_id"]
                proportion = wh_forecasts[wid] / max(total_forecast, 1)
                units = min(int(proportion * available * 0.8), wh["capacity"])  # keep 20% buffer
                allocations.append({"sku_id": sid, "warehouse_id": wid, "units_to_send": units})

        return {"forecasts": forecasts, "allocations": allocations}


# ══════════════════════════════════════════════════════════════════════════════
# Runners
# ══════════════════════════════════════════════════════════════════════════════

AGENTS = {
    "inventory_reorder": InventoryAgent(),
    "route_optimization": RouteAgent(),
    "demand_forecast_allocation": ForecastAgent(),
}


def run_offline(task_id: str) -> float:
    """Run the agent against the embedded environment (no server required)."""
    env = SupplyChainEnv()
    agent = AGENTS[task_id]

    obs = env.reset(task_id=task_id, seed=SEED)
    episode_rewards = []
    step = 0

    print(f"\n[offline] Task: {task_id}")
    while True:
        action = agent.act(obs)
        result = env.step(task_id=task_id, action=action)
        obs = result.observation
        episode_rewards.append(result.reward)
        step += 1
        print(f"  step {step:02d} | reward={result.reward:.4f} | done={result.done}")
        if result.done:
            break

    mean_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"  ── episode mean reward: {mean_reward:.4f}")
    return mean_reward


def run_online(task_id: str) -> float:
    """Run the agent against the HTTP API server."""
    if not REQUESTS_AVAILABLE:
        raise ImportError("pip install requests")
    agent = AGENTS[task_id]

    r = _requests.post(f"{BASE_URL}/reset", json={"task_id": task_id, "seed": SEED})
    r.raise_for_status()
    obs = r.json()["observation"]

    episode_rewards = []
    step = 0

    print(f"\n[online] Task: {task_id}")
    while True:
        action = agent.act(obs)
        r = _requests.post(f"{BASE_URL}/step", json={"task_id": task_id, "action": action})
        r.raise_for_status()
        data = r.json()
        obs = data["observation"]
        reward = data["reward"]
        done = data["done"]
        episode_rewards.append(reward)
        step += 1
        print(f"  step {step:02d} | reward={reward:.4f} | done={done}")
        if done:
            break

    mean_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"  ── episode mean reward: {mean_reward:.4f}")
    return mean_reward


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Supply Chain OpenEnv baseline inference")
    parser.add_argument(
        "--task",
        choices=["inventory_reorder", "route_optimization", "demand_forecast_allocation", "all"],
        default="all",
    )
    parser.add_argument("--offline", action="store_true", default=True,
                        help="Run against embedded env (default). Use --no-offline for HTTP server.")
    parser.add_argument("--no-offline", dest="offline", action="store_false")
    args = parser.parse_args()

    tasks = (
        ["inventory_reorder", "route_optimization", "demand_forecast_allocation"]
        if args.task == "all" else [args.task]
    )

    if args.offline and not OFFLINE_AVAILABLE:
        print("ERROR: offline mode requires env.py / models.py in the same directory.", file=sys.stderr)
        sys.exit(1)

    runner = run_offline if args.offline else run_online

    scores: Dict[str, float] = {}
    for task_id in tasks:
        scores[task_id] = runner(task_id)

    print("\n" + "═" * 50)
    print("FINAL SCORES (seed=42, reproducible)")
    print("═" * 50)
    for task_id, score in scores.items():
        difficulty = {"inventory_reorder": "easy", "route_optimization": "medium",
                      "demand_forecast_allocation": "hard"}[task_id]
        print(f"  {task_id:<35} [{difficulty:^6}]  {score:.4f}")
    overall = sum(scores.values()) / len(scores)
    print(f"  {'OVERALL MEAN':<35} {'':^8}  {overall:.4f}")
    print("═" * 50)

    # Write JSON results for CI/graders
    with open("scores.json", "w") as f:
        json.dump({"seed": SEED, "scores": scores, "overall": overall}, f, indent=2)
    print("\nResults written to scores.json")


if __name__ == "__main__":
    main()
