"""
env.py — Supply Chain OpenEnv core environment.

Three tasks of increasing difficulty:
  1. inventory_reorder   (easy)   — decide what to restock
  2. route_optimization  (medium) — sequence delivery stops
  3. demand_forecast_allocation (hard) — forecast + allocate inventory
"""

from __future__ import annotations
import math
import random
from typing import Any, Dict, Optional, Tuple

from models import (
    InventoryItem, InventoryObservation, ReorderAction,
    DeliveryStop, Vehicle, RouteObservation, VehicleRoute, RouteAction,
    SKU, Warehouse, SKUWarehouseHistory, CentralInventory,
    ForecastObservation, ForecastAction, AllocationDecision,
    StepResult, StateResponse, TaskId,
)


# ══════════════════════════════════════════════════════════════════════════════
# Task 1 — Inventory Reorder (Easy)
# ══════════════════════════════════════════════════════════════════════════════

class InventoryReorderTask:
    MAX_STEPS = 10

    def reset(self, seed: Optional[int] = None) -> dict:
        rng = random.Random(seed)
        self._step = 0
        self._day = 1

        self._items = [
            InventoryItem(
                sku_id=f"SKU-{i:03d}",
                current_stock=rng.randint(5, 80),
                reorder_point=rng.randint(15, 25),
                max_stock=100,
                lead_time_days=rng.randint(2, 5),
                demand_last_7d=[rng.randint(3, 12) for _ in range(7)],
                pending_order_qty=0,
            )
            for i in range(8)
        ]
        self._budget = 50_000.0
        return self._observe()

    def step(self, action: dict) -> Tuple[dict, float, bool, dict]:
        act = ReorderAction(**action)
        reward, info = self._grade(act)
        self._simulate_day()
        self._step += 1
        self._day += 1
        done = self._step >= self.MAX_STEPS
        return self._observe(), reward, done, info

    # ── internals ──────────────────────────────────────────────────────────────

    def _observe(self) -> dict:
        return InventoryObservation(
            day=self._day,
            items=self._items,
            budget_remaining=self._budget,
        ).model_dump()

    def _grade(self, act: ReorderAction) -> Tuple[float, dict]:
        """
        Reward components:
          - 0.5  for ordering below-reorder-point SKUs with sensible qty
          - 0.3  for not over-ordering (holding cost penalty)
          - 0.2  for staying within budget
        """
        needs_reorder = [it for it in self._items if it.current_stock < it.reorder_point]
        ordered_needed = sum(
            1 for it in needs_reorder
            if act.orders.get(it.sku_id, 0) > 0
        )
        coverage = ordered_needed / max(len(needs_reorder), 1)

        # penalise over-stock orders
        over_order_penalty = 0.0
        total_cost = 0.0
        for it in self._items:
            qty = act.orders.get(it.sku_id, 0)
            future_stock = it.current_stock + qty
            if future_stock > it.max_stock:
                over_order_penalty += (future_stock - it.max_stock) / it.max_stock
            total_cost += qty * 50  # flat unit cost

        over_order_score = max(0.0, 1.0 - over_order_penalty / max(len(self._items), 1))
        if total_cost <= self._budget:
            budget_score = 1.0
        elif self._budget <= 0.0:
            budget_score = 0.0
        else:
            budget_score = max(0.0, 1.0 - (total_cost - self._budget) / self._budget)
        self._budget -= min(total_cost, self._budget)

        reward = round(0.5 * coverage + 0.3 * over_order_score + 0.2 * budget_score, 4)
        return reward, {"coverage": coverage, "over_order_score": over_order_score, "budget_score": budget_score}

    def _simulate_day(self):
        for it in self._items:
            daily_demand = int(sum(it.demand_last_7d) / 7 * random.uniform(0.8, 1.2))
            it.current_stock = max(0, it.current_stock - daily_demand)
            it.demand_last_7d = it.demand_last_7d[1:] + [daily_demand]


# ══════════════════════════════════════════════════════════════════════════════
# Task 2 — Route Optimization (Medium)
# ══════════════════════════════════════════════════════════════════════════════

class RouteOptimizationTask:
    MAX_STEPS = 5

    def reset(self, seed: Optional[int] = None) -> dict:
        rng = random.Random(seed)
        self._step = 0

        # depot near a city centre
        self._depot_lat, self._depot_lon = 28.6139, 77.2090  # New Delhi

        n_stops = rng.randint(10, 16)
        self._stops = [
            DeliveryStop(
                stop_id=f"STOP-{i:02d}",
                lat=self._depot_lat + rng.uniform(-0.3, 0.3),
                lon=self._depot_lon + rng.uniform(-0.3, 0.3),
                demand_units=rng.randint(1, 8),
                time_window_open=rng.randint(0, 120),
                time_window_close=rng.randint(240, 480),
            )
            for i in range(n_stops)
        ]
        self._vehicles = [
            Vehicle(vehicle_id=f"VEH-{j}", capacity_units=rng.randint(20, 30))
            for j in range(3)
        ]
        self._best_distance: Optional[float] = None
        return self._observe()

    def step(self, action: dict) -> Tuple[dict, float, bool, dict]:
        act = RouteAction(**action)
        reward, info = self._grade(act)
        self._step += 1
        done = self._step >= self.MAX_STEPS
        return self._observe(), reward, done, info

    # ── internals ──────────────────────────────────────────────────────────────

    def _observe(self) -> dict:
        return RouteObservation(
            depot_lat=self._depot_lat,
            depot_lon=self._depot_lon,
            stops=self._stops,
            vehicles=self._vehicles,
        ).model_dump()

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2) -> float:
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        return R * 2 * math.asin(math.sqrt(a))

    def _route_distance(self, stop_ids: list) -> float:
        stop_map = {s.stop_id: s for s in self._stops}
        total = 0.0
        prev_lat, prev_lon = self._depot_lat, self._depot_lon
        for sid in stop_ids:
            s = stop_map.get(sid)
            if s:
                total += self._haversine(prev_lat, prev_lon, s.lat, s.lon)
                prev_lat, prev_lon = s.lat, s.lon
        total += self._haversine(prev_lat, prev_lon, self._depot_lat, self._depot_lon)
        return total

    def _grade(self, act: RouteAction) -> Tuple[float, dict]:
        stop_map = {s.stop_id: s for s in self._stops}
        cap_map = {v.vehicle_id: v.capacity_units for v in self._vehicles}

        all_assigned = set()
        capacity_violations = 0
        tw_violations = 0
        total_distance = 0.0

        for vr in act.routes:
            load = sum(stop_map[sid].demand_units for sid in vr.stop_sequence if sid in stop_map)
            cap = cap_map.get(vr.vehicle_id, 0)
            if load > cap:
                capacity_violations += 1

            # time window check (simplified: assume 1 min per km at 60 km/h)
            elapsed = 0.0
            prev_lat, prev_lon = self._depot_lat, self._depot_lon
            for sid in vr.stop_sequence:
                s = stop_map.get(sid)
                if not s:
                    continue
                dist_km = self._haversine(prev_lat, prev_lon, s.lat, s.lon)
                elapsed += (dist_km / 40.0) * 60  # 40 km/h → minutes
                if elapsed > s.time_window_close:
                    tw_violations += 1
                prev_lat, prev_lon = s.lat, s.lon

            all_assigned.update(vr.stop_sequence)
            total_distance += self._route_distance(vr.stop_sequence)

        coverage = len(all_assigned & {s.stop_id for s in self._stops}) / len(self._stops)

        # normalise distance vs naive (visit all from one vehicle)
        naive = self._route_distance([s.stop_id for s in self._stops])
        distance_score = min(1.0, naive / max(total_distance, 1.0))

        violation_penalty = 0.15 * capacity_violations + 0.1 * tw_violations
        reward = round(max(0.0, 0.4 * coverage + 0.6 * distance_score - violation_penalty), 4)
        return reward, {
            "coverage": coverage, "distance_score": distance_score,
            "total_distance_km": round(total_distance, 2),
            "capacity_violations": capacity_violations, "tw_violations": tw_violations,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Task 3 — Demand Forecast & Allocation (Hard)
# ══════════════════════════════════════════════════════════════════════════════

class DemandForecastTask:
    MAX_STEPS = 4  # 4 weeks

    def reset(self, seed: Optional[int] = None) -> dict:
        rng = random.Random(seed)
        self._step = 0
        self._week = 1

        self._skus = [
            SKU(sku_id=f"SKU-{i:02d}", name=f"Product-{i}",
                unit_cost=round(rng.uniform(10, 200), 2),
                holding_cost_per_unit_per_day=round(rng.uniform(0.05, 0.5), 3),
                stockout_cost_per_unit=round(rng.uniform(5, 50), 2))
            for i in range(5)
        ]
        self._warehouses = [
            Warehouse(warehouse_id=f"WH-{j}", region=r, capacity=rng.randint(300, 600))
            for j, r in enumerate(["North", "South", "East"])
        ]
        self._true_demand: Dict[str, Dict[str, int]] = {
            s.sku_id: {w.warehouse_id: rng.randint(20, 80) for w in self._warehouses}
            for s in self._skus
        }
        self._central: Dict[str, int] = {s.sku_id: rng.randint(150, 300) for s in self._skus}
        self._wh_stock: Dict[str, Dict[str, int]] = {
            s.sku_id: {w.warehouse_id: rng.randint(10, 60) for w in self._warehouses}
            for s in self._skus
        }
        return self._observe()

    def step(self, action: dict) -> Tuple[dict, float, bool, dict]:
        act = ForecastAction(**action)
        reward, info = self._grade(act)
        self._simulate_week(act)
        self._step += 1
        self._week += 1
        done = self._step >= self.MAX_STEPS
        return self._observe(), reward, done, info

    # ── internals ──────────────────────────────────────────────────────────────

    def _observe(self) -> dict:
        history = [
            SKUWarehouseHistory(
                sku_id=s.sku_id,
                warehouse_id=w.warehouse_id,
                sales_last_28d=[
                    int(self._true_demand[s.sku_id][w.warehouse_id] * random.uniform(0.7, 1.3))
                    for _ in range(28)
                ],
                current_stock=self._wh_stock[s.sku_id][w.warehouse_id],
                transit_stock=0,
            )
            for s in self._skus for w in self._warehouses
        ]
        central_inv = [
            CentralInventory(sku_id=s.sku_id, available_units=self._central[s.sku_id])
            for s in self._skus
        ]
        return ForecastObservation(
            week_number=self._week,
            skus=self._skus,
            warehouses=self._warehouses,
            history=history,
            central_inventory=central_inv,
        ).model_dump()

    def _grade(self, act: ForecastAction) -> Tuple[float, dict]:
        # 1. Forecast accuracy (MAPE-based)
        mape_scores = []
        for s in self._skus:
            for w in self._warehouses:
                true_d = self._true_demand[s.sku_id][w.warehouse_id]
                pred_d = act.forecasts.get(s.sku_id, {}).get(w.warehouse_id, 0)
                mape = abs(true_d - pred_d) / max(true_d, 1)
                mape_scores.append(max(0.0, 1.0 - mape))
        forecast_score = sum(mape_scores) / len(mape_scores)

        # 2. Allocation efficiency
        alloc_map: Dict[str, Dict[str, int]] = {}
        for a in act.allocations:
            alloc_map.setdefault(a.sku_id, {})[a.warehouse_id] = a.units_to_send

        service_levels = []
        for s in self._skus:
            for w in self._warehouses:
                demand = self._true_demand[s.sku_id][w.warehouse_id]
                stock = self._wh_stock[s.sku_id][w.warehouse_id]
                alloc = alloc_map.get(s.sku_id, {}).get(w.warehouse_id, 0)
                available = stock + alloc
                service_levels.append(min(1.0, available / max(demand, 1)))

        service_score = sum(service_levels) / len(service_levels)

        # 3. Budget constraint — penalise overspending from central
        for s in self._skus:
            total_sent = sum(alloc_map.get(s.sku_id, {}).values())
            if total_sent > self._central[s.sku_id]:
                service_score *= 0.5  # heavy penalty

        reward = round(0.4 * forecast_score + 0.6 * service_score, 4)
        return reward, {"forecast_score": forecast_score, "service_score": service_score}

    def _simulate_week(self, act: ForecastAction):
        alloc_map: Dict[str, Dict[str, int]] = {}
        for a in act.allocations:
            alloc_map.setdefault(a.sku_id, {})[a.warehouse_id] = a.units_to_send

        for s in self._skus:
            total_sent = sum(alloc_map.get(s.sku_id, {}).values())
            self._central[s.sku_id] = max(0, self._central[s.sku_id] - total_sent)
            # restock central partially
            self._central[s.sku_id] += random.randint(50, 120)

            for w in self._warehouses:
                demand = int(self._true_demand[s.sku_id][w.warehouse_id] * random.uniform(0.85, 1.15))
                alloc = alloc_map.get(s.sku_id, {}).get(w.warehouse_id, 0)
                self._wh_stock[s.sku_id][w.warehouse_id] = max(
                    0, self._wh_stock[s.sku_id][w.warehouse_id] + alloc - demand
                )

        # shift true demand slightly each week
        for s in self._skus:
            for w in self._warehouses:
                self._true_demand[s.sku_id][w.warehouse_id] = max(
                    5, self._true_demand[s.sku_id][w.warehouse_id] + random.randint(-8, 8)
                )


# ══════════════════════════════════════════════════════════════════════════════
# Master Environment
# ══════════════════════════════════════════════════════════════════════════════

class SupplyChainEnv:
    def __init__(self):
        self._tasks = {
            "inventory_reorder": InventoryReorderTask(),
            "route_optimization": RouteOptimizationTask(),
            "demand_forecast_allocation": DemandForecastTask(),
        }
        self._active: Optional[str] = None
        self._episode = 0
        self._step = 0
        self._total_reward = 0.0
        self._done = False
        self._last_obs: Optional[dict] = None

    def reset(self, task_id: TaskId, seed: Optional[int] = None) -> dict:
        if task_id not in self._tasks:
            raise ValueError(f"Unknown task: {task_id}")
        self._active = task_id
        self._episode += 1
        self._step = 0
        self._total_reward = 0.0
        self._done = False
        self._last_obs = self._tasks[task_id].reset(seed)
        return self._last_obs

    def step(self, task_id: TaskId, action: dict) -> StepResult:
        if task_id != self._active:
            raise ValueError(f"Active task is '{self._active}', not '{task_id}'")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")
        obs, reward, done, info = self._tasks[task_id].step(action)
        self._last_obs = obs
        self._total_reward += reward
        self._step += 1
        self._done = done
        return StepResult(observation=obs, reward=reward, done=done, info=info)

    def state(self) -> StateResponse:
        return StateResponse(
            task_id=self._active,
            episode=self._episode,
            step=self._step,
            total_reward=round(self._total_reward, 4),
            done=self._done,
            observation=self._last_obs,
        )
