"""
models.py — Typed models for the Supply Chain OpenEnv environment.
All observations, actions, and state objects are fully typed via Pydantic.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field


# ─── Shared primitives ────────────────────────────────────────────────────────

TaskId = Literal["inventory_reorder", "route_optimization", "demand_forecast_allocation"]


class SKU(BaseModel):
    sku_id: str
    name: str
    unit_cost: float
    holding_cost_per_unit_per_day: float
    stockout_cost_per_unit: float


class Warehouse(BaseModel):
    warehouse_id: str
    region: str
    capacity: int  # max units across all SKUs


# ─── Task 1: Inventory Reorder ─────────────────────────────────────────────────

class InventoryItem(BaseModel):
    sku_id: str
    current_stock: int
    reorder_point: int
    max_stock: int
    lead_time_days: int
    demand_last_7d: List[int] = Field(..., min_length=7, max_length=7)
    pending_order_qty: int = 0


class InventoryObservation(BaseModel):
    task_id: Literal["inventory_reorder"] = "inventory_reorder"
    day: int
    items: List[InventoryItem]
    budget_remaining: float


class ReorderAction(BaseModel):
    """For each SKU, how many units to order (0 = no order)."""
    orders: Dict[str, int]  # {sku_id: quantity}


# ─── Task 2: Route Optimization ────────────────────────────────────────────────

class DeliveryStop(BaseModel):
    stop_id: str
    lat: float
    lon: float
    demand_units: int
    time_window_open: int   # minutes from depot open (e.g. 0 = 8 AM)
    time_window_close: int  # minutes from depot open


class Vehicle(BaseModel):
    vehicle_id: str
    capacity_units: int
    speed_kmh: float = 40.0


class RouteObservation(BaseModel):
    task_id: Literal["route_optimization"] = "route_optimization"
    depot_lat: float
    depot_lon: float
    stops: List[DeliveryStop]
    vehicles: List[Vehicle]


class VehicleRoute(BaseModel):
    vehicle_id: str
    stop_sequence: List[str]  # ordered list of stop_ids


class RouteAction(BaseModel):
    routes: List[VehicleRoute]


# ─── Task 3: Demand Forecast & Allocation ──────────────────────────────────────

class SKUWarehouseHistory(BaseModel):
    sku_id: str
    warehouse_id: str
    sales_last_28d: List[int] = Field(..., min_length=28, max_length=28)
    current_stock: int
    transit_stock: int = 0  # in-transit from central


class CentralInventory(BaseModel):
    sku_id: str
    available_units: int


class ForecastObservation(BaseModel):
    task_id: Literal["demand_forecast_allocation"] = "demand_forecast_allocation"
    week_number: int
    skus: List[SKU]
    warehouses: List[Warehouse]
    history: List[SKUWarehouseHistory]
    central_inventory: List[CentralInventory]


class AllocationDecision(BaseModel):
    sku_id: str
    warehouse_id: str
    units_to_send: int


class ForecastAction(BaseModel):
    forecasts: Dict[str, Dict[str, int]]   # {sku_id: {warehouse_id: forecast}}
    allocations: List[AllocationDecision]


# ─── Generic Step / Reset / State wrappers ────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: TaskId
    seed: Optional[int] = None


class StepRequest(BaseModel):
    task_id: TaskId
    action: dict  # deserialized into task-specific action inside env


class StepResult(BaseModel):
    observation: dict
    reward: float = Field(..., ge=0.0, le=1.0)
    done: bool
    info: dict = {}


class StateResponse(BaseModel):
    task_id: Optional[TaskId]
    episode: int
    step: int
    total_reward: float
    done: bool
    observation: Optional[dict]
