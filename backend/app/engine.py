from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any
import math, random, time

ENVIRONMENT_FACTORS = {
    "clear": {"camera": 0.96, "lidar": 0.96, "radar": 0.94},
    "rain":  {"camera": 0.78, "lidar": 0.86, "radar": 0.91},
    "fog":   {"camera": 0.52, "lidar": 0.76, "radar": 0.91},
    "snow":  {"camera": 0.64, "lidar": 0.72, "radar": 0.87},
    "night": {"camera": 0.63, "lidar": 0.94, "radar": 0.94},
}

# This is explicitly a synthetic demonstration scene, not a real sensor dataset.
SCENARIOS = {
    "urban_intersection": "Urban Intersection — Jaywalking Pedestrian",
    "highway_following": "Highway Following — Lead Vehicle",
    "blind_spot_bicycle": "Blind Spot — Bicycle Crossing",
}

@dataclass
class Fault:
    kind: str = "none"
    severity: float = 0.0

class SimulatorEngine:
    def __init__(self):
        self.rng = random.Random(20261029)
        self.running = True
        self.tick_count = 0
        self.sim_time = 0.0
        self.speed_kmh = 44.0
        self.environment = "clear"
        self.scenario = "urban_intersection"
        self.fusion_mode = "centralized_ekf"
        self.faults: Dict[str, Fault] = {s: Fault() for s in ["camera", "lidar", "radar"]}
        self.events: List[dict] = []
        self.history: List[dict] = []
        self.aeb = "NONE"
        self._tracks: Dict[int, dict] = {}
        self._init_objects()
        self._snapshot = self._make_snapshot()

    def _init_objects(self):
        if self.scenario == "highway_following":
            self.objects = [
                {"id": 100, "class": "car", "x": 28.0, "y": 0.4, "vx": -1.0, "vy": 0.0},
                {"id": 101, "class": "truck", "x": 52.0, "y": -2.5, "vx": -0.4, "vy": 0.0},
                {"id": 102, "class": "car", "x": 65.0, "y": 2.2, "vx": 0.2, "vy": 0.0},
            ]
        elif self.scenario == "blind_spot_bicycle":
            self.objects = [
                {"id": 100, "class": "car", "x": 18.0, "y": 0.0, "vx": -0.5, "vy": 0.0},
                {"id": 101, "class": "bicycle", "x": 12.0, "y": 3.8, "vx": -1.0, "vy": -0.6},
                {"id": 102, "class": "pedestrian", "x": 21.0, "y": -4.2, "vx": -0.2, "vy": 0.1},
            ]
        else:
            self.objects = [
                {"id": 100, "class": "car", "x": 18.0, "y": 0.4, "vx": -1.2, "vy": 0.0},
                {"id": 101, "class": "pedestrian", "x": 12.0, "y": 3.4, "vx": -0.3, "vy": -0.7},
                {"id": 102, "class": "bicycle", "x": 22.0, "y": -4.2, "vx": -0.6, "vy": 0.4},
            ]
        self._tracks = {}

    def reset(self):
        self.tick_count = 0
        self.sim_time = 0.0
        self.aeb = "NONE"
        self.events = []
        self.history = []
        self._init_objects()
        self._snapshot = self._make_snapshot()
        return self._snapshot

    def set_scenario(self, scenario: str):
        if scenario not in SCENARIOS:
            raise ValueError("Unknown scenario")
        self.scenario = scenario
        self.reset()
        return self._snapshot

    def sensor_health(self, sensor: str) -> float:
        h = ENVIRONMENT_FACTORS[self.environment][sensor]
        fault = self.faults[sensor]
        if fault.kind == "dropout":
            h *= (1.0 - fault.severity)
        elif fault.kind == "noise":
            h *= (1.0 - 0.65 * fault.severity)
        elif fault.kind == "stale":
            h *= (1.0 - 0.55 * fault.severity)
        elif fault.kind == "occlusion":
            h *= (1.0 - 0.60 * fault.severity)
        return max(0.0, min(1.0, h))

    def _selected_sensors(self) -> List[str]:
        return {
            "camera_only": ["camera"],
            "lidar_only": ["lidar"],
            "radar_only": ["radar"],
            "camera_lidar": ["camera", "lidar"],
            "camera_radar": ["camera", "radar"],
            "lidar_radar": ["lidar", "radar"],
        }.get(self.fusion_mode, ["camera", "lidar", "radar"])

    def _noise(self, sensor: str) -> float:
        base = {"camera": 0.8, "lidar": 0.18, "radar": 0.65}[sensor]
        f = self.faults[sensor]
        env_penalty = 1.0 - ENVIRONMENT_FACTORS[self.environment][sensor]
        factor = 1.0 + env_penalty * 2.5 + (f.severity * 3.0 if f.kind == "noise" else 0.0)
        if f.kind == "occlusion" and sensor == "camera":
            factor += f.severity * 2.0
        return base * factor

    def _measurement(self, obj: dict, sensor: str) -> dict | None:
        fault = self.faults[sensor]
        health = self.sensor_health(sensor)
        if fault.kind == "dropout" and self.rng.random() < fault.severity:
            return None
        if sensor == "camera" and self.rng.random() > health:
            return None
        if sensor == "lidar" and self.rng.random() > max(0.12, health):
            return None
        if sensor == "radar" and self.rng.random() > max(0.2, health):
            return None
        sigma = self._noise(sensor)
        stale_shift = 0.0
        if fault.kind == "stale":
            stale_shift = fault.severity * 1.6
        x = obj["x"] - obj["vx"] * stale_shift + self.rng.gauss(0, sigma)
        y = obj["y"] - obj["vy"] * stale_shift + self.rng.gauss(0, sigma)
        confidence = max(0.05, min(0.99, health - self.rng.random() * 0.16))
        result = {"sensor": sensor, "x": round(x, 2), "y": round(y, 2), "confidence": round(confidence, 3)}
        if sensor == "radar":
            result["range_m"] = round(math.hypot(x, y), 2)
            result["doppler_mps"] = round(-obj["vx"] + self.rng.gauss(0, self._noise("radar") * 0.25), 2)
        if sensor == "lidar":
            result["points"] = max(0, int(180 * health / max(6, math.hypot(x, y)) + self.rng.randint(0, 20)))
        if sensor == "camera":
            result["bbox"] = [round(320 + y * 10, 1), round(210 + x * 2, 1), round(40 + confidence * 25, 1), round(70 + confidence * 35, 1)]
        return result

    def tick(self):
        if self.running:
            dt = 0.1
            self.sim_time += dt
            self.tick_count += 1
            # Small motion to make the demonstration dynamic.
            for obj in self.objects:
                obj["x"] = max(4, obj["x"] + obj["vx"] * dt)
                obj["y"] = max(-7, min(7, obj["y"] + obj["vy"] * dt))
            self._snapshot = self._make_snapshot()
            self.history.append({"time_s": round(self.sim_time, 2), **self._snapshot["metrics"]})
            self.history = self.history[-60:]
        return self._snapshot

    def _make_snapshot(self):
        measurements_by_sensor = {}
        for sensor in ["camera", "lidar", "radar"]:
            measurements_by_sensor[sensor] = []
        fused_objects = []
        selected = self._selected_sensors()
        for obj in self.objects:
            observations = {}
            for sensor in ["camera", "lidar", "radar"]:
                m = self._measurement(obj, sensor)
                if m is not None:
                    measurements_by_sensor[sensor].append({"object_id": obj["id"], "class": obj["class"], **m})
                    if sensor in selected:
                        observations[sensor] = m
            # Weighted object-level baseline, with weights derived from measured demo sensor health.
            weights = {s: self.sensor_health(s) * max(0.05, observations[s]["confidence"]) for s in observations}
            total = sum(weights.values())
            if total > 0:
                fx = sum(observations[s]["x"] * weights[s] for s in observations) / total
                fy = sum(observations[s]["y"] * weights[s] for s in observations) / total
                confidence = sum(observations[s]["confidence"] * weights[s] for s in observations) / total
            else:
                fx, fy, confidence = None, None, 0.0
            distance = math.hypot(fx, fy) if fx is not None else float("inf")
            closing = max(0.0, -obj["vx"])
            ttc = distance / closing if math.isfinite(distance) and closing > 0.1 else None
            risk = "CRITICAL" if ttc is not None and ttc < 2.0 and abs(fy or 99) < 2.0 else "HIGH" if distance < 12 and abs(fy or 99) < 4 else "MEDIUM" if distance < 28 else "LOW"
            fused_objects.append({
                "id": obj["id"], "class": obj["class"], "truth": {"x": round(obj["x"],2), "y": round(obj["y"],2)},
                "fused": None if fx is None else {"x": round(fx,2), "y": round(fy,2)},
                "confidence": round(confidence,3), "sources": list(observations.keys()), "risk": risk,
                "distance_m": None if not math.isfinite(distance) else round(distance,2), "ttc_s": None if ttc is None else round(ttc,2),
            })
        nearest = min((o["distance_m"] for o in fused_objects if o["distance_m"] is not None), default=99)
        self.aeb = "BRAKE_REQUEST" if nearest < 7 else "WARNING" if nearest < 13 else "NONE"
        if self.aeb != "NONE" and (not self.events or self.events[-1].get("aeb") != self.aeb):
            self.events.append({"id": len(self.events)+1, "time_s": round(self.sim_time,2), "type": "AEB_STATE", "aeb": self.aeb, "note": f"Synthetic demo AEB state: {self.aeb}; nearest object {nearest:.1f} m"})
            self.events = self.events[-100:]
        counts = {s: len(measurements_by_sensor[s]) for s in measurements_by_sensor}
        health = {s: round(self.sensor_health(s), 3) for s in measurements_by_sensor}
        average_health = sum(health[s] for s in selected) / max(1, len(selected))
        # Latency is a synthetic illustrative value, not a measured benchmark.
        latency = round(1.2 + (3.0 * (1-average_health)) + self.rng.random() * 0.7, 2)
        metrics = {"sensor_health": health, "detections": counts, "fusion_latency_ms_demo": latency,
                   "active_tracks": len(fused_objects), "nearest_object_m": None if nearest == 99 else round(nearest,2),
                   "aeb": self.aeb, "mode": self.fusion_mode}
        return {
            "timestamp": time.time(), "running": self.running, "tick": self.tick_count, "sim_time_s": round(self.sim_time,2),
            "speed_kmh": round(self.speed_kmh,1), "environment": self.environment, "scenario": self.scenario,
            "scenario_label": SCENARIOS[self.scenario], "fusion_mode": self.fusion_mode, "selected_sensors": selected,
            "sensors": {s: {"health": health[s], "count": counts[s], "fault": {"kind": self.faults[s].kind, "severity": self.faults[s].severity}} for s in health},
            "measurements": measurements_by_sensor, "objects": fused_objects, "metrics": metrics,
            "events": self.events[-12:], "history": self.history[-60:],
            "notice": "Synthetic sensor demonstration. It does not invoke the trained YOLO/LSTM checkpoints or represent a validated vehicle-control system."
        }

    def snapshot(self):
        return self._snapshot

engine = SimulatorEngine()
