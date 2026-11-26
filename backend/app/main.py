from __future__ import annotations
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
import os, io
from .engine import engine, SCENARIOS

app = FastAPI(title="AV Sensor Fusion Simulator API", version="0.1.0", description="Prototype API for simulator controls, synthetic sensor data, fusion baseline, replay, and experiment records.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class EnvRequest(BaseModel):
    environment: Literal["clear", "rain", "fog", "snow", "night"]
class ScenarioRequest(BaseModel):
    scenario: Literal["urban_intersection", "highway_following", "blind_spot_bicycle"]
class FusionRequest(BaseModel):
    mode: Literal["centralized_ekf", "adaptive_ekf", "late_track", "early_fusion", "camera_only", "lidar_only", "radar_only", "camera_lidar", "camera_radar", "lidar_radar"]
class ControlRequest(BaseModel):
    action: Literal["pause", "resume", "reset", "tick"]
    speed_kmh: float | None = Field(default=None, ge=0, le=160)
class FaultRequest(BaseModel):
    sensor: Literal["camera", "lidar", "radar"]
    kind: Literal["none", "noise", "dropout", "stale", "occlusion"]
    severity: float = Field(default=0.5, ge=0, le=1)
class IncidentRequest(BaseModel):
    event_type: str = Field(min_length=2, max_length=64)
    note: str = Field(default="Manual event marker", max_length=250)

@app.get("/api/health")
def health():
    yolo_path = os.getenv("YOLO_WEIGHTS", "")
    return {"status":"ok", "service":"av-sensor-fusion-api", "mode":"synthetic-demo", "models": {"yolo": "configured" if yolo_path and os.path.exists(yolo_path) else "not-configured", "lstm":"not-connected"}, "message":"API is available. Current sensor scene is synthetic demo data."}

@app.get("/api/state")
def get_state():
    return engine.snapshot()

@app.post("/api/tick")
def tick():
    return engine.tick()

@app.post("/api/control")
def control(req: ControlRequest):
    if req.speed_kmh is not None:
        engine.speed_kmh = req.speed_kmh
    if req.action == "pause": engine.running = False
    elif req.action == "resume": engine.running = True
    elif req.action == "reset":
        engine.running = True
        engine.reset()
    elif req.action == "tick":
        engine.tick()
    return engine.snapshot()

@app.post("/api/environment")
def set_environment(req: EnvRequest):
    engine.environment = req.environment
    engine._snapshot = engine._make_snapshot()
    return engine.snapshot()

@app.post("/api/scenario")
def set_scenario(req: ScenarioRequest):
    try: return engine.set_scenario(req.scenario)
    except ValueError as e: raise HTTPException(400, str(e))

@app.post("/api/fusion")
def set_fusion(req: FusionRequest):
    engine.fusion_mode = req.mode
    engine._snapshot = engine._make_snapshot()
    return engine.snapshot()

@app.post("/api/fault")
def set_fault(req: FaultRequest):
    engine.faults[req.sensor].kind = req.kind
    engine.faults[req.sensor].severity = req.severity if req.kind != "none" else 0.0
    engine._snapshot = engine._make_snapshot()
    return engine.snapshot()

@app.post("/api/incidents")
def add_incident(req: IncidentRequest):
    item = {"id": len(engine.events)+1, "time_s": engine.sim_time, "type": req.event_type, "aeb": engine.aeb, "note": req.note}
    engine.events.append(item)
    engine.events = engine.events[-100:]
    engine._snapshot = engine._make_snapshot()
    return {"event": item, "state": engine.snapshot()}

@app.get("/api/incidents")
def list_incidents():
    return {"events": engine.events}

@app.get("/api/benchmarks")
def benchmarks():
    # Values are the previously recorded model-level experiments, kept separate from synthetic live demo metrics.
    return {
      "source_note":"Recorded model-level experiments from the project notebooks; not live simulator metrics.",
      "yolo_kitti":{"dataset":"KITTI","train_images":5985,"validation_images":1496,"epochs":30,"precision":0.8491,"recall":0.7254,"map50":0.8198,"map50_95":0.5618},
      "yolo_dawn":{"dataset":"DAWN","train_images":821,"validation_images":206,"train_boxes":6407,"validation_boxes":1438,"epochs":30,"precision":0.5540,"recall":0.4980,"map50":0.5540,"map50_95":0.3580,"class_map50":{"Truck":0.546,"Person":0.648,"Bicycle":0.0825,"Car":0.876,"Motorcycle":0.614,"Bus":0.557}},
      "lstm":{"dataset":"synthetic trajectory sequences","sequences":13900,"train_sequences":11120,"test_sequences":2780,"input_timesteps":10,"input_features":["x","y","vx","vy","dx","dy"],"target":["next_dx","next_dy"],"movement_mae":0.0038,"movement_rmse":0.0050,"trajectory_mae":0.0038,"trajectory_rmse":0.0050,"ade":0.0062,"fde":0.0062,"epochs":30},
      "pointpillars":{"status":"in-progress","lidar_frames_prepared":1000,"sample_frame_points":125635,"trained_detection_model":False}
    }

@app.post("/api/inference/image")
async def infer_image(file: UploadFile = File(...)):
    """Optional real YOLO inference endpoint. Set YOLO_WEIGHTS and install ultralytics to enable it."""
    path = os.getenv("YOLO_WEIGHTS", "")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=503, detail="YOLO_WEIGHTS is not configured. Copy a trained .pt checkpoint into backend/models and set YOLO_WEIGHTS. Live dashboard sensor data remains synthetic demo data.")
    try:
        from ultralytics import YOLO
        from PIL import Image
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        model = YOLO(path)
        result = model.predict(image, verbose=False)[0]
        detections = []
        names = result.names
        if result.boxes is not None:
            for b in result.boxes:
                cls_id = int(b.cls.item())
                detections.append({"class": names.get(cls_id, str(cls_id)), "confidence": round(float(b.conf.item()), 4), "bbox_xyxy": [round(float(v), 2) for v in b.xyxy[0].tolist()]})
        return {"weights": os.path.basename(path), "count": len(detections), "detections": detections, "note":"Real YOLO inference for the uploaded image; this is separate from synthetic live simulator scene."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
