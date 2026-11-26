# Autonomous Vehicle Sensor Fusion Simulator

A modular prototype for exploring camera, LiDAR and radar sensing, adverse-weather effects, sensor fault injection, object-level fusion, replay and trajectory prediction.

## Repository structure

- `frontend/`: React + Vite dashboard
- `backend/`: FastAPI service and synthetic sensor simulation engine
- `ml-models/notebooks/`: available experiment notebook(s)
- `documentation/`: project report, submitted research paper and poster

## Run locally

Requirements: Python 3.12+ and Node.js 20+.

Backend:
```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Frontend (open a second terminal):
```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL printed in the terminal (usually http://localhost:5173). API docs are available at http://127.0.0.1:8000/docs.

## Important scope and reproducibility notes

The dashboard's live scene, sensor values, tracking and latency are synthetic demonstration data. They are not real sensor measurements or a validated autonomous-driving safety system.

Previously recorded model experiments are distinct from live demo output: YOLO11n was evaluated on KITTI and DAWN; the LSTM trajectory experiment used synthetic trajectories. The notebook included in this repository is the notebook file available at upload time; additional training notebooks and model checkpoints may be added separately when supplied.

PointPillars is represented in the project design, but a fully trained PointPillars detector and quantitative 3D detection benchmark are not included here. The current fusion baseline should not be described as a validated EKF/IMM-UKF implementation. No ISO 26262 certification is claimed.

## Documentation

See `documentation/` for the project report, submitted research-paper PDF and poster. Research-paper submission does not itself imply acceptance or publication.

## Data and model files

Large datasets and binary model weights are intentionally excluded from Git history. Do not commit KITTI archives, local environment folders, API keys, passwords or private credentials. Add large model artifacts through Git LFS or a separately documented download location after checking applicable dataset/model licences.
