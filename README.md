# Autonomous Vehicle Sensor Fusion Simulator

A modular prototype for exploring camera, LiDAR and radar sensing, adverse-weather effects, sensor fault injection, object-level fusion, replay and trajectory prediction.

## Online demo

- **Application:** https://av-sensor-fusion-simulator.ai.studio

## Colab notebooks

- [Colab notebook 1](https://colab.research.google.com/drive/1yx7Jld8i_kATpFByvR6wKDQIMXs8UcQ5?usp=sharing)
- [Colab notebook 2](https://colab.research.google.com/drive/12R9Tury8bcloMe21y63xWvZEWaXFgQIv?usp=sharing)
- [Colab notebook 3](https://colab.research.google.com/drive/1gqDA3taDH4pvM8XezDEpMkFj5BRiudhj)

The links must be shared with viewers who have permission to access them. Add brief notebook titles once the experiment represented by each notebook is confirmed.

## Repository structure

- `frontend/`: React + Vite dashboard
- `backend/`: FastAPI service and synthetic sensor simulation engine
- `ml-models/notebooks/`: experiment notebook(s)
- `documentation/`: project report, submitted research paper and poster (upload the final files here)

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

Previously recorded model experiments are distinct from live demo output: YOLO11n was evaluated on KITTI and DAWN; the LSTM trajectory experiment used synthetic trajectories. The notebooks linked above should be used to inspect the original experiment code and outputs.

PointPillars is represented in the project design, but a fully trained PointPillars detector and quantitative 3D detection benchmark are not included here. The current fusion baseline should not be described as a validated EKF/IMM-UKF implementation. No ISO 26262 certification is claimed.

## Documentation

Upload the final project report, submitted research-paper PDF and final poster to `documentation/`. Research-paper submission does not itself imply acceptance or publication.

## Data, model files and secrets

Large datasets and binary model weights are intentionally excluded from Git history. Do not commit KITTI archives, local environment folders, API keys, passwords or private credentials. Store secrets in local environment variables or a deployment secret manager; keep `.env` files out of Git. Rotate any key that may have been shared publicly. Add large model artifacts through Git LFS or a separately documented download location after checking applicable dataset/model licences.
