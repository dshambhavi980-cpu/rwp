# Kinematic Activity Streamlit App

This app uses your trained 64x89 action model with two modes:

- Upload mode: `.npy` pose arrays or `.mp4` videos
- Camera mode: browser webcam stream with live pose buffering and prediction

## Project Files

- `app.py`: Streamlit application
- `config.json`: Model architecture
- `model.weights.h5`: Trained weights
- `requirements.txt`: Python dependencies
- `Dockerfile`: Container image definition

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open: `http://localhost:8501`

## Run With Docker

```bash
docker build -t kinematic-streamlit .
docker run --rm -p 8501:8501 kinematic-streamlit
```

Open: `http://localhost:8501`

## Notes

- First run of video/camera mode downloads `yolov8m-pose.pt` automatically.
- In some Docker environments, webcam passthrough can be limited by browser/host permissions.
- `.npy` input supports `(frames,34)` and `(frames,17,2)` shapes.
