import json
import os
import tempfile
import threading
from collections import deque
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from keras.models import model_from_json
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
from ultralytics import YOLO

# Force CPU-only execution for cloud environments
os.environ["CUDA_VISIBLE_DEVICES"] = ""

TARGET_CLASSES = [
    "tricep Pushdown",
    "shoulder_stretch",
    "shoulder press",
    "rope_skipping",
    "Punching",
    "pull Up",
    "jumping_jacks",
]

OBJECT_CLASSES = [
    "none",
    "cup",
    "bottle",
    "sports ball",
    "cell phone",
    "fork",
    "spoon",
    "knife",
    "bowl",
    "sandwich",
    "remote",
    "book",
    "keyboard",
    "laptop",
    "scissors",
]

MAX_FRAMES = 64


def remove_quantization_config(config):
    if isinstance(config, dict):
        config.pop("quantization_config", None)
        for key, value in config.items():
            remove_quantization_config(value)
    elif isinstance(config, list):
        for item in config:
            remove_quantization_config(item)
    return config


@st.cache_resource
def load_action_model(config_path: str, weights_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config_json = json.load(f)

    config_json = remove_quantization_config(config_json)
    
    model = model_from_json(json.dumps(config_json))
    model.load_weights(weights_path)
    return model


@st.cache_resource
def load_pose_model(model_name: str = "yolov8n-pose.pt"):
    # Uses the same family as your notebook for keypoint extraction.
    return YOLO(model_name)


def normalize_pose_array(pose_data: np.ndarray) -> np.ndarray:
    if pose_data.ndim == 1:
        if pose_data.size % 34 != 0:
            raise ValueError("Invalid .npy shape. Expected flat data divisible by 34.")
        pose_data = pose_data.reshape(-1, 34)

    if pose_data.ndim == 2:
        if pose_data.shape[1] == 34:
            pose_data = pose_data.reshape(-1, 17, 2)
        elif pose_data.shape == (17, 2):
            pose_data = pose_data.reshape(1, 17, 2)
        else:
            raise ValueError("2D .npy must have shape (frames, 34) or (17, 2).")
    elif pose_data.ndim == 3:
        if pose_data.shape[1:] != (17, 2):
            raise ValueError("3D .npy must have shape (frames, 17, 2).")
    else:
        raise ValueError("Unsupported .npy dimensions. Use (frames,34) or (frames,17,2).")

    return pose_data.astype(np.float32)


def build_89d_features(raw_poses: np.ndarray) -> np.ndarray:
    obj_vector = np.zeros(15, dtype=np.float32)
    obj_vector[0] = 1.0

    features = []
    prev_norm_kpts = None

    for i in range(min(len(raw_poses), MAX_FRAMES)):
        current_kpts = raw_poses[i]

        nose = current_kpts[0]
        l_wrist = current_kpts[9]
        r_wrist = current_kpts[10]
        l_shoulder = current_kpts[5]
        r_shoulder = current_kpts[6]

        shoulder_dist = np.linalg.norm(l_shoulder - r_shoulder)
        scale = shoulder_dist if shoulder_dist > 0.01 else 1.0

        norm_kpts = np.copy(current_kpts)
        if np.sum(current_kpts) > 0:
            norm_kpts = (norm_kpts - current_kpts[0]) / scale

        skeleton_flat = norm_kpts.flatten()

        if prev_norm_kpts is None:
            velocity_flat = np.zeros(34, dtype=np.float32)
        else:
            velocity_flat = (norm_kpts - prev_norm_kpts).flatten()

        prev_norm_kpts = norm_kpts

        dist_l = np.linalg.norm(nose - l_wrist) / scale
        dist_r = np.linalg.norm(nose - r_wrist) / scale
        dir_l = (nose - l_wrist) / scale
        dir_r = (nose - r_wrist) / scale

        frame_matrix = np.concatenate(
            [
                skeleton_flat,
                velocity_flat,
                obj_vector,
                np.array([dist_l, dist_r], dtype=np.float32),
                dir_l,
                dir_r,
            ]
        )
        features.append(frame_matrix.astype(np.float32))

    if len(features) == 0:
        return np.zeros((1, MAX_FRAMES, 89), dtype=np.float32)

    padded = np.zeros((MAX_FRAMES, 89), dtype=np.float32)
    feature_arr = np.array(features, dtype=np.float32)
    padded[: feature_arr.shape[0], :] = feature_arr

    return padded[np.newaxis, ...]


def predict_from_pose_sequence(model, pose_sequence: np.ndarray):
    x = build_89d_features(pose_sequence)
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs


def extract_poses_from_video_file(video_path: str, pose_model) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    raw_poses = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (1280, 720))
        results = pose_model(frame_resized, verbose=False)[0]

        if results.keypoints is not None and len(results.keypoints.data) > 0:
            kpts = results.keypoints.data.cpu().numpy()[0][:, :2]
        else:
            kpts = np.zeros((17, 2), dtype=np.float32)

        raw_poses.append(kpts.astype(np.float32))

        if len(raw_poses) >= MAX_FRAMES:
            break

    cap.release()

    if len(raw_poses) < 5:
        raise ValueError("Not enough valid frames detected (need at least 5).")

    return np.array(raw_poses, dtype=np.float32)


class PoseVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose_model = load_pose_model()
        self.pose_buffer = deque(maxlen=MAX_FRAMES)
        self.lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        frame_resized = cv2.resize(image, (640, 480))
        results = self.pose_model(frame_resized, verbose=False)[0]

        if results.keypoints is not None and len(results.keypoints.data) > 0:
            kpts = results.keypoints.data.cpu().numpy()[0][:, :2].astype(np.float32)
            for point in kpts:
                x, y = int(point[0]), int(point[1])
                cv2.circle(frame_resized, (x, y), 3, (0, 255, 0), -1)
        else:
            kpts = np.zeros((17, 2), dtype=np.float32)

        with self.lock:
            self.pose_buffer.append(kpts)

        cv2.putText(
            frame_resized,
            f"Captured frames: {len(self.pose_buffer)}/{MAX_FRAMES}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return av.VideoFrame.from_ndarray(frame_resized, format="bgr24")

    def get_pose_sequence(self):
        with self.lock:
            if len(self.pose_buffer) < 5:
                return None
            return np.array(self.pose_buffer, dtype=np.float32)

    def get_buffer_size(self):
        with self.lock:
            return len(self.pose_buffer)


st.set_page_config(page_title="Kinematic Activity Classifier", layout="wide")
st.title("Kinematic Activity Classifier")
st.caption("Upload .npy/.mp4/.avi or use your camera for real-time activity prediction.")

root = Path(__file__).resolve().parent
config_path = root / "config.json"
weights_path = root / "model.weights.h5"

if not config_path.exists() or not weights_path.exists():
    st.error("Missing model files. Ensure config.json and model.weights.h5 are in the app directory.")
    st.stop()

model_89 = load_action_model(str(config_path), str(weights_path))

mode = st.sidebar.radio("Mode", ["Upload File (.npy/.mp4/.avi)", "Camera"])

if mode == "Upload File (.npy/.mp4/.avi)":
    st.subheader("Upload Mode")
    uploaded = st.file_uploader("Choose a file", type=["npy", "mp4", "avi"])

    if uploaded is not None:
        ext = uploaded.name.lower().split(".")[-1]

        if ext == "npy":
            try:
                pose_data = np.load(uploaded, allow_pickle=False)
                pose_seq = normalize_pose_array(pose_data)
                pred_idx, probs = predict_from_pose_sequence(model_89, pose_seq)

                st.success(f"Predicted Activity: {TARGET_CLASSES[pred_idx]} ({probs[pred_idx]*100:.2f}%)")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Activity": TARGET_CLASSES,
                            "Probability": probs,
                        }
                    ).sort_values("Probability", ascending=False),
                    hide_index=True,
                    use_container_width=True,
                )
                st.bar_chart(pd.DataFrame({"Probability": probs}, index=TARGET_CLASSES))
            except Exception as e:
                st.error(f"Failed to process .npy file: {e}")

        elif ext in ["mp4", "avi"]:
            try:
                if ext == "mp4":
                    st.video(uploaded)
                else:
                    st.info(f"Processing uploaded video: {uploaded.name} (Browser playback is natively unsupported for .avi files)")
                    
                pose_model = load_pose_model()

                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name

                pose_seq = extract_poses_from_video_file(tmp_path, pose_model)
                pred_idx, probs = predict_from_pose_sequence(model_89, pose_seq)

                st.success(f"Predicted Activity: {TARGET_CLASSES[pred_idx]} ({probs[pred_idx]*100:.2f}%)")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Activity": TARGET_CLASSES,
                            "Probability": probs,
                        }
                    ).sort_values("Probability", ascending=False),
                    hide_index=True,
                    use_container_width=True,
                )
                st.bar_chart(pd.DataFrame({"Probability": probs}, index=TARGET_CLASSES))
            except Exception as e:
                st.error(f"Failed to process video file: {e}")

else:
    st.subheader("Camera Mode")
    st.write("Start the camera stream, move through your activity, then click predict.")

    ctx = webrtc_streamer(
        key="activity-camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=PoseVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        if ctx.video_processor:
            st.metric("Buffered Frames", ctx.video_processor.get_buffer_size())
        else:
            st.info("Camera stream is not active yet.")

    with col2:
        if st.button("Predict Current Activity", type="primary"):
            if not ctx.video_processor:
                st.warning("Start the camera first.")
            else:
                pose_seq = ctx.video_processor.get_pose_sequence()
                if pose_seq is None:
                    st.warning("Need at least 5 frames with detectable person pose.")
                else:
                    pred_idx, probs = predict_from_pose_sequence(model_89, pose_seq)
                    st.success(
                        f"Predicted Activity: {TARGET_CLASSES[pred_idx]} ({probs[pred_idx]*100:.2f}%)"
                    )
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "Activity": TARGET_CLASSES,
                                "Probability": probs,
                            }
                        ).sort_values("Probability", ascending=False),
                        hide_index=True,
                        use_container_width=True,
                    )
                    st.bar_chart(pd.DataFrame({"Probability": probs}, index=TARGET_CLASSES))
