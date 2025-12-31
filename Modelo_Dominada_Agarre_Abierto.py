import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from collections import deque
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import json
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
CARPETA_VIDEOS = r"C:\repeticiones"
BASE_DIR = r"C:\modelo_fases_reglas"

os.makedirs(BASE_DIR, exist_ok=True)

RUTA_MODELO   = os.path.join(BASE_DIR, "modelo_fases.pkl")
RUTA_SCALER   = os.path.join(BASE_DIR, "scaler_fases.pkl")
RUTA_DATASET  = os.path.join(BASE_DIR, "dataset_fases.csv")
RUTA_COLUMNAS = os.path.join(BASE_DIR, "columnas_fases.json")

FOTOS_DIR = os.path.join(BASE_DIR, "fotos_fases")
os.makedirs(FOTOS_DIR, exist_ok=True)

mp_pose = mp.solutions.pose

# --------------------------------------------------
# UTILIDADES
# --------------------------------------------------
def angulo(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    den = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / den, -1, 1))))

def dist(p, q):
    return float(np.linalg.norm(np.array(p) - np.array(q)))

# --------------------------------------------------
# MOVIMIENTO VERTICAL (OPTICAL FLOW)
# --------------------------------------------------
def mov_vertical(video_path):
    cap = cv2.VideoCapture(video_path)
    ok, prev = cap.read()
    if not ok:
        return np.array([])

    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    mv = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mv.append(np.mean(flow[..., 1]))
        prev = gray

    cap.release()
    return np.array(mv)

# --------------------------------------------------
# DETECTAR INICIOS DE REPETICION
# --------------------------------------------------
def detecta_inicios(mov):
    if len(mov) == 0:
        return []

    s = pd.Series(mov).rolling(10, min_periods=1).mean().values
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)

    inicios, _ = find_peaks(s, distance=40, prominence=0.2)
    return inicios.tolist()

# --------------------------------------------------
# ASIGNAR FASES BIOMECANICAS
# --------------------------------------------------
def asigna_fases(frames, angulos):
    angulos = np.array(angulos)
    ang_max = np.max(angulos)

    frame2fase = {}
    for fr, ang in zip(frames, angulos):
        if ang >= 0.75 * ang_max:
            frame2fase[fr] = 1
        elif ang >= 0.45 * ang_max:
            frame2fase[fr] = 2
        else:
            frame2fase[fr] = 3

    return frame2fase

# --------------------------------------------------
# FEATURES POR FRAME
# --------------------------------------------------
def features_frame(lm, h, w, fps, hist):
    wri_l = np.array([lm[15].x * w, lm[15].y * h])
    wri_r = np.array([lm[16].x * w, lm[16].y * h])
    elb_l = np.array([lm[13].x * w, lm[13].y * h])
    elb_r = np.array([lm[14].x * w, lm[14].y * h])
    sho_l = np.array([lm[11].x * w, lm[11].y * h])
    sho_r = np.array([lm[12].x * w, lm[12].y * h])
    hip_l = np.array([lm[23].x * w, lm[23].y * h])
    hip_r = np.array([lm[24].x * w, lm[24].y * h])

    mid_hip = (hip_l + hip_r) / 2
    mid_sho = (sho_l + sho_r) / 2

    ang_l = angulo(sho_l, elb_l, wri_l)
    ang_r = angulo(sho_r, elb_r, wri_r)
    ang_h = angulo(elb_l, sho_l, elb_r)
    trunk = angulo(mid_hip, mid_sho, [mid_sho[0], 0])
    grip  = dist(wri_l, wri_r)

    for k, v in zip(
        ["ang_l", "ang_r", "ang_h", "grip", "trunk", "hip_y"],
        [ang_l, ang_r, ang_h, grip, trunk, mid_hip[1]]
    ):
        hist[k].append(v)

    feats, names = [], []
    for k in ["ang_l", "ang_r", "ang_h", "grip", "trunk"]:
        arr = np.array(hist[k])
        feats += [arr[-1], arr.mean(), arr.min(), arr.max(), np.std(arr)]
        names += [f"{k}_last", f"{k}_mean", f"{k}_min", f"{k}_max", f"{k}_std"]

    for k in ["ang_l", "ang_r", "ang_h", "hip_y"]:
        if len(hist[k]) >= 2:
            vel = (hist[k][-1] - hist[k][-2]) * fps
        else:
            vel = 0.0
        hist[k + "_vel"].append(vel)

        if len(hist[k + "_vel"]) >= 2:
            acc = (hist[k + "_vel"][-1] - hist[k + "_vel"][-2]) * fps
        else:
            acc = 0.0
        hist[k + "_acc"].append(acc)

        feats += [vel, np.mean(hist[k + "_vel"]), acc, np.mean(hist[k + "_acc"])]
        names += [f"{k}_vl", f"{k}_vm", f"{k}_al", f"{k}_am"]

    return np.array(feats), names

# --------------------------------------------------
# PROCESAR VIDEO COMPLETO
# --------------------------------------------------
def procesa_video(video_path, repeticiones, ventana=25):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    keys = [
        "ang_l", "ang_r", "ang_h", "grip", "trunk", "hip_y"
    ] + [k + s for k in ["ang_l", "ang_r", "ang_h", "hip_y"] for s in ["_vel", "_acc"]]

    hist = {k: deque(maxlen=ventana) for k in keys}

    frame2fase = {}

    with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
        for ini, fin in repeticiones:
            frames, angs = [], []

            for fr in range(ini, fin + 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
                ok, frame = cap.read()
                if not ok:
                    continue

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)
                if not res.pose_landmarks:
                    continue

                lm = res.pose_landmarks.landmark
                ang = angulo(
                    [lm[11].x * w, lm[11].y * h],
                    [lm[13].x * w, lm[13].y * h],
                    [lm[15].x * w, lm[15].y * h]
                )

                frames.append(fr)
                angs.append(ang)

            if len(frames) >= 10:
                frame2fase.update(asigna_fases(frames, angs))

    X, y = [], []
    frame_id = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame_id += 1
            if frame_id not in frame2fase:
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if not res.pose_landmarks:
                continue

            feat, col_names = features_frame(
                res.pose_landmarks.landmark, h, w, fps, hist
            )

            X.append(feat)
            y.append(frame2fase[frame_id])

    cap.release()
    return np.array(X), np.array(y), col_names

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":

    X_all, y_all = [], []
    col_names = None

    videos = [
        os.path.join(CARPETA_VIDEOS, v)
        for v in os.listdir(CARPETA_VIDEOS)
        if v.lower().endswith((".mp4", ".avi", ".mov"))
    ]

    print("Videos encontrados:", len(videos))

    for vid in videos:
        print("\nProcesando:", os.path.basename(vid))
        mov = mov_vertical(vid)
        inicios = detecta_inicios(mov)

        if len(inicios) < 2:
            print("  No hay repeticiones suficientes")
            continue

        repeticiones = []
        for i in range(len(inicios) - 1):
            repeticiones.append((inicios[i], inicios[i + 1] - 1))
        repeticiones.append((inicios[-1], len(mov) - 1))

        X, y, col_names = procesa_video(vid, repeticiones)
        if len(X) == 0:
            continue

        X_all.append(X)
        y_all.append(y)

    X_all = np.vstack(X_all)
    y_all = np.hstack(y_all)

    print("Distribucion:", np.unique(y_all, return_counts=True))

    df = pd.DataFrame(X_all, columns=col_names)
    df["fase"] = y_all
    df.to_csv(RUTA_DATASET, index=False)

    with open(RUTA_COLUMNAS, "w") as f:
        json.dump(col_names, f)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_all)

    modelo = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    modelo.fit(Xs, y_all)

    joblib.dump(modelo, RUTA_MODELO)
    joblib.dump(scaler, RUTA_SCALER)

    print("\n=== ENTRENAMIENTO FINALIZADO ===")
    print("Modelo :", RUTA_MODELO)
    print("Scaler :", RUTA_SCALER)
    print("Dataset:", RUTA_DATASET)
