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
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
CARPETA_VIDEOS = r"C:\dominadas_video"

BASE_DIR = r"C:\resultados_dominadas_uno"
os.makedirs(BASE_DIR, exist_ok=True)

RUTA_MODELO  = os.path.join(BASE_DIR, "modelo_fase_dominadas_rt.pkl")
RUTA_SCALER  = os.path.join(BASE_DIR, "scaler_fase_dominadas_rt.pkl")
RUTA_DATASET = os.path.join(BASE_DIR, "dataset_fase_dominadas_rt.csv")

mp_pose = mp.solutions.pose

# --------------------------------------------------
# UTILIDADES
# --------------------------------------------------
def calcular_angulo(a, b, c):
    a, b, c = map(np.array, (a, b, c))
    ba, bc = a - b, c - b
    den = np.linalg.norm(ba) * np.linalg.norm(bc)
    if den == 0:
        return 0.0
    cos = np.dot(ba, bc) / den
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

def vel_ang(hist, fps):
    return (hist[-1] - hist[-2]) * fps if len(hist) >= 2 else 0.0

def acc_ang(hist, fps):
    return (hist[-1] - hist[-2]) * fps if len(hist) >= 2 else 0.0

# --------------------------------------------------
# FASE POR CURVA 
# --------------------------------------------------
def fase_por_curva(hist):
    if len(hist) < hist.maxlen:
        return -1

    ang = np.array(hist)
    mn, mx = ang.min(), ang.max()
    amp = mx - mn + 1e-6
    a = ang[-1]

    if a <= mn + 0.2 * amp:
        return 1   # abajo
    if a >= mx - 0.2 * amp:
        return 3   # arriba
    return 2       # movimiento

# --------------------------------------------------
# FLUJO OPTICO (REPETICIONES)
# --------------------------------------------------
def extraer_movimiento_vertical(video_path):
    cap = cv2.VideoCapture(video_path)
    ok, prev = cap.read()
    if not ok:
        return np.array([])

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    mov = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        mov.append(np.mean(flow[..., 1]))
        prev_gray = gray

    cap.release()
    return np.array(mov)

def detectar_repeticiones(mov):
    if len(mov) == 0:
        return []

    mov = pd.Series(mov).rolling(10, min_periods=1).mean().values
    mov = (mov - mov.min()) / (mov.max() - mov.min() + 1e-8)

    peaks, _   = find_peaks(mov, distance=30, prominence=0.15)
    valleys, _ = find_peaks(-mov, distance=30, prominence=0.15)

    reps = []
    for i in range(min(len(peaks), len(valleys))):
        reps.append((int(min(peaks[i], valleys[i])),
                     int(max(peaks[i], valleys[i]))))
    return reps

# --------------------------------------------------
# EXTRACCION DE FEATURES
# --------------------------------------------------
def procesar_video(video_path, reps, ventana=25):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    ang_hist = deque(maxlen=ventana)
    vel_hist = deque(maxlen=5)

    X, y = [], []
    frame_id = 0

    with mp_pose.Pose(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as pose:

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame_id += 1

            if not any(ini <= frame_id <= fin for ini, fin in reps):
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if not res.pose_landmarks:
                continue

            lm = res.pose_landmarks.landmark
            ang = calcular_angulo(
                [lm[12].x*w, lm[12].y*h],
                [lm[14].x*w, lm[14].y*h],
                [lm[16].x*w, lm[16].y*h]
            )

            ang_hist.append(ang)
            vel = vel_ang(ang_hist, fps)
            vel_hist.append(vel)
            acc = acc_ang(vel_hist, fps)

            fase = fase_por_curva(ang_hist)
            if fase == -1:
                continue

            arr = np.array(ang_hist)

            X.append([
                ang,
                vel,
                acc,
                arr.mean(),
                arr.min(),
                arr.max(),
                np.mean(vel_hist)
            ])
            y.append(fase)

    cap.release()
    return np.array(X), np.array(y)

# --------------------------------------------------
# SOBREMUESTREO
# --------------------------------------------------
def balancear_dataset(X, y):
    clases, counts = np.unique(y, return_counts=True)
    max_c = counts.max()

    Xb, yb = [], []
    for c in clases:
        Xc = X[y == c]
        yc = y[y == c]

        idx = np.random.choice(len(Xc), max_c, replace=True)
        Xb.append(Xc[idx])
        yb.append(yc[idx])

    return np.vstack(Xb), np.hstack(yb)

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":

    X_total, y_total = [], []

    videos = [
        os.path.join(CARPETA_VIDEOS, v)
        for v in os.listdir(CARPETA_VIDEOS)
        if v.lower().endswith((".mp4", ".avi", ".mov"))
    ]

    print("Videos encontrados:", len(videos))

    for vid in videos:
        print("\nProcesando:", os.path.basename(vid))

        mov = extraer_movimiento_vertical(vid)
        reps = detectar_repeticiones(mov)

        print("  Repeticiones:", len(reps))
        if len(reps) == 0:
            continue

        X, y = procesar_video(vid, reps)
        if len(X) == 0:
            continue

        X_total.append(X)
        y_total.append(y)

    X_total = np.vstack(X_total)
    y_total = np.hstack(y_total)

    print("\nDistribucion original:", np.unique(y_total, return_counts=True))

    X_bal, y_bal = balancear_dataset(X_total, y_total)

    print("Distribucion balanceada:", np.unique(y_bal, return_counts=True))

    df = pd.DataFrame(X_bal, columns=[
        "ang", "vel", "acc",
        "ang_mean", "ang_min", "ang_max",
        "vel_mean"
    ])
    df["fase"] = y_bal
    df.to_csv(RUTA_DATASET, index=False)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_bal)

    modelo = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    modelo.fit(Xs, y_bal)

    joblib.dump(modelo, RUTA_MODELO)
    joblib.dump(scaler, RUTA_SCALER)

    print("\n=== ENTRENAMIENTO COMPLETADO ===")
    print("Modelo:", RUTA_MODELO)
    print("Scaler:", RUTA_SCALER)
    print("Dataset:", RUTA_DATASET)
