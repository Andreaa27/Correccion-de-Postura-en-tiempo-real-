import cv2
import numpy as np
import mediapipe as mp
import joblib
import pandas as pd
from scipy.signal import find_peaks
from collections import deque
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. RUTAS
# --------------------------------------------------
VIDEO      = r"C:\dominadas_prueba.mp4"
MODELO     = r"C:\modelo_fase_dominadas_rt.pkl"
SCALER     = r"C:\scaler_fase_dominadas_rt.pkl"
CSV_SALIDA = r"C:\evaluacion_curva_vs_ml.csv"

clf    = joblib.load(MODELO)
scaler = joblib.load(SCALER)
mp_pose = mp.solutions.pose

# --------------------------------------------------
# 2. UTILIDADES
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

def mean_safe(x): return float(np.mean(x)) if len(x) else 0.0
def min_safe(x):  return float(np.min(x)) if len(x) else 0.0
def max_safe(x):  return float(np.max(x)) if len(x) else 0.0

# --------------------------------------------------
# 3. FASE POR CURVA (GROUND TRUTH)
# --------------------------------------------------
def fase_por_curva(hist):
    if len(hist) < hist.maxlen:
        return -1
    ang = np.array(hist)
    mn, mx = ang.min(), ang.max()
    amp = mx - mn + 1e-6
    a = ang[-1]
    if a <= mn + 0.2 * amp:
        return 1  # abajo
    if a >= mx - 0.2 * amp:
        return 3  # arriba
    return 2  # movimiento

# --------------------------------------------------
# 4. PROCESAMIENTO FRAME A FRAME
# --------------------------------------------------
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
ang_hist = deque(maxlen=20)
vel_hist = deque(maxlen=10)

datos = []
frame_id = 0

with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        ang = vel = acc = 0.0
        fase_gt = -1
        fase_ml = -1

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            ang = calcular_angulo(
                [lm[12].x * w, lm[12].y * h],
                [lm[14].x * w, lm[14].y * h],
                [lm[16].x * w, lm[16].y * h]
            )

            ang_hist.append(ang)
            vel = vel_ang(ang_hist, fps)
            vel_hist.append(vel)
            acc = acc_ang(vel_hist, fps)

            # ---------- GT POR CURVA ----------
            fase_gt = fase_por_curva(ang_hist)

            # ---------- PREDICCION ML ----------
            if fase_gt != -1:
                X = np.array([[
                    ang,
                    vel,
                    acc,
                    mean_safe(ang_hist),
                    mean_safe(vel_hist),
                    min_safe(ang_hist),
                    max_safe(ang_hist)
                ]])
                Xs = scaler.transform(X)
                fase_ml = int(clf.predict(Xs)[0])

        datos.append([frame_id, ang, vel, acc, fase_gt, fase_ml])
        frame_id += 1

cap.release()

# --------------------------------------------------
# 5. MÉTRICAS Y CSV
# --------------------------------------------------
df = pd.DataFrame(datos, columns=["frame", "angulo", "vel", "acc", "fase_gt", "fase_ml"])
df = df[df["fase_gt"] != -1]
df.to_csv(CSV_SALIDA, index=False)

print("\nFrames evaluados:", len(df))
print("Coincidencia total (%):", np.mean(df["fase_gt"] == df["fase_ml"]) * 100)

cm = confusion_matrix(df["fase_gt"], df["fase_ml"], labels=[1, 2, 3])
print("\nMATRIZ DE CONFUSIÓN (GT Curva vs ML)")
print("Filas = Curva | Columnas = ML")
print(pd.DataFrame(cm, index=["Abajo", "Movimiento", "Arriba"], columns=["Abajo", "Movimiento", "Arriba"]))

print("\nREPORTE DE CLASIFICACIÓN:")
print(classification_report(df["fase_gt"], df["fase_ml"], target_names=["Abajo", "Movimiento", "Arriba"]))

print("\nCSV generado en:")
print(CSV_SALIDA)