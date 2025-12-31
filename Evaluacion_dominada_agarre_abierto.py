import cv2
import numpy as np
import mediapipe as mp
import joblib
import pandas as pd
from collections import deque
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
# RUTAS
# --------------------------------------------------
VIDEO = r"C:\dominada_video.mp4"

MODELO = r"C:\modelo_fases.pkl"
SCALER = r"C:\scaler_fases.pkl"

CSV_SALIDA = r"C:\evaluacion_gt_vs_ml.csv"

clf = joblib.load(MODELO)
scaler = joblib.load(SCALER)

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
# FEATURES (MISMAS 41 QUE ENTRENAMIENTO)
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
    grip = dist(wri_l, wri_r)

    for k, v in zip(
        ["ang_l", "ang_r", "ang_h", "grip", "trunk", "hip_y"],
        [ang_l, ang_r, ang_h, grip, trunk, mid_hip[1]]
    ):
        hist[k].append(v)

    feats = []

    for k in ["ang_l", "ang_r", "ang_h", "grip", "trunk"]:
        arr = np.array(hist[k])
        feats += [arr[-1], arr.mean(), arr.min(), arr.max(), np.std(arr)]

    for k in ["ang_l", "ang_r", "ang_h", "hip_y"]:
        vel = (hist[k][-1] - hist[k][-2]) * fps if len(hist[k]) >= 2 else 0.0
        hist[k + "_vel"].append(vel)

        acc = (hist[k + "_vel"][-1] - hist[k + "_vel"][-2]) * fps if len(hist[k + "_vel"]) >= 2 else 0.0
        hist[k + "_acc"].append(acc)

        feats += [
            vel,
            np.mean(hist[k + "_vel"]),
            acc,
            np.mean(hist[k + "_acc"])
        ]

    return np.array(feats)

# --------------------------------------------------
# GROUND TRUTH BIOMECANICO
# --------------------------------------------------
def fase_biomecanica(ang, vel, ang_min, ang_max,
                     vel_umbral=15,
                     ang_low=0.25,
                     ang_high=0.75):

    ang_norm = (ang - ang_min) / (ang_max - ang_min + 1e-6)

    if abs(vel) > vel_umbral:
        return 2  # Movimiento

    if ang_norm >= ang_high:
        return 3  # Arriba

    if ang_norm <= ang_low:
        return 1  # Abajo

    return 2

# --------------------------------------------------
# PREPASE PARA NORMALIZAR ANGULO
# --------------------------------------------------
cap_tmp = cv2.VideoCapture(VIDEO)
angulos = []

with mp_pose.Pose(min_detection_confidence=0.6,
                  min_tracking_confidence=0.6) as pose:

    while cap_tmp.isOpened():
        ok, frame = cap_tmp.read()
        if not ok:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            ang = angulo(
                [lm[11].x * w, lm[11].y * h],
                [lm[13].x * w, lm[13].y * h],
                [lm[15].x * w, lm[15].y * h]
            )
            angulos.append(ang)

cap_tmp.release()

ang_min = np.min(angulos)
ang_max = np.max(angulos)

# --------------------------------------------------
# PROCESAMIENTO FRAME A FRAME
# --------------------------------------------------
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

keys = (
    ["ang_l", "ang_r", "ang_h", "grip", "trunk", "hip_y"] +
    [k + s for k in ["ang_l", "ang_r", "ang_h", "hip_y"]
     for s in ["_vel", "_acc"]]
)

hist = {k: deque(maxlen=25) for k in keys}

datos = []
frame_id = 0

with mp_pose.Pose(min_detection_confidence=0.6,
                  min_tracking_confidence=0.6) as pose:

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

            ang = angulo(
                [lm[11].x * w, lm[11].y * h],
                [lm[13].x * w, lm[13].y * h],
                [lm[15].x * w, lm[15].y * h]
            )

            vel = (ang - hist["ang_l"][-1]) * fps if len(hist["ang_l"]) else 0.0

            fase_gt = fase_biomecanica(ang, vel, ang_min, ang_max)

            feat = features_frame(lm, h, w, fps, hist)
            Xs = scaler.transform(feat.reshape(1, -1))
            fase_ml = int(clf.predict(Xs)[0])

        datos.append([frame_id, ang, vel, fase_gt, fase_ml])
        frame_id += 1

cap.release()

# --------------------------------------------------
# METRICAS Y CSV
# --------------------------------------------------
df = pd.DataFrame(
    datos,
    columns=["frame", "angulo", "vel", "fase_gt", "fase_ml"]
)

df = df[df["fase_gt"] != -1]
df.to_csv(CSV_SALIDA, index=False)

print("\nFrames evaluados:", len(df))
print("Exactitud global (%):",
      np.mean(df["fase_gt"] == df["fase_ml"]) * 100)

cm = confusion_matrix(df["fase_gt"], df["fase_ml"], labels=[1, 2, 3])

print("\nMATRIZ DE CONFUSION (GT vs ML)")
print(pd.DataFrame(
    cm,
    index=["Abajo", "Movimiento", "Arriba"],
    columns=["Abajo", "Movimiento", "Arriba"]
))

print("\nREPORTE DE CLASIFICACION")
print(classification_report(
    df["fase_gt"],
    df["fase_ml"],
    target_names=["Abajo", "Movimiento", "Arriba"]
))

print("\nCSV generado en:")
print(CSV_SALIDA)
