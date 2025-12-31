import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import joblib
from collections import deque
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
VIDEO = r"C:\push_up_prueba.mp4"
RESULTADOS_DIR = r"C:\rangos_por_rep_push"
RUTA_MODELO = os.path.join(RESULTADOS_DIR, "modelo_fase.pkl")
RUTA_SCALER = os.path.join(RESULTADOS_DIR, "scaler_fase.pkl")
CSV_SALIDA = os.path.join(RESULTADOS_DIR, "evaluacion_curva_vs_ml_pushup.csv")

# ---------------- CARGAR MODELO ----------------
modelo = joblib.load(RUTA_MODELO)
scaler = joblib.load(RUTA_SCALER)
mp_pose = mp.solutions.pose

# ---------------- UTIL ----------------
def calcular_angulo(a, b, c):
    a, b, c = map(np.array, (a, b, c))
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    cos = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

# ---------------- GROUND TRUTH ----------------
amp_por_fase = {
    1: 45.61,   # solo codo
    2: 53.09,   # solo codo
    3: 21.59,   # codo + hombro
    4: 13.46    # codo + hombro
}

def fase_por_curva_hibrida(hist_codo, hist_hombro):
    if len(hist_codo) < hist_codo.maxlen:
        return -1

    ang_c = np.array(hist_codo)
    ang_ch = np.array([(c+h)/2 for c,h in zip(hist_codo, hist_hombro)])

    v_c = ang_c[-1] - ang_c[-2]
    v_ch = ang_ch[-1] - ang_ch[-2]

    mn_c, mx_c = ang_c.min(), ang_c.max()
    mn_ch, mx_ch = ang_ch.min(), ang_ch.max()

    # Estimación inicial por dirección
    fase_estim = 2 if v_c < 0 else 4

    if fase_estim in [1,2]:
        a = ang_c[-1]
        mn, mx = mn_c, mx_c
        amp = amp_por_fase[fase_estim]
    else:
        a = ang_ch[-1]
        mn, mx = mn_ch, mx_ch
        amp = amp_por_fase[fase_estim]

    if a >= mx - 0.15 * amp:
        return 1  # arriba
    if a <= mn + 0.15 * amp:
        return 3  # abajo
    return fase_estim

# ---------------- PROCESAMIENTO VIDEO ----------------
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

hist_codo = deque(maxlen=20)
hist_hombro = deque(maxlen=20)
datos = []
frame_id = 0

with mp_pose.Pose(min_detection_confidence=0.6,
                  min_tracking_confidence=0.6) as pose:

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        ang_codo = ang_hombro = ang_espalda = 0.0
        fase_gt = fase_ml = -1

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            hombro = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                      lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
            codo   = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                      lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
            muneca = [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                      lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]
            cadera = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
                      lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h]

            ang_codo = calcular_angulo(hombro, codo, muneca)
            ang_hombro = calcular_angulo(codo, hombro, cadera)
            ang_espalda = calcular_angulo(hombro, cadera, cadera)

            hist_codo.append(ang_codo)
            hist_hombro.append(ang_hombro)

            # GT híbrido
            fase_gt = fase_por_curva_hibrida(hist_codo, hist_hombro)

            # Predicción ML
            if fase_gt != -1:
                X = np.array([[ang_codo, ang_hombro, ang_espalda]])
                Xs = scaler.transform(X)
                fase_ml = int(modelo.predict(Xs)[0])

        datos.append([frame_id, ang_codo, ang_hombro, ang_espalda, fase_gt, fase_ml])
        frame_id += 1

cap.release()

# ---------------- MÉTRICAS ----------------
df = pd.DataFrame(datos, columns=["frame","codo","hombro","espalda","fase_gt","fase_ml"])
df = df[df["fase_gt"] != -1]
df.to_csv(CSV_SALIDA, index=False)

print("\nFrames evaluados:", len(df))
print("Coincidencia total (%):", np.mean(df["fase_gt"] == df["fase_ml"]) * 100)

cm = confusion_matrix(df["fase_gt"], df["fase_ml"], labels=[1,2,3,4])
print("\nMATRIZ DE CONFUSIÓN (GT Curva vs ML)")
print("Filas = Curva | Columnas = ML")
print(pd.DataFrame(cm, index=["Inicio","Descenso","Abajo","Subida"], columns=["Inicio","Descenso","Abajo","Subida"]))

print("\nREPORTE DE CLASIFICACIÓN:")
print(classification_report(df["fase_gt"], df["fase_ml"], target_names=["Inicio","Descenso","Abajo","Subida"]))

print("\nCSV generado en:", CSV_SALIDA)
