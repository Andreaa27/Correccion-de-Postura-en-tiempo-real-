# entrenamiento_rangos_por_rep.py
import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from scipy.signal import find_peaks
from pykalman import KalmanFilter
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
CARPETA_CORRECTOS = r"C:\push-up correcto"
CARPETA_SALIDA = r"C:\rangos_por_rep"

os.makedirs(CARPETA_SALIDA, exist_ok=True)

RUTA_MODELO = os.path.join(CARPETA_SALIDA, "modelo_fase.pkl")
RUTA_SCALER = os.path.join(CARPETA_SALIDA, "scaler_fase.pkl")
RUTA_RANGOS = os.path.join(CARPETA_SALIDA, "rangos_por_fase.npy")
RUTA_CSV_REPS = os.path.join(CARPETA_SALIDA, "angulos_por_rep.csv")

mp_pose = mp.solutions.pose

# ---------- UTIL ----------
def calcular_angulo(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom == 0:
        return 0.0
    cos = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

def suavizar_kalman(serie):
    serie = np.array(serie, dtype=float)
    if len(serie) < 2:
        return serie
    try:
        kf = KalmanFilter(initial_state_mean=serie[0], n_dim_obs=1)
        estado, _ = kf.smooth(serie)
        return estado.ravel()
    except:
        return serie

def extraer_angulos(ruta_video):
    cap = cv2.VideoCapture(ruta_video)
    datos = []
    frame_idx = 0
    with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                hombro = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                          lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
                codo = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                        lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
                muneca = [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                          lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]
                cadera = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
                          lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h]

                ang_codo = calcular_angulo(hombro, codo, muneca)
                ang_hombro = calcular_angulo(codo, hombro, cadera)
                ang_espalda = calcular_angulo(hombro, cadera, cadera)

                datos.append([frame_idx, ang_codo, ang_hombro, ang_espalda])
            else:
                datos.append([frame_idx, 0.0, 0.0, 0.0])

    cap.release()
    df = pd.DataFrame(datos, columns=["frame", "codo", "hombro", "espalda"])
    for c in ["codo", "hombro", "espalda"]:
        df[c] = suavizar_kalman(df[c].values)
    return df

def detectar_repeticiones(df):
    df_norm = df[["codo", "hombro", "espalda"]].copy()
    for col in df_norm.columns:
        minv, maxv = df_norm[col].min(), df_norm[col].max()
        df_norm[col] = (df_norm[col] - minv) / (maxv - minv) if maxv > minv else 0.0

    df["mov"] = (
        df_norm["codo"] * 0.6 +
        df_norm["hombro"] * 0.3 +
        df_norm["espalda"] * 0.1
    )
    df["mov_suav"] = df["mov"].rolling(9, min_periods=1).mean()

    pmax, _ = find_peaks(df["mov_suav"], distance=15, prominence=0.02)
    pmin, _ = find_peaks(-df["mov_suav"], distance=15, prominence=0.02)

    reps = []
    ult_fin = 0
    for fin in pmin:
        prev_max = [m for m in pmax if m < fin]
        if not prev_max:
            continue
        ini = prev_max[-1]
        if fin - ini < 20 or ini <= ult_fin:
            continue
        reps.append((ini, fin))
        ult_fin = fin
    return reps

def asignar_fases_rep(ini, fin):
    length = fin - ini + 1
    b2 = int(0.25 * length)
    b3 = int(0.45 * length)
    b4 = int(0.70 * length)
    indices = np.arange(ini, fin + 1)
    fases = np.zeros_like(indices, dtype=int)
    for i, idx in enumerate(indices):
        pos = i
        if pos <= b2:
            fases[i] = 1  # inicio
        elif pos <= b3:
            fases[i] = 2  # descenso
        elif pos <= b4:
            fases[i] = 3  # abajo
        else:
            fases[i] = 4  # subida
    return indices, fases

def extraer_angulos_por_rep(df, reps):
    filas = []
    for i, (ini, fin) in enumerate(reps, 1):
        indices, fases = asignar_fases_rep(ini, fin)
        for fase in [1, 2, 3, 4]:
            idx_fase = indices[fases == fase]
            if len(idx_fase) == 0:
                continue
            sub = df.iloc[idx_fase]
            filas.append({
                "Rep": i,
                "Fase": fase,
                "Codo_mean": sub["codo"].mean(),
                "Hombro_mean": sub["hombro"].mean(),
                "Espalda_mean": sub["espalda"].mean(),
            })
    return pd.DataFrame(filas)

def entrenar_modelo_y_rangos(df_reps):
    # Entrenar modelo de fase
    X = df_reps[["Codo_mean", "Hombro_mean", "Espalda_mean"]]
    y = df_reps["Fase"]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    modelo = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42)
    modelo.fit(Xs, y)

    # Rangos por fase
    rangos = {}
    for fase in sorted(df_reps["Fase"].unique()):
        sub = df_reps[df_reps["Fase"] == fase]
        rangos[fase] = {
            "codo": {"min": sub["Codo_mean"].min(), "max": sub["Codo_mean"].max(), "mean": sub["Codo_mean"].mean()},
            "hombro": {"min": sub["Hombro_mean"].min(), "max": sub["Hombro_mean"].max(), "mean": sub["Hombro_mean"].mean()},
            "espalda": {"min": sub["Espalda_mean"].min(), "max": sub["Espalda_mean"].max(), "mean": sub["Espalda_mean"].mean()},
        }

    joblib.dump(modelo, RUTA_MODELO)
    joblib.dump(scaler, RUTA_SCALER)
    np.save(RUTA_RANGOS, rangos)
    return modelo, scaler, rangos

# ---------- MAIN ----------
if __name__ == "__main__":
    print("=== EXTRAYENDO ÁNGULOS POR REPETICIÓN ===")
    all_reps = []

    for archivo in os.listdir(CARPETA_CORRECTOS):
        if not archivo.lower().endswith(('.mp4', '.avi', '.mov')):
            continue
        ruta = os.path.join(CARPETA_CORRECTOS, archivo)
        print("Procesando:", archivo)
        df = extraer_angulos(ruta)
        reps = detectar_repeticiones(df)
        if len(reps) == 0:
            print("  Sin repeticiones, saltando.")
            continue
        df_rep = extraer_angulos_por_rep(df, reps)
        df_rep["Video"] = archivo
        all_reps.append(df_rep)

    if not all_reps:
        print("No se encontraron repeticiones válidas.")
        raise SystemExit

    df_final = pd.concat(all_reps, ignore_index=True)
    df_final.to_csv(RUTA_CSV_REPS, index=False)

    print("Entrenando modelo y rangos...")
    modelo, scaler, rangos = entrenar_modelo_y_rangos(df_final)

    print(" Guardado:")
    print(" - Modelo:", RUTA_MODELO)
    print(" - Scaler:", RUTA_SCALER)
    print(" - Rangos:", RUTA_RANGOS)
    print(" - CSV por rep:", RUTA_CSV_REPS)