# entrenamiento_dominadas_sentado_FINAL_ULTRA.py
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
CARPETA_CORRECTOS = r"c:\dominada_abierta_correcto"
CARPETA_SALIDA = r"C:\rangos_por_rep_dominadaabierta"

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
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0: return 0.0
    coseno = np.dot(ba, bc) / denom
    return float(np.degrees(np.clip(coseno, -1.0, 1.0)))

def suavizar_kalman(serie):
    serie = np.array(serie, dtype=float)
    if len(serie) < 2: return serie
    try:
        kf = KalmanFilter(initial_state_mean=serie[0], n_dim_obs=1)
        estado, _ = kf.smooth(serie)
        return estado.ravel()
    except:
        return serie

# ---------- EXTRAER ÁNGULOS ----------
def extraer_angulos(ruta_video):
    cap = cv2.VideoCapture(ruta_video)
    datos = []
    frame_idx = 0
    with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok: break
            frame_idx += 1
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                hombro = np.array([lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                                   lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h])
                codo   = np.array([lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                                   lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h])
                muneca = np.array([lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                                   lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h])
                cadera = np.array([lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
                                   lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h])

                ang_hombro = calcular_angulo(codo, hombro, cadera)      # flex/ext brazo
                ang_tira   = calcular_angulo(muneca, hombro, cadera)    # elevación peso
                ang_espalda= calcular_angulo(hombro, cadera, cadera)    # postura

                datos.append([frame_idx, ang_hombro, ang_tira, ang_espalda])
            else:
                datos.append([frame_idx, 0.0, 0.0, 0.0])

    cap.release()
    df = pd.DataFrame(datos, columns=["frame", "hombro", "tira", "espalda"])
    for c in df.columns[1:]:
        df[c] = suavizar_kalman(df[c].values)
    return df

# ---------- DETECCIÓN ULTRA-ADAPTADA ----------
def detectar_repeticiones(df):
    df_norm = df[["hombro", "tira", "espalda"]].copy()
    for col in df_norm.columns:
        minv, maxv = df_norm[col].min(), df_norm[col].max()
        df_norm[col] = (df_norm[col] - minv) / (maxv - minv) if maxv > minv else 0.0

    df["mov"] = (
        (1 - df_norm["hombro"]) * 0.60 +
        (1 - df_norm["tira"])   * 0.35 +
        df_norm["espalda"]      * 0.05
    )
    df["mov_suav"] = df["mov"].rolling(2, min_periods=1).mean()

    pmax, _ = find_peaks(df["mov_suav"], distance=8, prominence=0.02, width=(2, 12))
    pmin, _ = find_peaks(-df["mov_suav"], distance=8, prominence=0.02, width=(2, 12))

    reps = []
    ult_fin = 0
    for imin in pmin:
        prev_max = pmax[pmax < imin]
        if prev_max.size == 0: continue
        imax = prev_max[-1]
        next_max = pmax[pmax > imin]
        if next_max.size == 0: continue
        fin = next_max[0]
        if fin - imax < 8 or imax <= ult_fin: continue
        reps.append((imax, fin))
        ult_fin = fin
    return reps

# ---------- FASES ----------
def asignar_fases_rep(ini, fin):
    length = fin - ini + 1
    b2 = int(0.25 * length)
    b3 = int(0.50 * length)
    b4 = int(0.75 * length)
    indices = np.arange(ini, fin + 1)
    fases = np.zeros_like(indices, dtype=int)
    for i, idx in enumerate(indices):
        pos = i
        if pos <= b2:      fases[i] = 1   # arriba (flex)
        elif pos <= b3:    fases[i] = 2   # bajando
        elif pos <= b4:    fases[i] = 3   # abajo (ext)
        else:              fases[i] = 4   # subiendo
    return indices, fases

# ---------- EXTRAER POR REP ----------
def extraer_angulos_por_rep(df, reps):
    filas = []
    for i, (ini, fin) in enumerate(reps, 1):
        indices, fases = asignar_fases_rep(ini, fin)
        for fase in [1, 2, 3, 4]:
            idx_fase = indices[fases == fase]
            if len(idx_fase) == 0: continue
            sub = df.iloc[idx_fase]
            filas.append({
                "Rep": i,
                "Fase": fase,
                "Hombro_mean": sub["hombro"].mean(),
                "Tira_mean": sub["tira"].mean(),
                "Espalda_mean": sub["espalda"].mean(),
            })
    return pd.DataFrame(filas)

# ---------- ENTRENAR ----------
def entrenar_modelo_y_rangos(df_reps):
    X = df_reps[["Hombro_mean", "Tira_mean", "Espalda_mean"]]
    y = df_reps["Fase"]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    modelo = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42)
    modelo.fit(Xs, y)

    rangos = {}
    for fase in sorted(df_reps["Fase"].unique()):
        sub = df_reps[df_reps["Fase"] == fase]
        rangos[fase] = {
            "hombro": {"min": sub["Hombro_mean"].min(), "max": sub["Hombro_mean"].max(), "mean": sub["Hombro_mean"].mean()},
            "tira": {"min": sub["Tira_mean"].min(), "max": sub["Tira_mean"].max(), "mean": sub["Tira_mean"].mean()},
            "espalda": {"min": sub["Espalda_mean"].min(), "max": sub["Espalda_mean"].max(), "mean": sub["Espalda_mean"].mean()},
        }

    joblib.dump(modelo, RUTA_MODELO)
    joblib.dump(scaler, RUTA_SCALER)
    np.save(RUTA_RANGOS, rangos)
    return modelo, scaler, rangos

# ---------- EXTRAS ----------
def guardar_frames_por_fase(ruta_video, df, reps, carpeta_salida):
    cap = cv2.VideoCapture(ruta_video)
    nombre_video = os.path.splitext(os.path.basename(ruta_video))[0]
    carpeta_frames = os.path.join(carpeta_salida, f"frames_{nombre_video}")
    os.makedirs(carpeta_frames, exist_ok=True)
    for i, (ini, fin) in enumerate(reps, 1):
        indices, fases = asignar_fases_rep(ini, fin)
        for fase in [1, 2, 3, 4]:
            idx_fase = indices[fases == fase]
            if len(idx_fase) == 0: continue
            idx_medio = idx_fase[len(idx_fase) // 2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx_medio)
            ret, frame = cap.read()
            if ret:
                ruta_frame = os.path.join(carpeta_frames, f"rep{i}_fase{fase}.jpg")
                cv2.imwrite(ruta_frame, frame)
    cap.release()

def guardar_video_con_anotaciones(ruta_video, df, reps, carpeta_salida):
    cap = cv2.VideoCapture(ruta_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nombre_video = os.path.splitext(os.path.basename(ruta_video))[0]
    ruta_out = os.path.join(carpeta_salida, f"{nombre_video}_anotado.mp4")
    out = cv2.VideoWriter(ruta_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    df["rep"] = 0; df["fase"] = 0
    for i, (ini, fin) in enumerate(reps, 1):
        indices, fases = asignar_fases_rep(ini, fin)
        df.loc[indices, "rep"] = i
        df.loc[indices, "fase"] = fases

    for idx, row in df.iterrows():
        cap.set(cv2.CAP_PROP_POS_FRAMES, row["frame"])
        ret, frame = cap.read()
        if not ret: continue
        texto = f"Rep: {int(row['rep'])} | Fase: {int(row['fase'])}"
        cv2.putText(frame, texto, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        out.write(frame)
    cap.release(); out.release()

# ---------- MAIN ----------
if __name__ == "__main__":
    print("=== EXTRAYENDO ÁNGULOS POR REPETICIÓN ===")
    all_reps = []
    videos = [v for v in os.listdir(CARPETA_CORRECTOS) if v.lower().endswith(('.mp4', '.avi', '.mov'))]
    for i, archivo in enumerate(videos):
        ruta = os.path.join(CARPETA_CORRECTOS, archivo)
        print("Procesando:", archivo)
        df = extraer_angulos(ruta)
        reps = detectar_repeticiones(df)
        print(f"  Picos max: {len(find_peaks(df['mov_suav'], distance=8, prominence=0.02)[0])}, "
              f"min: {len(find_peaks(-df['mov_suav'], distance=8, prominence=0.02)[0])}, "
              f"reps detectadas: {len(reps)}")
        if len(reps) == 0:
            print("  Sin repeticiones, saltando.")
            continue
        df_rep = extraer_angulos_por_rep(df, reps)
        df_rep["Video"] = archivo
        all_reps.append(df_rep)
        if i == 0:
            print("  Guardando frames y video anotado...")
            guardar_frames_por_fase(ruta, df, reps, CARPETA_SALIDA)
            guardar_video_con_anotaciones(ruta, df, reps, CARPETA_SALIDA)

    if not all_reps:
        print("No se encontraron repeticiones válidas.")
        raise SystemExit

    df_final = pd.concat(all_reps, ignore_index=True)
    df_final.to_csv(RUTA_CSV_REPS, index=False)

    print("Entrenando modelo y rangos...")
    modelo, scaler, rangos = entrenar_modelo_y_rangos(df_final)

    print("Guardado:")
    print(" - Modelo:", RUTA_MODELO)
    print(" - Scaler:", RUTA_SCALER)
    print(" - Rangos:", RUTA_RANGOS)
    print(" - CSV por rep:", RUTA_CSV_REPS)