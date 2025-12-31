# realtime.py
import os
import cv2
import numpy as np
import mediapipe as mp
from pykalman import KalmanFilter
import joblib

# ---------- CONFIG ----------
RESULTADOS_DIR = r"C:\rangos_por_rep_push"
RUTA_MODELO   = os.path.join(RESULTADOS_DIR, "modelo_fase.pkl")
RUTA_SCALER   = os.path.join(RESULTADOS_DIR, "scaler_fase.pkl")
RUTA_RANGOS   = os.path.join(RESULTADOS_DIR, "rangos_por_fase.npy")
RUTA_VIDEO    = os.path.join(RESULTADOS_DIR, "video_realtime_reps.mp4")

mp_pose = mp.solutions.pose

# ---------- RECURSOS ----------
def cargar_recursos():
    modelo = joblib.load(RUTA_MODELO)
    scaler = joblib.load(RUTA_SCALER)
    rangos = np.load(RUTA_RANGOS, allow_pickle=True).item()
    return modelo, scaler, rangos

# ---------- ÁNGULO ----------
def calcular_angulo(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0: return 0.0
    coseno = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(coseno, -1.0, 1.0))))

# ---------- SMOOTH ----------
def suavizar_kalman(serie):
    serie = np.array(serie, dtype=float)
    if len(serie) < 2: return serie
    try:
        kf = KalmanFilter(initial_state_mean=serie[0], n_dim_obs=1)
        estado, _ = kf.smooth(serie)
        return estado.ravel()
    except:
        return serie

# ---------- FEATURE ----------
def construir_feature_frame(curr_ang, _):
    return np.array([curr_ang["codo"], curr_ang["hombro"], curr_ang["espalda"]], dtype=float).reshape(1, -1)

# ---------- FEEDBACK ----------
def feedback_por_fase(angulos, fase_idx, rangos, slack=5.0):
    if fase_idx not in rangos:
        return ["Fase desconocida"]

    msgs = []
    for parte in ("codo", "hombro"):
        info = rangos[fase_idx].get(parte)
        if info is None:
            continue

        mn = float(info["min"]) - slack
        mx = float(info["max"]) + slack
        val = float(angulos[parte])

        if val < mn:
            msgs.append(f"{parte.capitalize()} muy bajo")
        elif val > mx:
            msgs.append(f"{parte.capitalize()} muy alto")

    return msgs if msgs else ["Postura correcta"]


# ---------- COLOR ----------
def color_por_fase(fase):
    return {1: (0, 255, 255), 2: (0, 165, 255), 3: (0, 0, 255), 4: (0, 255, 0)}.get(fase, (255, 255, 255))

# ---------- BARRA  ----------
def dibujar_barra_y_rep(frame, fase, reps):
    h, w = frame.shape[:2]
    ancho, alto = 300, 20
    x, y = 50, h - 50
    fase_norm = 0 if fase not in range(1, 5) else (fase - 1) / 3
    cv2.rectangle(frame, (x, y), (x + ancho, y + alto), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x + int(ancho * fase_norm), y + alto), color_por_fase(fase), -1)
    
# ---------- FSM REPETICIONES ----------
ESTADO_WAIT   = "WAIT_START"
ESTADO_INREP  = "IN_REP"
ESTADO_LOCKED = "LOCKED"

def init_fsm():
    return {
        "estado": ESTADO_WAIT,
        "frames_fase1": 0,
        "frames_rep": 0,
        "visitadas": set(),
        "reps": 0
    }

def actualizar_fsm(fsm, fase):
    MIN_FRAMES_REP = 10    # duracion minima de una rep
    RESET_FRAMES   = 8     # frames estables en fase 1 para reset

    if fase not in (1, 2, 3, 4):
        return fsm

    # -------- WAIT_START --------
    if fsm["estado"] == ESTADO_WAIT:
        if fase == 1:
            fsm["frames_fase1"] += 1
            if fsm["frames_fase1"] >= 3:
                fsm["estado"] = ESTADO_INREP
                fsm["frames_rep"] = 0
                fsm["visitadas"] = {1}
        else:
            fsm["frames_fase1"] = 0

    # -------- IN_REP --------
    elif fsm["estado"] == ESTADO_INREP:
        fsm["frames_rep"] += 1
        fsm["visitadas"].add(fase)

        # CONDICION A: todas las fases
        if fsm["visitadas"] >= {1, 2, 3, 4} and fase == 1 and fsm["frames_rep"] >= MIN_FRAMES_REP:
            fsm["reps"] += 1
            fsm["estado"] = ESTADO_LOCKED
            fsm["frames_fase1"] = 0

        # CONDICION B: vuelve a fase 1 tras movimiento (flexible)
        elif fase == 1 and fsm["frames_rep"] >= MIN_FRAMES_REP:
            if 3 in fsm["visitadas"] or 4 in fsm["visitadas"]:
                fsm["reps"] += 1
                fsm["estado"] = ESTADO_LOCKED
                fsm["frames_fase1"] = 0

    # -------- LOCKED --------
    elif fsm["estado"] == ESTADO_LOCKED:
        if fase == 1:
            fsm["frames_fase1"] += 1
            if fsm["frames_fase1"] >= RESET_FRAMES:
                fsm["estado"] = ESTADO_WAIT
                fsm["frames_fase1"] = 0
        else:
            fsm["frames_fase1"] = 0

    return fsm


# ---------- MAIN ----------
def main():
    modelo, scaler, rangos = cargar_recursos()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    ancho  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = 2.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(RUTA_VIDEO, fourcc, fps, (ancho, alto))

    window, hist_fases = [], []
    fase_prev = -1

    # ---- FSM REPETICIONES ----
    fsm = init_fsm()

    with mp_pose.Pose(min_detection_confidence=0.6,
                      min_tracking_confidence=0.6) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            angulos = {"codo": 0.0, "hombro": 0.0, "espalda": 0.0}
            fase_pred = -1
            feedback = ["Cuerpo no detectado"]

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                hombro = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                          lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
                codo   = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                          lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
                muneca = [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                          lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]
                cadera = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
                          lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h]

                angulos["codo"]    = calcular_angulo(hombro, codo, muneca)
                angulos["hombro"]  = calcular_angulo(codo, hombro, cadera)
                angulos["espalda"] = calcular_angulo(hombro, cadera, cadera)

                window.append(angulos.copy())
                if len(window) > 13:
                    window.pop(0)

                if len(window) >= 7:
                    X  = construir_feature_frame(angulos, window)
                    Xs = scaler.transform(X)
                    fase_pred = int(modelo.predict(Xs)[0])
                    feedback  = feedback_por_fase(angulos, fase_pred, rangos)

                hist_fases.append(fase_pred)
                if len(hist_fases) > 60:
                    hist_fases.pop(0)

                # ---- ACTUALIZAR FSM ----
                fsm = actualizar_fsm(fsm, fase_pred)

            # ---------- DIBUJOS ----------
            color_f = color_por_fase(fase_pred)
            cv2.putText(frame, f"Fase {fase_pred}", (w - 220, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_f, 3)

            dibujar_barra_y_rep(frame, fase_pred, fsm["reps"])

            cv2.putText(frame, f"Reps: {fsm['reps']}", (w - 220, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

            y = 80
            for msg in feedback[:4]:
                cv2.putText(frame, msg, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 255) if "muy" in msg else (0, 255, 0), 2)
                y += 25

            
            # ---------- ANGULOS  IZQUIERDA ----------
            #color_ang = (0, 255, 255)
            color_ang = (255, 180, 0)
  
            x_ang = 10
            y_ang = 200
            cv2.putText(frame, "ANGULOS", (x_ang, y_ang - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_ang, 2)

            cv2.putText(frame, f"CODO: {angulos['codo']:.0f}",
                      (x_ang, y_ang),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_ang, 2)
           

            cv2.putText(frame, f"HOMBRO: {angulos['hombro']:.0f}",
            (x_ang, y_ang + 45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_ang, 2)
            

            # ---- GRABAR Y MOSTRAR ----
            out.write(frame)
            cv2.imshow("Wall Push-Up", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    # ---- LIBERAR ----
    out.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Video guardado en: {RUTA_VIDEO}")


# ---------- EJECUTAR ----------
if __name__ == "__main__":
    main()