import os
import cv2
import numpy as np
import mediapipe as mp
import joblib
from collections import deque

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
BASE_DIR = r"C:\resultados_dominadas_uno"

RUTA_MODELO = os.path.join(BASE_DIR, "modelo_fase_dominadas_rt.pkl")
RUTA_SCALER = os.path.join(BASE_DIR, "scaler_fase_dominadas_rt.pkl")
RUTA_VIDEO = os.path.join(BASE_DIR, "video_resultado_final.mp4")

modelo = joblib.load(RUTA_MODELO)
scaler = joblib.load(RUTA_SCALER)

mp_pose = mp.solutions.pose

# --------------------------------------------------
#funciones
# --------------------------------------------------
def calcular_angulo(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    den = np.linalg.norm(ba) * np.linalg.norm(bc)
    if den == 0:
        return 0.0
    cos = np.dot(ba, bc) / den
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

def vel_ang(hist, fps):
    if len(hist) < 2:
        return 0.0
    return (hist[-1] - hist[-2]) * fps

def acc_ang(hist, fps):
    if len(hist) < 2:
        return 0.0
    return (hist[-1] - hist[-2]) * fps

def mean_safe(x):
    return float(np.mean(x)) if len(x) > 0 else 0.0

def min_safe(x):
    return float(np.min(x)) if len(x) > 0 else 0.0

def max_safe(x):
    return float(np.max(x)) if len(x) > 0 else 0.0

def fase_visual(f):
    return {1: "Abajo", 2: "Movimiento", 3: "Arriba"}.get(f, "-")

def color_fase(f):
    return {
        1: (0, 0, 255),
        2: (0, 165, 255),
        3: (0, 255, 0)
    }.get(f, (200, 200, 200))

# --------------------------------------------------
# FEEDBACK
# --------------------------------------------------
def evaluar_fase(fase, vel, acc, rep):
    mensajes = []

    if fase == 1 and vel < -15:
        mensajes.append(("Controla la bajada", 5))

    if fase == 2 and abs(acc) < 50:
        mensajes.append(("Movimiento muy lento", 3))

    if fase == 3 and vel > 10:
        mensajes.append(("Sube controlando", 4))

    if mensajes:
        mensajes.sort(key=lambda x: -x[1])
        msg, pen = mensajes[0]
        if msg not in rep["mensajes"]:
            rep["mensajes"].append(msg)
            rep["penalizacion"] += pen

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camara no disponible")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30

    out = cv2.VideoWriter(
        RUTA_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    ang_hist = deque(maxlen=20)
    vel_hist = deque(maxlen=10)

    reps = []
    rep = None
    rep_count = 0
    ultimo_msg = ""

    with mp_pose.Pose(min_detection_confidence=0.6,
                      min_tracking_confidence=0.6) as pose:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            ang = vel = acc = 0.0
            fase_final = -1

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

                # --------- PREDICCION ML (7 FEATURES) ----------
                X = np.array([[
                    ang,
                    vel,
                    acc,
                    mean_safe(ang_hist),
                    mean_safe(vel_hist),
                    min_safe(ang_hist),
                    max_safe(ang_hist)
                ]])

                X_scaled = scaler.transform(X)
                fase_final = int(modelo.predict(X_scaled)[0])

                # --------- LOGICA DE REP ----------
                if rep is None and fase_final == 1:
                    rep = {
                        "angulos": [],
                        "mensajes": [],
                        "penalizacion": 0
                    }

                if rep:
                    rep["angulos"].append(ang)
                    evaluar_fase(fase_final, vel, acc, rep)

                    if len(rep["angulos"]) > 25 and fase_final == 1:
                        ang_max = max(rep["angulos"])
                        ang_min = min(rep["angulos"])

                        if ang_max < 90:
                            rep["mensajes"].append("No subiste lo suficiente")
                            rep["penalizacion"] += 7

                        if ang_min > 150:
                            rep["mensajes"].append("No estiraste los brazos")
                            rep["penalizacion"] += 7

                        rep["score"] = max(
                            0,
                            100 - rep["penalizacion"] + np.random.randint(-5, 6)
                        )

                        ultimo_msg = rep["mensajes"][-1] if rep["mensajes"] else ""
                        reps.append(rep)
                        rep_count += 1
                        rep = None

            # ---------------- DRAW ----------------
            cv2.putText(frame, f"Reps: {rep_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, f"Fase: {fase_visual(fase_final)}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color_fase(fase_final), 2)

            cv2.putText(frame, f"Angulo: {ang:.1f}",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)

            cv2.putText(frame, f"Vel: {vel:.1f}  Acc: {acc:.1f}",
                        (20, 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (200, 200, 200), 2)

            if reps:
                cv2.putText(frame, f"Ult score: {reps[-1]['score']}",
                            (20, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 255), 2)

            if ultimo_msg:
                cv2.putText(frame, ultimo_msg,
                            (20, 230),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 200, 255), 2)

            out.write(frame)
            cv2.imshow("Dominadas - ML Tiempo Real", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\nREPS DETECTADAS:", len(reps))
    for i, r in enumerate(reps, 1):
        print(f"Rep {i} | Score {r['score']}")
        if r["mensajes"]:
            print(" -", r["mensajes"][-1])

    print("\nVideo guardado en:", RUTA_VIDEO)

# --------------------------------------------------
if __name__ == "__main__":
    main()
