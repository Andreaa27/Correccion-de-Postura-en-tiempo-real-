import os
import cv2
import numpy as np
import mediapipe as mp
import joblib
from collections import deque

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
BASE_DIR = r"C:\modelo_fases_reglas"

RUTA_MODELO = os.path.join(BASE_DIR, "modelo_fases.pkl")
RUTA_SCALER = os.path.join(BASE_DIR, "scaler_fases.pkl")

modelo = joblib.load(RUTA_MODELO)
scaler = joblib.load(RUTA_SCALER)

mp_pose = mp.solutions.pose

# --------------------------------------------------
# Funciones
# --------------------------------------------------
def angulo(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    den = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / den, -1, 1))))

def fase_txt(f):
    return {1: "Arriba", 2: "Transicion", 3: "Abajo"}.get(f, "-")

def color_fase(f):
    return {1:(0,0,255), 2:(0,165,255), 3:(0,255,0)}.get(f,(200,200,200))

def feedback_fase(fase, ang):
    if fase == 3:
        return "Extiende mas los brazos" if ang < 150 else "Buena extension"
    if fase == 2:
        return "Movimiento controlado"
    if fase == 1:
        return "Sube mas la barra" if ang > 70 else "Buena contraccion"
    return ""

# --------------------------------------------------
# FEATURES (MISMAS QUE EL ENTRENAMIENTO)
# --------------------------------------------------
def features_frame(lm, h, w, fps, hist):
    wri_l = np.array([lm[15].x*w, lm[15].y*h])
    wri_r = np.array([lm[16].x*w, lm[16].y*h])
    elb_l = np.array([lm[13].x*w, lm[13].y*h])
    elb_r = np.array([lm[14].x*w, lm[14].y*h])
    sho_l = np.array([lm[11].x*w, lm[11].y*h])
    sho_r = np.array([lm[12].x*w, lm[12].y*h])
    hip_l = np.array([lm[23].x*w, lm[23].y*h])
    hip_r = np.array([lm[24].x*w, lm[24].y*h])

    mid_hip = (hip_l + hip_r) / 2
    mid_sho = (sho_l + sho_r) / 2

    ang_l = angulo(sho_l, elb_l, wri_l)
    ang_r = angulo(sho_r, elb_r, wri_r)
    ang_h = angulo(elb_l, sho_l, elb_r)
    trunk = angulo(mid_hip, mid_sho, [mid_sho[0], 0])
    grip  = np.linalg.norm(wri_l - wri_r)

    for k,v in zip(
        ["ang_l","ang_r","ang_h","grip","trunk","hip_y"],
        [ang_l,ang_r,ang_h,grip,trunk,mid_hip[1]]
    ):
        hist[k].append(v)

    feats = []
    for k in ["ang_l","ang_r","ang_h","grip","trunk"]:
        arr = np.array(hist[k])
        feats += [arr[-1], arr.mean(), arr.min(), arr.max(), np.std(arr)]

    for k in ["ang_l","ang_r","ang_h","hip_y"]:
        vel = (hist[k][-1] - hist[k][-2]) * fps if len(hist[k]) > 1 else 0
        hist[k+"_vel"].append(vel)

        acc = (hist[k+"_vel"][-1] - hist[k+"_vel"][-2]) * fps if len(hist[k+"_vel"]) > 1 else 0
        hist[k+"_acc"].append(acc)

        feats += [vel, np.mean(hist[k+"_vel"]), acc, np.mean(hist[k+"_acc"])]

    return np.array(feats)

# --------------------------------------------------
# MAIN TIEMPO REAL  y  MAQUINA DE ESTADOS
# --------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la camara")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    keys = ["ang_l","ang_r","ang_h","grip","trunk","hip_y"] + \
           [k+s for k in ["ang_l","ang_r","ang_h","hip_y"] for s in ["_vel","_acc"]]

    hist = {k: deque(maxlen=25) for k in keys}

    # ---------- MAQUINA DE ESTADOS ----------
    ABAJO, SUBE, ARRIBA, BAJA = 0, 1, 2, 3
    estado = ABAJO

    repeticiones = 0
    cont_arriba = 0
    cont_abajo = 0
    FRAMES_ESTABLES = 5

    # ---------- POSICION TEXTO ----------
    X_IZQ = 30
    Y_BASE = int(h * 0.33)
    LINEA = 35

    with mp_pose.Pose(min_detection_confidence=0.6,
                      min_tracking_confidence=0.6) as pose:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            fase = -1
            ang = 0.0
            vel_ang = 0.0
            feedback = ""

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark

                ang = angulo(
                    [lm[11].x*w, lm[11].y*h],
                    [lm[13].x*w, lm[13].y*h],
                    [lm[15].x*w, lm[15].y*h]
                )

                feat = features_frame(lm, h, w, fps, hist)
                Xs = scaler.transform([feat])
                fase = int(modelo.predict(Xs)[0])

                feedback = feedback_fase(fase, ang)

                if len(hist["ang_l_vel"]) > 0:
                    vel_ang = hist["ang_l_vel"][-1]

                # ---------- TRANSICIONES ----------
                if estado == ABAJO and fase == 2:
                    estado = SUBE

                elif estado == SUBE and fase == 1:
                    cont_arriba += 1
                    if cont_arriba >= FRAMES_ESTABLES:
                        estado = ARRIBA
                        cont_arriba = 0

                elif estado == ARRIBA and fase == 2:
                    estado = BAJA

                elif estado == BAJA and fase == 3:
                    cont_abajo += 1
                    if cont_abajo >= FRAMES_ESTABLES:
                        repeticiones += 1
                        estado = ABAJO
                        cont_abajo = 0

            # ---------- DRAW ----------
            cv2.putText(frame, f"REPETICIONES: {repeticiones}",
                        (30, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 4)

            cv2.putText(frame, "Fase:",
                        (X_IZQ, Y_BASE),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

            cv2.putText(frame, fase_txt(fase),
                        (X_IZQ, Y_BASE + LINEA),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_fase(fase), 2)

            cv2.putText(frame, f"Angulo: {ang:.1f}",
                        (X_IZQ, Y_BASE + 2*LINEA),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            cv2.putText(frame, f"Velocidad: {vel_ang:.1f}",
                        (X_IZQ, Y_BASE + 3*LINEA),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            if feedback:
                cv2.putText(frame, feedback,
                            (X_IZQ, Y_BASE + 5*LINEA),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            cv2.imshow("Dominadas tiempo real", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


# --------------------------------------------------
if __name__ == "__main__":
    main()
