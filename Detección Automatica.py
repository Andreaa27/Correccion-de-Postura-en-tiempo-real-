#Detección Automatica
import cv2
import mediapipe as mp
import numpy as np
import subprocess
import sys

mp_pose = mp.solutions.pose

# -------- CONFIG --------
TIEMPO_ESTABLE = 1.0
UMBRAL_MOV = 8        # mas tolerante a vibracion
FPS_EST = 25        

SCRIPTS = {
    "pushup": r"C:\Users\Acer\OneDrive - Instituto Politecnico Nacional\Desktop\Proyecto\Wall_push_up\Retroalimentación_Wall__push_up.py",
    "dom_abierta": r"C:\Users\Acer\OneDrive - Instituto Politecnico Nacional\Desktop\6 semestre\proyecto\codigo\dominada_abierta\Retroalimentación_Dominada_Agarre_Abierto.py",
    "dom_neutra": r"C:\Users\Acer\OneDrive - Instituto Politecnico Nacional\Desktop\6 semestre\proyecto\codigo\dominadas_cerradas\Retroalimentacion_tiempo_real_dominada_neutra.py"
}

# -------- UTIL --------
def dist(a, b):
    return np.linalg.norm(a - b)

# -------- DETECCION EJERCICIO --------
def detectar_ejercicio(lm, w, h):
    sho_l = np.array([lm[11].x * w, lm[11].y * h])
    sho_r = np.array([lm[12].x * w, lm[12].y * h])
    nose  = np.array([lm[0].x * w, lm[0].y * h])

    ancho_hombros = abs(sho_l[0] - sho_r[0])
    centro_h = (sho_l + sho_r) / 2

    # perfil -> push up
    if abs(nose[0] - centro_h[0]) > ancho_hombros * 0.35:
        return "pushup"

    # frente o espalda
    if nose[1] < centro_h[1]:
        return "dom_neutra"     # viendo a camara
    else:
        return "dom_abierta"   # espalda a camara

# -------- MAIN --------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camara no disponible")
        return

    estable_frames = 0
    ejercicio = None
    sho_prev = None

    with mp_pose.Pose(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as pose:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            msg = "Colocate en la posicion"

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                ejercicio = detectar_ejercicio(lm, w, h)

                # estabilidad usando hombro derecho
                sho = np.array([lm[12].x * w, lm[12].y * h])

                if sho_prev is None:
                    sho_prev = sho
                    estable_frames = 0
                else:
                    movimiento = dist(sho, sho_prev)

                    if movimiento < UMBRAL_MOV:
                        estable_frames += 1
                    else:
                        estable_frames = 0

                    sho_prev = sho

                tiempo = estable_frames / FPS_EST
                msg = f"{ejercicio} | Estable {tiempo:.1f}s"

                if tiempo >= TIEMPO_ESTABLE:
                    for i in [3, 2, 1]:
                        temp = frame.copy()
                        cv2.putText(
                            temp, str(i),
                            (w // 2 - 40, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            3, (0, 255, 0), 6
                        )
                        cv2.imshow("Preparacion", temp)
                        cv2.waitKey(1000)

                    cap.release()
                    cv2.destroyAllWindows()

                    subprocess.call([sys.executable, SCRIPTS[ejercicio]])
                    return

            cv2.putText(
                frame, msg,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 255), 3
            )

            cv2.imshow("Seleccion de ejercicio", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
