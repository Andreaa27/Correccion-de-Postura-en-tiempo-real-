# Correccion de Postura en tiempo real 
Proyecto de análisis de ejercicios de fuerza (dominadas, push-ups y remo) mediante visión por computadora y aprendizaje automático. El sistema permite entrenar modelos de fases de movimiento, evaluar repeticiones con retroalimentación en tiempo real y comparar resultados con datos de referencia (ground truth). Este proyecto utiliza MediaPipe para la detección de poses y Random Forest Classifier para clasificar las fases de los movimientos durante los ejercicios de fuerza del tronco superior. Permite:

-Entrenamiento de modelos personalizados por tipo de ejercicio.

-Retroalimentación en tiempo real para corregir la postura durante la ejecución.

-Evaluación con Ground Truth para comparar la precisión del modelo y el seguimiento del movimiento.

-Detección automática de ejercicios físicos en tiempo real utilizando  MediaPipe Pose, el cual captura video desde la cámara web, identifica la postura del cuerpo humano mediante puntos clave como hombros y nariz, y a partir de su posición relativa determina qué ejercicio está realizando la persona, ya sea push up, dominada con agarre neutro o dominada con agarre abierto.
