# Verificación de métricas

Resultado: **ninguna de las dos afirmaciones se puede verificar con lo que hay en el repo.** No hay dataset, script de evaluación, logs ni tests (`git ls-files` solo contiene código del robot/visión/voz, README, requirements y 2 diagramas). La única fuente es texto en `README.md:100-107`, que remite a "el reporte del proyecto" (CE903), que no está en el repo.

## 1. ">95% de precisión en gestos"
| | |
|---|---|
| Cómo se mide | No hay código que lo mida. El README dice "Hand gesture classification on an unseen dataset: >95% accuracy" (`README.md:105`). |
| Conjunto de datos | "unseen dataset" sin nombre, tamaño ni ubicación. No está en el repo. |
| Número real | No existe número exacto en el repo; solo el texto ">95%". |
| Respaldo | Solo `README.md:105`. El modelo es MediaPipe `GestureRecognizer` (`vision_sim/vision_server/main.py:684-697`) cargado desde `gesture_recognizer.task` (archivo ausente). `sim.py:165` solo usa la categoría `Open_Palm`, que pertenece al modelo estándar de MediaPipe, por lo que probablemente no lo entrenó el equipo (confirmar). |
| Veredicto | **No verificable.** Úsala solo si el reporte CE903 trae el número, el dataset y el método; si el modelo es el preentrenado de Google, evita sugerir que lo entrenaste. |

## 2. "<5 mm de error de posicionamiento"
| | |
|---|---|
| Cómo se mide | No hay código que lo mida. El README dice "Camera-to-robot coordinate conversion: positioning error within 5mm" (`README.md:107`). |
| Qué mide realmente | El error de la **conversión de coordenadas cámara→mundo** (`panda_class.py:464-483`, `cam_to_world`), no el error de posicionamiento del robot. Dice "within 5mm" (≤5 mm), no "<5 mm". |
| Conjunto de pruebas | No especificado. |
| Número real | No hay número exacto en el repo. |
| Lo que sí muestra el código | `pose_tolerance = 0.02` (2 cm; `panda_class.py:261`): el robot da la pose por alcanzada dentro de ±2 cm (`wait_for_pose`, `panda_class.py:448`). La posición leída se redondea a 0.01 m (`panda_class.py:436`). Intrínsecos/extrínsecos de la cámara son constantes fijas de simulación (`panda_class.py:234-255`). Con eso el sistema no puede garantizar ni comprobar 5 mm. |
| Veredicto | **No verificable y con riesgo de sobreafirmar.** Redacción defendible: "conversión cámara→robot con error ≤5 mm (según el reporte)", o con lo que respalda el código: "tolerancia de posicionamiento de 2 cm en simulación". |

## Otras cifras del README (mismo estatus: sin respaldo en el repo)
- Voz→herramienta: ~95% (`README.md:104`)
- Reconocimiento de objetos en dataset no visto: >90% (`README.md:106`)

## Qué haría falta para respaldarlas
1. El reporte/test log de CE903 con tablas de resultados.
2. El dataset de prueba de gestos y un script de evaluación (matriz de confusión).
3. Un experimento de posición: comandar N puntos conocidos, medir la pose real y reportar error medio/máximo.
