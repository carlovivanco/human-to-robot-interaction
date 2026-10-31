# Export de portafolio

| Archivo | Estado | Fuente |
|---|---|---|
| `hri.mp4`, `hri.jpg` | **NO generados**: el repo no contiene grabaciones ni modelos | tú los generas (pasos abajo) |
| `metrics.md` | Generado | `README.md:100-107`, `panda_class.py`, `main.py` |
| `card.md` | Generado | imports, `requirements.txt`, `git log`, API de GitHub (repo público) |
| `../../scripts/export_demo.py`, `../../scripts/record_demo.sh` | Commiteados | — |

## Cómo producir el video (en tu máquina con ROS Noetic + franka_ros)
1. Grabar: `scripts/record_demo.sh screen` (pantalla de Gazebo + cámara de `sim.py`), o `scripts/record_demo.sh bag` (rosbag de la cámara). Di "panda, pass me the hammer" y muestra palma abierta. Ctrl+C para terminar. Salida en `portfolio_export/raw/`. Variables: `PANDA_WS`, `GAZEBO_WAIT`, `SIM_WAIT`; `--no-launch` si ya tienes Gazebo/sim corriendo.
2. Exportar:
   - Pantalla: `python3 scripts/export_demo.py convert --input portfolio_export/raw/demo_screen.mkv --start 3 --duration 8`
   - Bag/cámara con anotaciones del pipeline real: `python3 scripts/export_demo.py annotate --bag portfolio_export/raw/demo.bag --start 3 --duration 8` (requiere los modelos `vision_sim/models/tools-100-epochs.pt` y `gesture_recognizer.task`).
3. El script verifica 1280x720, 30 fps, H.264 yuv420p, +faststart, sin audio, 5-10 s, <8 MB y el JPG (calidad 85). Elige el cuadro del JPG con `--frame-at`.

Probado aquí solo con un video sintético de prueba (descartado); `annotate` no se pudo ejecutar sin modelos ni ROS.

## Te falta aportar
- La grabación (y, para `annotate`, los dos modelos).
- El reporte CE903 / test log si quieres usar cualquier cifra de `metrics.md`.
- Tu contribución concreta (`card.md`).
