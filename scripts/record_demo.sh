#!/usr/bin/env bash
# One command to run (and optionally record) the demo: Gazebo + sim.py (SIM=isaac for Isaac Sim).
#
#   scripts/record_demo.sh run      # run only
#   scripts/record_demo.sh screen   # + screen capture -> portfolio_export/raw/demo_screen.mkv
#   scripts/record_demo.sh bag      # + rosbag2 of the camera -> portfolio_export/raw/demo_bag/
#
# Press Enter when sim.py asks, then say "panda, pass me the hammer". Ctrl+C (or sim.py ending) stops everything.
# Then convert with scripts/export_demo.py (the command is printed at the end).
set -euo pipefail

MODE="${1:-}"
[[ "$MODE" == "run" || "$MODE" == "screen" || "$MODE" == "bag" ]] || { sed -n '2,9p' "$0"; exit 1; }

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/portfolio_export/raw"
WS="${PANDA_WS:-}"                   # colcon workspace (panda_ws), see README
if [[ -z "$WS" ]]; then
  for d in "$ROOT/../.." "$ROOT/.." "$HOME/Desktop/Robotics/TeamProject/panda_ws"; do
    [[ -f "$d/install/setup.bash" ]] && { WS="$(cd "$d" && pwd)"; break; }
  done
fi
[[ -n "$WS" ]] || { echo "workspace not found: export PANDA_WS=<path to panda_ws>"; exit 1; }

set +u
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"
[[ -f "$WS/.venv/bin/activate" ]] && source "$WS/.venv/bin/activate"
set -u
export YOLO_AUTOINSTALL=False        # otherwise ultralytics pip-installs numpy 2 and breaks cv_bridge
mkdir -p "$RAW"
PIDS=()

cleanup() {
  trap - INT TERM EXIT
  for pid in "${PIDS[@]:-}"; do kill -INT "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
  echo
  if [[ "$MODE" == "screen" ]]; then
    echo "Saved $RAW/demo_screen.mkv. Next:"
    echo "  python3 scripts/export_demo.py convert --input $RAW/demo_screen.mkv --start <s> --duration 45"
  elif [[ "$MODE" == "bag" ]]; then
    echo "Saved $RAW/demo_bag. Next:"
    echo "  python3 scripts/export_demo.py annotate --bag $RAW/demo_bag --start <s> --duration 8"
  fi
}
trap cleanup INT TERM EXIT

# SIM=isaac runs the same demo on Isaac Sim (sim_isaac/); default is Gazebo
if [[ "${SIM:-gazebo}" == "isaac" ]]; then
  export WHISPER_DEVICE=cpu  # Isaac needs the GPU memory
  ros2 launch "$ROOT/sim_isaac/demo.launch.py" &
  READY() { timeout 5 ros2 topic echo --once /joint_states >/dev/null 2>&1; }  # Isaac takes about a minute to start
  TRIES=120
else
  ros2 launch "$ROOT/sim_ros2/demo.launch.py" &
  READY() { ros2 control list_controllers 2>/dev/null | grep -q "arm_position_controller.*active"; }
  TRIES=60
fi
PIDS+=($!)
echo "Waiting for the simulator..."
for _ in $(seq $TRIES); do READY && break; sleep 2; done
READY || { echo "simulator did not start, see the output above"; exit 1; }
sleep 5  # the arm goes to its ready pose

if [[ "$MODE" == "screen" ]]; then
  command -v ffmpeg >/dev/null || { echo "ffmpeg missing"; exit 1; }
  SIZE="$(xdpyinfo | awk '/dimensions:/ {print $2}')"
  echo "Recording $SIZE from ${DISPLAY:-:0}"
  # audio: microphone (your voice); pick another with AUDIO_SRC=<name from `ffmpeg -sources pulse`>, or AUDIO_SRC=none
  AUDIO=(-f pulse -i "${AUDIO_SRC:-default}" -c:a aac -b:a 160k)
  [[ "${AUDIO_SRC:-}" == "none" ]] && AUDIO=(-an)
  ffmpeg -y -f x11grab -framerate 30 -video_size "$SIZE" -i "${DISPLAY:-:0}" "${AUDIO[@]}" \
    -c:v libx264 -preset ultrafast -crf 18 "$RAW/demo_screen.mkv" </dev/null &
  PIDS+=($!)
elif [[ "$MODE" == "bag" ]]; then
  ros2 bag record -o "$RAW/demo_bag" /camera/color/image_raw /camera/depth/image_raw &
  PIDS+=($!)
fi

cd "$ROOT"
python3 sim.py
