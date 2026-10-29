#!/usr/bin/env bash
# Record the real demo (Gazebo + sim.py). Run on the machine with ROS Noetic + franka_ros.
#
#   scripts/record_demo.sh screen [--no-launch]   # screen capture -> portfolio_export/raw/demo_screen.mkv
#   scripts/record_demo.sh bag    [--no-launch]   # rosbag of the camera -> portfolio_export/raw/demo.bag
#
# Stop with Ctrl+C. Then convert with scripts/export_demo.py (command printed at the end).
set -euo pipefail

MODE="${1:-}"; shift || true
LAUNCH=1
[[ "${1:-}" == "--no-launch" ]] && LAUNCH=0
[[ "$MODE" == "screen" || "$MODE" == "bag" ]] || { sed -n '2,8p' "$0"; exit 1; }

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/portfolio_export/raw"
WS="${PANDA_WS:-$ROOT/..}"          # catkin workspace root (panda_ws), see README
mkdir -p "$RAW"
PIDS=()

cleanup() {
  trap - INT TERM EXIT
  for pid in "${PIDS[@]:-}"; do kill -INT "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
  echo
  echo "Saved in $RAW. Next:"
  if [[ "$MODE" == "screen" ]]; then
    echo "  python3 scripts/export_demo.py convert --input $RAW/demo_screen.mkv --start <s> --duration 8"
  else
    echo "  python3 scripts/export_demo.py annotate --bag $RAW/demo.bag --start <s> --duration 8"
  fi
}
trap cleanup INT TERM EXIT

if [[ "$LAUNCH" == 1 ]]; then
  command -v roslaunch >/dev/null || { echo "source /opt/ros/noetic/setup.bash and $WS/devel/setup.bash first"; exit 1; }
  roslaunch franka_gazebo panda.launch x:=-0.35 \
    world:="$(rospack find franka_gazebo)/world/stone.sdf" \
    controller:=cartesian_impedance_example_controller rviz:=false &
  PIDS+=($!)
  echo "Waiting for Gazebo (${GAZEBO_WAIT:-30}s)..."
  sleep "${GAZEBO_WAIT:-30}"
  (cd "$ROOT" && python3 sim.py) &
  PIDS+=($!)
  echo "sim.py started. Wait for the camera window / 'press any key to start' prompt in its terminal."
  sleep "${SIM_WAIT:-20}"
fi

if [[ "$MODE" == "screen" ]]; then
  command -v ffmpeg >/dev/null || { echo "ffmpeg missing"; exit 1; }
  SIZE="$(xdpyinfo | awk '/dimensions:/ {print $2}')"
  echo "Recording $SIZE from ${DISPLAY:-:0} ... Ctrl+C to stop. Say the voice command now."
  ffmpeg -y -f x11grab -framerate 30 -video_size "$SIZE" -i "${DISPLAY:-:0}" \
    -an -c:v libx264 -preset ultrafast -crf 18 "$RAW/demo_screen.mkv" </dev/null &
  PIDS+=($!)
else
  echo "Recording camera topics ... Ctrl+C to stop. Say the voice command now."
  rosbag record -O "$RAW/demo.bag" /camera/color/image_raw /camera/depth/image_raw &
  PIDS+=($!)
fi
wait "${PIDS[-1]}"
