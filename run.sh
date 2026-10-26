#!/bin/sh
# Start the duck screen. Reuses microduck_rl's venv (mujoco, numpy, Pillow and the scenes are there),
# at the same default location as scripts/duck-sim; set DUCK_SIM_RL if it lives elsewhere.
set -e
RL="${DUCK_SIM_RL:-$HOME/Pollen/microduck_rl}"
PY="$RL/.venv/bin/python"
[ -x "$PY" ] || { echo "找不到 $PY —— microduck_rl 没装好？在那里运行 uv sync，或设 DUCK_SIM_RL" >&2; exit 1; }
cd "$(dirname "$0")"
exec "$PY" server.py "$@"
