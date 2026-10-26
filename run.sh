#!/bin/sh
# 启动鸭子屏幕。复用 microduck_rl 的 venv（mujoco、numpy、Pillow、场景文件都在那里），
# 和 scripts/duck-sim 用同一个默认位置；放在别处时设 DUCK_SIM_RL。
set -e
RL="${DUCK_SIM_RL:-$HOME/Pollen/microduck_rl}"
PY="$RL/.venv/bin/python"
[ -x "$PY" ] || { echo "找不到 $PY —— microduck_rl 没装好？在那里运行 uv sync，或设 DUCK_SIM_RL" >&2; exit 1; }
cd "$(dirname "$0")"
exec "$PY" server.py "$@"
