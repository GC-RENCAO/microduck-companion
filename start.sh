#!/bin/sh
# Start the simulated duck (no MuJoCo window — the browser draws it) and the duck screen.
# Ctrl+C stops both.
#
#   ./start.sh                        the apartment (default)
#   DUCK_SIM_SCENE= ./start.sh        an empty floor
#   DUCK_SIM_SCENE=ball ./start.sh    any scene_<name>.xml from microduck_rl, or a path
set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
DUCK_HOME="${DUCK_HOME:-$(dirname "$HERE")}"
export DUCK_SIM_RL="${DUCK_SIM_RL:-$DUCK_HOME/microduck_rl}"
export DUCK_SIM_SCENE="${DUCK_SIM_SCENE-apartment}"
SIM="$DUCK_HOME/microduck/scripts/duck-sim"

[ -x "$SIM" ] || { echo "no $SIM — run ./setup.sh first" >&2; exit 1; }
[ -d /opt/homebrew/opt/rustup/bin ] && PATH="/opt/homebrew/opt/rustup/bin:$PATH"

"$SIM" down >/dev/null 2>&1 || true
DUCK_SIM_VIEWER=0 "$SIM"
trap '"$SIM" down >/dev/null 2>&1' EXIT INT TERM

echo "open http://127.0.0.1:${DUCK_CHAT_PORT:-8765}"
"$HERE/run.sh"
