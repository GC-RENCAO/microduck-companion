#!/bin/sh
# One-time setup. Safe to run again: every step skips what is already done.
#
# Puts the two official Pollen Robotics repos next to this one (or in $DUCK_HOME):
#   microduck      the daemons (robotd & co.) — built here, run against a body in MuJoCo
#   microduck_rl   the simulator (duck-body), the scenes, and the venv duck-chat runs in
# then applies the small viewer patch (patches/), and pulls the local LLM if Ollama is installed.
set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
DUCK_HOME="${DUCK_HOME:-$(dirname "$HERE")}"
MICRODUCK="$DUCK_HOME/microduck"
RL="$DUCK_HOME/microduck_rl"
MODEL="${DUCK_LLM_MODEL:-qwen3.5:9b-mlx}"

say() { printf '\033[36m==\033[0m %s\n' "$*"; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# Homebrew's rustup is keg-only: its cargo is not on PATH until you add it.
[ -d /opt/homebrew/opt/rustup/bin ] && PATH="/opt/homebrew/opt/rustup/bin:$PATH"
command -v git >/dev/null || die "git is missing"
command -v uv >/dev/null || die "uv is missing — https://docs.astral.sh/uv/ (macOS: brew install uv)"
command -v cargo >/dev/null || die "Rust is missing — macOS: brew install rustup && rustup default stable (1.89+)"

say "official repos in $DUCK_HOME"
[ -d "$MICRODUCK/.git" ] || git clone https://github.com/pollen-robotics/microduck.git "$MICRODUCK"
[ -d "$RL/.git" ] || git clone --branch develop https://github.com/pollen-robotics/microduck_rl.git "$RL"

say "simulator venv (Python 3.12, MuJoCo, onnxruntime — a few GB the first time)"
(cd "$RL" && uv sync)

# macOS: MuJoCo's viewer launcher (mjpython) looks for libpython next to the venv, and uv's
# standalone Python keeps it in its own lib/. Only needed when you want the MuJoCo window.
if [ "$(uname -s)" = Darwin ]; then
    lib="$("$RL/.venv/bin/python" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
    for dylib in "$lib"/libpython3*.dylib; do
        [ -e "$dylib" ] && [ ! -e "$RL/.venv/$(basename "$dylib")" ] && ln -s "$dylib" "$RL/.venv/" && say "linked $(basename "$dylib") for mjpython"
    done
fi

# The viewer patch: a read-only `objects` op on the simulator, so balls and boxes move on screen.
# Optional — without it they stay where they started. Proposed upstream as a pull request.
PATCH="$HERE/patches/microduck_rl-objects-op.patch"
if git -C "$RL" apply --reverse --check "$PATCH" 2>/dev/null; then
    say "viewer patch already applied"
elif grep -q '"objects"' "$RL/src/mjlab_microduck/sim/body_server.py"; then
    say "simulator already has the objects op (upstream) — no patch needed"
elif git -C "$RL" apply --check "$PATCH" 2>/dev/null; then
    git -C "$RL" apply "$PATCH" && say "viewer patch applied"
else
    say "viewer patch does not apply to this microduck_rl — skipping (moving objects will stay put)"
fi

say "building the daemons"
(cd "$MICRODUCK" && cargo build -p robotd -p robotctl -p tof -p sounds -p configd -p updater)

if command -v ollama >/dev/null; then
    say "local LLM: $MODEL"
    ollama pull "$MODEL"
else
    say "Ollama not found — chat falls back to keyword mode. For the LLM: https://ollama.com, then: ollama pull $MODEL"
fi

say "done. Start it with: $HERE/start.sh"
