# Microduck Companion (duck-chat)

**A pet-like, LLM-driven companion for the [Pollen Robotics Microduck](https://github.com/pollen-robotics/microduck)** —
chat with the duck, wave at it through your webcam, and watch it find its own way to you through a 3D apartment.
Everything runs locally: the official daemons, the official simulator, and a local LLM.

English · [中文](README.zh-CN.md)

![The duck plans its own way from the kitchen to you in the living room](docs/demo-come-here.gif)

*You say “come here” from the living room; the duck, in the kitchen, plans a path through the doorway and stops in
front of you. Recorded in the official simulator, 4× speed.*

> **Unofficial community project.** Not affiliated with Pollen Robotics. It runs against the official
> simulator today and has **not been tested on a real robot yet** (see [Status](#status)).

## What it does

- **Duck screen** — the duck rendered in your browser with WebGL at native resolution and 60+ fps, mirrored live from
  the MuJoCo simulation, with a chat panel beside it.
- **Talk to it** — natural language (Chinese or English) through a local LLM in [Ollama](https://ollama.com)
  (default `qwen3.5:9b-mlx`). One call picks the actions (tool calling), a second one answers in character, streamed.
  The duck starts moving in ~2 s. Saying “stop” bypasses the LLM and acts instantly.
- **See you** — your webcam, processed in the browser with MediaPipe (nothing leaves the machine): its gaze follows your
  face and hand; gestures are filtered on finger detail (curl, orientation, drift) so a raised hand is not a command:
  ✋ held still = stop, 🫴 beckon = come here, 👍 = happy, ✌️ = roll, 🤟 = greet.
- **Act like a pet** — a 10 Hz behaviour layer with mood and energy: it turns to face you, looks for you when you leave
  and greets you when you come back, leans in when petted, flinches when startled, wanders when ignored, and walks
  back to its dock to nap. Falls are recovered automatically.
- **Live in a house** — the official apartment scene (6 rooms). A map is generated from the same scene file the physics
  uses; A* routes it through doorways and around furniture (and the stairwell). “Come here”, “go to the kitchen” and
  “go back to your dock” are planned paths, not straight lines into walls.

## Quick start

macOS (Apple silicon) or Linux, with `git`, [uv](https://docs.astral.sh/uv/), Rust 1.89+, and optionally Ollama.

```bash
git clone https://github.com/GC-RENCAO/microduck-companion.git duck-chat
cd duck-chat
./setup.sh   # once: clones the official repos next to this one, builds them, applies the viewer patch, pulls the LLM
./start.sh   # headless simulator in the apartment + the duck screen at http://127.0.0.1:8765 (Ctrl+C stops both)
```

`DUCK_HOME` changes where the official repos go; `DUCK_SIM_SCENE=` (empty) starts on an open floor instead.
Without Ollama the chat falls back to a keyword mode.

## How it fits the official architecture

The design follows the split described in microduck's
[architecture §5.3](https://github.com/pollen-robotics/microduck/blob/main/docs/design/architecture.md#53-server-side-agents-dont-force-them-through-webrtc):
the LLM is a **high-level controller off the robot**, reactive behaviour stays local, and `robotd` keeps motor authority.
Every command reaches the duck as the same JSON-RPC intents a gamepad sends (`robot.move`, `robot.do`, `robot.look`…).

```
browser ─ WebGL duck + chat + MediaPipe (camera never leaves the browser)
   │ HTTP
server.py ─ brain.py   local LLM (Ollama): plan actions → talk, streamed
          ─ life.py    mood, attention, autonomy, expressions (10 Hz, no LLM)
          ─ tracker.py hand/face perception, gestures
          ─ actions.py action queue, path following, fall recovery
          ─ navmap.py  map, rooms, places, A* — generated from the scene
          ─ pose.py    mirrors the simulated body to the browser
   │ unix socket (JSON-RPC intents)          │ TCP, read-only
robotd (official, unchanged)             duck-body (official MuJoCo simulator)
```

It is a working take on what microduck's roadmap calls **M9 — the autonomous brain**, and on the architecture proposed in
[microduck#454](https://github.com/pollen-robotics/microduck/issues/454).

## Measured in simulation

| | Result |
|---|---|
| Tool-calling accuracy (`bench.py`, 34 utterances incl. combos, impossible requests, English, places) | 34/34, isolated and in conversation |
| Time until the duck starts moving / first word | ~1.5–2.5 s / +0.5 s (M3 Pro, `qwen3.5:9b-mlx`) |
| Webcam → head reaction | hand detection 4.7 ms (GPU); head halfway in ~0.25 s, most of it the duck's own head response |
| “Come here” from the living room to you in the bedroom | living room → corridor → bedroom, 63 s, stopped 0.38 m from you, no collision, no fall |
| “Come here” with you standing outside the house | walked to the nearest reachable spot by the wall, 72 s |
| “Look at me” with you right behind it | turned in place in 3 s, drifted 0.13 m |

## Status

- **Simulation only so far.** It talks to `robotd` over the local unix socket, which is what `scripts/duck-sim` exposes.
  A real duck on your LAN needs a transport over the official remote path (WebRTC / rendezvous) — the next milestone.
- In simulation the duck knows exactly where it is; a real robot will need localisation (odometry + the visual SLAM the
  apartment scene is designed for).
- The LLM prompts, UI labels and the detailed notes ([README.zh-CN.md](README.zh-CN.md)) are in Chinese; the duck
  answers in whatever language you use. Code comments are being moved to English.

## Development

```bash
# the tests need the official scenes: point DUCK_SIM_RL at a microduck_rl checkout (setup.sh makes one)
DUCK_SIM_RL=../microduck_rl ../microduck_rl/.venv/bin/python -m pytest tests -q
```

They run offline in a few seconds — no simulator, no daemons, no Ollama (the LLM is a fake local server) — and cover
gesture recognition near misses (petting vs. “stop”, waving vs. beckoning), walk/turn conversion, the keyword parser,
the brain's tool handling, the apartment map (doorways, rooms, A* through doors) and the scene export.
CI runs the same suite on every push (`.github/workflows/tests.yml`).

### Calibrating gestures to your camera

The gesture thresholds (top of `tracker.py`) were tuned on one webcam. To check them on yours, turn the camera on,
then for ~20 s each: hold a raised palm close to the camera (stop), pet the air slowly with a flat hand, and beckon
with your fingers. Then:

```bash
curl -s localhost:8765/api/calibration | python3 -m json.tool   # p5/p50/p95 of each finger feature per gesture
curl -s -X POST localhost:8765/api/calibration                    # start a fresh recording
```

A stop palm should sit above `min_upright`/`min_size`; a beckon's `curl_swing` and `flips` above the `BECKON_*`
thresholds while its `drift` stays below. Please share the numbers in an issue if your camera needs different values.

## The upstream patch

`patches/microduck_rl-objects-op.patch` adds a read-only `objects` op to the simulator so a viewer can mirror balls and
boxes. `setup.sh` applies it if needed; without it those objects just stay where they started.
It is proposed upstream in [pollen-robotics/microduck_rl#83](https://github.com/pollen-robotics/microduck_rl/pull/83).

## License and credits

Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). Built on Pollen Robotics'
[microduck](https://github.com/pollen-robotics/microduck) and [microduck_rl](https://github.com/pollen-robotics/microduck_rl).
The duck's 3D meshes are © Pollen Robotics under **CC BY-NC-SA 4.0**; this repository does not include them — they are
read at runtime from your own `microduck_rl` checkout.

## Community

Questions and ideas are welcome in the issues here, and in the
[Pollen Robotics community Discord](https://discord.gg/2bAhWfXme9).
