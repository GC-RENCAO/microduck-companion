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
          ─ pose.py    mirrors the body to the browser (sim ground truth, or the robot's own robot.state)
          ─ transport.py  unix socket, or the official rendezvous over HTTP
   │ JSON-RPC intents                         │ TCP, read-only (sim only)
robotd (official, unchanged)             duck-body (official MuJoCo simulator)
```

It is a working take on what microduck's roadmap calls **M9 — the autonomous brain**, and on the architecture proposed in
[microduck#454](https://github.com/pollen-robotics/microduck/issues/454).

## A real duck, from anywhere

A duck you own is reached the way the official remote access works
([remote-access-design §3.8](https://github.com/pollen-robotics/microduck/blob/main/docs/design/remote-access-design.md)):
the robot signs in to a Hugging Face account and registers with the rendezvous Space; duck-chat signs in to the same
account and sends its JSON-RPC inside `peer` envelopes that the Space relays to the robot's `mediad`. Plain HTTP and
server-sent events — no WebRTC, no port forwarding, the same lane the official Spaces use.

```bash
robotctl account login          # on the duck, once (in the simulator: scripts/duck-sim ctl account login)
hf auth login                   # on your Mac, the same Hugging Face account (or set HF_TOKEN)
DUCK_TRANSPORT=rendezvous ./run.sh             # DUCK_ROBOT=<name> if the account has several ducks
```

With a real robot duck-chat switches to **real-robot mode** (`DUCK_WORLD=robot`):

- the 3D duck is drawn from what the robot reports — measured joints, IMU orientation, contact odometry — on a plain
  floor, because there is no house and no ground truth;
- the duck never walks on its own: it may be standing on a table and nothing here can see the edge. Chat, gestures,
  sounds, tricks, head tracking and body language all work; it walks only when you tell it to (`DUCK_ROAM=1` lifts this);
- “come here” and “look at me” are off, because the 3D camera is not a place in the robot's world.

The rendezvous allows 1200 requests a minute per token, and a 429 cuts off everything that token does. Over it,
continuous intents go at ~3 Hz instead of 10 and robot.state is read at 3 Hz, with a hard client-side stop at 1000.
`/api/status` reports the link: which duck, the last call's round trip, and requests in the last minute.

**Verified through the real Hugging Face Space**, with a simulated duck signed in exactly as a robot is
(`scripts/duck-sim ctl account login`, `mediad` registered with the rendezvous):

| | Measured |
|---|---|
| A call's round trip (Mac → Space → duck → back) | 350–510 ms |
| Requests a minute: idle / walking and talking | 54 / 239 |
| “Walk 3 seconds” | 0.32 m (0.36 m over the local socket) |
| “Turn left/right 90°” ×2 each | +90° +71° / −104° −95° |
| Roll, greeting, chat | all worked; the duck is listed as `simulated` in the account |

**Not yet verified on a physical duck.** To try the path without an account, `tests/fake_rendezvous.py` stands in for
the Space and the robot's relay and forwards to the simulator:

```bash
python tests/fake_rendezvous.py &              # relays to the simulator's socket
DUCK_TRANSPORT=rendezvous REACHY_CENTRAL_URL=http://127.0.0.1:8790 HF_TOKEN=local ./start.sh
```

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

- **Tested in simulation; the remote path is built but not yet run against the real Space or a physical duck**
  (see [A real duck, from anywhere](#a-real-duck-from-anywhere)).
- In simulation the duck knows exactly where it is; a real robot only has odometry from where it booted, so the house
  and navigation features stay in the simulator until there is localisation (the visual SLAM the apartment is designed for).
- The LLM prompts, UI labels and the detailed notes ([README.zh-CN.md](README.zh-CN.md)) are in Chinese; the duck
  answers in whatever language you use. Code comments are in English.

## Development

```bash
# the tests need the official scenes: point DUCK_SIM_RL at a microduck_rl checkout (setup.sh makes one)
DUCK_SIM_RL=../microduck_rl ../microduck_rl/.venv/bin/python -m pytest tests -q
```

They run offline in a few seconds — no simulator, no daemons, no Ollama (the LLM is a fake local server) — and cover
gesture recognition near misses (petting vs. “stop”, waving vs. beckoning), walk/turn conversion, the keyword parser,
the brain's tool handling, the apartment map (doorways, rooms, A* through doors), the scene export, both transports
(against a fake `robotd` and a fake rendezvous that copies the real one's rules: auth, stream-first, rate limits,
reconnects) and real-robot mode.
CI runs the same suite on every push (`.github/workflows/tests.yml`).

### Calibrating gestures to your camera

The gesture thresholds (top of `tracker.py`) were tuned on one webcam. To check them on yours, turn the camera on,
then for ~20 s each: hold a raised palm towards the camera (stop), pet the air slowly with a flat hand, and beckon
with your fingers. Then:

```bash
curl -s localhost:8765/api/calibration | python3 -m json.tool   # p5/p50/p95 of each finger feature per gesture
curl -s -X POST localhost:8765/api/calibration                    # start a fresh recording
```

Record one gesture at a time (POST to reset in between). Frames are grouped by MediaPipe's own label even below
its confidence cut (`min_score`); `confident` is the share that actually counts as that gesture. A stop palm should sit above `min_upright`/`min_size`; a beckon's `curl_swing` and `flips` above the `BECKON_*`
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
