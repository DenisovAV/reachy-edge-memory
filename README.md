# Reachy Edge Memory

A Reachy Mini robot that remembers what it sees and hears — who it has met,
what was said, what was in front of its camera — with the memory on the robot
itself, in [Qdrant Edge](https://qdrant.tech/documentation/edge/): an embedded
vector database that runs inside the robot's own process, with no server and
no network.

The models that do not fit on the robot's Raspberry Pi CM4 run on a laptop and
are called over HTTP; everything the robot remembers is stored and searched on
the robot. The talk this demo was built for, with the robot live on stage:
[Vector Space Stream, "Robots and Qdrant Edge"](https://www.youtube.com/watch?v=PxGlBlqTxJI&t=3709s).

## How it works

```
 Reachy Mini (Raspberry Pi CM4)                    Laptop (macOS, Apple silicon)
 ┌──────────────────────────────────────┐          ┌──────────────────────────────────┐
 │ voice loop       demo/run_demo.py    │  HTTP    │ demo/serve.py                    │
 │  listen → look → remember → answer   │ ───────▶ │  Whisper · Gemma 4 E2B · Inflect │
 │                                      │          │ demo/gpu_detect.py               │
 │ Qdrant Edge, three shards            │          │  YOLOX-Tiny (GPU) · YuNet        │
 │  memory/    what was said and seen   │          │ demo/embed_service.py            │
 │  people/    who it has met           │          │  SigLIP 2 · bge-small · HSFace   │
 │  knowledge/ what it was taught       │          │                                  │
 │                                      │  events  │ dashboard   demo/display/web.py  │
 │ camera + mic   demo/camera_service.py│ ───────▶ │  the audience's screen           │
 └──────────────────────────────────────┘          └──────────────────────────────────┘
```

- **The conversation** lives in the language model's context while it fits. As
  the context fills, the oldest exchanges move into the `memory` shard, embedded
  with bge-small. The model reaches them only by calling its `remember` tool.
- **What it saw** is stored as frames, with SigLIP 2 vectors, when the objects in
  view change or when it is asked to look. The words about a frame — the
  objects, the people in it, what the robot said about it — get a second, text
  vector on the same point.
- **Who it has met** is one point per person in the `people` shard: several shots
  of their face as a multivector, compared with MaxSim.
- **What it was taught** — facts about Qdrant and about the robot itself — ships
  as a snapshot (`demo/qdrant_knowledge.snapshot`) restored on every start. Each
  fact is stored with several phrasings of the questions it answers, as a
  multivector.

Any model can be moved onto the robot instead (`--on-robot`, below) except the
language model: Gemma 4 E2B is 2.5 GB, plus its context cache, against about
3 GB free on the robot.

## Requirements

- macOS on Apple silicon, Python 3.12, [uv](https://docs.astral.sh/uv/) and
  `ffmpeg` (`brew install ffmpeg`) — the laptop's camera, microphone and speaker
  go through ffmpeg and afplay.
- About 6 GB of disk for the models, downloaded from Hugging Face on first use.
- For the robot: a Reachy Mini (the Wireless one, with the CM4 inside), SSH
  access to it as `pollen` with a key, and the robot and the laptop on the same
  network. No robot? The simulator below stands in for it.

```bash
git clone https://github.com/qdrant-labs/reachy-edge-memory.git && cd reachy-edge-memory
uv sync
uv run python -m emulator.models      # download every model up front
```

Download them up front: otherwise the first start fetches about 3 GB while
`stage` waits for the models to come up, and on a slower connection it gives
up first. The face embedder, HSFace, has no ready-made LiteRT build: this step
builds it from its PyTorch weights (about two minutes, once, with PyTorch in a
temporary environment, not the project's). Skip the step and the first
`stage` does it. If it cannot be built, `stage` says so and goes on with faces
off: the robot still talks, remembers and recalls, it just calls everyone
"Person".

## Run it without a robot

The Reachy Mini simulator (MuJoCo) runs the robot's own daemon, so the demo
talks to it exactly as it talks to the real one. Install it in an environment
of its own, per Pollen's guide:
[Reachy Mini simulation](https://huggingface.co/docs/reachy_mini/platforms/simulation/get_started).
Then, in two terminals:

```bash
# 1. the robot, simulated — no camera or microphone of its own
mjpython -m reachy_mini.daemon.app.main --sim --no-media

# 2. everything else: the models, and the voice loop with the laptop's
#    camera, microphone and speaker
uv run python -m demo.stage --sim
```

Open <http://127.0.0.1:8091> and talk to it. The first start takes a minute
while the models load; the voice loop starts once they are up, and `stage`
says when it listens. It uses the system's default camera and microphone; for
another one, pass `--video` / `--audio` with the index
`ffmpeg -f avfoundation -list_devices true -i ""` prints. Ctrl-C stops it all.

What it remembers lives in `results/memory` and stays there between runs. The
dashboard's Restart button wipes it and starts over; the previous memory is
kept once, as `results/memory-previous`.

## Run it with a Raspberry Pi 5 — coming soon

The models on a Raspberry Pi 5 instead of the laptop, the robot simulated on
the Mac:

```bash
PI=raspberrypi.local ./demo/launch.sh
```

This mode is being moved onto the current services and tested on the board;
until it lands, the command says so and stops.

## Run it with the robot

Once, on the robot: the voice loop runs in the robot daemon's own Python, which
needs Qdrant Edge and Pillow.

```bash
ssh pollen@<robot-ip> /venvs/mini_daemon/bin/pip install qdrant-edge-py pillow
```

Then, on the laptop:

```bash
uv run python -m demo.stage --robot --robot-host <robot-ip>
```

This starts the laptop's services, copies `demo/` and `emulator/` to the robot,
starts its camera and microphone service, wakes it, and starts the voice loop
there (`scripts/robot_service.sh voice-start`). The dashboard is at
`http://<laptop-ip>:8091`; its Stream button puts the robot to sleep and wakes
it again. Ctrl-C stops everything and puts the robot to sleep.

The robot's memory lives on its own disk, in `~/reachy-demo/memory`, and stays
there between runs. The dashboard's Restart button wipes it and starts over;
the previous memory is kept once, as `memory-previous`.

### Moving models onto the robot

```bash
ON_ROBOT=detector,faces,asr,tts,embedder uv run python -m demo.stage --robot --robot-host <robot-ip>
```

Any subset of `asr`, `tts`, `detector`, `faces`, `embedder` runs on the robot
instead of the laptop: `scripts/robot_service.sh` installs the runtime there
and copies the models from the laptop's Hugging Face cache. `faces` builds the
face embedder on the laptop first if it is not there yet, and the start stops
if it cannot; `embedder` has the robot download SigLIP 2 and bge itself on
first use; `tts` installs espeak-ng with `sudo`.
The start log says where each model runs. The detector on the robot takes
about 0.6 s a frame on two of its four cores; recognition on the robot is
moonshine-tiny rather than Whisper, so it is faster there and less accurate.
To rehearse a placement without starting anything:

```bash
ROBOT=<robot-ip> ON_ROBOT=detector,asr scripts/robot_service.sh prepare
```

## What to ask it

Each kind of question is answered from a different place:

| Say | Answered from |
|---|---|
| "What do you see?", "Look to your left. What's there?" | the camera, right now — the head turns first |
| "What was on your left?", "What did I show you?" | the frames in `memory` |
| "What did we talk about?" | the conversation, and what moved out of it into `memory` |
| "Do you remember me?" | the face in front of it, matched in `people` |
| "What is Qdrant Edge?", "How does your memory work?" | `knowledge` |
| "Nod", "Show me you're happy" | nothing: the robot moves |

A face it has not met yet is asked its name, and remembered under it.

## Security

**Run this on a network you trust, and nowhere else.** None of the services
has authentication. The laptop's services bind every interface, so the robot
can reach them; anyone else on the network can too — to run the models, or to
read the dashboard, which shows the robot's camera and what it remembers. The
robot's camera service does the same with its live camera and microphone.

The dashboard refuses requests addressed to a host name it is not reached by
(which stops a web page from reaching it through DNS rebinding) and refuses a
button press from any page but its own; it has no other protection.

## Tests

```bash
uv run pytest
```

No robot, camera or model download needed.

## Layout

| | |
|---|---|
| `demo/run_demo.py` | the voice loop: listen, look, remember, answer |
| `demo/conversation.py`, `demo/chat_session.py` | the conversation, the tools, and how each memory question is answered |
| `demo/people.py`, `demo/knowledge.py` | who it has met; what it was taught |
| `demo/serve.py`, `demo/gpu_detect.py`, `demo/embed_service.py` | the laptop's model services |
| `demo/stage.py`, `scripts/robot_service.sh` | starting everything, on the laptop and on the robot |
| `demo/camera_service.py` | the robot's camera and microphone over HTTP |
| `demo/display/` | the dashboard |
| `emulator/models.py` | the models' files, by name, and where they come from |
| `emulator/edge_store.py`, `memory.py`, `frame_memory.py`, `face_memory.py` | the Qdrant Edge shards |

## See also

- [Qdrant Edge](https://qdrant.tech/documentation/edge/) — the documentation.
- [edge-mission-control](https://github.com/qdrant-labs/edge-mission-control) —
  another Qdrant Edge demo: a home robot's patrol, played from video, where
  every object it sees becomes a searchable memory.

## License

Apache-2.0 (see `LICENSE`). The models and several libraries have their own
licenses, and the voice uses GPL-licensed libraries at run time — see `NOTICE`.
