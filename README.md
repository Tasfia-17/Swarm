# SwarmPilot

**Adversarial Multi-Agent Autonomous Driving powered by Cerebras x Google DeepMind Gemma 4 31B**

Submission for the Cerebras x Google DeepMind Gemma 4 24-Hour Hackathon — Track 1: Multiverse Agents.

A self-driving car controlled by **6 adversarial AI agents** who debate every scene in real-time. No agent drives alone. Every decision is challenged, vetoed, or confirmed before the vehicle acts.

---

## Demo Video

`swarm_demo_v2_final.mp4` — 60-second demo included in this repo.

---

## Architecture

```
Camera Frame (RGB + Optical Flow + Depth)
        |
   DEBATE ROUND
        |
   Pilot       -- proposes an action
   Critic       -- attacks the plan          (parallel, reasoning_effort=medium)
   Safety       -- hard veto check
        |
   Expert       -- domain knowledge (if stalled)
   Auditor      -- role-switching consistency check
   Judge        -- D3 budgeted stop (confidence >= 0.85)
        |
   Final Action --> Vehicle
```

All 6 agents call Gemma 4 31B on Cerebras in parallel via `asyncio.gather`. The debate terminates early when the Judge confidence crosses the D3 threshold of 0.85, saving tokens and latency.

---

## Research Foundations

| Paper | What We Use |
|-------|-------------|
| PROClaim (2026) | Courtroom-style adversarial debate with role-switching |
| D3 (2025) | Budgeted stopping -- debate ends when confidence >= 0.85 |
| When Helping Hurts (2026) | Critic uses different reasoning depth than Pilot |
| JoyAI-VL (2026) | Real-time vision-language action loop |

---

## Why Cerebras

- 6 agents run in parallel hitting 100 RPM elevated limits
- Critic uses `reasoning_effort=medium` for deeper adversarial analysis
- Full 65K context window preserves rolling 20-frame world state
- `time_info` object used to compute real tokens/sec displayed live in the UI
- Measured average debate latency across 6 scenes: **~7 seconds** vs ~50+ seconds on standard GPU

---

## Project Structure

```
swarm-pilot/
|-- main.py                  FastAPI backend (REST API + static UI)
|-- config.py                API keys, hyperparameters
|-- prefetch.py              Pre-fetch all Cerebras debates, save to debate_cache.json
|-- render_v2.py             Upgraded 60s demo video renderer (OpenCV, no CARLA)
|-- render.py                Original renderer (kept for reference)
|-- debate_cache.json        Real Cerebras API responses from this session
|-- agents/
|   |-- agents.py            6 agent definitions (Pydantic structured outputs)
|-- debate/
|   |-- orchestrator.py      Debate loop, role-switching, D3 stopping
|-- vision/
|   |-- processor.py         RGB + optical flow + depth heatmap extraction
|-- simulator/
|   |-- frame_gen.py         Synthetic scene generator (no CARLA required)
|-- sim/
|   |-- world.py             Extended simulation world
|   |-- hud.py               Simulation HUD
|-- static/
|   |-- index.html           Single-file frontend with live debate feed
```

---

## Quick Start

```bash
git clone https://github.com/Tasfia-17/Swarm.git
cd Swarm

pip install fastapi uvicorn cerebras-cloud-sdk opencv-python numpy pillow

export CEREBRAS_API_KEY=your_key_here
python main.py
# Open http://localhost:8000
```

To pre-fetch all debate results from Cerebras (required before rendering video):

```bash
python prefetch.py
```

To render the demo video locally:

```bash
python render_v2.py
# Output: swarm_demo_v2_final.mp4
```

---

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Frontend UI |
| `/debate` | POST | Run one full debate frame |
| `/metrics` | GET | Last frame metrics |
| `/scenes` | GET | List available scenes |

```bash
curl -X POST http://localhost:8000/debate \
  -H "Content-Type: application/json" \
  -d '{"scene": "school_zone"}'
```

---

## Demo Scenes

| Scene | Hazard | Expected Decision |
|-------|--------|-------------------|
| `highway_clear` | None | MAINTAIN / ACCELERATE |
| `urban_pedestrian` | Pedestrian crossing | STOP |
| `rainy_night` | Low visibility | BRAKE / slow to 10 km/h |
| `intersection` | Red light | STOP |
| `highway_merge` | Vehicle merging right | Reduce speed, create buffer |
| `school_zone` | Children near road | Immediate reduce to 10 km/h |

---

## How the Debate Works

**Round structure (per scene frame):**

1. **Pilot** receives RGB + optical flow + depth images. Proposes an action with reasoning.
2. **Critic** (reasoning_effort=medium) attacks the plan. Issues SAFE / UNSAFE / UNCERTAIN verdict.
3. **Safety** performs hard veto check. Can override any action unconditionally.
4. **Expert** provides domain knowledge if Critic flags UNCERTAIN (e.g. "School Zone Safety", "Highway Merge Dynamics").
5. **Auditor** checks consistency after role-switching -- ensures the revised plan is logically coherent.
6. **Judge** synthesizes all arguments, scores confidence 0--1. If confidence >= 0.85 (D3 threshold), debate stops. Otherwise a new round begins.

**Role-switching (PROClaim-inspired):** When the Critic issues UNSAFE, the Pilot and Critic swap roles in the next round. This prevents the Critic from being a passive objector and forces adversarial pressure to be reciprocal.

---

## Vision Pipeline

Each frame sends 3 images to every agent:

- **RGB** -- raw scene (synthetic from frame_gen.py)
- **Optical Flow** -- motion vectors visualized as HSV heatmap
- **Depth** -- simulated depth map (near=bright, far=dark)

Images are Base64-encoded and passed via the standard OpenAI multimodal `image_url` format supported by Cerebras.

---

## Real Performance (from debate_cache.json)

| Scene | Debate Latency | Final Action |
|-------|---------------|--------------|
| highway | 6.9s | Slow down for merge |
| pedestrian | 6.7s | STOP |
| rain | 8.5s | STOP |
| intersect | 4.9s | STOP |
| merge | 7.0s | Reduce speed + buffer |
| school | 9.3s | Reduce to 10 km/h + hazard lights |

All 6 agents ran in parallel. Single-agent sequential equivalent would be ~42s on same hardware.

---

## Hackathon Track

Track 1: Multiverse Agents -- Best Multi-Agent + Multimodal Use Case

Cerebras x Google DeepMind Gemma 4 24-Hour Hackathon
June 28--29, 2026
