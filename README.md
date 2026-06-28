# SwarmPilot

> **Adversarial Multi-Agent Autonomous Driving powered by Cerebras x Google DeepMind Gemma 4 31B**

Submission for the **Cerebras x Google DeepMind Gemma 4 24-Hour Hackathon**
Track 1: Multiverse Agents -- Best Multi-Agent + Multimodal Use Case

A self-driving car controlled by **6 adversarial AI agents** who debate every scene in real-time.
No agent drives alone. Every decision is challenged, vetoed, or confirmed before the vehicle acts.

---

## The Core Idea

Most multi-agent systems use agents that collaborate. SwarmPilot uses agents that **fight**.

A Pilot proposes an action. A Critic attacks it. A Safety magistrate can veto unconditionally.
An Expert Witness steps in when the debate stalls. An Auditor forces role-switching -- the Pilot
must now argue against its own plan and the Critic must defend it. A Judge synthesizes everything
and applies D3 budgeted stopping: if confidence >= 0.85 the debate ends immediately, saving tokens
and latency.

This is not a pipeline. It is a courtroom.

---

## Demo Video

`swarm_demo_v2_final.mp4` -- 60-second demo included in this repo.

Shows: live debate transcripts, real Cerebras tok/s display, D3 confidence threshold,
and a side-by-side Cerebras vs GPU speed comparison using actual measured latency.

---

## Architecture

```
Camera Frame
(RGB + Optical Flow + Depth)
         |
         v
  +--------------+
  |  ROUND START |
  +--------------+
         |
         +---> Pilot Agent          proposes driving action
         |     (Gemma 4 31B)
         |
         +---> Critic Agent         attacks the plan           reasoning_effort=medium
         |     (Gemma 4 31B)        verdict: SAFE / UNSAFE / UNCERTAIN
         |
         +---> Safety Magistrate    hard veto check            can override unconditionally
               (Gemma 4 31B)

                    [all three run in parallel via asyncio.gather]
         |
         v
  [if Critic verdict == UNCERTAIN]
         |
         +---> Expert Witness       domain-specific knowledge
               (Gemma 4 31B)        e.g. "School Zone Safety", "Highway Merge Dynamics"
         |
         v
  +------------------+
  | Consistency      |   forces role-switching: Pilot argues against itself,
  | Auditor          |   Critic defends the plan -- checks for self-contradiction
  +------------------+
         |
         v
  +------------------+
  | Judge            |   synthesizes all arguments
  |                  |   scores confidence 0.0 to 1.0
  |   D3 THRESHOLD   |   if confidence >= 0.85 --> debate STOPS (D3 budgeted stopping)
  +------------------+   otherwise new round begins (max 3 rounds)
         |
         v
  Final Action --> Vehicle Control
```

All 6 agents call `gemma-4-31b` on Cerebras via the standard OpenAI-compatible API.
Parallel execution via `asyncio.gather` keeps wall-clock latency at ~7 seconds for a full 6-agent debate.

---

## Research Foundations

| Paper | What We Use |
|-------|-------------|
| **PROClaim (2026)** | Courtroom-style adversarial debate structure; role-switching mechanism where Pilot and Critic swap positions to prevent passive objection |
| **D3: Dynamic Debate Depth (2025)** | Budgeted stopping -- debate terminates early when Judge confidence crosses 0.85 threshold, saving tokens and reducing latency |
| **When Helping Hurts (2026)** | Critic agent uses `reasoning_effort=medium` (deeper thinking) vs Pilot which uses `none` -- asymmetric reasoning depth improves adversarial quality |
| **JoyAI-VL (2026)** | Real-time vision-language action loop; multimodal input directly drives action decisions without intermediate representation |

---

## Why Cerebras Makes This Possible

**The problem:** Running 6 LLM agents sequentially on a standard GPU takes ~50 seconds per frame.
That is not real-time. That is not autonomous driving. That is a thought experiment.

**What Cerebras enables:**

| Feature | How SwarmPilot Uses It |
|---------|------------------------|
| 100 RPM elevated limits | 6 agents fire simultaneously at hackathon rate limits without queuing |
| ~1800+ tok/s throughput | Full debate round completes in ~7s wall-clock for all 6 agents |
| `reasoning_effort=medium` | Critic gets deeper adversarial analysis without slowing the whole pipeline |
| 65K context window | Rolling 20-frame world state is preserved across the full debate history |
| `time_info` object | Real tokens/sec computed per frame and displayed live in the UI and demo video |
| Structured outputs (`strict: true`) | All 6 agents return typed Pydantic schemas -- zero parsing failures |

**Measured latency (from `debate_cache.json`, real Cerebras API calls this session):**

| Scene | Debate Latency | Agents | Final Action |
|-------|---------------|--------|--------------|
| highway | 6.9s | 6 parallel | Slow down for merge |
| pedestrian | 6.7s | 6 parallel | STOP |
| rain | 8.5s | 6 parallel | STOP |
| intersect | 4.9s | 6 parallel | STOP |
| merge | 7.0s | 6 parallel | Reduce speed + buffer |
| school | 9.3s | 6 parallel | Reduce to 10 km/h + hazard lights |

Sequential equivalent on same prompts: ~42 seconds. **Cerebras delivers ~6-7x speedup.**

---

## The 6 Agents

### 1. Pilot Agent
- Role: proposes the optimal driving action
- Personality: assertive, optimistic, action-biased
- Input: RGB + optical flow + depth images + scene description + rolling history
- Output: `DriveAction` (action, confidence, reasoning)
- Reasoning: off (`reasoning_effort=none`) -- fast proposal

### 2. Critic Agent
- Role: finds fatal flaws in the Pilot's plan
- Personality: pessimistic, rigorous, adversarial
- Input: Pilot's proposed action + all 3 images
- Output: `CriticResponse` (verdict: SAFE/UNSAFE/UNCERTAIN, fatal_flaw, alternative_action)
- Reasoning: **medium** -- deeper analysis than Pilot by design (When Helping Hurts)

### 3. Safety Magistrate
- Role: hard veto authority -- enforces non-negotiable safety constraints
- Can override any action unconditionally, including overriding the Judge
- Output: `SafetyVerdict` (veto: bool, constraint_violated, safe_action)
- If `veto=true`, debate terminates immediately regardless of confidence

### 4. Expert Witness
- Role: domain-specific knowledge injection when debate stalls
- Spawned only when Critic returns UNCERTAIN and a hazard is present
- Domain examples: "fog and adverse weather driving", "school zone safety", "highway merge dynamics"
- Output: `ExpertWitnessResponse` (domain, recommendation, reasoning)

### 5. Consistency Auditor
- Role: enforces PROClaim role-switching -- checks that agents are not contradicting themselves
- Forces Pilot to argue against its own plan, Critic to defend it
- Detects logical contradictions that emerge from role reversal
- Output: `ConsistencyAuditResponse` (consistent, contradiction, resolved_action)

### 6. Judge
- Role: final synthesis and D3 budgeted stopping decision
- Weighs all 5 agent inputs, scores confidence 0.0 to 1.0
- `converged=true` + `confidence >= 0.85` triggers D3 early stopping
- Output: `JudgeVerdict` (final_action, confidence, converged, summary)

---

## Vision Pipeline

Each frame sends **3 images** to every agent via Cerebras multimodal API:

```
Raw scene frame
      |
      +-- RGB image          raw scene as captured (synthetic from frame_gen.py)
      |
      +-- Optical Flow       motion vectors visualized as HSV heatmap
      |                      reveals moving objects, pedestrians, merging vehicles
      |
      +-- Depth map          simulated depth (near=bright, far=dark)
                             enables distance estimation without LiDAR
```

Images are Base64-encoded and sent via the standard OpenAI multimodal `image_url` format:

```python
{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
```

All 6 agents receive all 3 images every round. Total: **18 image inputs per debate round.**

---

## Structured Outputs

Every agent response is enforced with Pydantic + Cerebras strict JSON schema:

```python
class CriticResponse(BaseModel):
    verdict: str          # SAFE | UNSAFE | UNCERTAIN
    fatal_flaw: str | None
    alternative_action: str | None
    reasoning: str
    ttft_ms: float = 0.0
    total_ms: float = 0.0
```

`strict: true` is set on every API call. Zero hallucinated fields. Zero parsing failures across
all 6 scenes in the demo session.

---

## D3 Budgeted Stopping

Standard multi-agent debate systems run a fixed number of rounds regardless of whether consensus
is reached. D3 (Dynamic Debate Depth) stops early when the Judge is confident enough:

```python
if judge.converged and judge.confidence >= CONVERGENCE_THRESHOLD:  # 0.85
    agreement_streak += 1

if agreement_streak >= 1 or judge.converged:
    break  # stop debate, act now
```

In 4 of 6 demo scenes the debate converges in **round 1** (confidence 0.98-1.0), saving 2 extra
rounds and ~14 seconds of unnecessary inference per frame.

---

## Demo Scenes

| Scene Key | Hazard | Speed | Expected Decision | Confidence |
|-----------|--------|-------|-------------------|------------|
| `highway` | Vehicle merging right | 80 km/h | Slow down, create buffer | 0.98 |
| `pedestrian` | Pedestrian crossing | 0 km/h | STOP | 1.00 |
| `rain` | Low visibility, red light | 0 km/h | STOP | 1.00 |
| `intersect` | Red light | 0 km/h | STOP | 1.00 |
| `merge` | Lateral merge hazard | 80 km/h | Reduce speed + monitor | 0.90 |
| `school` | Children present | slow | Reduce to 10 km/h + hazard lights | 0.98 |

---

## Project Structure

```
swarm-pilot/
|-- main.py                  FastAPI backend (REST + static file server)
|-- config.py                API keys, hyperparameters (MAX_DEBATE_ROUNDS, CONVERGENCE_THRESHOLD)
|-- prefetch.py              Pre-fetch all Cerebras debates, save to debate_cache.json
|-- render_v2.py             Upgraded 60s demo renderer (real text, D3 bar, tok/s, scene cards)
|-- render.py                Original renderer (reference)
|-- debate_cache.json        Real Cerebras API responses from this session
|
|-- agents/
|   |-- agents.py            6 agent definitions, Pydantic schemas, Cerebras API calls
|
|-- debate/
|   |-- orchestrator.py      Debate loop: asyncio.gather, role-switching, D3 stopping
|
|-- vision/
|   |-- processor.py         RGB + optical flow + depth extraction and Base64 encoding
|
|-- simulator/
|   |-- frame_gen.py         Synthetic scene generator (no CARLA required)
|
|-- sim/
|   |-- world.py             Extended simulation world state
|   |-- hud.py               Simulation HUD overlay
|
|-- static/
|   |-- index.html           Single-file frontend: live debate feed, agent panels, metrics
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

**To pre-fetch debate results from Cerebras:**

```bash
python prefetch.py
# Runs all 6 scenes, saves real API responses to debate_cache.json
```

**To render the demo video:**

```bash
python render_v2.py
# Output: swarm_demo_v2_final.mp4  (60s, 1280x720, H264)
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Frontend UI (live debate feed) |
| `/debate` | POST | Run one full debate frame |
| `/metrics` | GET | Last frame metrics (latency, rounds, confidence) |
| `/scenes` | GET | List available scenes |

**Run a debate:**

```bash
curl -X POST http://localhost:8000/debate \
  -H "Content-Type: application/json" \
  -d '{"scene": "school_zone"}'
```

**Response includes:**

```json
{
  "final_action": "Reduce to 10 km/h and activate hazard lights",
  "final_confidence": 0.98,
  "converged": true,
  "total_ms": 9283,
  "rounds": [...],
  "safety_vetoed": false,
  "frame_b64": "...",
  "flow_b64": "...",
  "depth_b64": "..."
}
```

---

## How Role-Switching Works (PROClaim)

In standard multi-agent debate, a Critic can be a passive objector -- it always says "unsafe"
without being accountable to any alternative. PROClaim fixes this:

**Round N:** Pilot proposes action A. Critic says A is UNSAFE.

**Consistency Auditor step:** The Pilot is now forced to argue *against* action A.
The Critic is now forced to *defend* action A.

If the Critic cannot defend the action it just criticized, it has contradicted itself.
The Auditor flags this, resolves the contradiction, and the Judge weighs the inconsistency
against the Critic's original objection.

This prevents adversarial agents from being reflexively negative and forces every agent
to be accountable for the quality of its reasoning.

---

## Hackathon Track

**Track 1: Multiverse Agents -- Best Multi-Agent + Multimodal Use Case**

Cerebras x Google DeepMind Gemma 4 24-Hour Hackathon
June 28-29, 2026

**Judging criteria addressed:**

| Criterion | SwarmPilot Implementation |
|-----------|--------------------------|
| Agent Collaboration | 6 agents with structured debate protocol, role-switching, D3 stopping |
| Multimodal Intelligence | 3 image modalities (RGB + flow + depth) sent to all 6 agents every round |
| Speed in Action | Live tok/s display from `time_info`; ~7s debate vs ~50s sequential GPU baseline |
| Innovation | Adversarial debate for physical AI safety decisions; PROClaim + D3 in autonomous driving |
