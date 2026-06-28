"""
6-agent adversarial driving system.
Each agent is an async function that calls Cerebras gemma-4-31b with structured outputs.
"""
import asyncio
import json
import time
from openai import AsyncOpenAI
from pydantic import BaseModel
from config import CEREBRAS_API_KEY, MODEL

client = AsyncOpenAI(
    api_key=CEREBRAS_API_KEY,
    base_url="https://api.cerebras.ai/v1",
)


# ---------- Pydantic schemas ----------

class DriveAction(BaseModel):
    action: str          # ACCELERATE | BRAKE | STEER_LEFT | STEER_RIGHT | STOP | MAINTAIN
    confidence: float    # 0.0–1.0
    reasoning: str
    ttft_ms: float = 0.0
    total_ms: float = 0.0


class CriticResponse(BaseModel):
    verdict: str         # SAFE | UNSAFE | UNCERTAIN
    fatal_flaw: str | None
    alternative_action: str | None
    reasoning: str
    ttft_ms: float = 0.0
    total_ms: float = 0.0


class SafetyVerdict(BaseModel):
    veto: bool
    constraint_violated: str | None
    safe_action: str | None
    reasoning: str
    ttft_ms: float = 0.0
    total_ms: float = 0.0


class ExpertWitnessResponse(BaseModel):
    domain: str
    recommendation: str
    reasoning: str
    ttft_ms: float = 0.0
    total_ms: float = 0.0


class ConsistencyAuditResponse(BaseModel):
    consistent: bool
    contradiction: str | None
    resolved_action: str | None
    reasoning: str
    ttft_ms: float = 0.0
    total_ms: float = 0.0


class JudgeVerdict(BaseModel):
    final_action: str
    confidence: float
    converged: bool
    rounds_taken: int
    summary: str
    ttft_ms: float = 0.0
    total_ms: float = 0.0


# ---------- Shared helper ----------

def _make_image_msg(b64: str) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def _patch_schema(s: dict) -> dict:
    """Recursively add additionalProperties: false to all object schemas."""
    if isinstance(s, dict):
        if s.get("type") == "object":
            s["additionalProperties"] = False
        for v in s.values():
            _patch_schema(v)
    elif isinstance(s, list):
        for item in s:
            _patch_schema(item)
    return s


async def _call(system: str, user_text: str, images: list[str],
                schema: type, reasoning: str = "none") -> tuple[dict, dict]:
    content = [{"type": "text", "text": user_text}] + [_make_image_msg(b) for b in images]
    raw_schema = _patch_schema(schema.model_json_schema())
    t0 = time.monotonic()
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        response_format={"type": "json_schema", "json_schema": {
            "name": schema.__name__,
            "strict": True,
            "schema": raw_schema,
        }},
        reasoning_effort=reasoning,
        temperature=0.3,
    )
    elapsed = (time.monotonic() - t0) * 1000
    raw = json.loads(resp.choices[0].message.content)
    ti = getattr(resp, "time_info", None)
    if isinstance(ti, dict):
        ttft = ti.get("prompt_time", 0) * 1000
        total = ti.get("total_time", elapsed / 1000) * 1000
    elif ti is not None:
        ttft = getattr(ti, "prompt_time", 0) * 1000
        total = getattr(ti, "total_time", elapsed / 1000) * 1000
    else:
        ttft, total = 0.0, elapsed
    timing = {"ttft_ms": ttft, "total_ms": total}
    return raw, timing


# ---------- Agents ----------

async def pilot_agent(scene_desc: str, images: list[str], history: list[str]) -> DriveAction:
    hist = "\n".join(history[-4:]) if history else "No prior frames."
    raw, t = await _call(
        system="You are the Pilot Agent. You propose the optimal driving action. Be assertive and optimistic. Output JSON.",
        user_text=(
            f"Scene: {scene_desc}\nHistory: {hist}\n"
            "Propose the best driving action. Justify briefly."
        ),
        images=images,
        schema=DriveAction,
    )
    raw.update(t)
    return DriveAction(**raw)


async def critic_agent(scene_desc: str, pilot_action: str, images: list[str]) -> CriticResponse:
    raw, t = await _call(
        system="You are the Critic Agent. Your job is to find fatal flaws in the Pilot's plan. Be pessimistic and rigorous. Output JSON.",
        user_text=(
            f"Scene: {scene_desc}\nPilot proposed: {pilot_action}\n"
            "Find the worst-case failure. Is this action SAFE, UNSAFE, or UNCERTAIN?"
        ),
        images=images,
        schema=CriticResponse,
        reasoning="medium",  # Critic uses reasoning
    )
    raw.update(t)
    return CriticResponse(**raw)


async def safety_agent(scene_desc: str, pilot_action: str, critic_verdict: str, images: list[str]) -> SafetyVerdict:
    raw, t = await _call(
        system="You are the Safety Magistrate. You enforce hard constraints: never endanger life. Veto only if a hard constraint is violated. Output JSON.",
        user_text=(
            f"Scene: {scene_desc}\nPilot: {pilot_action}\nCritic: {critic_verdict}\n"
            "Does this action violate any safety constraint? Should you veto?"
        ),
        images=images,
        schema=SafetyVerdict,
    )
    raw.update(t)
    return SafetyVerdict(**raw)


async def expert_witness_agent(scene_desc: str, domain: str, images: list[str]) -> ExpertWitnessResponse:
    raw, t = await _call(
        system=f"You are an Expert Witness specializing in {domain}. Provide a domain-specific recommendation. Output JSON.",
        user_text=f"Scene: {scene_desc}\nWhat does a {domain} expert recommend for this situation?",
        images=images,
        schema=ExpertWitnessResponse,
    )
    raw.update(t)
    return ExpertWitnessResponse(**raw)


async def consistency_auditor_agent(
    scene_desc: str, pilot_action: str, critic_action: str, images: list[str]
) -> ConsistencyAuditResponse:
    """Role-switching: forces agents to argue the opposite position and checks consistency."""
    raw, t = await _call(
        system="You are the Consistency Auditor. You force role reversal: the Pilot must now argue against its own plan, and the Critic must defend it. Check if their arguments are self-consistent. Output JSON.",
        user_text=(
            f"Scene: {scene_desc}\n"
            f"Original Pilot plan: {pilot_action}\n"
            f"Original Critic objection: {critic_action}\n"
            "After role-switching, are these agents consistent or do they contradict themselves? "
            "Resolve to the most defensible action."
        ),
        images=images,
        schema=ConsistencyAuditResponse,
        reasoning="low",
    )
    raw.update(t)
    return ConsistencyAuditResponse(**raw)


async def judge_agent(
    scene_desc: str,
    pilot: DriveAction,
    critic: CriticResponse,
    safety: SafetyVerdict,
    auditor: ConsistencyAuditResponse,
    rounds: int,
    images: list[str],
) -> JudgeVerdict:
    summary = (
        f"Pilot: {pilot.action} (conf={pilot.confidence:.2f})\n"
        f"Critic: {critic.verdict} — {critic.fatal_flaw}\n"
        f"Safety veto: {safety.veto} — {safety.constraint_violated}\n"
        f"Auditor consistent: {auditor.consistent} — resolved: {auditor.resolved_action}"
    )
    raw, t = await _call(
        system="You are the Consensus Judge. Weigh all agent inputs and issue a final driving action. Apply D3 budgeted stopping: if confidence > 0.85 mark converged=true. Output JSON.",
        user_text=f"Scene: {scene_desc}\nDebate summary (round {rounds}):\n{summary}\nIssue final verdict.",
        images=images,
        schema=JudgeVerdict,
    )
    raw.update(t)
    raw["rounds_taken"] = rounds
    return JudgeVerdict(**raw)
