"""
Debate orchestrator — implements:
 - Parallel agent execution (asyncio.gather)
 - Role-switching via ConsistencyAuditor
 - Expert witness spawning when debate stalls
 - D3 budgeted stopping (stop when converged or max rounds)
"""
import asyncio
import time
from agents.agents import (
    pilot_agent, critic_agent, safety_agent,
    expert_witness_agent, consistency_auditor_agent, judge_agent,
    DriveAction, CriticResponse, SafetyVerdict,
    ConsistencyAuditResponse, JudgeVerdict,
)
from config import MAX_DEBATE_ROUNDS, CONVERGENCE_THRESHOLD

EXPERT_DOMAINS = {
    "low_visibility":    "fog and adverse weather driving",
    "pedestrian_crossing": "urban pedestrian safety",
    "red_light":         "traffic law compliance",
    "vehicle_merging":   "highway merge dynamics",
    "children":          "school zone safety",
}


async def run_debate(
    scene_desc: str,
    hazard: str | None,
    images: list[str],
    history: list[str],
) -> dict:
    t_start = time.monotonic()
    all_timings: list[float] = []
    rounds_log = []
    agreement_streak = 0

    pilot: DriveAction | None = None
    critic: CriticResponse | None = None
    safety: SafetyVerdict | None = None
    auditor: ConsistencyAuditResponse | None = None
    judge: JudgeVerdict | None = None

    for round_num in range(1, MAX_DEBATE_ROUNDS + 1):
        round_t = time.monotonic()

        # Round 1: Pilot, Critic, Safety in parallel
        if round_num == 1:
            pilot, critic, safety = await asyncio.gather(
                pilot_agent(scene_desc, images, history),
                critic_agent(scene_desc, "pending", images),
                safety_agent(scene_desc, "pending", "pending", images),
            )
            # Re-run critic with actual pilot action
            critic = await critic_agent(scene_desc, f"{pilot.action}: {pilot.reasoning}", images)
            safety = await safety_agent(
                scene_desc,
                f"{pilot.action}: {pilot.reasoning}",
                f"{critic.verdict}: {critic.fatal_flaw}",
                images,
            )
        else:
            # Subsequent rounds: re-evaluate with updated context
            pilot, critic = await asyncio.gather(
                pilot_agent(scene_desc, images, history + [f"Round {round_num-1} action: {judge.final_action if judge else 'none'}"]),
                critic_agent(scene_desc, f"{pilot.action}: {pilot.reasoning}" if pilot else "none", images),
            )
            safety = await safety_agent(
                scene_desc,
                f"{pilot.action}: {pilot.reasoning}",
                f"{critic.verdict}: {critic.fatal_flaw}",
                images,
            )

        all_timings.extend([pilot.total_ms, critic.total_ms, safety.total_ms])

        # Spawn Expert Witness if debate is contested
        expert = None
        if critic.verdict in ("UNSAFE", "UNCERTAIN") and hazard and hazard in EXPERT_DOMAINS:
            expert = await expert_witness_agent(scene_desc, EXPERT_DOMAINS[hazard], images)
            all_timings.append(expert.total_ms)

        # Consistency Auditor (role-switching)
        auditor = await consistency_auditor_agent(
            scene_desc,
            f"{pilot.action}: {pilot.reasoning}",
            f"{critic.alternative_action or 'none'}: {critic.reasoning}",
            images,
        )
        all_timings.append(auditor.total_ms)

        # Judge verdict
        judge = await judge_agent(scene_desc, pilot, critic, safety, auditor, round_num, images)
        all_timings.append(judge.total_ms)

        round_elapsed = (time.monotonic() - round_t) * 1000
        rounds_log.append({
            "round": round_num,
            "pilot_action": pilot.action,
            "pilot_confidence": pilot.confidence,
            "critic_verdict": critic.verdict,
            "critic_flaw": critic.fatal_flaw,
            "safety_veto": safety.veto,
            "auditor_consistent": auditor.consistent,
            "expert_domain": expert.domain if expert else None,
            "judge_action": judge.final_action,
            "judge_confidence": judge.confidence,
            "round_ms": round_elapsed,
            "pilot_reasoning": pilot.reasoning,
            "critic_reasoning": critic.reasoning,
            "safety_reasoning": safety.reasoning,
            "auditor_reasoning": auditor.reasoning,
            "judge_summary": judge.summary,
        })

        # D3 budgeted stopping
        if judge.converged and judge.confidence >= 0.85:
            agreement_streak += 1
        else:
            agreement_streak = 0

        if agreement_streak >= CONVERGENCE_THRESHOLD or judge.converged:
            break

        # Safety veto overrides — no more debate needed
        if safety.veto:
            break

    total_ms = (time.monotonic() - t_start) * 1000
    avg_agent_ms = sum(all_timings) / len(all_timings) if all_timings else 0

    return {
        "final_action": judge.final_action if judge else "STOP",
        "final_confidence": judge.confidence if judge else 0.0,
        "converged": judge.converged if judge else False,
        "rounds": rounds_log,
        "total_ms": total_ms,
        "avg_agent_ms": avg_agent_ms,
        "safety_vetoed": safety.veto if safety else False,
        "scene": scene_desc,
        "hazard": hazard,
    }
