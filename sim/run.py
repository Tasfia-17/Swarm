"""
SwarmPilot Live Simulation → MP4
Run:  python sim/run.py
Captures 60-second pygame simulation to swarm_pilot_sim.mp4
"""
import sys, os, asyncio, threading, math, random, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pygame
import numpy as np
import cv2

from sim.cfg import *
from sim.world import (draw_road, draw_ego, NPC, Pedestrian,
                       TireMarks, Particles, glow, draw_rounded_rect)
from sim.hud  import (speedometer, agent_pill, debate_panel,
                      action_badge, top_bar, _fonts)

from simulator.frame_gen import get_frame
from vision.processor    import process as vision_process
from debate.orchestrator import run_debate


# ── Debate state ─────────────────────────────────────────────────────

class DebateState:
    def __init__(self):
        self.action     = "MAINTAIN"
        self.confidence = 0.70
        self.bubbles: list[dict] = []
        self.agent_active: dict  = {k: False for k in AGENT_COL}
        self.thinking   = False
        self.total_ms   = 0
        self.round_num  = 0
        self._lock      = threading.Lock()
        self._history: list[str] = []

    def push(self, agent, text, verdict=None):
        with self._lock:
            self.bubbles.append({"agent": agent, "text": text, "verdict": verdict})
            if len(self.bubbles) > 30:
                self.bubbles.pop(0)
            self.agent_active[agent] = True

    def set_result(self, action, confidence, ms, round_num):
        with self._lock:
            self.action     = action
            self.confidence = confidence
            self.total_ms   = ms
            self.round_num  = round_num
            self.thinking   = False

    def get_bubbles(self):
        with self._lock:
            return list(self.bubbles)

    def get_active(self):
        with self._lock:
            return dict(self.agent_active)

    def reset_active(self):
        with self._lock:
            for k in self.agent_active:
                self.agent_active[k] = False
            self.thinking = True


ds = DebateState()


# ── Scene script ─────────────────────────────────────────────────────

SCRIPT = [
    # (duration_s, scene_key, hazard, spawn_npcs, spawn_ped, spawn_merging)
    (5,  "highway_clear",    None,                 True,  False, False),
    (8,  "urban_pedestrian", "pedestrian_crossing",True,  True,  False),
    (7,  "rainy_night",      "low_visibility",     True,  False, False),
    (8,  "intersection",     "red_light",          False, True,  False),
    (8,  "highway_merge",    "vehicle_merging",    True,  False, True),
    (8,  "school_zone",      "children",           True,  True,  False),
    (6,  "highway_clear",    None,                 True,  False, False),
    (10, "outro",            None,                 True,  False, False),
]


# ── Background debate thread ──────────────────────────────────────────

def _debate_worker(scene: str, hazard: str | None):
    ds.reset_active()
    ds.push("pilot",   "Analyzing road ahead...", None)
    ds.push("critic",  "Scanning for risks...",   None)
    ds.push("safety",  "Checking constraints...", None)

    async def _run():
        frame_np, meta = get_frame(scene, random.randint(0, 20))
        vis = vision_process(frame_np)
        images = [vis["rgb"], vis["flow"], vis["depth"]]
        sd = f"Scene: {meta['name']}. Speed: {meta['speed']} km/h. Hazard: {hazard or 'none'}."
        result = await run_debate(scene_desc=sd, hazard=hazard,
                                  images=images, history=ds._history[-6:])
        return result

    try:
        result = asyncio.run(_run())
        for r in result.get("rounds", []):
            ds.push("pilot",   r.get("pilot_reasoning","")[:60], None)
            ds.push("critic",  r.get("critic_reasoning","")[:60], r.get("critic_verdict"))
            if r.get("safety_veto"):
                ds.push("safety", "VETO: " + r.get("safety_reasoning","")[:48], "UNSAFE")
            else:
                ds.push("safety", r.get("safety_reasoning","")[:60], "SAFE")
            if r.get("expert_domain"):
                ds.push("expert", f"Domain: {r['expert_domain'][:40]}", None)
            ds.push("auditor", r.get("auditor_reasoning","")[:60], None)
            ds.push("judge",   r.get("judge_summary","")[:60], None)

        ds.set_result(result["final_action"],
                      result["final_confidence"],
                      result["total_ms"],
                      len(result.get("rounds",[])))
        ds._history.append(f"{scene}: {result['final_action']}")
    except Exception as e:
        ds.push("judge", f"Error: {str(e)[:50]}", "UNSAFE")
        ds.set_result("MAINTAIN", 0.5, 0, 0)


# ── Main simulation ───────────────────────────────────────────────────

def run(output="swarm_pilot_sim.mp4", headless=True):
    os.environ.setdefault("SDL_VIDEODRIVER", "offscreen" if headless else "x11")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    pygame.init()
    pygame.font.init()

    if headless:
        screen = pygame.Surface((W, H))
    else:
        screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("SwarmPilot")

    clock  = pygame.time.Clock()

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, RECORD_FPS, (W, H))

    particles  = Particles()
    tire_marks = TireMarks()

    scroll_y   = 0.0
    ego_x      = LANE_XS[1] + 20   # start in lane 2
    ego_y      = H * 0.72
    ego_speed  = 0.0
    target_speed = 60.0

    npcs: list[NPC] = []
    peds: list[Pedestrian] = []

    # Spawn initial NPCs
    for _ in range(6):
        lane = random.randint(0, NUM_LANES-1)
        npcs.append(NPC(lane, random.randint(-400, -50), random.uniform(30, 70)))

    scene_idx   = 0
    scene_start = 0.0
    scene_key, hazard = SCRIPT[0][1], SCRIPT[0][2]
    debate_timer = 0.0

    # Fire first debate
    threading.Thread(target=_debate_worker, args=(scene_key, hazard), daemon=True).start()

    t_total = 0.0
    frame_count = 0
    TOTAL_DURATION = sum(s[0] for s in SCRIPT)
    print(f"🎬 Recording simulation → {output}")
    print(f"   Scenes: {len(SCRIPT)}  |  Duration: {TOTAL_DURATION}s")

    running = True
    FIXED_DT = 1.0 / FPS  # fixed timestep for deterministic 60s video

    while running and t_total < TOTAL_DURATION:
        if not headless:
            clock.tick(FPS)
        dt = FIXED_DT
        t_total += dt
        debate_timer += dt

        # ── Events ─────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # ── Scene progression ──────────────────────────────────
        scene_elapsed = t_total - scene_start
        if scene_elapsed >= SCRIPT[scene_idx][0] and scene_idx < len(SCRIPT)-1:
            scene_idx   += 1
            scene_start  = t_total
            scene_key    = SCRIPT[scene_idx][1]
            hazard       = SCRIPT[scene_idx][2]
            spawn_npcs   = SCRIPT[scene_idx][3]
            spawn_ped    = SCRIPT[scene_idx][4]
            spawn_merge  = SCRIPT[scene_idx][5]
            debate_timer = 0.0

            if scene_key != "outro":
                threading.Thread(target=_debate_worker,
                                 args=(scene_key, hazard), daemon=True).start()
            if spawn_ped and scene_key != "outro":
                peds.append(Pedestrian(ego_y - 180))
            if spawn_merge and scene_key != "outro":
                npcs.append(NPC(NUM_LANES-1, int(ego_y - 120),
                                random.uniform(25, 45), ORANGE))
            if spawn_npcs and len(npcs) < 4:
                for _ in range(2):
                    lane = random.randint(0, NUM_LANES-1)
                    npcs.append(NPC(lane, random.randint(-300,-50),
                                    random.uniform(30,65)))

        # ── Physics ────────────────────────────────────────────
        action = ds.action
        if action in ("BRAKE", "STOP"):
            target_speed = 0.0
        elif action == "ACCELERATE":
            target_speed = 95.0
        elif action == "STEER_LEFT":
            target_speed = 55.0
            ego_x += (-1.5) * 40 * dt
        elif action == "STEER_RIGHT":
            target_speed = 55.0
            ego_x += 1.5 * 40 * dt
        else:
            target_speed = 58.0

        ego_speed += (target_speed - ego_speed) * min(2.0*dt, 1.0)
        ego_x = max(ROAD_LEFT+18, min(ROAD_RIGHT-18, ego_x))
        scroll_y += ego_speed * dt * 1.2

        # Spawn NPCs periodically
        if random.random() < 0.015 and len(npcs) < 8 and scene_key != "outro":
            lane = random.randint(0, NUM_LANES-1)
            npcs.append(NPC(lane, -60, random.uniform(25, 65)))

        # Update / cull NPCs
        for n in npcs:
            n.update(dt, ego_speed)
        npcs = [n for n in npcs if n.y < H + 100]

        # Pedestrians
        for p in peds:
            p.update(dt)
        peds = [p for p in peds if p.x < ROAD_RIGHT + 100]

        # Tire marks
        if action in ("BRAKE","STOP"):
            tire_marks.add(ego_x, ego_y, scroll_y)

        # ── Render ─────────────────────────────────────────────
        if scene_key == "outro":
            _draw_outro(screen, t_total, ds)
        else:
            draw_road(screen, scroll_y, scene_key)
            tire_marks.draw(screen, scroll_y)
            for n in npcs:
                n.draw(screen)
            for p in peds:
                p.draw(screen, t_total)
            draw_ego(screen, ego_x, ego_y, action, t_total, particles)
            particles.update_draw(screen, dt)
            _draw_right_panel(screen, ds, ego_speed, t_total, action, scene_key, hazard)
            top_bar(screen, scene_key, hazard, t_total, 1.0/(dt+0.001))

        # Capture every 2nd frame → 30fps video
        frame_count += 1
        if frame_count % 2 == 0:
            pix = pygame.surfarray.array3d(screen)
            frame_bgr = cv2.cvtColor(
                np.transpose(pix, (1,0,2)), cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

        if not headless:
            pygame.display.flip()

    writer.release()
    pygame.quit()

    # Re-encode for compatibility
    final = output.replace(".mp4", "_final.mp4")
    os.system(f'ffmpeg -y -i "{output}" '
              f'-c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p '
              f'-f lavfi -i anullsrc=r=44100:cl=stereo '
              f'-c:a aac -b:a 128k -shortest '
              f'"{final}" 2>/dev/null')
    if os.path.exists(final):
        os.remove(output)
        print(f"\n✅ {final}  ({os.path.getsize(final)//1024}KB)")
    else:
        print(f"\n✅ {output}")


def _draw_right_panel(surf, ds: DebateState, speed, t, action, scene, hazard):
    """Right side: debate bubbles + speedometer + agent pills + action."""
    PX = 800   # panel starts at x=800
    draw_rounded_rect(surf, DARK_PANEL, (PX, 46, W-PX, H-46), 0, 210)
    pygame.draw.line(surf, (40,40,70), (PX,46), (PX, H), 1)

    # Action badge
    action_badge(surf, PX + (W-PX)//2, 54, action)

    # Confidence bar
    bx, by = PX+10, 104
    bw = W-PX-20
    pygame.draw.rect(surf, (30,30,45), (bx, by, bw, 12), border_radius=5)
    fw2 = int(bw * ds.confidence)
    bar_col = NEON if ds.confidence > 0.7 else ORANGE if ds.confidence > 0.4 else RED
    pygame.draw.rect(surf, bar_col, (bx, by, fw2, 12), border_radius=5)
    f = pygame.font.Font(None, 16)
    surf.blit(f.render(f"CONSENSUS  {ds.confidence*100:.0f}%  |  {ds.total_ms:.0f}ms", True, GRAY),
              (bx, by+14))

    # Agent pills
    for i, (key, label) in enumerate([
        ("pilot","🚗 PILOT"), ("critic","🔍 CRITIC"), ("safety","✅ SAFETY"),
        ("expert","🎓 EXPERT"), ("auditor","🔄 AUDITOR"), ("judge","⚖ JUDGE")
    ]):
        active  = ds.get_active().get(key, False)
        think   = ds.thinking and key in ("pilot","critic")
        agent_pill(surf, PX+10, 134 + i*28, key, label, active, think)

    # Speedometer
    speedometer(surf, PX + (W-PX)//2, H-80, speed)

    # Debate panel
    debate_panel(surf, PX+4, 310, W-PX-8, H-310-130, ds.get_bubbles())


def _draw_outro(surf, t, ds: DebateState):
    surf.fill(BG)
    pulse = 0.7 + 0.3*math.sin(t*4)
    cx, cy = W//2, H//2 - 60
    for r2 in [120, 90, 60]:
        glow(surf, (0, int(200*pulse), int(100*pulse)), (cx,cy), r2, 30)

    f1 = pygame.font.Font(None, 72)
    f2 = pygame.font.Font(None, 28)
    f3 = pygame.font.Font(None, 20)
    t1 = f1.render("⚡ SWARMPILOT", True, CYAN)
    surf.blit(t1, (cx - t1.get_width()//2, cy - 30))
    t2 = f2.render("Adversarial Multi-Agent Autonomous Driving", True, WHITE)
    surf.blit(t2, (cx - t2.get_width()//2, cy + 42))
    t3 = f2.render("Cerebras × Gemma 4 31B  ·  6 Parallel Agents", True, NEON)
    surf.blit(t3, (cx - t3.get_width()//2, cy + 74))

    tags = "@Cerebras  @googlegemma  #g4hackathon-multiverse-agents"
    t4 = f3.render(tags, True, GRAY)
    surf.blit(t4, (cx - t4.get_width()//2, cy + 115))


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "swarm_pilot_sim.mp4"
    run(out, headless=True)
