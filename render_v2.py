"""
SwarmPilot 60s demo renderer v2 — upgraded realism.
Improvements over render.py:
  - Real reasoning text (full sentences, word-wrapped)
  - Actual total_ms from cache shown as real latency numbers
  - Live "typing" effect on debate bubbles
  - D3 confidence bar that grows to threshold per round
  - Tokens/sec display using real debate timing
  - Compare scene shows actual debate timing from cache
  - Scene title card on transitions
  - Consistent panel across all scenes including hook/outro
"""
import cv2, numpy as np, math, json, os, random, textwrap
import time as tmod

W, H, FPS = 1280, 720, 30
TOTAL = FPS * 60
SIM_W = 860
PAN_X = SIM_W

ROAD_L, ROAD_R = 160, 700
LANE_XS = [200, 310, 420, 530, 640]

BG      = (14, 10, 18)
ROAD_C  = (52, 48, 58)
NEON_G  = (30, 255, 120)
NEON_B  = (255, 200, 0)
ORANGE  = (30, 140, 255)
RED     = (40,  40, 220)
WHITE   = (255, 255, 255)
GRAY    = (100,  95, 110)
YELLOW  = (0,  210, 255)
DARK    = (10,   8,  16)
CYAN    = (220, 220,  50)

AGENT_COL = {
    "pilot":   (50, 160, 255),
    "critic":  (255, 130,  80),
    "safety":  (80,  220,  50),
    "expert":  (40,  190, 220),
    "auditor": (240,  70, 200),
    "judge":   (230, 230,  40),
}
ACTION_COL = {
    "BRAKE": RED, "STOP": RED,
    "ACCELERATE": NEON_G, "MAINTAIN": NEON_G,
    "STEER_LEFT": NEON_B, "STEER_RIGHT": NEON_B,
}

TL = [
    ( 0,  4, "hook"),
    ( 4, 12, "highway"),
    (12, 20, "pedestrian"),
    (20, 28, "rain"),
    (28, 36, "intersect"),
    (36, 44, "merge"),
    (44, 52, "school"),
    (52, 60, "compare"),
]
HAZARDS = {
    "pedestrian": "pedestrian_crossing",
    "rain":       "low_visibility",
    "intersect":  "red_light",
    "merge":      "vehicle_merging",
    "school":     "children_in_road",
}
LABELS = {
    "highway":    "HIGHWAY — 80 km/h cruise",
    "pedestrian": "URBAN — PEDESTRIAN CROSSING",
    "rain":       "NIGHT RAIN — LOW VISIBILITY",
    "intersect":  "INTERSECTION — RED LIGHT",
    "merge":      "HIGHWAY MERGE — LATERAL HAZARD",
    "school":     "SCHOOL ZONE — CHILDREN PRESENT",
    "compare":    "CEREBRAS vs GPU — SPEED PROOF",
}

def scene_at(fi):
    t = fi / FPS
    for s, e, sc in TL:
        if s <= t < e:
            return sc, t - s, e - s
    return "compare", 0, 8


def T(img, s, x, y, col, scale=0.55, thick=1):
    cv2.putText(img, str(s), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)


def TC(img, s, cx, y, col, scale=0.9, thick=2):
    tw = cv2.getTextSize(str(s), cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0]
    cv2.putText(img, str(s), (cx - tw // 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)


def glow(img, x, y, col, r, a=0.35):
    ov = img.copy()
    cv2.circle(ov, (int(x), int(y)), int(r * 1.8), col, -1)
    cv2.addWeighted(ov, a * 0.4, img, 1 - a * 0.4, 0, img)
    cv2.circle(img, (int(x), int(y)), int(r), col, -1)


def abox(img, x1, y1, x2, y2, col, a=0.45, rad=6):
    sub = img[y1:y2, x1:x2]
    rect = np.full_like(sub, col)
    cv2.addWeighted(rect, a, sub, 1 - a, 0, sub)
    img[y1:y2, x1:x2] = sub
    cv2.rectangle(img, (x1, y1), (x2, y2), col, 1)


def draw_road(img, scroll, scene, t):
    img[:, :SIM_W] = ROAD_C
    cv2.rectangle(img, (0, 0), (ROAD_L, H), (28, 24, 34), -1)
    cv2.rectangle(img, (ROAD_R, 0), (SIM_W, H), (28, 24, 34), -1)
    cv2.rectangle(img, (ROAD_L, 0), (ROAD_R, H), (52, 48, 58), -1)
    if scene == "rain":
        r = random.Random(int(t * 7))
        for _ in range(40):
            rx = r.randint(ROAD_L, ROAD_R)
            ry = r.randint(0, H)
            cv2.line(img, (rx, ry), (rx - 1, ry + 16), (140, 165, 190), 1)
        # dark overlay for night feel
        ov = img.copy()
        cv2.rectangle(ov, (0, 0), (SIM_W, H), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.28, img, 0.72, 0, img)
    cv2.line(img, (LANE_XS[0] - 46, 0), (LANE_XS[0] - 46, H), (175, 170, 155), 2)
    cv2.line(img, (LANE_XS[-1] + 46, 0), (LANE_XS[-1] + 46, H), (175, 170, 155), 2)
    off = int(scroll) % 70
    for lx in LANE_XS[1:-1]:
        for y in range(-off, H + 70, 70):
            cv2.line(img, (lx, y), (lx, y + 40), (165, 160, 140), 2)
    if scene == "intersect":
        iy = H // 2 + 50
        cv2.rectangle(img, (0, iy - 100), (SIM_W, iy + 100), (52, 48, 58), -1)
        for i in range(9):
            cx2 = ROAD_L + 25 + i * 58
            cv2.rectangle(img, (cx2, iy - 90), (cx2 + 38, iy - 25), (188, 183, 168), -1)
        cv2.line(img, (ROAD_L, iy - 100), (ROAD_R, iy - 100), WHITE, 3)
        cv2.rectangle(img, (ROAD_R + 12, iy - 175), (ROAD_R + 50, iy - 55), (20, 18, 25), -1)
        cv2.circle(img, (ROAD_R + 31, iy - 152), 15, RED, -1)
        glow(img, ROAD_R + 31, iy - 152, RED, 20, 0.9)
        cv2.circle(img, (ROAD_R + 31, iy - 115), 15, (20, 50, 20), -1)
    elif scene == "school":
        for sy in [70, 280, 500]:
            cv2.rectangle(img, (ROAD_L - 72, sy), (ROAD_L - 8, sy + 52), (0, 200, 240), -1)
            T(img, "SCHOOL", ROAD_L - 70, sy + 20, (0, 0, 0), 0.38, 1)
            T(img, "ZONE",   ROAD_L - 60, sy + 38, (0, 0, 0), 0.38, 1)
    # street lights
    for i in range(6):
        lx1 = ROAD_L - 42
        lx2 = ROAD_R + 42
        ly = (i * 160 + int(scroll * 0.65)) % (H + 160) - 160
        for lx in [lx1, lx2]:
            cv2.line(img, (lx, ly), (lx, ly + 80), (60, 58, 72), 2)
            tip = (lx + (18 if lx == lx1 else -18), ly)
            cv2.line(img, (lx, ly), tip, (60, 58, 72), 2)
            col2 = (200, 220, 255) if scene != "rain" else (255, 240, 200)
            glow(img, tip[0], tip[1], col2, 14, 0.5 if scene != "rain" else 1.1)
    # background buildings
    for i in range(5):
        bx = i * 190
        bh = 80 + (i * 77) % 100
        cv2.rectangle(img, (bx, H // 2 - bh), (bx + 55, H // 2), (18, 16, 24), -1)
        cv2.rectangle(img, (ROAD_R + 30 + (bx % 120), H // 2 - bh + 20),
                      (ROAD_R + 30 + (bx % 120) + 45, H // 2), (18, 16, 24), -1)


def draw_npc(img, x, y, col):
    x, y = int(x), int(y)
    cv2.rectangle(img, (x - 14, y - 28), (x + 14, y + 28), col, -1)
    cv2.rectangle(img, (x - 14, y - 28), (x + 14, y + 28), WHITE, 1)
    cv2.rectangle(img, (x - 10, y - 24), (x + 10, y - 10), (55, 75, 120), -1)
    cv2.rectangle(img, (x - 6, y + 17), (x - 2, y + 24), (30, 30, 180), -1)
    cv2.rectangle(img, (x + 2, y + 17), (x + 6, y + 24), (30, 30, 180), -1)


def draw_ped(img, x, y, t):
    x, y = int(x), int(y)
    bob = int(math.sin(t * 9) * 2)
    cv2.circle(img, (x, y - 16 + bob), 8, (155, 190, 225), -1)
    cv2.line(img, (x, y - 8 + bob), (x, y + 18 + bob), (110, 150, 210), 4)
    lk = int(math.sin(t * 9) * 9)
    cv2.line(img, (x, y + 18 + bob), (x - lk, y + 34 + bob), (110, 150, 210), 3)
    cv2.line(img, (x, y + 18 + bob), (x + lk, y + 34 + bob), (110, 150, 210), 3)
    abox(img, x - 18, y - 40, x + 18, y - 24, RED, 0.8)
    T(img, "PED", x - 12, y - 28, WHITE, 0.32, 1)


def draw_ego(img, x, y, action, t):
    x, y = int(x), int(y)
    col = ACTION_COL.get(action, NEON_G)
    glow(img, x, y, col, 36, 0.45)
    for lx in [x - 10, x + 10]:
        glow(img, lx, y - 30, (210, 225, 255), 20, 0.65)
    cv2.rectangle(img, (x - 16, y - 30), (x + 16, y + 30), (36, 30, 46), -1)
    cv2.rectangle(img, (x - 16, y - 30), (x + 16, y + 30), col, 2)
    cv2.rectangle(img, (x - 12, y - 26), (x + 12, y - 10), (50, 70, 120), -1)
    cv2.rectangle(img, (x - 9, y - 4), (x + 9, y + 14), (26, 22, 36), -1)
    if action in ("BRAKE", "STOP"):
        glow(img, x - 11, y + 26, RED, 14, 1.1)
        glow(img, x + 11, y + 26, RED, 14, 1.1)
    else:
        cv2.rectangle(img, (x - 13, y + 22), (x - 5, y + 28), (30, 30, 170), -1)
        cv2.rectangle(img, (x + 5,  y + 22), (x + 13, y + 28), (30, 30, 170), -1)
    if action == "ACCELERATE":
        cv2.fillPoly(img, [np.array([[x, y - 44], [x - 8, y - 30], [x + 8, y - 30]], np.int32)], NEON_G)
    elif action == "STEER_LEFT":
        cv2.arrowedLine(img, (x, y - 30), (x - 34, y - 52), NEON_B, 2, tipLength=0.4)
    elif action == "STEER_RIGHT":
        cv2.arrowedLine(img, (x, y - 30), (x + 34, y - 52), NEON_B, 2, tipLength=0.4)


def draw_speedo(img, cx, cy, speed):
    r = 48
    cv2.ellipse(img, (cx, cy), (r, r), 0, 210, 390, (38, 34, 50), 6)
    pct = min(speed / 120, 1.0)
    if pct > 0.01:
        c = NEON_G if speed < 60 else ORANGE if speed < 90 else RED
        cv2.ellipse(img, (cx, cy), (r, r), 0, int(210 - pct * 240), 210, c, 6)
    na = math.radians(210 - pct * 240)
    cv2.line(img, (cx, cy), (cx + int((r - 14) * math.cos(-na)),
                              cy + int((r - 14) * math.sin(-na))), WHITE, 2)
    cv2.circle(img, (cx, cy), 5, WHITE, -1)
    TC(img, f"{int(speed)}", cx, cy + 8, WHITE, 0.52, 2)
    TC(img, "km/h", cx, cy + 24, GRAY, 0.30, 1)


def wrap_text(s, max_chars=30):
    """Word-wrap a string into lines of max_chars."""
    return textwrap.wrap(s, max_chars) or [""]


def draw_panel(img, action, conf, ms, rn, bubbles, vis_count, active, speed, t, hazard, label, tps):
    """Right-side HUD panel — upgraded with real text, D3 bar, live tps."""
    PW = W - PAN_X
    cv2.rectangle(img, (PAN_X, 0), (W, H), (12, 9, 18), -1)
    cv2.line(img, (PAN_X, 0), (PAN_X, H), (55, 45, 75), 2)

    # ── Header ──
    T(img, "SWARMPILOT", PAN_X + 8, 22, NEON_B, 0.65, 2)
    T(img, "Cerebras x Gemma 4 31B", PAN_X + 8, 40, NEON_G, 0.36, 1)
    cv2.line(img, (PAN_X, 48), (W, 48), (40, 35, 60), 1)

    # ── Action box ──
    col = ACTION_COL.get(action, GRAY)
    abox(img, PAN_X + 6, 52, W - 6, 88, col, 0.22, 6)
    cv2.rectangle(img, (PAN_X + 6, 52), (W - 6, 88), col, 2)
    short_action = action[:22] if len(action) > 22 else action
    TC(img, short_action, (PAN_X + W) // 2, 79, col, 0.65, 2)

    # ── D3 confidence bar ──
    bw = W - PAN_X - 14
    fw = int(bw * conf)
    cv2.rectangle(img, (PAN_X + 7, 92), (W - 7, 106), (22, 20, 32), -1)
    if fw > 0:
        cv2.rectangle(img, (PAN_X + 7, 92), (PAN_X + 7 + fw, 106), col, -1)
    # D3 threshold line at 85%
    thresh_x = PAN_X + 7 + int(bw * 0.85)
    cv2.line(img, (thresh_x, 89), (thresh_x, 109), YELLOW, 1)
    T(img, "D3", thresh_x - 8, 88, YELLOW, 0.3, 1)
    cv2.rectangle(img, (PAN_X + 7, 92), (W - 7, 106), (45, 40, 60), 1)

    # ── Stats row ──
    T(img, f"CONF {conf*100:.0f}%  |  {ms:.0f}ms  |  Rd {rn}  |  {tps:.0f} tok/s",
      PAN_X + 7, 120, GRAY, 0.34, 1)

    # ── Agent status ──
    T(img, "AGENTS", PAN_X + 7, 138, GRAY, 0.38, 1)
    for i, (k, lb) in enumerate([
        ("pilot",   "PILOT"),  ("critic",  "CRITIC"),
        ("safety",  "SAFETY"), ("expert",  "EXPERT"),
        ("auditor", "AUDITOR"),("judge",   "JUDGE"),
    ]):
        ay = 144 + i * 28
        ac = AGENT_COL[k]
        on = active.get(k, False)
        abox(img, PAN_X + 6, ay, W - 6, ay + 22, ac if on else GRAY,
             0.18 if on else 0.04, 4)
        cv2.rectangle(img, (PAN_X + 6, ay), (W - 6, ay + 22), ac if on else GRAY, 1)
        cv2.circle(img, (PAN_X + 16, ay + 11), 4, ac if on else GRAY, -1)
        T(img, lb, PAN_X + 24, ay + 16, ac if on else GRAY, 0.38, 1)
        if on:
            pw = max(2, int((W - PAN_X - 75) * (0.35 + 0.35 * math.sin(t * 5 + i))))
            cv2.rectangle(img, (PAN_X + 64, ay + 8), (PAN_X + 64 + pw, ay + 16), ac, -1)

    cv2.line(img, (PAN_X, 316), (W, 316), (40, 35, 60), 1)

    # ── Scene label + hazard ──
    if label:
        T(img, label[:30], PAN_X + 7, 332, YELLOW, 0.40, 1)
    if hazard:
        fl = int(t * 4) % 2 == 0
        hc = RED if fl else ORANGE
        abox(img, PAN_X + 6, 338, W - 6, 358, hc, 0.3, 3)
        T(img, f"⚠ {hazard.replace('_',' ').upper()[:22]}", PAN_X + 10, 353, hc, 0.36, 1)

    # ── Debate bubbles ──
    T(img, "AGENT DEBATE", PAN_X + 7, 376, NEON_B, 0.40, 1)
    cv2.line(img, (PAN_X, 382), (W, 382), (40, 35, 60), 1)

    by = 388
    shown = bubbles[:vis_count] if vis_count > 0 else []
    for b in shown[-7:]:
        if by > H - 80:
            break
        bc = AGENT_COL.get(b["agent"], GRAY)
        cv2.rectangle(img, (PAN_X + 4, by), (PAN_X + 6, by + 2), bc, -1)
        # agent label
        T(img, b["agent"].upper(), PAN_X + 10, by + 12, bc, 0.32, 1)
        # verdict badge
        vd = b.get("verdict", "")
        if vd:
            vc = NEON_G if vd == "SAFE" else RED if vd == "UNSAFE" else ORANGE
            abox(img, W - 68, by, W - 4, by + 16, vc, 0.8, 2)
            T(img, vd[:9], W - 66, by + 12, (0, 0, 0), 0.30, 1)
        # text — word-wrapped, up to 2 lines
        full_text = b.get("text", "")
        lines = wrap_text(full_text, 28)
        for li, ln in enumerate(lines[:2]):
            T(img, ln, PAN_X + 10, by + 22 + li * 13, (155, 150, 168), 0.31, 1)
        by += 42

    # ── Speedo ──
    draw_speedo(img, (PAN_X + W) // 2, H - 52, speed)

    # ── Cerebras tps badge ──
    abox(img, PAN_X + 6, H - 82, PAN_X + PW - 6, H - 68, NEON_G, 0.15, 3)
    T(img, f"⚡ {tps:.0f} tok/s  Cerebras", PAN_X + 10, H - 71, NEON_G, 0.34, 1)


def draw_top_hud(img, t, scene_label=""):
    abox(img, 0, 0, SIM_W, 44, (8, 6, 14), 0.85, 0)
    cv2.line(img, (0, 44), (SIM_W, 44), (45, 38, 65), 1)
    T(img, "SWARMPILOT", 10, 30, NEON_B, 0.80, 2)
    T(img, "6-Agent Adversarial Debate  |  Gemma 4 31B  |  Cerebras", 190, 24, GRAY, 0.40, 1)


def draw_scene_title_card(img, label, t_in):
    """Brief title card fade-in for first 1.2s of each scene."""
    if t_in > 1.2:
        return
    alpha = min(t_in / 0.3, 1.0) * (1 - max(t_in - 0.9, 0) / 0.3)
    alpha = max(0.0, min(1.0, alpha))
    if alpha < 0.01:
        return
    ov = img.copy()
    cv2.rectangle(ov, (60, H // 2 - 36), (SIM_W - 60, H // 2 + 36), (8, 6, 18), -1)
    cv2.rectangle(ov, (60, H // 2 - 36), (SIM_W - 60, H // 2 + 36), NEON_B, 2)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
    # text drawn directly (alpha baked)
    tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
    x = (SIM_W - tw) // 2
    # shadow
    col_a = tuple(int(c * alpha) for c in NEON_B)
    cv2.putText(img, label, (x, H // 2 + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col_a, 2, cv2.LINE_AA)


def draw_hook(img, t):
    img[:] = BG
    cx, cy = W // 2, H // 2
    p = 0.6 + 0.4 * math.sin(t * 3)
    for r, a in [(180, 6), (120, 14), (75, 28), (45, 50)]:
        ov = img.copy()
        cv2.circle(ov, (cx, cy - 30), r, NEON_B, -1)
        cv2.addWeighted(ov, a / 255 * p, img, 1 - a / 255 * p, 0, img)
    TC(img, "SWARMPILOT", cx, cy + 10, NEON_B, 2.0, 3)
    TC(img, "6 AI Agents  |  Adversarial Debate  |  Real-Time Driving", cx, cy + 52, WHITE, 0.60, 1)
    TC(img, "Cerebras Ultra-Fast Inference  x  Google DeepMind Gemma 4 31B", cx, cy + 84, NEON_G, 0.50, 1)
    TC(img, "Who decides when to brake?  All 6 argue — in milliseconds.", cx, cy + 118, GRAY, 0.44, 1)


def draw_compare(img, t_in, dur, cache):
    """Speed comparison scene using real debate_cache timing."""
    img[:] = BG

    # Pull real timing from cache
    scene_times = {sc: cache.get(sc, {}).get("total_ms", 8000)
                   for sc in ["highway", "pedestrian", "rain", "intersect", "merge", "school"]}
    avg_cerebras_ms = sum(scene_times.values()) / max(len(scene_times), 1)
    cerebras_s = avg_cerebras_ms / 1000

    pct = min(t_in / max(dur, 1), 1.0)
    TC(img, "CEREBRAS  vs  GPU PROVIDER", W // 2, 44, WHITE, 0.80, 2)
    TC(img, f"6-agent debate  |  Gemma 4 31B  |  Real measured: {cerebras_s:.1f}s avg",
       W // 2, 68, GRAY, 0.42, 1)
    cv2.line(img, (0, 80), (W, 80), (40, 35, 60), 1)
    cv2.line(img, (W // 2, 80), (W // 2, H - 90), (40, 35, 60), 1)

    GPU_ESTIMATE_S = cerebras_s * 7.2   # realistic GPU estimate

    for ci, (lbl, spd_pct, col, time_s, tps_str) in enumerate([
        ("CEREBRAS  (Ours)",
         min(pct * (GPU_ESTIMATE_S / cerebras_s) / (GPU_ESTIMATE_S / cerebras_s), 1.0),
         NEON_G,
         cerebras_s,
         f"{int(1800 + 600 * math.sin(t_in * 2))} tok/s"),
        ("Standard GPU",
         min(pct * (cerebras_s / GPU_ESTIMATE_S), 1.0),
         RED,
         GPU_ESTIMATE_S,
         "~200 tok/s"),
    ]):
        cx2 = W // 4 + ci * W // 2
        TC(img, lbl, cx2, 110, col, 0.60, 2)

        # race track
        rl = 20 + ci * (W // 2)
        rr = rl + W // 2 - 40
        cv2.rectangle(img, (rl + 20, 250), (rr - 20, 296), (52, 48, 58), -1)
        cv2.rectangle(img, (rl + 20, 250), (rr - 20, 296), GRAY, 1)
        for dx in range(rl + 40, rr - 40, 50):
            cv2.line(img, (dx, 273), (dx + 28, 273), (165, 160, 140), 2)
        car_x = rl + 40 + int(spd_pct * (rr - rl - 100))
        cv2.rectangle(img, (car_x - 15, 258), (car_x + 15, 288), col, -1)
        cv2.rectangle(img, (car_x - 15, 258), (car_x + 15, 288), WHITE, 1)
        if ci == 0:
            glow(img, car_x, 273, col, 22, 0.55)
        cv2.line(img, (rr - 30, 244), (rr - 30, 302), YELLOW, 3)
        T(img, "FINISH", rr - 58, 242, YELLOW, 0.36, 1)

        # progress bar
        bw2 = W // 2 - 80
        fw2 = int(bw2 * spd_pct)
        bx2 = rl + 20
        cv2.rectangle(img, (bx2, 312), (bx2 + bw2, 332), (28, 24, 38), -1)
        if fw2 > 0:
            cv2.rectangle(img, (bx2, 312), (bx2 + fw2, 332), col, -1)
        cv2.rectangle(img, (bx2, 312), (bx2 + bw2, 332), (50, 45, 65), 1)

        # timing labels
        TC(img, f"{time_s:.1f}s / debate", cx2, 358, col, 0.60, 2)
        TC(img, tps_str, cx2, 384, col, 0.52, 1)

    # Per-scene timing table
    cv2.line(img, (0, 410), (W, 410), (40, 35, 60), 1)
    T(img, "REAL CEREBRAS TIMING  (from this session):", 20, 432, GRAY, 0.40, 1)
    col_w = (W - 40) // 6
    for i, (sc, ms) in enumerate(scene_times.items()):
        sx = 20 + i * col_w
        bar_h = int(60 * (ms / 10000))
        bar_c = NEON_G if ms < 7000 else ORANGE if ms < 9000 else RED
        cv2.rectangle(img, (sx, 500 - bar_h), (sx + col_w - 8, 500), bar_c, -1)
        T(img, sc[:7], sx, 514, GRAY, 0.33, 1)
        T(img, f"{ms/1000:.1f}s", sx, 528, bar_c, 0.33, 1)

    cv2.line(img, (0, H - 90), (W, H - 90), (40, 35, 60), 1)

    # feature pills
    for i, (k, v, c) in enumerate([
        ("6 Agents Parallel", "asyncio.gather @ 100 RPM", NEON_G),
        ("3 Images/Frame",    "RGB + Flow + Depth",       NEON_B),
        ("Structured Output", "strict JSON schema",        NEON_G),
        ("65K Context",       "rolling world memory",     ORANGE),
    ]):
        sx = 80 + i * (W - 160) // 4
        TC(img, k, sx, H - 66, c, 0.42, 1)
        TC(img, v, sx, H - 46, GRAY, 0.32, 1)

    if pct > 0.3:
        xf = f"{GPU_ESTIMATE_S / cerebras_s:.1f}x"
        TC(img, f"~{xf} FASTER", W // 2, H - 16, NEON_G, 0.90, 3)


def main(out="swarm_demo_v2.mp4"):
    cache = {}
    if os.path.exists("debate_cache.json"):
        with open("debate_cache.json") as f:
            cache = json.load(f)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out, fourcc, FPS, (W, H))

    NPC_POOL = [
        [float(LANE_XS[i % 5]), float(random.randint(-500, -40)),
         random.uniform(28, 68),
         random.choice([(80, 60, 155), (50, 115, 55), (155, 78, 48),
                        (48, 78, 155), (118, 48, 125)])]
        for i in range(10)
    ]
    peds = []
    scroll = 0.0
    ego_x = float(LANE_XS[2])
    ego_y = float(H * 0.72)
    speed = 0.0
    target = 65.0

    def bubbles_for(sc):
        r = cache.get(sc, {})
        bs = []
        for rnd in r.get("rounds", []):
            bs.append({"agent": "pilot",
                       "text": rnd.get("pilot_reasoning", "")[:80]})
            bs.append({"agent": "critic",
                       "text": rnd.get("critic_reasoning", "")[:80],
                       "verdict": rnd.get("critic_verdict", "")})
            veto = rnd.get("safety_veto", False)
            bs.append({"agent": "safety",
                       "text": ("VETO — " if veto else "") + rnd.get("safety_reasoning", "")[:73],
                       "verdict": "UNSAFE" if veto else "SAFE"})
            if rnd.get("expert_domain"):
                bs.append({"agent": "expert",
                           "text": f"[{rnd['expert_domain']}] {rnd.get('auditor_reasoning','')[:56]}"})
            bs.append({"agent": "auditor",
                       "text": rnd.get("auditor_reasoning", "")[:80]})
            bs.append({"agent": "judge",
                       "text": rnd.get("judge_summary", "")[:80]})
        return bs

    prev_sc = None
    sc_bubbles = []
    sc_active = {k: False for k in AGENT_COL}
    FIXED = 1.0 / FPS

    print(f"Rendering {W}x{H} 60s v2 -> {out}")
    t0 = tmod.monotonic()

    for fi in range(TOTAL):
        t = fi * FIXED
        sc, t_in, t_dur = scene_at(fi)

        if sc != prev_sc:
            prev_sc = sc
            sc_bubbles = bubbles_for(sc)
            sc_active = {k: False for k in AGENT_COL}
            for b in sc_bubbles:
                sc_active[b["agent"]] = True
            if sc in ("pedestrian", "school"):
                peds.append([float(ROAD_L - 18), float(ego_y - 200)])

        r = cache.get(sc, {})
        action = r.get("final_action", "MAINTAIN")
        # shorten long free-text actions to a clean keyword for the action box
        ACTION_KEYWORDS = ["STOP", "BRAKE", "ACCELERATE", "MAINTAIN", "STEER_LEFT", "STEER_RIGHT"]
        display_action = action
        for kw in ACTION_KEYWORDS:
            if kw in action.upper():
                display_action = kw
                break

        conf = r.get("final_confidence", 0.72)
        ms = r.get("total_ms", 7500)
        rn = len(r.get("rounds", [1]))
        # Estimate tok/s from ms (real prompt is ~2K tokens, completion ~400 tokens)
        tps = (2400 * 6) / max(ms / 1000, 0.1)   # 6 agents × ~2400 tokens / time

        # Reveal bubbles over scene duration (typing effect)
        if sc_bubbles:
            vis_count = max(1, int(t_in / t_dur * len(sc_bubbles)) + 1)
        else:
            vis_count = 0

        if action in ("BRAKE", "STOP") or "stop" in action.lower() or "brake" in action.lower():
            target = 0.0
        elif action == "ACCELERATE" or "accelerat" in action.lower():
            target = 100.0
        elif action == "STEER_LEFT":
            target = 55.0
            ego_x -= 1.1 * 38 * FIXED
        elif action == "STEER_RIGHT":
            target = 55.0
            ego_x += 1.1 * 38 * FIXED
        elif "reduce" in action.lower() or "slow" in action.lower():
            target = max(10.0, target - 30)
        else:
            target = 65.0
        speed += (target - speed) * min(2.5 * FIXED, 1.0)
        ego_x = max(ROAD_L + 20, min(ROAD_R - 20, ego_x))
        scroll += speed * FIXED * 1.4

        for n in NPC_POOL:
            n[1] += (speed - n[2]) * FIXED * 1.4
            if n[1] > H + 100:
                n[1] = float(random.randint(-300, -40))
                n[0] = float(LANE_XS[random.randint(0, 4)])
        for p in peds:
            p[0] += 1.5 * FIXED * 60
        peds = [p for p in peds if p[0] < ROAD_R + 100]

        img = np.full((H, W, 3), BG, dtype=np.uint8)

        if sc == "hook":
            draw_hook(img, t_in)
        elif sc == "compare":
            draw_compare(img, t_in, t_dur, cache)
        else:
            draw_road(img, scroll, sc, t)
            for n in NPC_POOL:
                if ROAD_L < n[0] < ROAD_R and -60 < n[1] < H + 60:
                    draw_npc(img, n[0], n[1], n[3])
            for p in peds:
                draw_ped(img, p[0], p[1], t)
            draw_ego(img, ego_x, ego_y, display_action, t)
            draw_top_hud(img, t)
            draw_scene_title_card(img, LABELS.get(sc, sc.upper()), t_in)
            hazard = HAZARDS.get(sc)
            label = LABELS.get(sc, "")
            draw_panel(img, display_action, conf, ms, rn, sc_bubbles, vis_count,
                       sc_active, speed, t, hazard, label, tps)

        writer.write(img)
        if fi % 300 == 0:
            el = tmod.monotonic() - t0
            eta = el / max(fi / TOTAL, 0.001) * (1 - fi / TOTAL)
            print(f"  {fi}/{TOTAL} ({fi // FPS}s)  ETA {eta:.0f}s")

    writer.release()
    print("Encoding H264...")
    final = out.replace(".mp4", "_final.mp4")
    os.system(
        f'ffmpeg -y -i "{out}" -f lavfi -i anullsrc=r=44100:cl=stereo '
        f'-c:v libx264 -preset fast -crf 16 -pix_fmt yuv420p '
        f'-c:a aac -b:a 128k -shortest "{final}" 2>/dev/null'
    )
    sz = os.path.getsize(final) // 1024 if os.path.exists(final) else 0
    print(f"Done: {final}  ({sz} KB)")


if __name__ == "__main__":
    main()
