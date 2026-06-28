"""
World renderer: scrolling road, ego car, NPC vehicles,
pedestrians, tire marks, neon glow, particles.
"""
import pygame, math, random
from sim.cfg import *

# ── helpers ──────────────────────────────────────────────────────────

def glow(surf, color, pos, r, alpha=80):
    for dr in [r+12, r+6, r]:
        s = pygame.Surface((dr*2+4, dr*2+4), pygame.SRCALPHA)
        a = alpha if dr == r else alpha // 3
        pygame.draw.circle(s, (*color, a), (dr+2, dr+2), dr)
        surf.blit(s, (pos[0]-dr-2, pos[1]-dr-2), special_flags=pygame.BLEND_ADD)

def draw_rounded_rect(surf, color, rect, radius=10, alpha=255):
    s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
    pygame.draw.rect(s, (*color, alpha), (0,0,rect[2],rect[3]), border_radius=radius)
    surf.blit(s, (rect[0], rect[1]))

# ── Tire marks ───────────────────────────────────────────────────────

class TireMarks:
    def __init__(self):
        self.surf = pygame.Surface((W, H*4), pygame.SRCALPHA)

    def add(self, x, y, scroll_y):
        wy = int(y + scroll_y)
        if 0 < wy < H*4:
            pygame.draw.circle(self.surf, (60,60,60,120), (int(x-8), wy), 3)
            pygame.draw.circle(self.surf, (60,60,60,120), (int(x+8), wy), 3)

    def draw(self, surf, scroll_y):
        surf.blit(self.surf, (0, -scroll_y % (H*4)))

# ── Particle system ──────────────────────────────────────────────────

class Particle:
    __slots__ = ('x','y','vx','vy','life','color','r')
    def __init__(self, x, y, color):
        self.x, self.y = x, y
        ang = random.uniform(0, math.pi*2)
        spd = random.uniform(1, 5)
        self.vx, self.vy = math.cos(ang)*spd, math.sin(ang)*spd
        self.life = random.randint(20, 50)
        self.color = color
        self.r = random.randint(2, 5)

class Particles:
    def __init__(self):
        self.ps: list[Particle] = []

    def emit(self, x, y, color, n=8):
        for _ in range(n):
            self.ps.append(Particle(x, y, color))

    def update_draw(self, surf, dt):
        alive = []
        for p in self.ps:
            p.x += p.vx; p.y += p.vy
            p.vy += 0.1
            p.life -= 1
            if p.life > 0:
                a = int(255 * p.life / 50)
                s = pygame.Surface((p.r*2, p.r*2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*p.color, a), (p.r, p.r), p.r)
                surf.blit(s, (int(p.x-p.r), int(p.y-p.r)), special_flags=pygame.BLEND_ADD)
                alive.append(p)
        self.ps = alive

# ── NPC vehicles ─────────────────────────────────────────────────────

class NPC:
    def __init__(self, lane, y, speed, color=None):
        self.lane = lane
        self.x = LANE_XS[lane % NUM_LANES]
        self.y = y
        self.speed = speed
        self.color = color or random.choice([(180,60,60),(60,100,200),(200,160,40),(80,160,80)])
        self.w, self.h = 28, 52

    def update(self, dt, ego_speed):
        self.y += (ego_speed - self.speed) * dt * 60

    def draw(self, surf):
        r = pygame.Rect(self.x - self.w//2, self.y - self.h//2, self.w, self.h)
        pygame.draw.rect(surf, self.color, r, border_radius=5)
        pygame.draw.rect(surf, WHITE, r, 1, border_radius=5)
        # taillights
        pygame.draw.rect(surf, RED, (r.left+3, r.bottom-8, 8, 5), border_radius=2)
        pygame.draw.rect(surf, RED, (r.right-11, r.bottom-8, 8, 5), border_radius=2)

# ── Pedestrian ───────────────────────────────────────────────────────

class Pedestrian:
    def __init__(self, y):
        self.x = ROAD_LEFT - 10
        self.y = y
        self.speed = random.uniform(0.8, 1.6)
        self.phase = random.uniform(0, math.pi*2)

    def update(self, dt):
        self.x += self.speed * dt * 60

    def draw(self, surf, t):
        bx, by = int(self.x), int(self.y)
        # head
        pygame.draw.circle(surf, (230, 190, 160), (bx, by), 9)
        # body bob
        by2 = by + 9 + int(math.sin(t*8 + self.phase)*2)
        pygame.draw.line(surf, (160, 180, 230), (bx, by+9), (bx, by+30), 4)
        # legs
        lk = int(math.sin(t*8 + self.phase) * 10)
        pygame.draw.line(surf, (160, 180, 230), (bx, by+30), (bx-lk, by+46), 3)
        pygame.draw.line(surf, (160, 180, 230), (bx, by+30), (bx+lk, by+46), 3)
        # warning box
        draw_rounded_rect(surf, RED, (bx-18, by-28, 36, 16), 4, 180)
        font = pygame.font.Font(None, 16)
        surf.blit(font.render("PED", True, WHITE), (bx-11, by-26))

# ── Road ─────────────────────────────────────────────────────────────

def draw_road(surf, scroll_y, scene):
    # Shoulders
    surf.fill(BG)
    pygame.draw.rect(surf, CURB,  (ROAD_LEFT-20, 0, 20, H))
    pygame.draw.rect(surf, CURB,  (ROAD_RIGHT,   0, 20, H))
    pygame.draw.rect(surf, ROAD,  (ROAD_LEFT, 0, ROAD_RIGHT-ROAD_LEFT, H))

    # Dashed center lines
    offset = int(scroll_y) % 60
    for lx in LANE_XS[1:-1]:
        for y in range(-offset, H+60, 60):
            pygame.draw.rect(surf, (*LANE_MARK, 180),
                             (lx-2, y, 4, 35))

    # Edge lines
    pygame.draw.line(surf, LANE_MARK, (LANE_XS[0]-40, 0), (LANE_XS[0]-40, H), 3)
    pygame.draw.line(surf, LANE_MARK, (LANE_XS[-1]+40, 0), (LANE_XS[-1]+40, H), 3)

    # Scene-specific road decoration
    if scene == "intersection":
        iy = H//2
        # Cross road
        pygame.draw.rect(surf, ROAD, (0, iy-80, W, 160))
        # Stop line
        pygame.draw.line(surf, WHITE, (ROAD_LEFT, iy-80), (ROAD_RIGHT, iy-80), 4)
        # Crosswalk
        for i in range(8):
            cx2 = ROAD_LEFT + 20 + i*34
            pygame.draw.rect(surf, (210,210,210), (cx2, iy-75, 22, 50))

    elif scene == "school_zone":
        # Yellow road markings
        for lx in LANE_XS[1:-1]:
            for y in range(-offset, H+60, 60):
                pygame.draw.rect(surf, YELLOW, (lx-2, y, 4, 35))
        # School zone sign
        font = pygame.font.Font(None, 18)
        for sy in [150, 400, 600]:
            pygame.draw.rect(surf, YELLOW, (ROAD_LEFT-70, sy, 65, 40), border_radius=4)
            surf.blit(font.render("SCHOOL", True, (0,0,0)), (ROAD_LEFT-68, sy+5))
            surf.blit(font.render("ZONE",   True, (0,0,0)), (ROAD_LEFT-60, sy+20))

    elif scene == "rainy_night":
        # Wet road sheen
        for y in range(0, H, 4):
            alpha = int(20 + 10*math.sin(y*0.05 + scroll_y*0.01))
            s2 = pygame.Surface((ROAD_RIGHT-ROAD_LEFT, 2), pygame.SRCALPHA)
            s2.fill((100, 120, 160, alpha))
            surf.blit(s2, (ROAD_LEFT, y))

# ── Ego car ──────────────────────────────────────────────────────────

def draw_ego(surf, x, y, action, t, particles):
    col = ACTION_COL.get(action, NEON)
    w2, h2 = 30, 56

    # Headlight glow
    for lx in [x-10, x+10]:
        glow(surf, (200, 220, 255), (int(lx), int(y-h2//2)), 18, 60)

    # Body
    body = pygame.Rect(x-w2//2, y-h2//2, w2, h2)
    pygame.draw.rect(surf, (40, 45, 60), body, border_radius=6)
    pygame.draw.rect(surf, col, body, 2, border_radius=6)

    # Windshield
    pygame.draw.rect(surf, (60, 80, 120, 180),
                     (x-w2//2+4, y-h2//2+6, w2-8, 16), border_radius=3)

    # Neon underglow
    glow(surf, col, (int(x), int(y)), 24, 35)

    # Brake lights
    if action in ("BRAKE", "STOP"):
        glow(surf, RED, (int(x-10), int(y+h2//2-6)), 10, 120)
        glow(surf, RED, (int(x+10), int(y+h2//2-6)), 10, 120)

    # Tire marks when braking
    if action in ("BRAKE", "STOP") and random.random() < 0.4:
        particles.emit(x-10, y+h2//2, (60, 60, 60))
        particles.emit(x+10, y+h2//2, (60, 60, 60))

    # Speed arrow
    if action == "ACCELERATE":
        pygame.draw.polygon(surf, NEON, [
            (x, y-h2//2-18), (x-8, y-h2//2-4), (x+8, y-h2//2-4)])

    # SWARM label on car
    font = pygame.font.SysFont("monospace", 9, bold=True)
    surf.blit(font.render("AI", True, col), (x-6, y-4))
