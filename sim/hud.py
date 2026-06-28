"""HUD: debate panel, speedometer arc, agent pills, action badge."""
import pygame, math
from sim.cfg import *

_font_sm  = None
_font_med = None
_font_lg  = None
_font_xl  = None

def _fonts():
    global _font_sm, _font_med, _font_lg, _font_xl
    if _font_sm is None:
        _font_sm  = pygame.font.Font(None, 18)
        _font_med = pygame.font.Font(None, 22)
        _font_lg  = pygame.font.Font(None, 28)
        _font_xl  = pygame.font.Font(None, 44)


def draw_rounded_rect(surf, color, rect, r=8, alpha=255):
    s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
    pygame.draw.rect(s, (*color, alpha), (0,0,rect[2],rect[3]), border_radius=r)
    surf.blit(s, (rect[0], rect[1]))


def speedometer(surf, cx, cy, speed, max_speed=120):
    """Arc gauge."""
    r = 52
    # Background arc
    pygame.draw.arc(surf, (40,40,55),
                    (cx-r, cy-r, r*2, r*2), math.radians(210), math.radians(510), 6)
    # Speed arc
    pct = min(speed / max_speed, 1.0)
    end_a = math.radians(210 - pct * 240)
    col = NEON if speed < 60 else ORANGE if speed < 90 else RED
    if pct > 0.01:
        pygame.draw.arc(surf, col,
                        (cx-r, cy-r, r*2, r*2), end_a, math.radians(210), 6)
    # Needle
    needle_deg = 210 - pct * 240
    nx = cx + int((r-14) * math.cos(math.radians(needle_deg)))
    ny = cy - int((r-14) * math.sin(math.radians(needle_deg)))
    pygame.draw.line(surf, WHITE, (cx, cy), (nx, ny), 2)
    pygame.draw.circle(surf, GRAY, (cx, cy), 5)
    # Text
    _fonts()
    surf.blit(_font_lg.render(f"{int(speed)}", True, WHITE), (cx-18, cy+6))
    surf.blit(_font_sm.render("km/h", True, GRAY), (cx-16, cy+28))


def agent_pill(surf, x, y, key, label, active, thinking):
    """One agent status row."""
    _fonts()
    col = AGENT_COL[key]
    alpha = 255 if active else 60
    # Dot
    glow_s = pygame.Surface((20,20), pygame.SRCALPHA)
    a2 = alpha
    pygame.draw.circle(glow_s, (*col, a2), (10,10), 7)
    surf.blit(glow_s, (x, y+3))
    # Label
    surf.blit(_font_sm.render(label, True, (*col, alpha) if alpha < 255
              else col), (x+20, y+4))
    # Thinking animation
    if thinking:
        dots = "•" * (int(pygame.time.get_ticks()/300) % 4)
        surf.blit(_font_sm.render(dots, True, col), (x+115, y+4))


def debate_panel(surf, x, y, w, h, bubbles: list):
    """Scrolling debate log panel."""
    _fonts()
    draw_rounded_rect(surf, DARK_PANEL, (x, y, w, h), 10, 210)
    pygame.draw.rect(surf, (40,40,70), (x, y, w, h), 1, border_radius=10)

    surf.blit(_font_med.render("AGENT DEBATE", True, CYAN), (x+10, y+8))
    pygame.draw.line(surf, (40,40,70), (x, y+28), (x+w, y+28), 1)

    by = y + 36
    for b in bubbles[-10:]:
        if by > y+h-30:
            break
        col = AGENT_COL.get(b["agent"], GRAY)
        # Left stripe
        pygame.draw.rect(surf, col, (x+4, by, 3, 42))
        # Agent name
        surf.blit(_font_sm.render(b["agent"].upper(), True, col), (x+12, by+2))
        # Verdict badge
        if b.get("verdict"):
            vc = NEON if b["verdict"]=="SAFE" else RED if b["verdict"]=="UNSAFE" else ORANGE
            draw_rounded_rect(surf, vc, (x+w-62, by+1, 56, 17), 4, 200)
            surf.blit(_font_sm.render(b["verdict"], True, (0,0,0)), (x+w-58, by+3))
        # Text lines
        txt = b.get("text","")[:54]
        lines = [txt[i:i+27] for i in range(0,len(txt),27)]
        for li, line in enumerate(lines[:2]):
            surf.blit(_font_sm.render(line, True, (170,170,185)), (x+12, by+16+li*13))
        by += 46


def action_badge(surf, cx, y, action):
    _fonts()
    col = ACTION_COL.get(action, GRAY)
    w2 = 180
    draw_rounded_rect(surf, (15,15,25), (cx-w2//2, y, w2, 40), 10, 220)
    pygame.draw.rect(surf, col, (cx-w2//2, y, w2, 40), 2, border_radius=10)
    txt = _font_xl.render(action, True, col)
    surf.blit(txt, (cx - txt.get_width()//2, y+4))


def top_bar(surf, scene, hazard, t, fps):
    _fonts()
    draw_rounded_rect(surf, DARK_PANEL, (0, 0, W, 46), 0, 230)
    pygame.draw.line(surf, (40,40,70), (0,46), (W,46), 1)
    surf.blit(_font_lg.render("⚡ SWARMPILOT", True, CYAN), (10, 10))
    surf.blit(_font_sm.render("6-Agent Adversarial Driving · Cerebras × Gemma 4 31B",
                              True, GRAY), (190, 16))
    # FPS
    surf.blit(_font_sm.render(f"{fps:.0f}fps", True, NEON), (W-70, 16))
    # Hazard flash
    if hazard:
        flash = int(t*4)%2==0
        hcol = RED if flash else ORANGE
        draw_rounded_rect(surf, (25,8,8), (W-260, 5, 180, 36), 6, 200)
        surf.blit(_font_med.render(f"⚠ {hazard.replace('_',' ').upper()[:18]}",
                                   True, hcol), (W-252, 14))
