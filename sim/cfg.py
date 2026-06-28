"""Simulation constants."""
W, H = 1280, 720
FPS  = 60
RECORD_FPS = 30   # output video fps (every 2nd frame)

# Colors (RGB for pygame)
BG         = (10,  10,  15)
ROAD       = (45,  45,  55)
LANE_MARK  = (200, 200, 180)
CURB       = (80,  80,  95)
NEON       = (0,   255, 136)
CYAN       = (0,   191, 255)
ORANGE     = (255, 140, 30)
RED        = (220, 40,  40)
WHITE      = (255, 255, 255)
GRAY       = (100, 100, 115)
PURPLE     = (180, 80,  220)
YELLOW     = (255, 210, 0)
DARK_PANEL = (12,  12,  22)

AGENT_COL = {
    "pilot":   (255, 160, 50),
    "critic":  (80,  120, 255),
    "safety":  (60,  210, 90),
    "expert":  (220, 180, 40),
    "auditor": (180, 70,  230),
    "judge":   (40,  220, 220),
}

ACTION_COL = {
    "BRAKE":       (220, 40, 40),
    "STOP":        (220, 40, 40),
    "ACCELERATE":  (60,  210, 90),
    "MAINTAIN":    (0,   255, 136),
    "STEER_LEFT":  (0,   191, 255),
    "STEER_RIGHT": (0,   191, 255),
}

# Road layout
ROAD_LEFT  = 280
ROAD_RIGHT = 780
LANE_XS    = [350, 460, 570, 680]   # lane centers
NUM_LANES  = 4
