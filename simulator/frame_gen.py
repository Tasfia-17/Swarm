"""
Synthetic driving scene generator — no CARLA required.
Produces RGB frames that look like a forward-facing dashcam.
"""
import base64
import random
import time
import io
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

SCENES = [
    {"name": "highway_clear",    "hazard": None,              "speed": 80},
    {"name": "urban_pedestrian", "hazard": "pedestrian_crossing", "speed": 40},
    {"name": "rainy_night",      "hazard": "low_visibility",  "speed": 30},
    {"name": "intersection",     "hazard": "red_light",       "speed": 0},
    {"name": "highway_merge",    "hazard": "vehicle_merging", "speed": 65},
    {"name": "school_zone",      "hazard": "children",        "speed": 20},
]

_prev_frame: np.ndarray | None = None


def _draw_scene(scene: dict, frame_idx: int) -> np.ndarray:
    w, h = 640, 480
    img = Image.new("RGB", (w, h), (135, 206, 235))  # sky blue
    draw = ImageDraw.Draw(img)

    # Road
    draw.rectangle([0, h // 2, w, h], fill=(80, 80, 80))
    # Lane markings
    for x in range(0, w, 60):
        offset = (frame_idx * 4) % 60
        draw.rectangle([x - offset, h // 2 + 20, x - offset + 30, h // 2 + 30], fill=(255, 255, 255))

    # Horizon line / vanishing point buildings
    for i in range(5):
        bx = i * 130 - 20
        draw.rectangle([bx, h // 2 - 80, bx + 60, h // 2], fill=(150 + i * 10, 140, 130))

    # Hazard rendering
    hazard = scene.get("hazard")
    if hazard == "pedestrian_crossing":
        # Pedestrian stripes
        for i in range(6):
            draw.rectangle([200 + i * 20, h // 2, 215 + i * 20, h // 2 + 60], fill=(255, 255, 255))
        # Walking person
        px = 310 + (frame_idx % 20)
        draw.ellipse([px, h // 2 + 5, px + 20, h // 2 + 25], fill=(255, 200, 150))  # head
        draw.rectangle([px + 5, h // 2 + 25, px + 15, h // 2 + 55], fill=(0, 100, 200))  # body
    elif hazard == "low_visibility":
        overlay = Image.new("RGBA", (w, h), (200, 200, 220, 160))
        img = img.convert("RGBA")
        img = Image.alpha_composite(img, overlay)
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
    elif hazard == "red_light":
        draw.ellipse([300, h // 2 - 120, 340, h // 2 - 80], fill=(255, 0, 0))
    elif hazard == "vehicle_merging":
        vx = w - 100 - (frame_idx % 30) * 2
        draw.rectangle([vx, h // 2 + 10, vx + 80, h // 2 + 50], fill=(200, 50, 50))
    elif hazard == "children":
        for i in range(3):
            cx = 200 + i * 60 + (frame_idx % 10)
            draw.ellipse([cx, h // 2 + 5, cx + 16, h // 2 + 21], fill=(255, 220, 180))
            draw.rectangle([cx + 3, h // 2 + 21, cx + 13, h // 2 + 40], fill=(100, 200, 100))

    # HUD
    draw.rectangle([0, 0, w, 30], fill=(0, 0, 0))
    draw.text((10, 5), f"Scene: {scene['name']}  |  Speed: {scene['speed']} km/h  |  Frame: {frame_idx}", fill=(0, 255, 0))
    if hazard:
        draw.text((10, 455), f"⚠ HAZARD: {hazard.upper()}", fill=(255, 50, 50))

    return np.array(img)


def get_frame(scene_name: str | None = None, frame_idx: int = 0) -> tuple[np.ndarray, dict]:
    global _prev_frame
    scene = next((s for s in SCENES if s["name"] == scene_name), None) or random.choice(SCENES)
    frame = _draw_scene(scene, frame_idx)
    result = (frame, scene)
    _prev_frame = frame
    return result


def get_prev_frame() -> np.ndarray | None:
    return _prev_frame


def frame_to_b64(frame: np.ndarray, fmt: str = "jpeg") -> str:
    _, buf = cv2.imencode(f".{fmt}", frame)
    return base64.b64encode(buf).decode()


def list_scenes() -> list[str]:
    return [s["name"] for s in SCENES]
