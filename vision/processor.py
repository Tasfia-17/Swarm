"""
Vision processor: takes an RGB frame, produces three base64 images:
  1. RGB (original)
  2. Optical flow heatmap (Farneback)
  3. Depth heatmap (gradient-based proxy — no GPU/MiDaS required)
"""
import base64
import numpy as np
import cv2

_prev_gray: np.ndarray | None = None


def _optical_flow_heatmap(curr: np.ndarray, prev: np.ndarray | None) -> np.ndarray:
    gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    if prev is None:
        prev_gray = gray
    else:
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(curr)
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _depth_heatmap(frame: np.ndarray) -> np.ndarray:
    """Proxy depth via Laplacian gradient magnitude — near objects have high gradients."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = np.abs(lap)
    lap = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Invert: high gradient = close = warm color
    lap = 255 - lap
    depth_color = cv2.applyColorMap(lap, cv2.COLORMAP_INFERNO)
    return depth_color


def process(frame_rgb: np.ndarray, prev_frame: np.ndarray | None = None) -> dict[str, str]:
    """Returns dict with keys rgb, flow, depth — each a base64 JPEG string."""
    # OpenCV uses BGR internally
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    flow_img = _optical_flow_heatmap(frame_bgr, prev_frame)
    depth_img = _depth_heatmap(frame_bgr)

    def _enc(img: np.ndarray) -> str:
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(buf).decode()

    return {
        "rgb":   _enc(frame_bgr),
        "flow":  _enc(flow_img),
        "depth": _enc(depth_img),
    }
