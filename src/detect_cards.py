"""Card detection - tries multiple methods, always returns a result."""

import cv2
import numpy as np
from typing import List, Dict


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load: {path}")
    return img


def order_corners(pts):
    pts = pts.reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    return np.array([pts[np.argmin(s)], pts[np.argmin(diff)], 
                     pts[np.argmax(s)], pts[np.argmax(diff)]], dtype=np.float32)


def find_card_contour(gray, method='otsu'):
    h, w = gray.shape
    
    if method == 'otsu':
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(gray[thresh == 255]) < np.mean(gray[thresh == 0]):
            thresh = cv2.bitwise_not(thresh)
    elif method == 'adaptive':
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 51, 5)
    elif method == 'canny':
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        thresh = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=2)
    else:
        return None
    
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = h * w * 0.05
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if rw > 0 and rh > 0:
            if rw > rh: rw, rh = rh, rw
            aspect = rw / rh
            if 0.4 < aspect < 0.95:
                box = cv2.boxPoints(rect)
                return order_corners(box), area
    return None


def detect_cards(image: np.ndarray) -> List[Dict]:
    """Find card using multiple methods. Always returns at least one result."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Try multiple detection methods
    for method in ['otsu', 'adaptive', 'canny']:
        result = find_card_contour(gray, method)
        if result:
            corners, area = result
            return [{"corners": corners, "area": area}]
    
    # Fallback: assume card fills most of image, use center crop
    margin_w = int(w * 0.05)
    margin_h = int(h * 0.05)
    corners = np.array([
        [margin_w, margin_h],
        [w - margin_w, margin_h],
        [w - margin_w, h - margin_h],
        [margin_w, h - margin_h]
    ], dtype=np.float32)
    
    return [{"corners": corners, "area": h * w * 0.9}]