"""Smart corner extraction - checks all 4 orientations."""

import cv2
import numpy as np
from typing import Tuple, List

RANK_SIZE = (45, 60)
SUIT_SIZE = (35, 35)


def get_all_corners(warped: np.ndarray) -> List[np.ndarray]:
    """Return warped image in all 4 rotations."""
    return [
        warped,
        cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(warped, cv2.ROTATE_180),
        cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]


def find_rank_suit_in_corner(gray: np.ndarray):
    """Find rank and suit contours in top-left corner of image."""
    h, w = gray.shape
    corner = gray[0:int(h*0.4), 0:int(w*0.4)]
    
    _, binary = cv2.threshold(corner, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = 150
    max_area = corner.shape[0] * corner.shape[1] * 0.4
    
    boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if min_area < area < max_area:
            boxes.append((y, x, bw, bh))
    
    boxes.sort(key=lambda b: b[0])  # Sort by Y
    return boxes, corner


def extract_rank_region(gray: np.ndarray) -> np.ndarray:
    """Extract rank from top-left corner."""
    boxes, corner = find_rank_suit_in_corner(gray)
    
    if len(boxes) >= 1:
        y, x, bw, bh = boxes[0]
        if bh > corner.shape[0] * 0.35:
            bh = int(bh * 0.55)
        pad = 2
        y1, x1 = max(0, y-pad), max(0, x-pad)
        y2, x2 = min(corner.shape[0], y+bh+pad), min(corner.shape[1], x+bw+pad)
        return corner[y1:y2, x1:x2]
    
    return gray[0:66, 0:60]


def extract_suit_region(gray: np.ndarray) -> np.ndarray:
    """Extract suit from below rank."""
    boxes, corner = find_rank_suit_in_corner(gray)
    
    if len(boxes) >= 2:
        y, x, bw, bh = boxes[1]
        pad = 2
        y1, x1 = max(0, y-pad), max(0, x-pad)
        y2, x2 = min(corner.shape[0], y+bh+pad), min(corner.shape[1], x+bw+pad)
        return corner[y1:y2, x1:x2]
    elif len(boxes) == 1:
        y, x, bw, bh = boxes[0]
        if bh > corner.shape[0] * 0.35:
            split = int(bh * 0.55)
            return corner[y+split:y+bh, x:x+bw]
    
    return gray[54:87, 10:40]


def preprocess(region: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Convert to binary, resize."""
    if region is None or region.size == 0:
        return np.zeros((size[1], size[0]), dtype=np.uint8)
    _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    resized = cv2.resize(binary, size, interpolation=cv2.INTER_AREA)
    _, final = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    return final