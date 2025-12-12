"""Card warping module."""

import cv2
import numpy as np

CARD_W, CARD_H = 200, 300


def warp_card(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Warp card to 200x300 with rank/suit in top-left."""
    corners = corners.astype(np.float32)
    
    # Handle landscape
    w = np.linalg.norm(corners[1] - corners[0])
    h = np.linalg.norm(corners[3] - corners[0])
    if w > h:
        corners = np.roll(corners, -1, axis=0)
    
    dst = np.array([[0, 0], [CARD_W-1, 0], [CARD_W-1, CARD_H-1], [0, CARD_H-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(image, M, (CARD_W, CARD_H))
    
    # Check orientation - rank/suit should be in top-left
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # Compare top-left vs bottom-right
    tl = gray[5:70, 5:50]
    br = gray[CARD_H-70:CARD_H-5, CARD_W-50:CARD_W-5]
    
    tl_ink = np.sum(tl < 150)
    br_ink = np.sum(br < 150)
    
    if br_ink > tl_ink:
        warped = cv2.rotate(warped, cv2.ROTATE_180)
    
    return warped