# warp_and_extract_detected.py
import cv2
from pathlib import Path
from warp_card import warp_card
from extract_corner import extract_and_preprocess
from detect_cards import load_image, detect_cards

# Load image
img_path = "data/raw/images/2C31.jpg"
img = load_image(img_path)

# Detect all cards
cards = detect_cards(img)
if not cards:
    print("No cards detected.")
    exit()

# Make output directories
Path("output/warped").mkdir(parents=True, exist_ok=True)
Path("debug/corner_extraction").mkdir(parents=True, exist_ok=True)

for i, card in enumerate(cards):
    # Warp the card
    warped = warp_card(img, card["corners"])
    warped_path = f"output/warped/{i}_{Path(img_path).stem}.png"
    cv2.imwrite(warped_path, warped)
    print(f"Warped card saved: {warped_path}")

    # Extract and preprocess corner
    corner = extract_and_preprocess(warped, debug=True, debug_path=f"debug/corner_extraction/{i}")
    corner_path = f"debug/corner_extraction/{i}_corner.png"
    cv2.imwrite(corner_path, corner)
    print(f"Corner extracted and saved: {corner_path}")
