"""Card recognition pipeline - outputs labeled images."""

import argparse
from pathlib import Path
import cv2

from detect_cards import load_image, detect_cards
from warp_card import warp_card
from template_matching import TemplateMatcher


def batch_process(image_dir: str, template_dir: str = "data/templates",
                  output_dir: str = "data/output"):
    """Process all images and save labeled versions."""
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    matcher = TemplateMatcher(template_dir)
    
    files = sorted(image_dir.glob("*.jpg"))
    total = len(files)
    
    for i, img_path in enumerate(files):
        try:
            img = load_image(str(img_path))
            cards = detect_cards(img)
            
            if not cards:
                # Still save with "?" label
                out = img.copy()
                cv2.putText(out, "?", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            else:
                warped = warp_card(img, cards[0]["corners"])
                rank, suit, score = matcher.match_best_orientation(warped)
                label = f"{rank}{suit}"
                
                out = img.copy()
                corners = cards[0]["corners"].astype(int)
                cv2.polylines(out, [corners.reshape(-1, 1, 2)], True, (0, 255, 0), 3)
                x, y = corners[0]
                cv2.putText(out, label, (x, max(50, y - 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            
            out_path = output_dir / f"{img_path.stem}_labeled.jpg"
            cv2.imwrite(str(out_path), out)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i+1}/{total}")
                
        except Exception as e:
            print(f"Error: {img_path.name}: {e}")
    
    print(f"\nDone! {total} labeled images saved to {output_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", default="data/raw/images")
    p.add_argument("--template_dir", default="data/templates")
    p.add_argument("--output_dir", default="data/output")
    args = p.parse_args()
    
    batch_process(args.image_dir, args.template_dir, args.output_dir)