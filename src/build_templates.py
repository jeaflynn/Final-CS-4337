"""Fast template building - stops once we have all templates."""

import cv2
from pathlib import Path

from detect_cards import load_image, detect_cards
from warp_card import warp_card
from extract_corner import extract_rank_region, extract_suit_region, preprocess, RANK_SIZE, SUIT_SIZE


def parse_filename(name):
    stem = Path(name).stem
    ranks = {'2','3','4','5','6','7','8','9','10','J','Q','K','A'}
    suits = {'C','D','H','S'}
    if stem.startswith('10'):
        r, s = '10', stem[2] if len(stem) > 2 else None
    else:
        r, s = (stem[0] if stem else None), (stem[1] if len(stem) > 1 else None)
    return (r, s) if r in ranks and s in suits else (None, None)


def build_templates(image_dir: str, output_dir: str):
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    (output_dir / "ranks").mkdir(parents=True, exist_ok=True)
    (output_dir / "suits").mkdir(parents=True, exist_ok=True)
    
    need_ranks = {'2','3','4','5','6','7','8','9','10','J','Q','K','A'}
    need_suits = {'C','D','H','S'}
    
    for img_path in sorted(image_dir.glob("*.jpg")):
        if not need_ranks and not need_suits:
            break  # Done!
        
        rank, suit = parse_filename(img_path.name)
        if not rank:
            continue
        
        want_r = rank in need_ranks
        want_s = suit in need_suits
        if not want_r and not want_s:
            continue
        
        try:
            img = load_image(str(img_path))
            cards = detect_cards(img)
            if not cards:
                continue
            
            warped = warp_card(img, cards[0]["corners"])
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            
            if want_r:
                region = extract_rank_region(gray)
                if region is not None and region.size > 100:
                    tmpl = preprocess(region, RANK_SIZE)
                    cv2.imwrite(str(output_dir / "ranks" / f"{rank}.png"), tmpl)
                    need_ranks.discard(rank)
                    print(f"Rank {rank}")
            
            if want_s:
                region = extract_suit_region(gray)
                if region is not None and region.size > 100:
                    tmpl = preprocess(region, SUIT_SIZE)
                    cv2.imwrite(str(output_dir / "suits" / f"{suit}.png"), tmpl)
                    need_suits.discard(suit)
                    print(f"Suit {suit}")
        except:
            continue
    
    print(f"\nDone! Missing ranks: {need_ranks}, Missing suits: {need_suits}")


if __name__ == "__main__":
    import sys
    img_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/images"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "data/templates"
    build_templates(img_dir, out_dir)