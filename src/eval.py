"""Evaluate card recognition."""

import argparse
from pathlib import Path

from detect_cards import load_image, detect_cards
from warp_card import warp_card
from template_matching import TemplateMatcher


def parse_gt(name):
    stem = Path(name).stem
    ranks = {'2','3','4','5','6','7','8','9','10','J','Q','K','A'}
    suits = {'C','D','H','S'}
    if stem.startswith('10'):
        r, s = '10', stem[2] if len(stem) > 2 else None
    else:
        r, s = (stem[0] if stem else None), (stem[1] if len(stem) > 1 else None)
    return (r, s) if r in ranks and s in suits else (None, None)


def evaluate(image_dir: str, template_dir: str, limit: int = None):
    matcher = TemplateMatcher(template_dir)
    
    files = [f for f in sorted(Path(image_dir).glob("*.jpg")) if parse_gt(f.name)[0]]
    if limit:
        files = files[:limit]
    
    total = len(files)
    det_fail = 0
    rank_ok = suit_ok = both_ok = 0
    
    for f in files:
        gt_r, gt_s = parse_gt(f.name)
        
        try:
            img = load_image(str(f))
            cards = detect_cards(img)
            
            if not cards:
                det_fail += 1
                continue
            
            warped = warp_card(img, cards[0]["corners"])
            pred_r, pred_s, score = matcher.match_best_orientation(warped)
            
            if pred_r == gt_r:
                rank_ok += 1
            if pred_s == gt_s:
                suit_ok += 1
            if pred_r == gt_r and pred_s == gt_s:
                both_ok += 1
                
        except:
            det_fail += 1
    
    tested = total - det_fail
    print(f"\nTotal: {total}, Detection failed: {det_fail}, Tested: {tested}")
    if tested > 0:
        print(f"Rank: {rank_ok}/{tested} = {100*rank_ok/tested:.1f}%")
        print(f"Suit: {suit_ok}/{tested} = {100*suit_ok/tested:.1f}%")
        print(f"Both: {both_ok}/{tested} = {100*both_ok/tested:.1f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", default="data/raw/images")
    p.add_argument("--template_dir", default="data/templates")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    evaluate(args.image_dir, args.template_dir, args.limit)