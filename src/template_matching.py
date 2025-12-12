"""Template matching - tries all 4 rotations and picks best."""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

from extract_corner import (RANK_SIZE, SUIT_SIZE, preprocess, 
                            get_all_corners, extract_rank_region, extract_suit_region)


class TemplateMatcher:
    def __init__(self, template_dir: str):
        self.ranks: Dict[str, np.ndarray] = {}
        self.suits: Dict[str, np.ndarray] = {}
        self._load(Path(template_dir))
    
    def _load(self, path: Path):
        for f in (path / "ranks").glob("*.png"):
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, RANK_SIZE)
                _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                self.ranks[f.stem] = img
        
        for f in (path / "suits").glob("*.png"):
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, SUIT_SIZE)
                _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                self.suits[f.stem] = img
        
        print(f"Loaded {len(self.ranks)} ranks, {len(self.suits)} suits")
    
    def match_best_orientation(self, warped: np.ndarray) -> Tuple[str, str, float]:
        """Try all 4 rotations, return best (rank, suit, combined_score)."""
        best_rank, best_suit, best_score = "?", "?", -999
        
        for rotated in get_all_corners(warped):
            gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY) if len(rotated.shape) == 3 else rotated
            
            rank_region = extract_rank_region(gray)
            suit_region = extract_suit_region(gray)
            
            r, rs = self._match(preprocess(rank_region, RANK_SIZE), self.ranks)
            s, ss = self._match(preprocess(suit_region, SUIT_SIZE), self.suits)
            
            combined = rs + ss
            if combined > best_score:
                best_score = combined
                best_rank, best_suit = r, s
        
        return best_rank, best_suit, best_score
    
    def match_rank(self, region: np.ndarray) -> Tuple[str, float]:
        return self._match(preprocess(region, RANK_SIZE), self.ranks)
    
    def match_suit(self, region: np.ndarray) -> Tuple[str, float]:
        return self._match(preprocess(region, SUIT_SIZE), self.suits)
    
    def _match(self, query: np.ndarray, templates: Dict) -> Tuple[str, float]:
        if not templates or query is None or query.size == 0:
            return "?", -999
        
        best_name, best_score = "?", -999
        for name, tmpl in templates.items():
            res = cv2.matchTemplate(query, tmpl, cv2.TM_CCOEFF_NORMED)
            score = float(res[0, 0])
            if score > best_score:
                best_score = score
                best_name = name
        
        return best_name, best_score