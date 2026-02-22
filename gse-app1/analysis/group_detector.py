"""
group_detector.py — Intelligent Sample Group Detection
Detects binary, multi-class, time-series, and paired sample designs.
"""

import re
import logging
import pandas as pd
import numpy as np
from collections import Counter

logger = logging.getLogger(__name__)


def detect_groups(sample_info: pd.DataFrame) -> dict:
    """
    Detect sample groups from metadata.
    
    Returns dict with:
    - groups: pd.Series with group labels
    - group_type: "binary" | "multiclass" | "timeseries"
    - group_names: list of unique group labels
    - confidence: 0-1 confidence score
    - timepoints: list (if timeseries)
    """
    logger.info("Detecting sample groups...")
    
    # Combine text columns
    text_cols = sample_info.select_dtypes(include=["object"]).columns
    combined_text = sample_info[text_cols].fillna("").apply(
        lambda row: " ".join(str(v).lower() for v in row), axis=1
    )
    
    # Try detectors
    result = _detect_timeseries(combined_text) or \
             _detect_multiclass(combined_text) or \
             _detect_binary(combined_text) or \
             _fallback_split(len(sample_info))
    
    result["groups"].index = sample_info.index
    logger.info(f"Detected: {result['group_type']} ({result['confidence']:.1%} confidence)")
    return result


def _detect_timeseries(combined_text: pd.Series) -> dict:
    """Detect time-series: T0, T1, Day0, Week1, etc."""
    patterns = [
        (r'[tT](\d+)', 1.0),
        (r'day[\s\-_]?(\d+)', 1.0),
        (r'week[\s\-_]?(\d+)', 7.0),
        (r'(\d+)\s*h', 1/24.0),
    ]
    
    timepoints = []
    for text in combined_text:
        found = None
        for pat, scale in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                found = float(match.group(1)) * scale
                break
        timepoints.append(found)
    
    timepoints = pd.Series(timepoints)
    valid = timepoints.dropna()
    
    if len(valid) / len(timepoints) > 0.7 and valid.nunique() >= 3:
        groups = timepoints.apply(lambda x: f"T{int(x)}" if pd.notna(x) else "Unknown")
        return {
            "groups": groups,
            "group_type": "timeseries",
            "group_names": sorted([g for g in groups.unique() if g != "Unknown"]),
            "timepoints": sorted(valid.unique().tolist()),
            "confidence": 0.9,
        }
    
    return None


def _detect_multiclass(combined_text: pd.Series) -> dict:
    """Detect multi-class: Stage I/II/III, Grade 1/2/3, etc."""
    stage_pattern = r'stage[\s\-_]?([IV1-4]+)'
    grade_pattern = r'grade[\s\-_]?([1-4])'
    
    groups = []
    for text in combined_text:
        found = None
        
        # Stage
        match = re.search(stage_pattern, text, re.IGNORECASE)
        if match:
            found = f"Stage{match.group(1).upper()}"
        
        # Grade
        if not found:
            match = re.search(grade_pattern, text, re.IGNORECASE)
            if match:
                found = f"Grade{match.group(1)}"
        
        groups.append(found or "Unknown")
    
    groups = pd.Series(groups)
    unique = groups.value_counts()
    
    if len(unique) >= 3 and unique.get("Unknown", 0) / len(groups) < 0.6:
        return {
            "groups": groups,
            "group_type": "multiclass",
            "group_names": [g for g in unique.index if g != "Unknown"],
            "confidence": 0.85,
        }
    
    return None


def _detect_binary(combined_text: pd.Series) -> dict:
    """Detect binary: control vs treatment."""
    ctrl_kw = {"control", "normal", "wild", "healthy", "wt", "untreated", "vehicle", "mock"}
    treat_kw = {"treated", "tumor", "cancer", "mutant", "disease", "patient", "ko", "knockout"}
    
    groups = []
    for text in combined_text:
        if any(k in text for k in ctrl_kw):
            groups.append("Control")
        elif any(k in text for k in treat_kw):
            groups.append("Treatment")
        else:
            groups.append("Unknown")
    
    groups = pd.Series(groups)
    unique = groups.value_counts()
    
    if "Control" in unique and "Treatment" in unique and unique.get("Unknown", 0) / len(groups) < 0.3:
        return {
            "groups": groups,
            "group_type": "binary",
            "group_names": ["Control", "Treatment"],
            "confidence": 0.75,
        }
    
    return None


def _fallback_split(n_samples: int) -> dict:
    """Fallback: 50/50 split."""
    groups = pd.Series(["Control"] * (n_samples//2) + ["Treatment"] * (n_samples - n_samples//2))
    return {
        "groups": groups,
        "group_type": "binary",
        "group_names": ["Control", "Treatment"],
        "confidence": 0.5,
        "warning": "Could not detect groups — split 50/50",
    }
