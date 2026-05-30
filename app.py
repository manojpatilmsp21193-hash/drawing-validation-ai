# =========================================================
# PHASE 1 : IMPORTS + OCR + PREPROCESSING
# Includes AI OCR Correction + Strict DOWN/UP Angle Logic
# EasyOCR deployment-safe model path added
# =========================================================

import streamlit as st
import cv2
import numpy as np
import easyocr
from PIL import Image
import pandas as pd
import ssl
import certifi
import re
import json
import io
import math
import os

ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)

st.set_page_config(
    page_title="VeriCAD-AI Powered CAD Drawing Validation & Correction Platform",
    layout="wide"
)

st.title("🏭 VeriCAD-AI Powered CAD Drawing Validation & Correction Platform")

# =========================================================
# DEPLOYMENT-SAFE EASYOCR LOADING
# =========================================================

EASYOCR_MODEL_DIR = os.path.join(
    os.getcwd(),
    "easyocr_models"
)

@st.cache_resource
def load_ocr():
    os.makedirs(
        EASYOCR_MODEL_DIR,
        exist_ok=True
    )

    return easyocr.Reader(
        ["en"],
        gpu=False,
        model_storage_directory=EASYOCR_MODEL_DIR,
        download_enabled=True,
        verbose=False
    )

reader = load_ocr()

uploaded_file = st.file_uploader(
    "Upload CAD Drawing Image",
    type=["png", "jpg", "jpeg"]
)

def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return image, img_bgr

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    denoise = cv2.fastNlMeansDenoising(
        gray,
        None,
        25,
        7,
        21
    )

    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharpen = cv2.filter2D(
        denoise,
        -1,
        sharpen_kernel
    )

    thresh = cv2.adaptiveThreshold(
        sharpen,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    edges = cv2.Canny(
        sharpen,
        50,
        150
    )

    return gray, thresh, edges

# =========================================================
# AI OCR CORRECTION LAYER
# =========================================================

def correct_ocr_text(text):
    t = text.upper().strip()

    corrections = {
        "EXTRUSI0N": "EXTRUSION",
        "EXTRUSLON": "EXTRUSION",
        "EXTRUS1ON": "EXTRUSION",
        "EXTRUSlON": "EXTRUSION",

        "ALUMlNIUM": "ALUMINIUM",
        "ALUM1NIUM": "ALUMINIUM",
        "ALUMIN1UM": "ALUMINIUM",
        "ALUMINlUM": "ALUMINIUM",

        "D0WN": "DOWN",
        "DOWM": "DOWN",
        "D0WM": "DOWN",
        "DWN": "DOWN",
        "UPP": "UP",

        "FLAT PATIERN": "FLAT PATTERN",
        "FLAT PATTFRN": "FLAT PATTERN",
        "FIAT PATTERN": "FLAT PATTERN",

        "MATERlAL": "MATERIAL",
        "MATER1AL": "MATERIAL",
        "FINlSH": "FINISH",
        "FIN1SH": "FINISH",

        "AN0DIZE": "ANODIZE",
        "ANODlZE": "ANODIZE",
        "ANOD1ZE": "ANODIZE",
        "CLEAR AN0DIZE": "CLEAR ANODIZE",

        "S1OT": "SLOT",
        "SL0T": "SLOT",
        "5LOT": "SLOT",

        "ClRCLIP": "CIRCLIP",
        "C1RCLIP": "CIRCLIP",
        "GR00VE": "GROOVE",
        "GRO0VE": "GROOVE",

        "RZ.0": "R2.0",
        "RZ.O": "R2.0",
        "ØI0": "Ø10",
        "ØL0": "Ø10",
        "O10": "Ø10"
    }

    for wrong, right in corrections.items():
        t = t.replace(wrong, right)

    t = re.sub(r"\s+", " ", t).strip()

    return t

# =========================================================
# OCR FUNCTION
# =========================================================

def run_ocr(thresh):
    results = reader.readtext(
        thresh,
        detail=1,
        paragraph=False
    )

    parsed = []

    for bbox, text, conf in results:
        x1 = int(bbox[0][0])
        y1 = int(bbox[0][1])
        x2 = int(bbox[2][0])
        y2 = int(bbox[2][1])

        parsed.append({
            "text": correct_ocr_text(text),
            "raw_text": text.strip(),
            "confidence": round(float(conf), 3),
            "bbox": bbox,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": abs(x2 - x1),
            "height": abs(y2 - y1)
        })

    return parsed

def joined_ocr_text(ocr_items):
    return " ".join([
        i["text"]
        for i in ocr_items
    ]).upper()

def compact_ocr_text(ocr_items):
    return (
        joined_ocr_text(ocr_items)
        .replace(" ", "")
        .replace(".", "")
        .replace(":", "")
        .replace("-", "")
        .replace("°", "")
    )

# =========================================================
# STRICT AI-ASSISTED BEND NOTE DETECTION
# DOWN / UP must be followed by angle 1 to 360
# =========================================================

def has_down_or_up_notation(ocr_items):

    for item in ocr_items:
        txt = item["text"].upper()
        raw = item.get("raw_text", "").upper()

        combined = txt + " " + raw

        combined = combined.replace("D0WN", "DOWN")
        combined = combined.replace("DOWM", "DOWN")
        combined = combined.replace("D0WM", "DOWN")
        combined = combined.replace("DWN", "DOWN")
        combined = combined.replace("UPP", "UP")

        compact = (
            combined.replace(" ", "")
            .replace(".", "")
            .replace("°", "")
            .replace(":", "")
            .replace("-", "")
        )

        down_match = re.search(
            r"\bDOWN\s*(\d{1,3})\b",
            combined
        )

        up_match = re.search(
            r"\bUP\s*(\d{1,3})\b",
            combined
        )

        if down_match:
            angle = int(down_match.group(1))

            if 1 <= angle <= 360:
                return True

        if up_match:
            angle = int(up_match.group(1))

            if 1 <= angle <= 360:
                return True

        compact_down = re.search(
            r"DOWN(\d{1,3})",
            compact
        )

        compact_up = re.search(
            r"UP(\d{1,3})",
            compact
        )

        if compact_down:
            angle = int(compact_down.group(1))

            if 1 <= angle <= 360:
                return True

        if compact_up:
            angle = int(compact_up.group(1))

            if 1 <= angle <= 360:
                return True

    return False

def has_bl_notation(ocr_items):
    return has_down_or_up_notation(ocr_items)

# =========================================================
# DRAWING TYPE DETECTION
# =========================================================

def is_extrusion_drawing(ocr_items):
    text = joined_ocr_text(ocr_items)
    compact = compact_ocr_text(ocr_items)

    extrusion_keywords = [
        "ALUMINIUMEXTRUSION",
        "ALUMINUMEXTRUSION",
        "ALUMINIUMEXTRUSI0N",
        "ALUMINUMEXTRUSI0N",
        "EXTRUSION",
        "EXTRUSI0N",
        "EXTRUS"
    ]

    if any(k in compact for k in extrusion_keywords):
        return True

    has_aluminium = (
        "ALUMINIUM" in text
        or "ALUMINUM" in text
        or "ALUM" in text
    )

    has_extrusion = (
        "EXTRUSION" in text
        or "EXTRUSI0N" in text
        or "EXTRUS" in text
    )

    return has_aluminium and has_extrusion

def has_flat_pattern_note(ocr_items):
    return "FLAT PATTERN" in joined_ocr_text(ocr_items)

def is_sheet_metal_drawing(ocr_items):
    return (
        has_down_or_up_notation(ocr_items)
        or
        has_flat_pattern_note(ocr_items)
    )

# =========================================================
# DIMENSION EXTRACTION
# =========================================================

def extract_dimensions(ocr_items):
    dimensions = []

    for item in ocr_items:
        raw_text = item["text"]
        text = raw_text.upper().replace(" ", "")

        matches = re.findall(
            r"(Ø|DIA|R|C)?\s*(\d+(\.\d+)?)",
            text
        )

        for match in matches:
            symbol = match[0]
            value = float(match[1])

            dimensions.append({
                "raw_text": raw_text,
                "original_raw_text": item.get("raw_text", raw_text),
                "value": value,
                "symbol": symbol,
                "is_diameter": symbol in ["Ø", "DIA"] or "Ø" in text,
                "is_radius": symbol == "R" or text.startswith("R"),
                "is_chamfer": symbol == "C" or "CHAMFER" in text,
                "bbox": item["bbox"],
                "confidence": item["confidence"],
                "x1": item["x1"],
                "y1": item["y1"],
                "x2": item["x2"],
                "y2": item["y2"]
            })

    return dimensions
# =========================================================
# PHASE 2 : GEOMETRY DETECTION
# Paste this AFTER Phase 1
# =========================================================

# =========================================================
# HOUGH LINE EXTRACTION
# =========================================================
def get_hough_lines(edges):
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=70,
        minLineLength=20,
        maxLineGap=8
    )

    if lines is None:
        return []

    return [line[0] for line in lines]

# =========================================================
# HOLE DETECTION
# =========================================================
def detect_holes(edges, img):
    output = img.copy()

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    holes = []
    img_area = img.shape[0] * img.shape[1]

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < img_area * 0.00002 or area > img_area * 0.02:
            continue

        perimeter = cv2.arcLength(cnt, True)

        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if 0.65 <= circularity <= 1.25:
            (x, y), radius = cv2.minEnclosingCircle(cnt)

            if radius < 5:
                continue

            holes.append({
                "center_x": int(x),
                "center_y": int(y),
                "radius_px": int(radius),
                "area": round(area, 2),
                "circularity": round(circularity, 2)
            })

            cv2.circle(
                output,
                (int(x), int(y)),
                int(radius),
                (0, 255, 0),
                2
            )

            cv2.circle(
                output,
                (int(x), int(y)),
                3,
                (0, 0, 255),
                -1
            )

    return holes, output

# =========================================================
# SHAFT REGION DETECTION
# Used only for circlip groove checking
# =========================================================
def detect_shaft_regions(edges, img):
    h, w = img.shape[:2]

    lines = get_hough_lines(edges)

    horizontal_lines = []

    for x1, y1, x2, y2 in lines:
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if dx > w * 0.28 and dy <= 6:
            horizontal_lines.append({
                "x1": min(x1, x2),
                "x2": max(x1, x2),
                "y": int((y1 + y2) / 2),
                "length": dx
            })

    shaft_regions = []

    for i in range(len(horizontal_lines)):
        for j in range(i + 1, len(horizontal_lines)):
            a = horizontal_lines[i]
            b = horizontal_lines[j]

            y_gap = abs(a["y"] - b["y"])
            x_overlap = (
                min(a["x2"], b["x2"])
                -
                max(a["x1"], b["x1"])
            )

            if 18 <= y_gap <= h * 0.30 and x_overlap > w * 0.30:
                x1 = max(a["x1"], b["x1"])
                x2 = min(a["x2"], b["x2"])
                y1 = min(a["y"], b["y"])
                y2 = max(a["y"], b["y"])

                shaft_regions.append({
                    "x1": x1,
                    "x2": x2,
                    "y1": y1,
                    "y2": y2,
                    "width": x2 - x1,
                    "height": y2 - y1
                })

    shaft_regions = sorted(
        shaft_regions,
        key=lambda r: r["width"] * r["height"],
        reverse=True
    )

    return shaft_regions[:3]

# =========================================================
# CIRCLIP GROOVE DETECTION
# =========================================================
def detect_circlip_groove(edges, img, ocr_items=None):
    features = {
        "circlip_groove_present": False,
        "circlip_bbox": None,
        "groove_line_count": 0,
        "shaft_region_found": False
    }

    # Fully exclude extrusion drawings
    if ocr_items is not None and is_extrusion_drawing(ocr_items):
        return features

    # Do not treat sheet-metal / slot drawings as shaft circlip drawings
    if ocr_items is not None:
        text = joined_ocr_text(ocr_items)

        if (
            has_down_or_up_notation(ocr_items)
            or "SLOT" in text
            or "FLAT PATTERN" in text
        ):
            return features

    h, w = img.shape[:2]

    shaft_regions = detect_shaft_regions(edges, img)

    if not shaft_regions:
        return features

    features["shaft_region_found"] = True

    lines = get_hough_lines(edges)
    vertical_lines = []

    for x1, y1, x2, y2 in lines:
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if dx > 8 or dy < 18:
            continue

        x_mid = int((x1 + x2) / 2)

        vertical_lines.append({
            "x": x_mid,
            "y1": min(y1, y2),
            "y2": max(y1, y2),
            "height": abs(y2 - y1)
        })

    groove_candidates = []

    for region in shaft_regions:
        sx1 = region["x1"]
        sx2 = region["x2"]
        sy1 = region["y1"]
        sy2 = region["y2"]

        shaft_len = sx2 - sx1
        shaft_ht = sy2 - sy1

        valid_v_lines = []

        for line in vertical_lines:
            inside_x = (
                sx1 + shaft_len * 0.10
                <
                line["x"]
                <
                sx2 - shaft_len * 0.10
            )

            crosses_shaft = (
                line["y1"] <= sy1 + shaft_ht * 0.30
                and line["y2"] >= sy2 - shaft_ht * 0.30
            )

            if inside_x and crosses_shaft:
                valid_v_lines.append(line)

        for i in range(len(valid_v_lines)):
            for j in range(i + 1, len(valid_v_lines)):
                a = valid_v_lines[i]
                b = valid_v_lines[j]

                gap = abs(a["x"] - b["x"])

                y_overlap = (
                    min(a["y2"], b["y2"])
                    -
                    max(a["y1"], b["y1"])
                )

                if not (5 <= gap <= 35):
                    continue

                if y_overlap < shaft_ht * 0.50:
                    continue

                avg_x = int((a["x"] + b["x"]) / 2)

                if (
                    avg_x < sx1 + shaft_len * 0.15
                    or avg_x > sx2 - shaft_len * 0.15
                ):
                    continue

                groove_candidates.append({
                    "score": y_overlap - gap,
                    "x1": min(a["x"], b["x"]),
                    "x2": max(a["x"], b["x"]),
                    "y1": max(min(a["y1"], b["y1"]), sy1 - 10),
                    "y2": min(max(a["y2"], b["y2"]), sy2 + 10)
                })

    if groove_candidates:
        best = max(
            groove_candidates,
            key=lambda g: g["score"]
        )

        features["circlip_groove_present"] = True
        features["groove_line_count"] = len(groove_candidates)

        features["circlip_bbox"] = [
            [max(best["x1"] - 18, 0), max(best["y1"] - 18, 0)],
            [min(best["x2"] + 18, w), max(best["y1"] - 18, 0)],
            [min(best["x2"] + 18, w), min(best["y2"] + 18, h)],
            [max(best["x1"] - 18, 0), min(best["y2"] + 18, h)]
        ]

    return features

# =========================================================
# TRUE SLOT DETECTION
# Definition:
# - closed obround / elongated hole
# - two long straight sides
# - two rounded ends
# - not extrusion
# - not outer border
# - not shaft circlip groove
# =========================================================
def detect_slot_features(edges, img, ocr_items=None):
    features = {
        "slot_present": False,
        "slot_bboxes": [],
        "slot_count": 0
    }

    # Fully exclude extrusion drawings
    if ocr_items is not None and is_extrusion_drawing(ocr_items):
        return features

    h, w = img.shape[:2]

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    slot_candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 100:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        if bw < 15 or bh < 10:
            continue

        # Reject large borders/views
        if bw > w * 0.22 or bh > h * 0.22:
            continue

        long_side = max(bw, bh)
        short_side = min(bw, bh)
        aspect = long_side / max(short_side, 1)

        # Slot must be elongated but not just a long line
        if aspect < 1.7 or aspect > 6.0:
            continue

        rect_area = bw * bh

        if rect_area == 0:
            continue

        fill_ratio = area / rect_area

        if fill_ratio < 0.32 or fill_ratio > 0.88:
            continue

        perimeter = cv2.arcLength(cnt, True)

        if perimeter == 0:
            continue

        approx = cv2.approxPolyDP(
            cnt,
            0.012 * perimeter,
            True
        )

        # Slot has curved ends, so contour should not be simple rectangle
        if len(approx) < 7:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)

        if hull_area == 0:
            continue

        solidity = area / hull_area

        if solidity < 0.72:
            continue

        if short_side < 8:
            continue

        # Ellipse-fit confidence for rounded-end geometry
        if len(cnt) >= 5:
            try:
                ellipse = cv2.fitEllipse(cnt)
                (_, _), (axis1, axis2), _ = ellipse

                ellipse_aspect = max(axis1, axis2) / max(min(axis1, axis2), 1)

                if ellipse_aspect < 1.5 or ellipse_aspect > 7.0:
                    continue

            except Exception:
                continue

        slot_candidates.append({
            "bbox": [
                [x, y],
                [x + bw, y],
                [x + bw, y + bh],
                [x, y + bh]
            ],
            "x": x,
            "y": y,
            "w": bw,
            "h": bh,
            "area": round(area, 2),
            "aspect": round(aspect, 2)
        })

    # Remove duplicate detections
    filtered = []

    for cand in slot_candidates:
        duplicate = False

        for existing in filtered:
            if (
                abs(cand["x"] - existing["x"]) < 10
                and abs(cand["y"] - existing["y"]) < 10
            ):
                duplicate = True
                break

        if not duplicate:
            filtered.append(cand)

    if filtered:
        features["slot_present"] = True
        features["slot_bboxes"] = [s["bbox"] for s in filtered]
        features["slot_count"] = len(filtered)

    return features

# =========================================================
# RADIUS FEATURE DETECTION
# =========================================================
def detect_radius_features(edges, img):
    features = {
        "radius_feature_present": False,
        "radius_bbox": None
    }

    h, w = img.shape[:2]

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    radius_candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 800:
            continue

        perimeter = cv2.arcLength(cnt, True)

        if perimeter == 0:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        if bw < w * 0.12 or bh < h * 0.12:
            continue

        if bw > w * 0.90 or bh > h * 0.90:
            continue

        approx = cv2.approxPolyDP(
            cnt,
            0.008 * perimeter,
            True
        )

        if len(approx) >= 8:
            radius_candidates.append(
                (x, y, x + bw, y + bh)
            )

    if radius_candidates:
        features["radius_feature_present"] = True

        x1 = min(c[0] for c in radius_candidates)
        y1 = min(c[1] for c in radius_candidates)
        x2 = max(c[2] for c in radius_candidates)
        y2 = max(c[3] for c in radius_candidates)

        features["radius_bbox"] = [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ]

    return features

# =========================================================
# CHAMFER FEATURE DETECTION
# =========================================================
def detect_chamfer_features(edges, img):
    features = {
        "chamfer_feature_present": False,
        "chamfer_bbox": None
    }

    h, w = img.shape[:2]
    lines = get_hough_lines(edges)

    diagonal_lines = []

    for x1, y1, x2, y2 in lines:
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0:
            continue

        angle = abs(
            np.degrees(
                np.arctan2(dy, dx)
            )
        )

        length = np.sqrt(dx * dx + dy * dy)

        if 30 <= angle <= 60 and length > 25:
            diagonal_lines.append(
                (x1, y1, x2, y2)
            )

    if diagonal_lines:
        features["chamfer_feature_present"] = True

        xs = []
        ys = []

        for x1, y1, x2, y2 in diagonal_lines:
            xs.extend([x1, x2])
            ys.extend([y1, y2])

        features["chamfer_bbox"] = [
            [max(min(xs) - 20, 0), max(min(ys) - 20, 0)],
            [min(max(xs) + 20, w), max(min(ys) - 20, 0)],
            [min(max(xs) + 20, w), min(max(ys) + 20, h)],
            [max(min(xs) - 20, 0), min(max(ys) + 20, h)]
        ]

    return features

# =========================================================
# COMPLETE GEOMETRY FEATURE DETECTION
# =========================================================
def detect_geometry_features(edges, img, ocr_items=None):
    features = {
        "radius_feature_present": False,
        "chamfer_feature_present": False,
        "radius_bbox": None,
        "chamfer_bbox": None,
        "circlip_groove_present": False,
        "circlip_bbox": None,
        "groove_line_count": 0,
        "shaft_region_found": False,
        "slot_present": False,
        "slot_bboxes": [],
        "slot_count": 0
    }

    extrusion = False

    if ocr_items is not None:
        extrusion = is_extrusion_drawing(ocr_items)

    # For extrusion drawings, skip radius / chamfer / slot / circlip feature checks.
    if not extrusion:
        radius_features = detect_radius_features(edges, img)
        features.update(radius_features)

        chamfer_features = detect_chamfer_features(edges, img)
        features.update(chamfer_features)

        circlip_features = detect_circlip_groove(
            edges,
            img,
            ocr_items
        )
        features.update(circlip_features)

        slot_features = detect_slot_features(
            edges,
            img,
            ocr_items
        )
        features.update(slot_features)

    return features
# =========================================================
# PHASE 3 : VALIDATION ENGINE
# Paste this AFTER Phase 2
# =========================================================

RULES = {
    "expected_hole_count_min": 1
}

DIN471_SHAFT_GROOVE_TABLE = {
    10: {"d2": 9.6, "d2_tol": "-0.11", "m": 1.10, "m_tol": "H13"},
    11: {"d2": 10.5, "d2_tol": "-0.11", "m": 1.10, "m_tol": "H13"},
    12: {"d2": 11.5, "d2_tol": "-0.11", "m": 1.10, "m_tol": "H13"},
    13: {"d2": 12.4, "d2_tol": "-0.11", "m": 1.10, "m_tol": "H13"},
    14: {"d2": 13.4, "d2_tol": "-0.11", "m": 1.10, "m_tol": "H13"},
    15: {"d2": 14.3, "d2_tol": "-0.11", "m": 1.10, "m_tol": "H13"},
    16: {"d2": 15.2, "d2_tol": "-0.11", "m": 1.10, "m_tol": "H13"},
    17: {"d2": 16.2, "d2_tol": "-0.11", "m": 1.10, "m_tol": "H13"},
    18: {"d2": 17.0, "d2_tol": "-0.11", "m": 1.30, "m_tol": "H13"},
    19: {"d2": 18.0, "d2_tol": "-0.11", "m": 1.30, "m_tol": "H13"},
    20: {"d2": 19.0, "d2_tol": "-0.13", "m": 1.30, "m_tol": "H13"},
    21: {"d2": 20.0, "d2_tol": "-0.13", "m": 1.30, "m_tol": "H13"},
    22: {"d2": 21.0, "d2_tol": "-0.13", "m": 1.30, "m_tol": "H13"},
    23: {"d2": 22.0, "d2_tol": "-0.15", "m": 1.30, "m_tol": "H13"},
    24: {"d2": 22.9, "d2_tol": "-0.21", "m": 1.30, "m_tol": "H13"},
    25: {"d2": 23.9, "d2_tol": "-0.21", "m": 1.30, "m_tol": "H13"},
    26: {"d2": 24.9, "d2_tol": "-0.21", "m": 1.30, "m_tol": "H13"},
    27: {"d2": 25.6, "d2_tol": "-0.21", "m": 1.30, "m_tol": "H13"},
    28: {"d2": 26.6, "d2_tol": "-0.21", "m": 1.60, "m_tol": "H13"},
    29: {"d2": 27.6, "d2_tol": "-0.21", "m": 1.60, "m_tol": "H13"},
    30: {"d2": 28.6, "d2_tol": "-0.21", "m": 1.60, "m_tol": "H13"},
    31: {"d2": 29.3, "d2_tol": "-0.21", "m": 1.60, "m_tol": "H13"},
    32: {"d2": 30.3, "d2_tol": "-0.25", "m": 1.60, "m_tol": "H13"},
    33: {"d2": 31.3, "d2_tol": "-0.25", "m": 1.60, "m_tol": "H13"},
    34: {"d2": 32.3, "d2_tol": "-0.25", "m": 1.60, "m_tol": "H13"},
    35: {"d2": 33.0, "d2_tol": "-0.25", "m": 1.60, "m_tol": "H13"},
    36: {"d2": 34.0, "d2_tol": "-0.25", "m": 1.85, "m_tol": "H13"},
    37: {"d2": 35.0, "d2_tol": "-0.25", "m": 1.85, "m_tol": "H13"},
    38: {"d2": 36.0, "d2_tol": "-0.25", "m": 1.85, "m_tol": "H13"},
    39: {"d2": 37.0, "d2_tol": "-0.25", "m": 1.85, "m_tol": "H13"},
    40: {"d2": 37.5, "d2_tol": "-0.25", "m": 1.85, "m_tol": "H13"},
}


def approximate_bbox(img_shape, position):
    h, w = img_shape[:2]

    if position == "center_distance":
        return [[int(w*0.25), int(h*0.60)], [int(w*0.62), int(h*0.60)], [int(w*0.62), int(h*0.74)], [int(w*0.25), int(h*0.74)]]

    if position == "dowel_callout":
        return [[int(w*0.40), int(h*0.02)], [int(w*0.86), int(h*0.02)], [int(w*0.86), int(h*0.20)], [int(w*0.40), int(h*0.20)]]

    if position == "finish":
        return [[int(w*0.20), int(h*0.82)], [int(w*0.75), int(h*0.82)], [int(w*0.75), int(h*0.96)], [int(w*0.20), int(h*0.96)]]

    if position == "flat_pattern":
        return [[int(w*0.35), int(h*0.68)], [int(w*0.75), int(h*0.68)], [int(w*0.75), int(h*0.88)], [int(w*0.35), int(h*0.88)]]

    return None


def find_finish_bbox(ocr_items, img_shape):
    finish_items = [
        item for item in ocr_items
        if "FINISH" in item["text"].upper()
        or "CLEAR" in item["text"].upper()
        or "ZINC" in item["text"].upper()
        or "ANODIZE" in item["text"].upper()
        or "ANODISE" in item["text"].upper()
    ]

    if finish_items:
        x1 = min(i["x1"] for i in finish_items)
        y1 = min(i["y1"] for i in finish_items)
        x2 = max(i["x2"] for i in finish_items)
        y2 = max(i["y2"] for i in finish_items)
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    return approximate_bbox(img_shape, "finish")


def detect_shaft_diameter(dimensions):
    candidates = []

    for d in dimensions:
        raw = d["raw_text"].upper()

        if d["is_diameter"] and 10 <= d["value"] <= 40:
            candidates.append(d["value"])

        elif ("SHAFT" in raw or "DIA" in raw or "Ø" in raw) and 10 <= d["value"] <= 40:
            candidates.append(d["value"])

    if candidates:
        return int(round(max(candidates)))

    fallback = [
        d["value"] for d in dimensions
        if 10 <= d["value"] <= 40
        and not d["is_radius"]
        and not d["is_chamfer"]
    ]

    if fallback:
        return int(round(max(fallback)))

    return None


def build_circlip_suggestion(shaft_size):
    ref = DIN471_SHAFT_GROOVE_TABLE.get(shaft_size)

    if not ref:
        return "Circlip groove detected. Shaft size not found in embedded Ø10–Ø40 DIN 471 reference table."

    return (
        f"Suggested as per DIN 471-style reference for shaft Ø{shaft_size}: "
        f"groove diameter d2 = {ref['d2']} mm, d2 tolerance {ref['d2_tol']}, "
        f"groove width m = {ref['m']} mm, width tolerance {ref['m_tol']}."
    )


def check_dowel_center_tolerance(ocr_items, dimensions, issues, issue_id, img_shape):
    for item in ocr_items:
        text = item["text"].upper().replace(" ", "")

        if "H7" in text or "THRU" in text or "DOWEL" in text:
            continue

        patterns = re.findall(r"(\d+(?:\.\d+)?)(?:±|\+/-|\+-|\+)(\d+(?:\.\d+)?)", text)

        for base, tol in patterns:
            base_value = float(base)
            tol_value = float(tol)

            if 20 <= base_value < 150 and abs(tol_value - 0.013) > 0.0005:
                issues.append({
                    "id": issue_id,
                    "severity": "Critical",
                    "type": "Dowel Hole Center Distance Tolerance",
                    "message": "Wrong Tolerance value",
                    "suggestion": "Use ±0.013 for center distance below 150 mm",
                    "bbox": approximate_bbox(img_shape, "center_distance")
                })
                return issue_id + 1, True

            if 150 <= base_value <= 300 and abs(tol_value - 0.020) > 0.0005:
                issues.append({
                    "id": issue_id,
                    "severity": "Critical",
                    "type": "Dowel Hole Center Distance Tolerance",
                    "message": "Wrong Tolerance value",
                    "suggestion": "Use ±0.020 for center distance between 150 and 300 mm",
                    "bbox": approximate_bbox(img_shape, "center_distance")
                })
                return issue_id + 1, True

    return issue_id, False


def check_dowel_decimal_rule(ocr_items, issues, issue_id, img_shape):
    joined_text = joined_ocr_text(ocr_items)

    if "DOWEL" not in joined_text:
        return issue_id

    dowel_matches = re.findall(r"Ø\s*(\d+(?:\.\d+)?)", joined_text)

    if not dowel_matches:
        dowel_matches = re.findall(r"2X\s*(?:Ø)?\s*(\d+(?:\.\d+)?)\s*H7", joined_text)

    for value in dowel_matches:
        decimal_count = len(value.split(".")[1]) if "." in value else 0

        if decimal_count != 3:
            issues.append({
                "id": issue_id,
                "severity": "High",
                "type": "Dowel Hole Decimal",
                "message": "Tolerance decimals are wrong",
                "suggestion": "Use 3 decimals for dowel-hole size, e.g. Ø10.000",
                "bbox": approximate_bbox(img_shape, "dowel_callout")
            })
            issue_id += 1

    return issue_id


def check_material_finish(ocr_items, issues, issue_id, img_shape):
    joined_text = joined_ocr_text(ocr_items)

    material_is_aluminium = (
        "MATERIAL" in joined_text
        and ("ALUMINIUM" in joined_text or "ALUMINUM" in joined_text)
    )

    finish_is_clear_anodize = (
        "FINISH" in joined_text
        and ("CLEAR ANODIZE" in joined_text or "CLEAR ANODISE" in joined_text)
    )

    if material_is_aluminium and not finish_is_clear_anodize:
        issues.append({
            "id": issue_id,
            "severity": "High",
            "type": "Material / Finish",
            "message": "Material is ALUMINIUM but finish is not CLEAR ANODIZE",
            "suggestion": "Please check the finish",
            "bbox": find_finish_bbox(ocr_items, img_shape),
            "auto_correct": True
        })
        issue_id += 1

    return issue_id


def check_flat_pattern_rule(ocr_items, issues, issue_id, img_shape):
    if has_down_or_up_notation(ocr_items) and not has_flat_pattern_note(ocr_items):
        issues.append({
            "id": issue_id,
            "severity": "High",
            "type": "Flat Pattern",
            "message": "DOWN/UP bend notation found but FLAT PATTERN description is missing",
            "suggestion": "Mention FLAT PATTERN",
            "bbox": approximate_bbox(img_shape, "flat_pattern")
        })
        issue_id += 1

    return issue_id


def has_slot_annotation(ocr_items):
    text = joined_ocr_text(ocr_items)

    return (
        "SLOT" in text
        or "SLOTTED" in text
        or "OBROUND" in text
        or re.search(r"\d+(\.\d+)?\s*SLOT", text) is not None
    )


def check_slot_annotation(ocr_items, geometry_features, issues, issue_id):
    if is_extrusion_drawing(ocr_items):
        return issue_id

    if not geometry_features.get("slot_present", False):
        return issue_id

    if has_slot_annotation(ocr_items):
        return issue_id

    bbox = geometry_features["slot_bboxes"][0] if geometry_features.get("slot_bboxes") else None

    issues.append({
        "id": issue_id,
        "severity": "High",
        "type": "Slot Annotation",
        "message": "Slot feature exists but slot annotation is missing",
        "suggestion": "Slot dimension missing",
        "bbox": bbox
    })

    return issue_id + 1


def check_radius_rule(ocr_items, geometry_features, issues, issue_id):
    if is_extrusion_drawing(ocr_items):
        return issue_id

    joined_text = joined_ocr_text(ocr_items)
    compact_text = joined_text.replace(" ", "")

    radius_annotation_found = (
        re.search(r"\bR\s*\d+(\.\d+)?", joined_text) is not None
        or re.search(r"\d+\s*X\s*R\s*\d+(\.\d+)?", joined_text) is not None
        or re.search(r"\d+X?R\d+(\.\d+)?", compact_text) is not None
    )

    if geometry_features.get("radius_feature_present", False) and not radius_annotation_found:
        issues.append({
            "id": issue_id,
            "severity": "High",
            "type": "Radius Annotation",
            "message": "Radius feature exists but radius annotation is missing",
            "suggestion": "Add radius callout, e.g. R2.0 or 4X R2.0",
            "bbox": geometry_features.get("radius_bbox")
        })
        issue_id += 1

    return issue_id


def check_chamfer_rule(ocr_items, geometry_features, issues, issue_id):
    if is_extrusion_drawing(ocr_items):
        return issue_id

    joined_text = joined_ocr_text(ocr_items)

    chamfer_annotation_found = (
        "CHAMFER" in joined_text
        or re.search(r"\bC\s*\d+(\.\d+)?", joined_text)
        or re.search(r"\d+(\.\d+)?\s*[Xx]\s*45", joined_text)
    )

    if geometry_features.get("chamfer_feature_present", False) and not chamfer_annotation_found:
        issues.append({
            "id": issue_id,
            "severity": "Medium",
            "type": "Chamfer Annotation",
            "message": "Chamfer feature exists but chamfer annotation is missing",
            "suggestion": "Add chamfer callout, e.g. C1.0 or 1X45°",
            "bbox": geometry_features.get("chamfer_bbox")
        })
        issue_id += 1

    return issue_id


def check_circlip_groove_annotation(ocr_items, dimensions, geometry_features, issues, issue_id):
    if is_extrusion_drawing(ocr_items):
        return issue_id

    joined_text = joined_ocr_text(ocr_items)

    if (
        has_down_or_up_notation(ocr_items)
        or "SLOT" in joined_text
        or "FLAT PATTERN" in joined_text
    ):
        return issue_id

    if not geometry_features.get("circlip_groove_present", False):
        return issue_id

    shaft_size = detect_shaft_diameter(dimensions)

    if shaft_size is None or shaft_size not in DIN471_SHAFT_GROOVE_TABLE:
        issues.append({
            "id": issue_id,
            "severity": "High",
            "type": "Circlip Groove",
            "message": "Circlip groove detected but shaft diameter could not be matched to DIN 471 table",
            "suggestion": "Add shaft diameter and circlip groove callout. Embedded table supports shaft Ø10 to Ø40.",
            "bbox": geometry_features.get("circlip_bbox")
        })
        return issue_id + 1

    ref = DIN471_SHAFT_GROOVE_TABLE[shaft_size]
    expected_d2 = ref["d2"]
    expected_m = ref["m"]

    circlip_words_found = (
        "CIRCLIP" in joined_text
        or "RETAINING RING" in joined_text
        or "GROOVE" in joined_text
        or "DIN 471" in joined_text
    )

    numeric_values = [d["value"] for d in dimensions]

    d2_found = any(abs(v - expected_d2) <= 0.08 for v in numeric_values)
    m_found = any(abs(v - expected_m) <= 0.06 for v in numeric_values)

    if not circlip_words_found and not d2_found and not m_found:
        issues.append({
            "id": issue_id,
            "severity": "Critical",
            "type": "Circlip Groove Annotation",
            "message": "Circlip groove annotation is missing",
            "suggestion": build_circlip_suggestion(shaft_size),
            "bbox": geometry_features.get("circlip_bbox")
        })
        return issue_id + 1

    if not d2_found or not m_found:
        missing = []

        if not d2_found:
            missing.append("groove diameter d2")

        if not m_found:
            missing.append("groove width m")

        issues.append({
            "id": issue_id,
            "severity": "Critical",
            "type": "Circlip Groove Annotation",
            "message": "Circlip groove annotation is wrong or incomplete",
            "suggestion": build_circlip_suggestion(shaft_size) + " Missing/incorrect: " + ", ".join(missing),
            "bbox": geometry_features.get("circlip_bbox")
        })
        return issue_id + 1

    return issue_id


def check_extrusion_table_dimensions(ocr_items, issues, issue_id):
    if not is_extrusion_drawing(ocr_items):
        return issue_id

    table_header_items = [
        i for i in ocr_items
        if (
            "DESCRIPTION" in i["text"].upper()
            or "LENGTH" in i["text"].upper()
            or "ITEM" in i["text"].upper()
            or "QTY" in i["text"].upper()
        )
    ]

    if not table_header_items:
        return issue_id

    table_top = min(i["y1"] for i in table_header_items) - 20
    table_bottom = max(i["y2"] for i in table_header_items) + 140

    table_values = []
    drawing_values = []

    for item in ocr_items:
        text = item["text"].strip().upper()

        values = re.findall(r"\b(\d{2,4})(?:\.0)?\b", text)

        for v in values:
            value = int(v)

            if not (50 <= value <= 2000):
                continue

            value_obj = {
                "value": value,
                "bbox": item["bbox"],
                "raw_text": item["text"],
                "original_raw_text": item.get("raw_text", item["text"]),
                "x1": item["x1"],
                "y1": item["y1"]
            }

            if table_top <= item["y1"] <= table_bottom:
                table_values.append(value_obj)
            else:
                drawing_values.append(value_obj)

    drawing_set = {v["value"] for v in drawing_values}
    checked = set()

    for tv in table_values:
        value = tv["value"]

        if value in checked:
            continue

        checked.add(value)

        if value not in drawing_set:
            issues.append({
                "id": issue_id,
                "severity": "High",
                "type": "Table Dimension Check",
                "message": f"Table dimension {value} is not available in drawing view",
                "suggestion": f"{value} dimension is missing",
                "bbox": tv["bbox"]
            })
            issue_id += 1

    return issue_id


def check_extrusion_decimal_format(ocr_items, issues, issue_id):
    if not is_extrusion_drawing(ocr_items):
        return issue_id

    for item in ocr_items:
        text = item["text"].strip()

        matches = re.findall(r"\b\d{2,4}\.0\b", text)

        for m in matches:
            issues.append({
                "id": issue_id,
                "severity": "Medium",
                "type": "Extrusion Dimension Format",
                "message": f"Extrusion drawing dimension has decimal format: {m}",
                "suggestion": "Round off to nearest decimal",
                "bbox": item["bbox"]
            })
            issue_id += 1

    return issue_id


def check_basic_geometry(ocr_items, holes, issues, issue_id):
    if is_extrusion_drawing(ocr_items):
        return issue_id

    if len(holes) < RULES["expected_hole_count_min"]:
        issues.append({
            "id": issue_id,
            "severity": "High",
            "type": "Geometry",
            "message": "No circular holes detected",
            "suggestion": "Verify CAD geometry or scan quality",
            "bbox": None
        })
        issue_id += 1

    return issue_id


def validate(ocr_items, holes, dimensions, geometry_features, img_shape):
    issues = []
    issue_id = 1

    extrusion = is_extrusion_drawing(ocr_items)

    issue_id, _ = check_dowel_center_tolerance(ocr_items, dimensions, issues, issue_id, img_shape)
    issue_id = check_dowel_decimal_rule(ocr_items, issues, issue_id, img_shape)
    issue_id = check_material_finish(ocr_items, issues, issue_id, img_shape)
    issue_id = check_extrusion_table_dimensions(ocr_items, issues, issue_id)
    issue_id = check_extrusion_decimal_format(ocr_items, issues, issue_id)

    if not extrusion:
        issue_id = check_circlip_groove_annotation(ocr_items, dimensions, geometry_features, issues, issue_id)
        issue_id = check_slot_annotation(ocr_items, geometry_features, issues, issue_id)
        issue_id = check_radius_rule(ocr_items, geometry_features, issues, issue_id)
        issue_id = check_chamfer_rule(ocr_items, geometry_features, issues, issue_id)
        issue_id = check_flat_pattern_rule(ocr_items, issues, issue_id, img_shape)

    issue_id = check_basic_geometry(ocr_items, holes, issues, issue_id)

    return issues

# =========================================================
# PHASE 4 : STREAMLIT UI + EXPORT
# Paste this AFTER Phase 3
# =========================================================

def get_label_position(x1, y1, img_shape):
    h, w = img_shape[:2]
    label_x = max(25, x1 - 40)
    label_y = max(35, y1 - 35)

    if label_x < 30:
        label_x = min(w - 30, x1 + 40)

    return label_x, label_y

def annotate_issues(img, issues, corrected_material=False):
    annotated = img.copy()

    for issue in issues:
        if corrected_material and issue["type"] == "Material / Finish":
            continue

        bbox = issue.get("bbox")

        if bbox:
            x1 = int(bbox[0][0])
            y1 = int(bbox[0][1])
            x2 = int(bbox[2][0])
            y2 = int(bbox[2][1])

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)

            label_x, label_y = get_label_position(x1, y1, img.shape)

            cv2.circle(annotated, (label_x, label_y), 18, (0, 0, 255), -1)

            cv2.putText(
                annotated,
                str(issue["id"]),
                (label_x - 7, label_y + 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

    return annotated

def auto_correct_material_finish(img, issues):
    corrected = img.copy()

    material_issue = None

    for issue in issues:
        if issue["type"] == "Material / Finish":
            material_issue = issue
            break

    if material_issue is None:
        return corrected

    bbox = material_issue.get("bbox")

    if bbox:
        x1 = int(bbox[0][0])
        y1 = int(bbox[0][1])
        x2 = int(bbox[2][0])
        y2 = int(bbox[2][1])

        cv2.rectangle(corrected, (x1, y1), (x2, y2), (255, 255, 255), -1)

        cv2.putText(
            corrected,
            "FINISH :- CLEAR ANODIZE",
            (x1, y1 + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 160, 0),
            2
        )

    return corrected

def annotate_corrected_material_green(img, issues):
    annotated = img.copy()

    for issue in issues:
        if issue["type"] == "Material / Finish":
            bbox = issue.get("bbox")

            if bbox:
                x1 = int(bbox[0][0])
                y1 = int(bbox[0][1])

                label_x, label_y = get_label_position(x1, y1, img.shape)

                cv2.circle(annotated, (label_x, label_y), 18, (0, 160, 0), -1)

                cv2.putText(
                    annotated,
                    str(issue["id"]),
                    (label_x - 7, label_y + 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

    return annotated

def make_report(issues, holes, dimensions, geometry_features, ocr_items):
    report = {
        "summary": {
            "total_issues": len(issues),
            "status": "FAILED" if issues else "PASSED",
            "detected_holes": len(holes),
            "detected_dimensions": len(dimensions),
            "is_extrusion_drawing": is_extrusion_drawing(ocr_items),
            "is_sheet_metal_drawing": is_sheet_metal_drawing(ocr_items),
            "down_up_with_angle_present": has_down_or_up_notation(ocr_items),
            "flat_pattern_note_present": has_flat_pattern_note(ocr_items),
            "radius_feature_present": geometry_features.get("radius_feature_present", False),
            "chamfer_feature_present": geometry_features.get("chamfer_feature_present", False),
            "circlip_groove_present": geometry_features.get("circlip_groove_present", False),
            "shaft_region_found": geometry_features.get("shaft_region_found", False),
            "slot_present": geometry_features.get("slot_present", False),
            "slot_count": geometry_features.get("slot_count", 0)
        },
        "issues": issues,
        "holes": holes,
        "dimensions": dimensions,
        "geometry_features": geometry_features,
        "rules": RULES,
        "din471_reference_table_used": DIN471_SHAFT_GROOVE_TABLE,
        "ocr_text_corrected": joined_ocr_text(ocr_items),
        "ocr_items": ocr_items
    }

    return json.dumps(report, indent=4, default=str)

# =========================================================
# MAIN APP FLOW
# =========================================================

if uploaded_file:
    original_pil, img = load_image(uploaded_file)

    if "corrected_img" not in st.session_state:
        st.session_state.corrected_img = img.copy()

    if "uploaded_name" not in st.session_state:
        st.session_state.uploaded_name = uploaded_file.name

    if "material_corrected" not in st.session_state:
        st.session_state.material_corrected = False

    if st.session_state.uploaded_name != uploaded_file.name:
        st.session_state.corrected_img = img.copy()
        st.session_state.uploaded_name = uploaded_file.name
        st.session_state.material_corrected = False

    gray, thresh, edges = preprocess(img)

    ocr_items = run_ocr(thresh)

    holes, hole_img = detect_holes(edges, img)

    dimensions = extract_dimensions(ocr_items)

    geometry_features = detect_geometry_features(edges, img, ocr_items)

    issues = validate(
        ocr_items,
        holes,
        dimensions,
        geometry_features,
        img.shape
    )

    annotated = annotate_issues(img, issues)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📄 Input",
        "🔍 OCR",
        "⚙ Geometry",
        "⚠ Validation",
        "🛠 Auto Correct",
        "🖼 Final Comparison",
        "📤 Export"
    ])

    with tab1:
        st.subheader("Original Drawing")
        st.image(original_pil, use_column_width=True)

        st.subheader("Preprocessed Image")
        st.image(thresh, use_column_width=True)

    with tab2:
        st.subheader("OCR Results With AI Correction")

        if ocr_items:
            df_ocr = pd.DataFrame([
                {
                    "Corrected Text": i["text"],
                    "Raw OCR Text": i.get("raw_text", ""),
                    "Confidence": i["confidence"],
                    "Height px": i["height"],
                    "X1": i["x1"],
                    "Y1": i["y1"]
                }
                for i in ocr_items
            ])

            st.dataframe(df_ocr)
            st.subheader("Joined Corrected OCR Text")
            st.write(joined_ocr_text(ocr_items))

            st.subheader("Drawing Type Detection")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Aluminium Extrusion",
                    is_extrusion_drawing(ocr_items)
                )

            with col2:
                st.metric(
                    "DOWN / UP With Angle Present",
                    has_down_or_up_notation(ocr_items)
                )

            with col3:
                st.metric(
                    "Flat Pattern Present",
                    has_flat_pattern_note(ocr_items)
                )

        else:
            st.warning("No OCR text detected")

    with tab3:
        st.subheader("Detected Geometry")

        st.image(
            cv2.cvtColor(hole_img, cv2.COLOR_BGR2RGB),
            use_column_width=True
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Detected Holes", len(holes))

        with c2:
            st.metric(
                "Radius Feature",
                geometry_features.get("radius_feature_present", False)
            )

        with c3:
            st.metric(
                "Chamfer Feature",
                geometry_features.get("chamfer_feature_present", False)
            )

        with c4:
            st.metric(
                "Slot Feature",
                geometry_features.get("slot_present", False)
            )

        c5, c6, c7 = st.columns(3)

        with c5:
            st.metric(
                "Slot Count",
                geometry_features.get("slot_count", 0)
            )

        with c6:
            st.metric(
                "Shaft Region",
                geometry_features.get("shaft_region_found", False)
            )

        with c7:
            st.metric(
                "Circlip Groove",
                geometry_features.get("circlip_groove_present", False)
            )

        st.subheader("Detected Dimensions")

        if dimensions:
            df_dim = pd.DataFrame([
                {
                    "Corrected Text": d["raw_text"],
                    "Original OCR": d.get("original_raw_text", ""),
                    "Value": d["value"],
                    "Diameter?": d["is_diameter"],
                    "Radius?": d["is_radius"],
                    "Chamfer?": d["is_chamfer"],
                    "Confidence": d["confidence"]
                }
                for d in dimensions
            ])

            st.dataframe(df_dim)
        else:
            st.warning("No dimensions detected")

        st.subheader("Raw Geometry Dictionary")
        st.json(geometry_features)

    with tab4:
        st.subheader("Validation Result")

        if issues:
            st.error("❌ CAD Drawing Failed Validation")

            issue_df = pd.DataFrame([
                {
                    "ID": i["id"],
                    "Severity": i["severity"],
                    "Type": i["type"],
                    "Message": i["message"],
                    "Suggestion": i["suggestion"]
                }
                for i in issues
            ])

            st.dataframe(issue_df)

            st.subheader("Issue Numbers Marked on Drawing")

            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                use_column_width=True
            )

        else:
            st.success("✔ CAD Drawing Passed Validation")

    with tab5:
        st.subheader("Auto Correction")

        material_issue_exists = any(
            i["type"] == "Material / Finish"
            for i in issues
        )

        if material_issue_exists:
            st.warning("Material / Finish error detected")

            if st.button("Auto Correct Material Finish"):
                st.session_state.corrected_img = auto_correct_material_finish(
                    img,
                    issues
                )

                st.session_state.material_corrected = True

                st.success("Finish corrected to CLEAR ANODIZE")

            preview = annotate_issues(
                st.session_state.corrected_img,
                issues,
                corrected_material=st.session_state.material_corrected
            )

            if st.session_state.material_corrected:
                preview = annotate_corrected_material_green(preview, issues)

            st.image(
                cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                use_column_width=True
            )

        else:
            st.success("No material finish auto-correction required")

    with tab6:
        st.subheader("Original vs Corrections Required")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Original Uploaded Drawing")
            st.image(original_pil, use_column_width=True)

        with col2:
            st.write("Drawing With Corrections Required")

            comparison_img = annotate_issues(
                st.session_state.corrected_img,
                issues,
                corrected_material=st.session_state.material_corrected
            )

            if st.session_state.material_corrected:
                comparison_img = annotate_corrected_material_green(
                    comparison_img,
                    issues
                )

            st.image(
                cv2.cvtColor(comparison_img, cv2.COLOR_BGR2RGB),
                use_column_width=True
            )

    with tab7:
        st.subheader("Export Inspection Report")

        report_json = make_report(
            issues,
            holes,
            dimensions,
            geometry_features,
            ocr_items
        )

        st.download_button(
            "Download JSON Report",
            report_json,
            file_name="cad_validation_report.json",
            mime="application/json"
        )

        export_img = annotate_issues(
            st.session_state.corrected_img,
            issues,
            corrected_material=st.session_state.material_corrected
        )

        if st.session_state.material_corrected:
            export_img = annotate_corrected_material_green(export_img, issues)

        annotated_rgb = cv2.cvtColor(export_img, cv2.COLOR_BGR2RGB)
        pil_out = Image.fromarray(annotated_rgb)

        buffer = io.BytesIO()
        pil_out.save(buffer, format="PNG")

        st.download_button(
            "Download Corrected / Marked Drawing",
            buffer.getvalue(),
            file_name="cad_corrected_marked_result.png",
            mime="image/png"
        )

else:
    st.info("Upload a CAD drawing image to begin.")

