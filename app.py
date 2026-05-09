import os
import uuid
import json
import math

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
from werkzeug.utils import secure_filename


# =========================
# Load config from .env
# =========================
load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "").strip()
ROBOFLOW_API_URL = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com").strip()
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "").strip()
ROBOFLOW_WORKFLOW_ID = os.getenv("ROBOFLOW_WORKFLOW_ID", "").strip()

UPLOAD_DIR = os.path.join("static", "uploads")
RESULT_DIR = os.path.join("static", "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB

client = InferenceHTTPClient(api_url=ROBOFLOW_API_URL, api_key=ROBOFLOW_API_KEY)


# =========================
# Roboflow + parsing helpers
# =========================
def run_workflow(image_path: str) -> dict:
    if not ROBOFLOW_API_KEY:
        raise RuntimeError("Missing ROBOFLOW_API_KEY in .env")
    if not ROBOFLOW_WORKSPACE:
        raise RuntimeError("Missing ROBOFLOW_WORKSPACE in .env")
    if not ROBOFLOW_WORKFLOW_ID:
        raise RuntimeError("Missing ROBOFLOW_WORKFLOW_ID in .env")

    return client.run_workflow(
        workspace_name=ROBOFLOW_WORKSPACE,
        workflow_id=ROBOFLOW_WORKFLOW_ID,
        images={"image": image_path},
        use_cache=True
    )


def _find_bbox_objects(obj):
    """
    Recursively find bbox-like dicts. Accepts objects that have x,y,width,height and class/label/class_name.
    """
    found = []
    if isinstance(obj, dict):
        keys = obj.keys()
        if {"x", "y", "width", "height"}.issubset(keys) and (
            ("class" in keys) or ("label" in keys) or ("class_name" in keys)
        ):
            found.append(obj)
        for v in obj.values():
            found.extend(_find_bbox_objects(v))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_find_bbox_objects(item))
    return found


def normalize_predictions(workflow_result) -> list[dict]:
    """
    Returns list of:
      {class, confidence, x, y, width, height}
    """
    preds_raw = _find_bbox_objects(workflow_result)
    normalized = []

    for p in preds_raw:
        cls = p.get("class") or p.get("label") or p.get("class_name") or "object"
        conf = p.get("confidence", p.get("conf", p.get("score", 0.0)))

        try:
            conf = float(conf)
        except Exception:
            continue

        try:
            x = float(p["x"])
            y = float(p["y"])
            w = float(p["width"])
            h = float(p["height"])
        except Exception:
            continue

        normalized.append({
            "class": str(cls),
            "confidence": conf,
            "x": x,
            "y": y,
            "width": w,
            "height": h
        })

    # Deduplicate
    seen = set()
    unique = []
    for p in normalized:
        k = (p["class"], round(p["confidence"], 4),
             round(p["x"], 1), round(p["y"], 1), round(p["width"], 1), round(p["height"], 1))
        if k not in seen:
            seen.add(k)
            unique.append(p)

    return unique


# =========================
# Image IO (Windows-safe)
# =========================
def read_image_cv(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image_cv(path: str, image_bgr: np.ndarray) -> None:
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("Failed to encode output image.")
    buf.tofile(path)


# =========================
# Filtering (FAST FIXES)
# =========================
def xywh_to_xyxy(p):
    x, y, w, h = p["x"], p["y"], p["width"], p["height"]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return x1, y1, x2, y2


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def nms(preds, iou_thresh=0.45, class_aware=True):
    """
    Non-max suppression. Keeps highest-confidence boxes, removes overlaps.
    """
    if not preds:
        return []

    preds_sorted = sorted(preds, key=lambda p: p["confidence"], reverse=True)
    kept = []

    for p in preds_sorted:
        p_box = xywh_to_xyxy(p)
        ok = True
        for k in kept:
            if class_aware and (k["class"] != p["class"]):
                continue
            if iou_xyxy(p_box, xywh_to_xyxy(k)) >= iou_thresh:
                ok = False
                break
        if ok:
            kept.append(p)

    return kept


def apply_fast_fixes(predictions, img_w, img_h,
                     min_conf=0.80,
                     min_area_frac=0.008,
                     nms_iou=0.45,
                     max_total=20,
                     max_per_class=12):
    """
    Applies:
    - confidence filter
    - min area filter
    - NMS
    - cap per class
    - cap total
    """
    img_area = img_w * img_h

    # 1) confidence filter
    preds = [p for p in predictions if p["confidence"] >= min_conf]

    # 2) min area filter
    min_area = min_area_frac * img_area
    preds = [p for p in preds if (p["width"] * p["height"]) >= min_area]

    # 3) NMS
    preds = nms(preds, iou_thresh=nms_iou, class_aware=True)

    # 4) sort & cap per class + total
    preds.sort(key=lambda p: p["confidence"], reverse=True)

    per_class = {}
    filtered = []
    for p in preds:
        c = p["class"]
        per_class[c] = per_class.get(c, 0)
        if per_class[c] >= max_per_class:
            continue
        filtered.append(p)
        per_class[c] += 1
        if len(filtered) >= max_total:
            break

    return filtered


# =========================
# Drawing
# =========================
def color_for_class(cls: str):
    c = cls.lower()
    if "coral" in c:
        return (0, 255, 0)      # green
    if "artifact" in c or "structure" in c:
        return (255, 160, 0)    # orange-ish
    return (0, 255, 255)        # yellow


def draw_boxes(image_bgr: np.ndarray, predictions: list[dict], small_mode: bool) -> np.ndarray:
    out = image_bgr.copy()
    H, W = out.shape[:2]

    thickness = 1 if small_mode else 2
    font_scale = 0.45 if small_mode else 0.60
    font_thickness = 1 if small_mode else 2

    for p in predictions:
        cls = p["class"]
        conf = p["confidence"]

        x1, y1, x2, y2 = xywh_to_xyxy(p)

        x1 = int(max(0, min(W - 1, x1)))
        y1 = int(max(0, min(H - 1, y1)))
        x2 = int(max(0, min(W - 1, x2)))
        y2 = int(max(0, min(H - 1, y2)))

        col = color_for_class(cls)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, thickness)

        # small labels: keep but small; if you want labels OFF, I can add a toggle too
        label = f"{cls} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        top = max(0, y1 - th - 6)
        cv2.rectangle(out, (x1, top), (x1 + tw + 6, y1), col, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), font_thickness)

    return out


# =========================
# Routes
# =========================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/detect", methods=["POST"])
def api_detect():
    try:
        if "image" not in request.files:
            return jsonify({"ok": False, "error": "No image uploaded"}), 400

        f = request.files["image"]
        if not f or f.filename == "":
            return jsonify({"ok": False, "error": "Empty filename"}), 400

        filename = secure_filename(f.filename)
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXTS:
            return jsonify({"ok": False, "error": "Unsupported file type. Use jpg/png/webp."}), 400

        # UI parameters (fast fixes)
        def get_float(name, default, lo=None, hi=None):
            v = request.form.get(name, default)
            try:
                v = float(v)
            except Exception:
                v = float(default)
            if lo is not None:
                v = max(lo, v)
            if hi is not None:
                v = min(hi, v)
            return v

        def get_int(name, default, lo=None, hi=None):
            v = request.form.get(name, default)
            try:
                v = int(float(v))
            except Exception:
                v = int(default)
            if lo is not None:
                v = max(lo, v)
            if hi is not None:
                v = min(hi, v)
            return v

        min_conf = get_float("min_conf", 0.80, 0.0, 1.0)
        min_area_frac = get_float("min_area_frac", 0.008, 0.0, 0.50)
        nms_iou = get_float("nms_iou", 0.45, 0.0, 1.0)
        max_total = get_int("max_total", 20, 1, 300)
        max_per_class = get_int("max_per_class", 12, 1, 300)
        small_mode = request.form.get("small_mode", "1") == "1"

        uid = str(uuid.uuid4())
        upload_path = os.path.join(UPLOAD_DIR, f"{uid}{ext}")
        result_path = os.path.join(RESULT_DIR, f"{uid}.jpg")

        f.save(upload_path)

        img = read_image_cv(upload_path)
        if img is None:
            return jsonify({"ok": False, "error": "Could not read image."}), 400

        H, W = img.shape[:2]

        # Run workflow -> normalize
        workflow_result = run_workflow(upload_path)
        raw_predictions = normalize_predictions(workflow_result)

        # Apply fast fixes
        predictions = apply_fast_fixes(
            raw_predictions, img_w=W, img_h=H,
            min_conf=min_conf,
            min_area_frac=min_area_frac,
            nms_iou=nms_iou,
            max_total=max_total,
            max_per_class=max_per_class
        )

        # Draw filtered
        out = draw_boxes(img, predictions, small_mode=small_mode)
        write_image_cv(result_path, out)

        original_url = "/" + upload_path.replace("\\", "/")
        result_url = "/" + result_path.replace("\\", "/")

        return jsonify({
            "ok": True,
            "original_url": original_url,
            "result_url": result_url,
            "predictions": predictions,
            "counts": {
                "raw": len(raw_predictions),
                "final": len(predictions)
            },
            "params": {
                "min_conf": min_conf,
                "min_area_frac": min_area_frac,
                "nms_iou": nms_iou,
                "max_total": max_total,
                "max_per_class": max_per_class,
                "small_mode": small_mode
            }
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)