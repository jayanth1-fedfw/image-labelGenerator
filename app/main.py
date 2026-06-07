from flask import Flask, request, jsonify, render_template
import os
import base64
import json
import requests
from dotenv import load_dotenv
from db import init_db, save_result, get_all_results

load_dotenv()
app = Flask(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_VISION_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
VALID_MODES = {"labels", "objects", "describe"}
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


def confidence_tier(confidence):
    try:
        score = float(confidence)
    except (TypeError, ValueError):
        return "low"

    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def normalize_mode(value):
    mode = (value or "labels").lower().strip()
    return mode if mode in VALID_MODES else "labels"


def image_to_data_url(image_bytes, mime_type):
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    safe_mime_type = mime_type if mime_type in ALLOWED_IMAGE_TYPES else "image/jpeg"
    return f"data:{safe_mime_type};base64,{encoded_image}"


def extract_json(content):
    if not content:
        return {}

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start:end + 1])
        raise


def normalize_labels(items):
    labels = []
    for item in items or []:
        if isinstance(item, str):
            label = item.strip()
            confidence = 0.5
        elif isinstance(item, dict):
            label = str(item.get("label") or item.get("name") or "").strip()
            confidence = item.get("confidence", 0.5)
        else:
            continue

        if not label:
            continue

        try:
            confidence = round(float(confidence), 3)
        except (TypeError, ValueError):
            confidence = 0.5

        labels.append({
            "label": label,
            "confidence": max(0, min(confidence, 1)),
            "tier": confidence_tier(confidence)
        })

    return labels[:12]


def normalize_objects(items):
    objects = []
    for item in items or []:
        if isinstance(item, str):
            name = item.strip()
            count = 1
            confidence = 0.5
        elif isinstance(item, dict):
            name = str(item.get("name") or item.get("label") or "").strip()
            count = item.get("count", 1)
            confidence = item.get("confidence", 0.5)
        else:
            continue

        if not name:
            continue

        try:
            count = max(1, int(count))
        except (TypeError, ValueError):
            count = 1

        try:
            confidence = round(float(confidence), 3)
        except (TypeError, ValueError):
            confidence = 0.5

        objects.append({
            "name": name,
            "count": count,
            "confidence": max(0, min(confidence, 1))
        })

    return objects[:12]


def analyze_image_openai(image_bytes, mime_type, mode):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured. Add it in Render environment variables. If you reused the old slot, HF_API_KEY also works as a temporary fallback.")

    prompt = (
        "Analyze this image for an image label generator app. Return only JSON with this exact shape: "
        "{\"labels\":[{\"label\":string,\"confidence\":number}],"
        "\"objects\":[{\"name\":string,\"count\":number,\"confidence\":number}],"
        "\"description\":string}. Confidence values must be between 0 and 1. "
        f"The active user mode is {mode}, so make that part especially accurate."
    )

    payload = {
        "model": OPENAI_MODEL,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": "You are a precise computer vision assistant. Return compact valid JSON only."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_data_url(image_bytes, mime_type),
                            "detail": "low"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 700
    }

    response = requests.post(
        OPENAI_API_URL,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=45
    )

    if response.status_code >= 400:
        try:
            error_message = response.json().get("error", {}).get("message", response.text)
        except ValueError:
            error_message = response.text
        raise RuntimeError(f"OpenAI vision request failed: {error_message[:300]}")

    data = response.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
    parsed = extract_json(content)

    if mode == "describe":
        return str(parsed.get("description") or "No description returned.")
    if mode == "objects":
        return normalize_objects(parsed.get("objects"))
    return normalize_labels(parsed.get("labels"))


@app.route("/")
def index():
    try:
        results = get_all_results(limit=6)
    except Exception as exc:
        print(f"Database read skipped: {exc}")
        results = []
    return render_template("index.html", results=results)


@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided."}), 400

        image = request.files["image"]
        if not image.filename:
            return jsonify({"error": "Empty filename."}), 400

        if image.content_type not in ALLOWED_IMAGE_TYPES:
            return jsonify({"error": f"Unsupported type: {image.content_type}"}), 415

        mode = normalize_mode(request.args.get("mode"))
        image_bytes = image.read()
        results = analyze_image_openai(image_bytes, image.content_type, mode)
        record_id = save_result(filename=image.filename, gcs_url="local", labels=results, mode=mode)

        return jsonify({
            "success": True,
            "id": record_id,
            "filename": image.filename,
            "mode": mode,
            "results": results
        }), 200

    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 502
    except Exception as exc:
        print(f"Unexpected upload error: {exc}")
        return jsonify({"error": "Unexpected server error while analyzing the image."}), 500


@app.route("/history", methods=["GET"])
def history():
    limit = request.args.get("limit", 50, type=int)
    try:
        rows = get_all_results(limit=limit)
        return jsonify([
            {
                "id": row[0],
                "filename": row[1],
                "gcs_url": row[2],
                "labels": row[3],
                "mode": row[4],
                "created_at": str(row[5])
            }
            for row in rows
        ]), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "provider": "openai", "model": OPENAI_MODEL}), 200


try:
    init_db()
except Exception as exc:
    print(f"Database init skipped: {exc}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)