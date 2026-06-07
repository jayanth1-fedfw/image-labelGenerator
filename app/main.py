from flask import Flask, request, jsonify, render_template
import os
import base64
import json
import requests
from dotenv import load_dotenv
from db import init_db, save_result, get_all_results
from object_catalog import analyze_image_local

load_dotenv()
app = Flask(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_VISION_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

HF_API_KEY = os.getenv("HF_API_KEY")
HF_API_URL = "https://api-inference.huggingface.co/models"
HF_LABEL_MODEL = os.getenv("HF_LABEL_MODEL", "google/vit-large-patch16-224")
HF_OBJECT_MODEL = os.getenv("HF_OBJECT_MODEL", "facebook/detr-resnet-50")
HF_DESCRIBE_MODEL = os.getenv("HF_DESCRIBE_MODEL", "Salesforce/blip-image-captioning-large")

DEFAULT_PROVIDER = os.getenv("AI_PROVIDER", "auto").lower().strip()
VALID_PROVIDERS = {"auto", "openai", "huggingface", "local"}
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


def normalize_provider(value):
    provider = (value or DEFAULT_PROVIDER or "auto").lower().strip()
    return provider if provider in VALID_PROVIDERS else "auto"


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


def clamp_confidence(confidence):
    try:
        score = round(float(confidence), 3)
    except (TypeError, ValueError):
        score = 0.5
    return max(0, min(score, 1))


def normalize_labels(items):
    labels = []
    for item in items or []:
        if isinstance(item, str):
            label = item.strip()
            confidence = 0.5
        elif isinstance(item, dict):
            label = str(item.get("label") or item.get("name") or "").strip()
            confidence = item.get("confidence", item.get("score", 0.5))
        else:
            continue

        if not label:
            continue

        confidence = clamp_confidence(confidence)
        labels.append({
            "label": label.split(",")[0].strip(),
            "confidence": confidence,
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
            confidence = item.get("confidence", item.get("score", 0.5))
        else:
            continue

        if not name:
            continue

        try:
            count = max(1, int(count))
        except (TypeError, ValueError):
            count = 1

        objects.append({
            "name": name,
            "count": count,
            "confidence": clamp_confidence(confidence)
        })

    return objects[:12]


def analyze_image_openai(image_bytes, mime_type, mode):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

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


def call_huggingface_model(model, image_bytes):
    if not HF_API_KEY:
        raise RuntimeError("HF_API_KEY is not configured.")

    response = requests.post(
        f"{HF_API_URL}/{model}",
        headers={"Authorization": f"Bearer {HF_API_KEY}"},
        data=image_bytes,
        timeout=60
    )

    if response.status_code >= 400:
        try:
            payload = response.json()
            error_message = payload.get("error", response.text) if isinstance(payload, dict) else response.text
        except ValueError:
            error_message = response.text
        raise RuntimeError(f"Hugging Face request failed: {error_message[:300]}")

    payload = response.json()
    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(f"Hugging Face model error: {str(payload['error'])[:300]}")
    return payload


def analyze_image_huggingface(image_bytes, mode):
    if mode == "objects":
        payload = call_huggingface_model(HF_OBJECT_MODEL, image_bytes)
        return normalize_objects(payload)

    if mode == "describe":
        payload = call_huggingface_model(HF_DESCRIBE_MODEL, image_bytes)
        if isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, dict):
                return str(first.get("generated_text") or first.get("label") or "No description returned.")
            return str(first)
        if isinstance(payload, dict):
            return str(payload.get("generated_text") or payload.get("label") or "No description returned.")
        return "No description returned."

    payload = call_huggingface_model(HF_LABEL_MODEL, image_bytes)
    return normalize_labels(payload)


def analyze_image(image_bytes, mime_type, mode, provider, filename=""):
    if provider == "openai":
        return analyze_image_openai(image_bytes, mime_type, mode), "openai"

    if provider == "huggingface":
        return analyze_image_huggingface(image_bytes, mode), "huggingface"

    if provider == "local":
        return analyze_image_local(image_bytes, filename, mode), "local"

    errors = []
    if OPENAI_API_KEY:
        try:
            return analyze_image_openai(image_bytes, mime_type, mode), "openai"
        except RuntimeError as exc:
            errors.append(str(exc))

    if HF_API_KEY:
        try:
            return analyze_image_huggingface(image_bytes, mode), "huggingface"
        except RuntimeError as exc:
            errors.append(str(exc))

    try:
        return analyze_image_local(image_bytes, filename, mode), "local"
    except Exception as exc:
        errors.append(f"Local fallback failed: {exc}")

    if errors:
        raise RuntimeError("All providers failed. " + " | ".join(errors[-3:]))
    raise RuntimeError("No provider returned results.")


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
        provider = normalize_provider(request.args.get("provider"))
        image_bytes = image.read()
        results, used_provider = analyze_image(image_bytes, image.content_type, mode, provider, image.filename)
        record_id = None
        try:
            record_id = save_result(filename=image.filename, gcs_url="local", labels=results, mode=mode)
        except Exception as exc:
            print(f"Database save skipped: {exc}")

        return jsonify({
            "success": True,
            "id": record_id,
            "filename": image.filename,
            "mode": mode,
            "provider": used_provider,
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
    return jsonify({
        "status": "healthy",
        "default_provider": DEFAULT_PROVIDER,
        "providers": {
            "openai": bool(OPENAI_API_KEY),
            "huggingface": bool(HF_API_KEY),
            "local": True
        },
        "models": {
            "openai": OPENAI_MODEL,
            "hf_labels": HF_LABEL_MODEL,
            "hf_objects": HF_OBJECT_MODEL,
            "hf_describe": HF_DESCRIBE_MODEL
        }
    }), 200


try:
    init_db()
except Exception as exc:
    print(f"Database init skipped: {exc}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)