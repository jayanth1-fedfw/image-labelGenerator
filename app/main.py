import os
import requests
import json
from flask import Flask, request, jsonify, render_template
from db import init_db, save_result, get_all_results
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL_URL = "https://api-inference.huggingface.co/models/google/vit-large-patch16-224"


# ─── Hugging Face Label Detection (FREE) ──────────────────────
def detect_labels_hf(image_bytes, content_type):
    """Send image to Hugging Face Vision model and return labels."""
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    response = requests.post(
        HF_MODEL_URL,
        headers=headers,
        data=image_bytes
    )

    if response.status_code != 200:
        raise RuntimeError(f"Hugging Face API error: {response.text}")

    predictions = response.json()

    if isinstance(predictions, dict) and "error" in predictions:
        raise RuntimeError(f"Model error: {predictions['error']}")

    labels = [
        {
            "label": item["label"].split(",")[0].strip(),
            "confidence": round(item["score"], 3),
            "tier": (
                "high" if item["score"] >= 0.85 else
                "medium" if item["score"] >= 0.50 else
                "low"
            )
        }
        for item in predictions[:12]
    ]
    return labels


# ─── Routes ───────────────────────────────────────────────────

@app.route("/")
def index():
    try:
        results = get_all_results()
    except Exception as e:
        results = []
        print(f"DB fetch error: {e}")
    return render_template("index.html", results=results)


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    allowed_types = {"image/jpeg", "image/png", "image/webp"}
    if file.content_type not in allowed_types:
        return jsonify({"error": f"Unsupported type: {file.content_type}"}), 415

    try:
        image_bytes = file.read()

        # Detect labels using Hugging Face (free)
        labels = detect_labels_hf(image_bytes, file.content_type)

        # Save to Supabase PostgreSQL
        record_id = save_result(
            filename=file.filename,
            gcs_url="local",
            labels=labels,
            mode="labels"
        )

        return jsonify({
            "success": True,
            "id": record_id,
            "filename": file.filename,
            "results": labels
        }), 200

    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@app.route("/history", methods=["GET"])
def history():
    limit = request.args.get("limit", 50, type=int)
    try:
        rows = get_all_results(limit=limit)
        return jsonify([
            {
                "id": r[0],
                "filename": r[1],
                "gcs_url": r[2],
                "labels": r[3],
                "mode": r[4],
                "created_at": str(r[5])
            }
            for r in rows
        ]), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# ─── Startup ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("Starting Flask on port 8080...")
    app.run(host="0.0.0.0", port=8080, debug=False)