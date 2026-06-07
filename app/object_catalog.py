import io
import os
import re
from collections import Counter

try:
    from PIL import Image, ImageStat
except ImportError:
    Image = None
    ImageStat = None

COMMON_OBJECTS = [
    ("person", "people", "human", "man", "woman", "child", "face", "selfie"),
    ("bicycle", "bike", "cycle"),
    ("car", "vehicle", "auto", "automobile", "taxi"),
    ("motorcycle", "motorbike", "scooter"),
    ("bus", "coach"),
    ("truck", "lorry"),
    ("train", "rail", "railway"),
    ("airplane", "plane", "aircraft"),
    ("boat", "ship", "vessel"),
    ("traffic light", "signal"),
    ("stop sign", "stop"),
    ("bench", "seat"),
    ("bird", "pigeon", "sparrow"),
    ("cat", "kitten"),
    ("dog", "puppy"),
    ("horse",),
    ("sheep",),
    ("cow",),
    ("elephant",),
    ("bear",),
    ("zebra",),
    ("giraffe",),
    ("backpack", "bag", "schoolbag"),
    ("umbrella",),
    ("handbag", "purse"),
    ("tie",),
    ("suitcase", "luggage"),
    ("frisbee",),
    ("skis",),
    ("snowboard",),
    ("sports ball", "ball", "football", "basketball", "cricket"),
    ("kite",),
    ("baseball bat", "bat"),
    ("baseball glove", "glove"),
    ("skateboard",),
    ("surfboard",),
    ("tennis racket", "racket"),
    ("bottle", "water bottle"),
    ("wine glass", "glass"),
    ("cup", "mug"),
    ("fork",),
    ("knife",),
    ("spoon",),
    ("bowl",),
    ("banana",),
    ("apple",),
    ("sandwich",),
    ("orange",),
    ("broccoli",),
    ("carrot",),
    ("hot dog",),
    ("pizza",),
    ("donut",),
    ("cake",),
    ("chair",),
    ("couch", "sofa"),
    ("potted plant", "plant"),
    ("bed",),
    ("dining table", "table", "desk"),
    ("toilet",),
    ("tv", "television", "monitor", "screen"),
    ("laptop", "computer"),
    ("mouse", "computer mouse"),
    ("remote", "remote control"),
    ("keyboard",),
    ("cell phone", "phone", "mobile", "smartphone"),
    ("microwave",),
    ("oven",),
    ("toaster",),
    ("sink",),
    ("refrigerator", "fridge"),
    ("book", "notebook"),
    ("clock", "watch"),
    ("vase",),
    ("scissors",),
    ("teddy bear", "toy"),
    ("hair drier", "dryer"),
    ("toothbrush",),
    ("door",),
    ("window",),
    ("curtain",),
    ("mirror",),
    ("pillow",),
    ("blanket",),
    ("wardrobe", "closet", "cabinet"),
    ("shelf",),
    ("lamp", "light"),
    ("fan",),
    ("camera",),
    ("headphones", "earphones"),
    ("shoes", "shoe", "sneakers"),
    ("shirt", "clothes", "clothing"),
    ("pants", "jeans"),
    ("hat", "cap"),
    ("wallet",),
    ("keys", "key"),
    ("pen", "pencil"),
    ("paper", "document"),
    ("medicine", "pill", "tablet"),
    ("guitar",),
    ("piano",),
]

OBJECT_DESCRIPTIONS = {
    "person": "a human subject, useful for people-focused image labeling",
    "cell phone": "a handheld mobile device commonly seen in daily photos",
    "laptop": "a portable computer often found on desks and tables",
    "chair": "a seating object commonly found indoors",
    "dining table": "a flat furniture surface such as a table or desk",
    "bed": "sleeping furniture commonly visible in room images",
    "book": "printed or notebook material often found on shelves or desks",
    "bottle": "a drink or container object with a narrow top",
    "cup": "a small drink container such as a mug or glass",
    "tv": "a display screen or monitor-like object",
    "backpack": "a carry bag used for school, travel, or daily items",
    "door": "an entry or room boundary object",
    "window": "a glass opening or bright rectangular room feature",
    "pillow": "a soft cushion often found on beds or couches",
    "blanket": "soft fabric used on beds or seating areas",
    "wardrobe": "storage furniture such as a cabinet or closet",
    "lamp": "a lighting object or visible light source",
    "shirt": "upper-body clothing or visible fabric",
}

COLOR_NAMES = {
    "black": (35, 35, 35),
    "white": (235, 235, 235),
    "gray": (128, 128, 128),
    "red": (190, 55, 55),
    "orange": (220, 130, 45),
    "yellow": (220, 205, 70),
    "green": (70, 150, 80),
    "blue": (65, 105, 185),
    "purple": (135, 85, 170),
    "pink": (220, 120, 170),
    "brown": (120, 80, 45),
}


def _tokenize(value):
    return set(re.findall(r"[a-z0-9]+", (value or "").lower()))


def _confidence_tier(confidence):
    if confidence >= 0.7:
        return "high"
    if confidence >= 0.4:
        return "medium"
    return "low"


def _label(label, confidence):
    confidence = max(0, min(round(float(confidence), 3), 1))
    return {"label": label, "confidence": confidence, "tier": _confidence_tier(confidence)}


def _object(name, confidence, count=1):
    return {"name": name, "count": count, "confidence": max(0, min(round(float(confidence), 3), 1))}


def _nearest_color(rgb):
    def distance(item):
        _, target = item
        return sum((rgb[i] - target[i]) ** 2 for i in range(3))

    return min(COLOR_NAMES.items(), key=distance)[0]


def _image_profile(image_bytes):
    if Image is None:
        return {
            "available": False,
            "width": 0,
            "height": 0,
            "orientation": "unknown",
            "brightness": "unknown",
            "colors": [],
        }

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            width, height = image.size
            thumb = image.resize((1, 1))
            dominant = thumb.getpixel((0, 0))
            stat = ImageStat.Stat(image.resize((64, 64)))
            brightness_value = sum(stat.mean) / 3
            orientation = "landscape" if width > height else "portrait" if height > width else "square"
            brightness = "bright" if brightness_value >= 175 else "dark" if brightness_value <= 75 else "moderate lighting"
            colors = [_nearest_color(dominant)]

            return {
                "available": True,
                "width": width,
                "height": height,
                "orientation": orientation,
                "brightness": brightness,
                "colors": colors,
            }
    except Exception:
        return {
            "available": False,
            "width": 0,
            "height": 0,
            "orientation": "unknown",
            "brightness": "unknown",
            "colors": [],
        }


def _filename_matches(filename):
    tokens = _tokenize(os.path.splitext(filename or "")[0])
    matches = []
    for names in COMMON_OBJECTS:
        primary = names[0]
        aliases = set()
        for name in names:
            aliases.update(_tokenize(name))
        if tokens & aliases:
            matches.append(primary)
    return matches


def _scene_hints(profile):
    hints = ["image"]
    if profile["available"]:
        hints.append(f"{profile['orientation']} photo")
        hints.append(profile["brightness"])
        hints.extend(f"{color} color tones" for color in profile["colors"])
    return hints


def _daily_object_guesses(profile):
    guesses = ["person", "chair", "dining table", "cell phone", "book", "bottle"]
    if profile["available"] and profile["orientation"] == "landscape":
        guesses.extend(["bed", "couch", "tv"])
    if profile["available"] and profile["orientation"] == "portrait":
        guesses.extend(["door", "window", "wardrobe"])
    if profile["available"] and "brown" in profile["colors"]:
        guesses.extend(["dining table", "door", "cabinet"])
    if profile["available"] and "white" in profile["colors"]:
        guesses.extend(["paper", "pillow", "wall"])
    return guesses


def analyze_image_local(image_bytes, filename, mode):
    profile = _image_profile(image_bytes)
    filename_matches = _filename_matches(filename)
    guessed_objects = filename_matches + _daily_object_guesses(profile)
    ordered = []
    seen = set()
    for name in guessed_objects:
        if name not in seen:
            seen.add(name)
            ordered.append(name)

    if mode == "objects":
        objects = []
        for index, name in enumerate(ordered[:10]):
            confidence = 0.66 if name in filename_matches else max(0.28, 0.46 - (index * 0.025))
            objects.append(_object(name, confidence))
        return objects

    scene_hints = _scene_hints(profile)
    if mode == "describe":
        size_text = "unknown size"
        if profile["available"]:
            size_text = f"{profile['width']}x{profile['height']} {profile['orientation']} image"
        object_text = ", ".join(ordered[:8])
        color_text = ", ".join(profile["colors"]) or "unknown"
        return (
            "No-AI catalog fallback: this is a "
            f"{size_text} with {profile['brightness']} and dominant {color_text} tones. "
            "Possible daily objects from the local common-object catalog include: "
            f"{object_text}. These are safe fallback hints, not AI-confirmed detections."
        )

    labels = [_label(name, 0.64 if name in filename_matches else 0.42) for name in ordered[:8]]
    labels.extend(_label(hint, 0.36) for hint in scene_hints[:4])
    return labels[:12]