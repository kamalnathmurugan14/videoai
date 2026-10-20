"""Static catalogs: voices, templates, caption styles, avatars."""
import json

from ..config import ASSETS_DIR

# (voice id, label, language label, gender)
VOICES = [
    ("en-US-AriaNeural", "Aria", "English (US)", "female"),
    ("en-US-GuyNeural", "Guy", "English (US)", "male"),
    ("en-US-JennyNeural", "Jenny", "English (US)", "female"),
    ("en-GB-SoniaNeural", "Sonia", "English (UK)", "female"),
    ("en-GB-RyanNeural", "Ryan", "English (UK)", "male"),
    ("en-AU-NatashaNeural", "Natasha", "English (Australia)", "female"),
    ("en-IN-NeerjaNeural", "Neerja", "English (India)", "female"),
    ("en-IN-PrabhatNeural", "Prabhat", "English (India)", "male"),
    ("es-ES-ElviraNeural", "Elvira", "Spanish (Spain)", "female"),
    ("es-MX-JorgeNeural", "Jorge", "Spanish (Mexico)", "male"),
    ("fr-FR-DeniseNeural", "Denise", "French", "female"),
    ("fr-FR-HenriNeural", "Henri", "French", "male"),
    ("de-DE-KatjaNeural", "Katja", "German", "female"),
    ("de-DE-ConradNeural", "Conrad", "German", "male"),
    ("it-IT-ElsaNeural", "Elsa", "Italian", "female"),
    ("pt-BR-FranciscaNeural", "Francisca", "Portuguese (Brazil)", "female"),
    ("hi-IN-SwaraNeural", "Swara", "Hindi", "female"),
    ("hi-IN-MadhurNeural", "Madhur", "Hindi", "male"),
    ("ta-IN-PallaviNeural", "Pallavi", "Tamil", "female"),
    ("ta-IN-ValluvarNeural", "Valluvar", "Tamil", "male"),
    ("ja-JP-NanamiNeural", "Nanami", "Japanese", "female"),
    ("ko-KR-SunHiNeural", "SunHi", "Korean", "female"),
    ("zh-CN-XiaoxiaoNeural", "Xiaoxiao", "Chinese (Mandarin)", "female"),
    ("ar-SA-ZariyahNeural", "Zariyah", "Arabic", "female"),
]
VOICE_IDS = {v[0] for v in VOICES}

CAPTION_STYLES = [
    {"id": "bold", "name": "Bold highlight", "description": "Big white text, current word in yellow"},
    {"id": "classic", "name": "Classic", "description": "White text on a dark bar"},
    {"id": "minimal", "name": "Minimal", "description": "Clean white text with soft shadow"},
]

ASPECTS = [
    {"id": "16:9", "name": "Landscape", "hint": "YouTube, web"},
    {"id": "9:16", "name": "Portrait", "hint": "TikTok, Reels, Shorts"},
    {"id": "1:1", "name": "Square", "hint": "Instagram, LinkedIn"},
]

GRADIENTS = [
    ["#4f46e5", "#9333ea"], ["#0ea5e9", "#22d3ee"], ["#f97316", "#ec4899"],
    ["#10b981", "#14b8a6"], ["#1e293b", "#475569"], ["#f43f5e", "#fb923c"],
]

TEMPLATES = [
    {
        "id": "explainer",
        "name": "Explainer",
        "description": "Introduce a problem, explain the idea, and close with a takeaway.",
        "aspect_ratio": "16:9",
        "background": {"type": "gradient", "color": "#0ea5e9", "color2": "#22d3ee"},
        "scenes": [
            "Hi there! Ever wondered how this works? In the next minute, I'll walk you through it.",
            "First, the problem. Most people spend hours on something that should only take minutes.",
            "Here is the idea: break it into small steps, and let simple tools do the heavy lifting.",
            "That's all it takes. Try it today, and see the difference for yourself.",
        ],
    },
    {
        "id": "social-promo",
        "name": "Social promo",
        "description": "Short, punchy vertical video with a hook and a call to action.",
        "aspect_ratio": "9:16",
        "background": {"type": "gradient", "color": "#f97316", "color2": "#ec4899"},
        "scenes": [
            "Stop scrolling! This will change how you create videos.",
            "No camera. No studio. Just type your script and press render.",
            "Follow for more tips, and share this with a friend who needs it!",
        ],
    },
    {
        "id": "product-intro",
        "name": "Product intro",
        "description": "Present your product: what it is, why it matters, and how to get it.",
        "aspect_ratio": "16:9",
        "background": {"type": "gradient", "color": "#4f46e5", "color2": "#9333ea"},
        "scenes": [
            "Meet your new favourite tool. Simple, fast, and built for people like you.",
            "It saves you time, cuts the busywork, and helps you focus on what matters.",
            "Getting started is free and takes less than a minute. Visit our site and give it a try.",
        ],
    },
]


def load_avatars() -> list[dict]:
    path = ASSETS_DIR / "avatars" / "avatars.json"
    return json.loads(path.read_text()) if path.exists() else []


def get_avatar(avatar_id: str) -> dict | None:
    return next((a for a in load_avatars() if a["id"] == avatar_id), None)


def load_sceneries() -> list[dict]:
    path = ASSETS_DIR / "sceneries" / "sceneries.json"
    return json.loads(path.read_text()) if path.exists() else []


def get_scenery(scenery_id: str) -> dict | None:
    return next((s for s in load_sceneries() if s["id"] == scenery_id), None)
