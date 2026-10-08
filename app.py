import os
import json
import uuid
import time
import re
import subprocess
import threading
import shutil
import hashlib
import math
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from google import genai
from google.genai import types

from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# APP
# ============================================================

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JOBS = {}
JOBS_LOCK = threading.Lock()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

BASE_URL = os.getenv(
    "BASE_URL",
    ""
).strip().rstrip("/")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
).strip()

MAX_SCENES = 12
MAX_DIALOGUE_LINES = 100
MAX_VISIBLE_CHARACTERS = 8

WIDTH = 480
HEIGHT = 854

# Higher FPS gives smoother animation.
FPS = 12


# ============================================================
# BASIC HELPERS
# ============================================================

def now():
    return time.time()


def safe_filename(value):
    value = str(value or "")
    value = re.sub(
        r"[^a-zA-Z0-9._-]+",
        "_",
        value
    )
    return value[:120] or "video"


def normalize_name(value):
    return re.sub(
        r"\s+",
        " ",
        str(value or "").strip()
    )


def name_key(value):
    return normalize_name(value).lower()


def deterministic_number(value):
    digest = hashlib.md5(
        str(value).encode("utf-8")
    ).hexdigest()

    return int(
        digest[:10],
        16
    )


def clamp(value, minimum, maximum):
    try:
        return max(
            minimum,
            min(maximum, value)
        )
    except Exception:
        return minimum


# ============================================================
# FONT
# ============================================================

def get_font(size=22, bold=False):

    candidates = []

    if bold:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ]
    else:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]

    for path in candidates:

        if os.path.exists(path):

            try:
                return ImageFont.truetype(
                    path,
                    size
                )
            except Exception:
                pass

    return ImageFont.load_default()


# ============================================================
# COLOR SAFETY
# ============================================================

COLOR_MAP = {
    "black": "#151515",
    "white": "#FFFFFF",

    "red": "#D64545",
    "dark red": "#8F2020",
    "light red": "#F06A6A",

    "blue": "#3978D4",
    "dark blue": "#214C91",
    "light blue": "#73B7E6",
    "navy": "#23395D",

    "green": "#3E9B61",
    "dark green": "#23613B",
    "light green": "#7BC98D",

    "yellow": "#F2C94C",
    "dark yellow": "#B99116",
    "light yellow": "#F8E27A",

    "orange": "#E67E22",
    "dark orange": "#A94F12",
    "light orange": "#F6A45C",

    "purple": "#8055B8",
    "dark purple": "#513274",
    "light purple": "#B58ADD",

    "pink": "#E58AAA",
    "dark pink": "#A94C6D",
    "light pink": "#F2B8CA",

    "brown": "#7A4B2A",
    "dark brown": "#5A321E",
    "light brown": "#A96B42",
    "deep brown": "#4A2818",

    "gray": "#777777",
    "grey": "#777777",
    "dark gray": "#444444",
    "dark grey": "#444444",
    "light gray": "#BBBBBB",
    "light grey": "#BBBBBB",

    "teal": "#3C8D8D",
    "cyan": "#42AFC0",
    "gold": "#D6A62E",
    "beige": "#D9C3A1",
    "cream": "#F2E4C8",
    "maroon": "#722F37",
    "olive": "#68733C",
}


def color_to_hex(value, fallback="#777777"):

    if value is None:
        return fallback

    text = str(value).strip().lower()

    if not text:
        return fallback

    if re.fullmatch(
        r"#[0-9a-f]{6}",
        text
    ):
        return text.upper()

    if re.fullmatch(
        r"#[0-9a-f]{3}",
        text
    ):
        return (
            "#"
            + text[1] * 2
            + text[2] * 2
            + text[3] * 2
        ).upper()

    rgb_match = re.fullmatch(
        r"rgb\s*\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)",
        text
    )

    if rgb_match:

        r = clamp(
            int(rgb_match.group(1)),
            0,
            255
        )

        g = clamp(
            int(rgb_match.group(2)),
            0,
            255
        )

        b = clamp(
            int(rgb_match.group(3)),
            0,
            255
        )

        return f"#{r:02X}{g:02X}{b:02X}"

    if text in COLOR_MAP:
        return COLOR_MAP[text]

    for color_name in sorted(
        COLOR_MAP.keys(),
        key=len,
        reverse=True
    ):

        if color_name in text:
            return COLOR_MAP[color_name]

    return fallback


# ============================================================
# CHARACTER DESIGN
# ============================================================

SKIN_TONES = [
    "#3D2418",
    "#512F20",
    "#693C27",
    "#7B4A30",
    "#8D5838",
    "#A96B42",
    "#C4875B",
    "#D49A6A",
    "#E0AE83",
]

SHIRT_COLORS = [
    "#D94A4A",
    "#3E75C6",
    "#3E9B61",
    "#D5A72A",
    "#7C55B7",
    "#E27B2F",
    "#2C8791",
    "#A95075",
]

PANTS_COLORS = [
    "#29384F",
    "#354A35",
    "#503D2D",
    "#4D4D55",
    "#202020",
    "#6A4C3B",
]

HAIR_COLORS = [
    "#15100D",
    "#2A1710",
    "#3A2115",
    "#51311F",
    "#6A452D",
]

HAIR_STYLES = [
    "short",
    "curly",
    "afro",
    "braids",
    "bob",
    "long",
    "high",
    "side",
]


def build_character_profile(
    name,
    index,
    raw_profile=None
):

    name = normalize_name(name)

    seed = deterministic_number(name)

    skin = SKIN_TONES[
        (seed + index)
        % len(SKIN_TONES)
    ]

    shirt = SHIRT_COLORS[
        (seed // 7 + index)
        % len(SHIRT_COLORS)
    ]

    pants = PANTS_COLORS[
        (seed // 13 + index)
        % len(PANTS_COLORS)
    ]

    hair = HAIR_COLORS[
        (seed // 19 + index)
        % len(HAIR_COLORS)
    ]

    hairstyle = HAIR_STYLES[
        (seed // 23 + index)
        % len(HAIR_STYLES)
    ]

    gender = ""
    age = ""

    if isinstance(raw_profile, dict):

        gender = str(
            raw_profile.get(
                "gender",
                ""
            )
        ).strip()

        age = str(
            raw_profile.get(
                "age",
                ""
            )
        ).strip()

        supplied_skin = (
            raw_profile.get("skin_tone")
            or raw_profile.get("skin")
        )

        supplied_shirt = (
            raw_profile.get("clothing_color")
            or raw_profile.get("shirt_color")
        )

        supplied_pants = (
            raw_profile.get("pants_color")
            or raw_profile.get("trouser_color")
        )

        supplied_hair = (
            raw_profile.get("hair_color")
            or raw_profile.get("hair")
        )

        supplied_style = (
            raw_profile.get("hair_style")
            or raw_profile.get("hairstyle")
        )

        if supplied_skin:
            skin = color_to_hex(
                supplied_skin,
                skin
            )

        if supplied_shirt:
            shirt = color_to_hex(
                supplied_shirt,
                shirt
            )

        if supplied_pants:
            pants = color_to_hex(
                supplied_pants,
                pants
            )

        if supplied_hair:
            hair = color_to_hex(
                supplied_hair,
                hair
            )

        if supplied_style:

            style = str(
                supplied_style
            ).lower()

            for possible in HAIR_STYLES:

                if possible in style:
                    hairstyle = possible
                    break

    return {
        "name": name,
        "skin": skin,
        "shirt": shirt,
        "pants": pants,
        "hair": hair,
        "hairstyle": hairstyle,
        "gender": gender,
        "age": age,
        "clothing": (
            f"{shirt} shirt and "
            f"{pants} trousers"
        ),
        "clothing_type": "shirt",
        "current_outfit": {
            "shirt": shirt,
            "pants": pants
        }
    }


# ============================================================
# CLOTHING
# ============================================================

def apply_clothing_instruction(
    profile,
    clothing_text
):

    if not clothing_text:
        return profile

    text = str(
        clothing_text
    ).lower()

    found_colors = []

    for color_name in sorted(
        COLOR_MAP.keys(),
        key=len,
        reverse=True
    ):

        if color_name in text:

            if COLOR_MAP[color_name] not in found_colors:

                found_colors.append(
                    COLOR_MAP[color_name]
                )

    if found_colors:
        profile["shirt"] = found_colors[0]

    if len(found_colors) >= 2:
        profile["pants"] = found_colors[1]

    if "dress" in text:
        profile["clothing_type"] = "dress"

    elif "suit" in text:
        profile["clothing_type"] = "suit"

    elif "jacket" in text:
        profile["clothing_type"] = "jacket"

    elif "uniform" in text:
        profile["clothing_type"] = "uniform"

    elif "traditional" in text:
        profile["clothing_type"] = "traditional"

    elif "t-shirt" in text or "tee" in text:
        profile["clothing_type"] = "tshirt"

    elif "shirt" in text:
        profile["clothing_type"] = "shirt"

    profile["clothing_prompt"] = clothing_text

    profile["current_outfit"] = {
        "shirt": profile["shirt"],
        "pants": profile["pants"]
    }

    return profile


# ============================================================
# JSON
# ============================================================

def clean_json(text):

    text = str(
        text or ""
    ).strip()

    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"^```\s*",
        "",
        text
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        text = text[
            start:end + 1
        ]

    return text


# ============================================================
# EXPLICIT SCRIPT DETECTION
# ============================================================

def looks_like_explicit_script(text):

    if not text:
        return False

    text = str(text)

    if re.search(
        r"(?im)^\s*(scene|escena)\s*\d*",
        text
    ):
        return True

    if re.search(
        r"(?im)^\s*[^:\n]{1,60}\s+(?:says|asks|replies|responds)\s*:",
        text
    ):
        return True

    if re.search(
        r"(?m)^\s*[^:\n]{1,60}\s*:\s*[\"“].+[\"”]\s*$",
        text
    ):
        return True

    if re.search(
        r"(?im)^\s*(action|visual|location|setting|camera|time)\s*:",
        text
    ):
        return True

    return False


# ============================================================
# RESERVED LABELS
# ============================================================

RESERVED_LABELS = {
    "scene",
    "location",
    "time",
    "action",
    "visual",
    "camera",
    "setting",
    "characters",
    "character",
    "narrator",
}


# ============================================================
# SCRIPT DIALOGUE EXTRACTION
# ============================================================

def extract_natural_dialogue(line):
    """
    Supports:

    Sarah says: "Hello."
    David says: "Yes."
    David stands up and says: "Come here."
    Sarah walks into the room and says: "Hi."

    Returns:
        character name,
        dialogue,
        action
    """

    line = str(line or "").strip()

    if not line:
        return None

    # --------------------------------------------------------
    # Quoted dialogue
    # --------------------------------------------------------

    quoted = re.search(
        r'["“](.*?)["”]\s*$',
        line
    )

    if not quoted:
        return None

    dialogue_text = quoted.group(1).strip()

    if not dialogue_text:
        return None

    before = line[
        :quoted.start()
    ].strip()

    before = re.sub(
        r"[:\-–—]\s*$",
        "",
        before
    ).strip()

    # --------------------------------------------------------
    # "David says"
    # --------------------------------------------------------

    speaker_match = re.match(
        r"^(.+?)\s+"
        r"(?:says|said|asks|asked|replies|replied|"
        r"responds|responded|shouts|shouted|"
        r"yells|yelled|whispers|whispered)"
        r"\s*$",
        before,
        flags=re.IGNORECASE
    )

    if speaker_match:

        speaker = normalize_name(
            speaker_match.group(1)
        )

        return {
            "speaker": speaker,
            "text": dialogue_text,
            "action": "",
            "emotion": "talking"
        }

    # --------------------------------------------------------
    # "David stands up and says"
    # --------------------------------------------------------

    speaker_match = re.match(
        r"^(.+?)\s+"
        r"(.+?)\s+"
        r"(?:says|said|asks|asked|replies|replied|"
        r"responds|responded|shouts|shouted|"
        r"yells|yelled|whispers|whispered)"
        r"\s*$",
        before,
        flags=re.IGNORECASE
    )

    if speaker_match:

        speaker = normalize_name(
            speaker_match.group(1)
        )

        action = normalize_name(
            speaker_match.group(2)
        )

        return {
            "speaker": speaker,
            "text": dialogue_text,
            "action": action,
            "emotion": "talking"
        }

    return None


# ============================================================
# EXPLICIT SCRIPT PARSER
# ============================================================

def parse_explicit_script(
    script,
    category="General",
    language="English"
):

    script = str(
        script or ""
    ).strip()

    if not script:
        return None

    if not looks_like_explicit_script(script):
        return None

    lines = script.splitlines()

    scenes = []
    current_scene = None

    character_names = []
    character_profiles = {}

    def new_scene(
        number=None,
        title=""
    ):

        scene_number = (
            number
            if number is not None
            else len(scenes) + 1
        )

        return {
            "scene": scene_number,
            "title": (
                title
                or f"Scene {scene_number}"
            ),
            "location": "open area",
            "time": "day",
            "camera": "wide shot",
            "action": "",
            "characters_present": [],
            "dialogue": [],
            "character_states": {}
        }

    def ensure_scene():

        nonlocal current_scene

        if current_scene is None:
            current_scene = new_scene()

    def add_character(name):

        name = normalize_name(name)

        if not name:
            return

        if name_key(name) in RESERVED_LABELS:
            return

        if not any(
            name_key(x) == name_key(name)
            for x in character_names
        ):
            character_names.append(name)

    def add_present(
        scene,
        name
    ):

        name = normalize_name(name)

        add_character(name)

        if not any(
            name_key(x)
            == name_key(name)
            for x in scene[
                "characters_present"
            ]
        ):
            scene[
                "characters_present"
            ].append(name)

    for raw_line in lines:

        line = raw_line.strip()

        if not line:
            continue

        # ====================================================
        # SCENE HEADER
        # ====================================================

        scene_match = re.match(
            r"^\s*(?:SCENE|Scene|ESCENA)\s*"
            r"(\d+)?\s*"
            r"(?:[:\-–—]\s*)?"
            r"(.*)$",
            line
        )

        if scene_match:

            if current_scene is not None:

                scenes.append(
                    current_scene
                )

            number_text = (
                scene_match.group(1)
            )

            title = (
                scene_match.group(2).strip()
                or f"Scene {len(scenes) + 1}"
            )

            number = (
                int(number_text)
                if number_text
                else len(scenes) + 1
            )

            current_scene = new_scene(
                number,
                title
            )

            continue

        ensure_scene()

        # ====================================================
        # LOCATION
        # ====================================================

        location_match = re.match(
            r"^\s*(?:LOCATION|SETTING)\s*:\s*(.+)$",
            line,
            re.IGNORECASE
        )

        if location_match:

            current_scene["location"] = (
                location_match.group(1).strip()
            )

            continue

        # ====================================================
        # TIME
        # ====================================================

        time_match = re.match(
            r"^\s*TIME\s*:\s*(.+)$",
            line,
            re.IGNORECASE
        )

        if time_match:

            current_scene["time"] = (
                time_match.group(1).strip()
            )

            continue

        # ====================================================
        # CAMERA
        # ====================================================

        camera_match = re.match(
            r"^\s*CAMERA\s*:\s*(.+)$",
            line,
            re.IGNORECASE
        )

        if camera_match:

            current_scene["camera"] = (
                camera_match.group(1).strip()
            )

            continue

        # ====================================================
        # ACTION / VISUAL
        # ====================================================

        action_match = re.match(
            r"^\s*(?:ACTION|VISUAL)\s*:\s*(.+)$",
            line,
            re.IGNORECASE
        )

        if action_match:

            action_text = (
                action_match.group(1).strip()
            )

            if current_scene["action"]:
                current_scene["action"] += " "

            current_scene["action"] += action_text

            continue

        # ====================================================
        # NATURAL DIALOGUE
        # ====================================================

        natural_dialogue = extract_natural_dialogue(
            line
        )

        if natural_dialogue:

            speaker = normalize_name(
                natural_dialogue["speaker"]
            )

            if (
                speaker
                and name_key(speaker)
                not in RESERVED_LABELS
            ):

                add_present(
                    current_scene,
                    speaker
                )

                current_scene[
                    "dialogue"
                ].append(
                    natural_dialogue
                )

                # Save action too
                if natural_dialogue.get(
                    "action"
                ):

                    if current_scene["action"]:
                        current_scene["action"] += " "

                    current_scene["action"] += (
                        speaker
                        + " "
                        + natural_dialogue["action"]
                        + "."
                    )

                continue

        # ====================================================
        # STANDARD:
        #
        # David: "Hello"
        # ====================================================

        standard_match = re.match(
            r'^\s*([^:]{1,60}?)\s*:\s*'
            r'["“](.*?)["”]\s*$',
            line
        )

        if standard_match:

            speaker = normalize_name(
                standard_match.group(1)
            )

            dialogue_text = (
                standard_match.group(2).strip()
            )

            if (
                speaker
                and name_key(speaker)
                not in RESERVED_LABELS
                and dialogue_text
            ):

                add_present(
                    current_scene,
                    speaker
                )

                current_scene[
                    "dialogue"
                ].append({
                    "speaker": speaker,
                    "text": dialogue_text,
                    "action": "",
                    "emotion": "talking"
                })

                continue

        # ====================================================
        # PLAIN:
        #
        # David: Hello
        # ====================================================

        plain_match = re.match(
            r"^\s*([^:]{1,60}?)\s*:\s*(.+?)\s*$",
            line
        )

        if plain_match:

            speaker = normalize_name(
                plain_match.group(1)
            )

            dialogue_text = (
                plain_match.group(2).strip()
            )

            if (
                speaker
                and name_key(speaker)
                not in RESERVED_LABELS
                and dialogue_text
            ):

                add_present(
                    current_scene,
                    speaker
                )

                current_scene[
                    "dialogue"
                ].append({
                    "speaker": speaker,
                    "text": dialogue_text,
                    "action": "",
                    "emotion": "talking"
                })

                continue

        # ====================================================
        # STAR ACTION
        # ====================================================

        if (
            line.startswith("*")
            and line.endswith("*")
        ):

            action = line.strip(
                "* "
            ).strip()

            if current_scene["action"]:
                current_scene["action"] += " "

            current_scene["action"] += action

            continue

        # ====================================================
        # NORMAL ACTION
        # ====================================================

        if current_scene["action"]:
            current_scene["action"] += " "

        current_scene["action"] += line

        # Try to detect names from obvious action
        for known_name in character_names:

            if name_key(known_name) in name_key(line):
                add_present(
                    current_scene,
                    known_name
                )

    if current_scene is not None:
        scenes.append(current_scene)

    if not scenes:
        return None

    # ========================================================
    # FIND CHARACTERS FROM DIALOGUE/ACTIONS
    # ========================================================

    for scene in scenes:

        for name in scene.get(
            "characters_present",
            []
        ):
            add_character(name)

        for dialogue in scene.get(
            "dialogue",
            []
        ):

            add_character(
                dialogue.get(
                    "speaker",
                    ""
                )
            )

    # ========================================================
    # CREATE CHARACTER PROFILES
    # ========================================================

    characters = []

    for index, name in enumerate(
        character_names
    ):

        profile = build_character_profile(
            name,
            index
        )

        characters.append(
            profile
        )

        character_profiles[
            name_key(name)
        ] = profile

    # ========================================================
    # CLOTHING CHANGES
    # ========================================================

    clothing_patterns = [

        r"([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9 _'-]{0,50})\s+"
        r"(?:is\s+now\s+wearing|changes\s+into|"
        r"wears|puts\s+on|is\s+dressed\s+in)\s+"
        r"(.+?)(?:\.|$)",

        r"([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9 _'-]{0,50})\s+"
        r"(?:is\s+wearing|wearing)\s+"
        r"(.+?)(?:\.|$)"
    ]

    # ========================================================
    # IMPORTANT:
    # Carry clothing state from scene to scene.
    # ========================================================

    state_profiles = {}

    for key, profile in character_profiles.items():

        state_profiles[key] = dict(profile)

    for scene in scenes:

        scene["character_states"] = {}

        scene_text = (
            str(scene.get("action", ""))
        )

        for dialogue in scene.get(
            "dialogue",
            []
        ):

            scene_text += " "
            scene_text += str(
                dialogue.get(
                    "action",
                    ""
                )
            )

        # Detect clothing instructions
        for pattern in clothing_patterns:

            matches = re.findall(
                pattern,
                scene_text,
                flags=re.IGNORECASE
            )

            for character_name, clothing in matches:

                key = name_key(
                    character_name
                )

                profile = state_profiles.get(
                    key
                )

                if profile:

                    apply_clothing_instruction(
                        profile,
                        clothing
                    )

        # Snapshot current clothing
        for name in scene.get(
            "characters_present",
            []
        ):

            key = name_key(name)

            profile = state_profiles.get(
                key
            )

            if profile:

                scene[
                    "character_states"
                ][key] = {
                    "shirt": profile["shirt"],
                    "pants": profile["pants"],
                    "clothing_type": profile.get(
                        "clothing_type",
                        "shirt"
                    )
                }

    # Apply final profile state
    for key, profile in state_profiles.items():

        if key in character_profiles:

            character_profiles[key].update(
                profile
            )

    return {
        "title": scenes[0].get(
            "title",
            "Cartoon Movie"
        ),
        "category": category,
        "language": language,
        "characters": list(
            character_profiles.values()
        ),
        "scenes": scenes,
        "_explicit_script": True
    }


# ============================================================
# GEMINI
# ============================================================

def get_gemini_client():

    if not GEMINI_API_KEY:

        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    return genai.Client(
        api_key=GEMINI_API_KEY
    )


def generate_story(
    category,
    topic,
    duration,
    language
):

    client = get_gemini_client()

    prompt = f"""
You are a professional cartoon movie director,
screenwriter and storyboard artist.

Create a structured 2D cartoon movie.

CATEGORY:
{category}

USER IDEA:
{topic}

TARGET DURATION:
{duration} seconds

LANGUAGE:
{language}

RULES:

1. Use character names that fit the user's story.

2. Never hard-code names such as Kofi or Amina
   unless the user actually requested those names.

3. Keep every character visually consistent.

4. Characters must have different appearances.

5. Every scene must have a real environment.

6. Characters can:
   - walk
   - enter
   - leave
   - sit
   - stand
   - face another character
   - point
   - wave
   - hold objects
   - perform simple actions.

7. When characters talk, they should face each other.

8. Only the current speaker should have talking
   mouth animation.

9. Use cinematic camera shots.

10. Make it feel like a cartoon movie rather than
    a slideshow.

11. Return ONLY valid JSON.

JSON:

{{
  "title": "Movie title",

  "characters": [
    {{
      "name": "character name",
      "gender": "optional",
      "age": "optional",
      "skin_tone": "brown",
      "hair_color": "black",
      "hair_style": "short",
      "clothing_color": "blue"
    }}
  ],

  "scenes": [
    {{
      "scene": 1,
      "title": "Scene title",
      "location": "living room",
      "time": "day",
      "camera": "wide shot",
      "action": "Character enters the room.",
      "characters_present": [
        "character name"
      ],
      "dialogue": [
        {{
          "speaker": "character name",
          "text": "Hello.",
          "action": "walks toward another character",
          "emotion": "happy",
          "facing": "another character"
        }}
      ]
    }}
  ]
}}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.15,
            response_mime_type="application/json"
        )
    )

    raw = clean_json(
        response.text
    )

    try:
        story = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(
            "Gemini returned invalid JSON: "
            + str(exc)
        )

    return normalize_story(
        story,
        language
    )


# ============================================================
# STORY NORMALIZATION
# ============================================================

def normalize_story(
    story,
    language="English"
):

    if not isinstance(
        story,
        dict
    ):

        raise ValueError(
            "Gemini returned invalid story data."
        )

    raw_characters = story.get(
        "characters",
        []
    )

    characters = []
    character_map = {}

    # ========================================================
    # CHARACTERS
    # ========================================================

    for item in raw_characters:

        if isinstance(
            item,
            str
        ):

            name = normalize_name(
                item
            )

            raw_profile = {}

        elif isinstance(
            item,
            dict
        ):

            name = normalize_name(
                item.get(
                    "name",
                    ""
                )
            )

            raw_profile = item

        else:
            continue

        if not name:
            continue

        key = name_key(name)

        if key in character_map:
            continue

        profile = build_character_profile(
            name,
            len(characters),
            raw_profile
        )

        characters.append(
            profile
        )

        character_map[key] = profile

    # ========================================================
    # SCENES
    # ========================================================

    raw_scenes = story.get(
        "scenes",
        []
    )

    scenes = []

    for scene_index, raw_scene in enumerate(
        raw_scenes[:MAX_SCENES],
        start=1
    ):

        if not isinstance(
            raw_scene,
            dict
        ):
            continue

        location = normalize_name(
            raw_scene.get(
                "location",
                "open area"
            )
        )

        title = normalize_name(
            raw_scene.get(
                "title",
                location
            )
        )

        time_of_day = normalize_name(
            raw_scene.get(
                "time",
                "day"
            )
        )

        camera = normalize_name(
            raw_scene.get(
                "camera",
                "wide shot"
            )
        )

        action = str(
            raw_scene.get(
                "action",
                ""
            )
        ).strip()

        present = []

        for name in raw_scene.get(
            "characters_present",
            []
        ):

            clean_name = normalize_name(
                name
            )

            if not clean_name:
                continue

            key = name_key(
                clean_name
            )

            if key not in character_map:

                profile = build_character_profile(
                    clean_name,
                    len(characters)
                )

                characters.append(
                    profile
                )

                character_map[key] = profile

            if key not in [
                name_key(x)
                for x in present
            ]:

                present.append(
                    clean_name
                )

        dialogue = []

        for line in raw_scene.get(
            "dialogue",
            []
        )[:MAX_DIALOGUE_LINES]:

            if not isinstance(
                line,
                dict
            ):
                continue

            speaker = normalize_name(
                line.get(
                    "speaker",
                    ""
                )
            )

            text = str(
                line.get(
                    "text",
                    ""
                )
            ).strip()

            if not speaker or not text:
                continue

            key = name_key(
                speaker
            )

            if key not in character_map:

                profile = build_character_profile(
                    speaker,
                    len(characters)
                )

                characters.append(
                    profile
                )

                character_map[key] = profile

            if key not in [
                name_key(x)
                for x in present
            ]:

                present.append(
                    speaker
                )

            dialogue.append({
                "speaker": speaker,
                "text": text,
                "action": str(
                    line.get(
                        "action",
                        ""
                    )
                ).strip(),
                "emotion": str(
                    line.get(
                        "emotion",
                        "talking"
                    )
                ).strip(),
                "facing": normalize_name(
                    line.get(
                        "facing",
                        ""
                    )
                )
            })

        scenes.append({
            "scene": scene_index,
            "title": title,
            "location": location,
            "time": time_of_day,
            "camera": camera,
            "action": action,
            "characters_present": present,
            "dialogue": dialogue,
            "character_states": {}
        })

    if not scenes:

        raise ValueError(
            "No scenes were generated."
        )

    return {
        "title": story.get(
            "title",
            "Cartoon Movie"
        ),
        "category": story.get(
            "category",
            "General"
        ),
        "language": language,
        "characters": characters,
        "scenes": scenes,
        "_explicit_script": False
    }


# ============================================================
# ENVIRONMENT
# ============================================================

def detect_location(location):

    text = str(
        location or ""
    ).lower()

    if any(x in text for x in [
        "living room",
        "lounge",
        "sitting room"
    ]):
        return "living_room"

    if any(x in text for x in [
        "bedroom",
        "sleeping room"
    ]):
        return "bedroom"

    if "kitchen" in text:
        return "kitchen"

    if any(x in text for x in [
        "classroom",
        "school"
    ]):
        return "classroom"

    if any(x in text for x in [
        "office",
        "workplace"
    ]):
        return "office"

    if any(x in text for x in [
        "restaurant",
        "cafe",
        "cafeteria"
    ]):
        return "restaurant"

    if any(x in text for x in [
        "hospital",
        "clinic"
    ]):
        return "hospital"

    if any(x in text for x in [
        "shop",
        "store",
        "market"
    ]):
        return "shop"

    if any(x in text for x in [
        "street",
        "road",
        "town"
    ]):
        return "street"

    if any(x in text for x in [
        "beach",
        "sea",
        "ocean"
    ]):
        return "beach"

    if any(x in text for x in [
        "park",
        "garden"
    ]):
        return "park"

    if "farm" in text:
        return "farm"

    return "generic_room"


# ============================================================
# BACKGROUND
# ============================================================

def draw_scene_background(
    draw,
    scene
):

    location = detect_location(
        scene.get("location")
    )

    time_text = str(
        scene.get(
            "time",
            "day"
        )
    ).lower()

    night = any(
        word in time_text
        for word in [
            "night",
            "evening",
            "midnight"
        ]
    )

    if night:
        wall = "#1C2740"
        floor = "#3B3440"
    else:
        wall = "#DCE8F0"
        floor = "#B99068"

    draw.rectangle(
        [0, 0, WIDTH, HEIGHT],
        fill=wall
    )

    draw.rectangle(
        [
            0,
            int(HEIGHT * 0.68),
            WIDTH,
            HEIGHT
        ],
        fill=floor
    )

    if location == "living_room":

        draw.rectangle(
            [30, 100, WIDTH - 30, 600],
            outline="#A8B3BA",
            width=3
        )

        draw.rectangle(
            [55, 155, 175, 310],
            fill="#8FC7E7",
            outline="#FFFFFF",
            width=5
        )

        draw.line(
            [115, 155, 115, 310],
            fill="#FFFFFF",
            width=4
        )

        draw.line(
            [55, 232, 175, 232],
            fill="#FFFFFF",
            width=4
        )

        draw.rounded_rectangle(
            [180, 515, 420, 660],
            radius=20,
            fill="#8D5960",
            outline="#563A3E",
            width=4
        )

        draw.rectangle(
            [195, 485, 405, 565],
            fill="#9E6970",
            outline="#563A3E",
            width=4
        )

        draw.rectangle(
            [70, 625, 170, 645],
            fill="#704B32"
        )

        draw.rectangle(
            [85, 645, 100, 720],
            fill="#704B32"
        )

        draw.rectangle(
            [140, 645, 155, 720],
            fill="#704B32"
        )

    elif location == "bedroom":

        draw.rectangle(
            [140, 490, 420, 650],
            fill="#D5DCE8",
            outline="#657185",
            width=4
        )

        draw.rectangle(
            [155, 430, 300, 520],
            fill="#F2F2F2",
            outline="#657185",
            width=4
        )

        draw.rectangle(
            [40, 190, 145, 470],
            fill="#72513C",
            outline="#4D372A",
            width=4
        )

        draw.rectangle(
            [390, 300, 398, 480],
            fill="#51483E"
        )

        draw.polygon(
            [
                (360, 300),
                (430, 300),
                (415, 250),
                (375, 250)
            ],
            fill="#E7C96E"
        )

    elif location == "kitchen":

        draw.rectangle(
            [25, 110, WIDTH - 25, 290],
            fill="#B87A4B",
            outline="#6F442C",
            width=4
        )

        for x in [80, 180, 280, 380]:

            draw.line(
                [x, 110, x, 290],
                fill="#6F442C",
                width=3
            )

        draw.rectangle(
            [20, 410, WIDTH - 20, 500],
            fill="#777A7E"
        )

        draw.rectangle(
            [210, 420, 330, 470],
            fill="#A9B0B5",
            outline="#4F5558",
            width=3
        )

        draw.rectangle(
            [40, 300, 145, 650],
            fill="#E3E5E8",
            outline="#73777A",
            width=4
        )

        draw.line(
            [40, 470, 145, 470],
            fill="#73777A",
            width=3
        )

    elif location == "classroom":

        draw.rectangle(
            [55, 110, WIDTH - 55, 330],
            fill="#2F5445",
            outline="#503C2E",
            width=6
        )

        draw.rectangle(
            [165, 500, 350, 580],
            fill="#8C603D"
        )

        for x in [45, 280]:

            draw.rectangle(
                [x, 600, x + 150, 660],
                fill="#9C6C45"
            )

    elif location == "office":

        draw.rectangle(
            [45, 120, 205, 340],
            fill="#91C6E6",
            outline="#FFFFFF",
            width=5
        )

        draw.rectangle(
            [115, 500, 390, 580],
            fill="#704A30"
        )

        draw.rectangle(
            [145, 580, 165, 700],
            fill="#704A30"
        )

        draw.rectangle(
            [335, 580, 355, 700],
            fill="#704A30"
        )

        draw.rectangle(
            [205, 400, 330, 490],
            fill="#333A40"
        )

        draw.rectangle(
            [260, 490, 275, 510],
            fill="#333A40"
        )

    elif location == "restaurant":

        for cx in [120, 360]:

            draw.ellipse(
                [
                    cx - 60,
                    515,
                    cx + 60,
                    575
                ],
                fill="#754A2F",
                outline="#4B3020",
                width=4
            )

            draw.rectangle(
                [
                    cx - 8,
                    575,
                    cx + 8,
                    690
                ],
                fill="#754A2F"
            )

    elif location == "hospital":

        draw.rectangle(
            [110, 475, 390, 600],
            fill="#E9EEF4",
            outline="#788797",
            width=4
        )

        draw.rectangle(
            [110, 425, 250, 500],
            fill="#FFFFFF",
            outline="#788797",
            width=4
        )

        draw.rectangle(
            [350, 260, 430, 390],
            fill="#303A43"
        )

        draw.line(
            [
                360,
                330,
                375,
                330,
                385,
                305,
                400,
                350,
                415,
                330
            ],
            fill="#71D38D",
            width=3
        )

    elif location == "shop":

        for x in [35, 180, 325]:

            draw.rectangle(
                [
                    x,
                    160,
                    x + 100,
                    600
                ],
                fill="#8A5A37",
                outline="#523A28",
                width=4
            )

            for y in [250, 360, 470]:

                draw.line(
                    [x, y, x + 100, y],
                    fill="#523A28",
                    width=3
                )

    elif location == "street":

        draw.rectangle(
            [20, 200, 150, 680],
            fill="#C78D70"
        )

        draw.rectangle(
            [170, 150, 300, 680],
            fill="#8AA1B5"
        )

        draw.rectangle(
            [320, 230, 460, 680],
            fill="#C4A35B"
        )

        draw.rectangle(
            [0, 680, WIDTH, HEIGHT],
            fill="#4D4D50"
        )

        for x in range(
            0,
            WIDTH,
            100
        ):

            draw.rectangle(
                [
                    x,
                    765,
                    x + 50,
                    775
                ],
                fill="#E7DDA8"
            )

    elif location == "beach":

        draw.rectangle(
            [0, 350, WIDTH, 680],
            fill="#5EA8C8"
        )

        draw.rectangle(
            [0, 680, WIDTH, HEIGHT],
            fill="#D8C18B"
        )

        draw.ellipse(
            [40, 180, 130, 270],
            fill="#F2C94C"
        )

    elif location == "park":

        for x, y in [
            (50, 360),
            (390, 330),
            (240, 290)
        ]:

            draw.rectangle(
                [x - 8, y, x + 8, y + 130],
                fill="#69452D"
            )

            draw.ellipse(
                [
                    x - 55,
                    y - 60,
                    x + 55,
                    y + 35
                ],
                fill="#4F8A4E"
            )

        draw.rectangle(
            [0, 680, WIDTH, HEIGHT],
            fill="#70A65A"
        )

    elif location == "farm":

        draw.rectangle(
            [0, 650, WIDTH, HEIGHT],
            fill="#71944A"
        )

        draw.rectangle(
            [150, 370, 360, 650],
            fill="#C89B6D"
        )

        draw.polygon(
            [
                (125, 370),
                (385, 370),
                (255, 245)
            ],
            fill="#914D3A"
        )

    else:

        draw.rectangle(
            [45, 130, WIDTH - 45, 600],
            outline="#B4B4B4",
            width=3
        )

        draw.rectangle(
            [70, 180, 180, 320],
            fill="#91C8E6",
            outline="#FFFFFF",
            width=5
        )

        draw.rectangle(
            [145, 560, 365, 620],
            fill="#765038"
        )

    label = normalize_name(
        scene.get(
            "location",
            "Scene"
        )
    )

    font = get_font(
        16,
        True
    )

    label_width = min(
        WIDTH - 30,
        max(
            150,
            25 + len(label[:28]) * 9
        )
    )

    draw.rounded_rectangle(
        [
            15,
            15,
            label_width,
            50
        ],
        radius=10,
        fill="#000000"
    )

    draw.text(
        [25, 22],
        label[:28],
        fill="#FFFFFF",
        font=font
    )


# ============================================================
# CAMERA
# ============================================================

def camera_mode(scene):

    camera = str(
        scene.get(
            "camera",
            ""
        )
    ).lower()

    if "close" in camera:
        return "close"

    if "medium" in camera:
        return "medium"

    if "over" in camera:
        return "over"

    if "two" in camera:
        return "two"

    if "wide" in camera:
        return "wide"

    return "wide"


# ============================================================
# CHARACTER POSITIONS
# ============================================================

def base_character_positions(count):

    positions = [
        (75, 555),
        (175, 555),
        (275, 555),
        (375, 555),

        (110, 640),
        (210, 640),
        (310, 640),
        (410, 640)
    ]

    return positions[:count]


def calculate_position(
    base_x,
    base_y,
    action,
    character_index
):

    text = str(
        action or ""
    ).lower()

    x = base_x
    y = base_y

    if "walks left" in text:
        x -= 45

    elif "walks right" in text:
        x += 45

    elif "walks toward" in text:
        x += 20

    elif "approaches" in text:
        x += 20

    elif "enters" in text:
        x += 25

    elif (
        "exits" in text
        or "leaves" in text
    ):
        x += 70

    if any(
        word in text
        for word in [
            "sits",
            "sitting",
            "seated",
            "on the sofa",
            "on the couch"
        ]
    ):
        y += 40

    if "stands" in text:
        y -= 15

    return (
        clamp(
            x,
            40,
            WIDTH - 40
        ),
        clamp(
            y,
            350,
            HEIGHT - 90
        )
    )


# ============================================================
# FACING
# ============================================================

def determine_facing(
    character_name,
    speaker,
    dialogue,
    all_names
):

    explicit = normalize_name(
        dialogue.get(
            "facing",
            ""
        )
    )

    if explicit:

        for name in all_names:

            if name_key(name) == name_key(
                explicit
            ):

                return name

    if name_key(character_name) == name_key(
        speaker
    ):

        for name in all_names:

            if name_key(name) != name_key(
                character_name
            ):
                return name

    return ""


# ============================================================
# CHARACTER DRAWING
# ============================================================

def draw_character(
    draw,
    profile,
    x,
    y,
    scale=1.0,
    facing="right",
    talking=False,
    emotion="neutral",
    seated=False,
    walking=False,
    animation_phase=0
):

    skin = color_to_hex(
        profile.get("skin"),
        "#8D5838"
    )

    shirt = color_to_hex(
        profile.get("shirt"),
        "#3E75C6"
    )

    pants = color_to_hex(
        profile.get("pants"),
        "#29384F"
    )

    hair = color_to_hex(
        profile.get("hair"),
        "#15100D"
    )

    clothing_type = profile.get(
        "clothing_type",
        "shirt"
    )

    # ========================================================
    # DETERMINISTIC ANIMATION
    # ========================================================

    bob = 0

    if walking:

        bob = int(
            math.sin(
                animation_phase * 2.0
            ) * 3
        )

    if talking:

        bob += int(
            math.sin(
                animation_phase * 3.0
            ) * 2
        )

    y += bob

    s = scale

    # ========================================================
    # HEAD
    # ========================================================

    head_w = int(
        74 * s
    )

    head_h = int(
        86 * s
    )

    head_left = int(
        x - head_w / 2
    )

    head_top = int(
        y - 175 * s
    )

    head_right = int(
        x + head_w / 2
    )

    head_bottom = int(
        y - 89 * s
    )

    draw.ellipse(
        [
            head_left,
            head_top,
            head_right,
            head_bottom
        ],
        fill=skin,
        outline="#3C2418",
        width=max(
            1,
            int(2 * s)
        )
    )

    # ========================================================
    # HAIR
    # ========================================================

    hairstyle = profile.get(
        "hairstyle",
        "short"
    )

    hair_box = [
        head_left - int(4 * s),
        head_top - int(8 * s),
        head_right + int(4 * s),
        head_top + int(34 * s)
    ]

    if hairstyle == "afro":

        draw.ellipse(
            hair_box,
            fill=hair
        )

    elif hairstyle == "curly":

        for dx in range(
            -25,
            30,
            12
        ):

            draw.ellipse(
                [
                    x + int(dx * s) - 14,
                    head_top - 8,
                    x + int(dx * s) + 14,
                    head_top + 30
                ],
                fill=hair
            )

    elif hairstyle == "long":

        draw.ellipse(
            hair_box,
            fill=hair
        )

        draw.rectangle(
            [
                head_left - int(5 * s),
                head_top + int(15 * s),
                head_left + int(14 * s),
                head_bottom
            ],
            fill=hair
        )

        draw.rectangle(
            [
                head_right - int(14 * s),
                head_top + int(15 * s),
                head_right + int(5 * s),
                head_bottom
            ],
            fill=hair
        )

    else:

        draw.ellipse(
            hair_box,
            fill=hair
        )

    # ========================================================
    # EYES
    # ========================================================

    eye_y = int(
        head_top + 45 * s
    )

    eye_offset = int(
        18 * s
    )

    eye_positions = [
        int(x - eye_offset),
        int(x + eye_offset)
    ]

    if facing == "left":
        eye_shift = -int(4 * s)

    elif facing == "right":
        eye_shift = int(4 * s)

    else:
        eye_shift = 0

    eye_radius = max(
        2,
        int(5 * s)
    )

    for eye_x in eye_positions:

        draw.ellipse(
            [
                eye_x - eye_radius,
                eye_y - eye_radius,
                eye_x + eye_radius,
                eye_y + eye_radius
            ],
            fill="#FFFFFF"
        )

        pupil = max(
            2,
            int(2.5 * s)
        )

        draw.ellipse(
            [
                eye_x + eye_shift - pupil,
                eye_y - pupil,
                eye_x + eye_shift + pupil,
                eye_y + pupil
            ],
            fill="#111111"
        )

    # ========================================================
    # MOUTH
    # ========================================================

    mouth_y = int(
        head_top + 68 * s
    )

    mouth_w = int(
        20 * s
    )

    if talking:

        mouth_open = (
            7
            if animation_phase % 2 < 1
            else 11
        )

        draw.ellipse(
            [
                int(x - mouth_w / 2),
                mouth_y - int(6 * s),
                int(x + mouth_w / 2),
                mouth_y + int(mouth_open * s)
            ],
            fill="#4C1717"
        )

    else:

        draw.arc(
            [
                int(x - mouth_w / 2),
                mouth_y - int(4 * s),
                int(x + mouth_w / 2),
                mouth_y + int(8 * s)
            ],
            0,
            180,
            fill="#421C18",
            width=max(
                1,
                int(2 * s)
            )
        )

    # ========================================================
    # BODY
    # ========================================================

    body_top = int(
        y - 85 * s
    )

    body_bottom = int(
        y + 60 * s
    )

    if clothing_type == "dress":

        draw.polygon(
            [
                (
                    int(x - 35 * s),
                    body_top
                ),
                (
                    int(x + 35 * s),
                    body_top
                ),
                (
                    int(x + 65 * s),
                    body_bottom
                ),
                (
                    int(x - 65 * s),
                    body_bottom
                )
            ],
            fill=shirt,
            outline="#49352A"
        )

    else:

        draw.rounded_rectangle(
            [
                int(x - 36 * s),
                body_top,
                int(x + 36 * s),
                body_bottom
            ],
            radius=max(
                4,
                int(12 * s)
            ),
            fill=shirt,
            outline="#49352A",
            width=max(
                1,
                int(2 * s)
            )
        )

    # ========================================================
    # ARMS
    # ========================================================

    arm_y = int(
        body_top + 35 * s
    )

    if talking:

        draw.line(
            [
                int(x + 30 * s),
                arm_y,
                int(x + 55 * s),
                arm_y - int(28 * s)
            ],
            fill=skin,
            width=max(
                3,
                int(9 * s)
            )
        )

    else:

        draw.line(
            [
                int(x - 30 * s),
                arm_y,
                int(x - 45 * s),
                arm_y + int(40 * s)
            ],
            fill=skin,
            width=max(
                3,
                int(8 * s)
            )
        )

        draw.line(
            [
                int(x + 30 * s),
                arm_y,
                int(x + 45 * s),
                arm_y + int(40 * s)
            ],
            fill=skin,
            width=max(
                3,
                int(8 * s)
            )
        )

    # ========================================================
    # LEGS
    # ========================================================

    leg_top = body_bottom

    if seated:

        draw.line(
            [
                int(x - 15 * s),
                leg_top,
                int(x - 45 * s),
                leg_top + int(45 * s)
            ],
            fill=pants,
            width=max(
                5,
                int(13 * s)
            )
        )

        draw.line(
            [
                int(x + 15 * s),
                leg_top,
                int(x + 45 * s),
                leg_top + int(45 * s)
            ],
            fill=pants,
            width=max(
                5,
                int(13 * s)
            )
        )

    else:

        walk_offset = 0

        if walking:

            walk_offset = int(
                math.sin(
                    animation_phase * 2.2
                ) * 10
            )

        draw.line(
            [
                int(x - 15 * s),
                leg_top,
                int(
                    x - 22 * s
                    + walk_offset
                ),
                leg_top + int(75 * s)
            ],
            fill=pants,
            width=max(
                5,
                int(13 * s)
            )
        )

        draw.line(
            [
                int(x + 15 * s),
                leg_top,
                int(
                    x + 22 * s
                    - walk_offset
                ),
                leg_top + int(75 * s)
            ],
            fill=pants,
            width=max(
                5,
                int(13 * s)
            )
        )

    # ========================================================
    # NAME
    # ========================================================

    name = profile.get(
        "name",
        ""
    )

    font = get_font(
        max(
            12,
            int(13 * s)
        ),
        True
    )

    bbox = draw.textbbox(
        (0, 0),
        name,
        font=font
    )

    text_width = (
        bbox[2] - bbox[0]
    )

    label_x = int(
        x - text_width / 2
    )

    label_y = int(
        y + 75 * s
    )

    draw.rounded_rectangle(
        [
            label_x - 5,
            label_y - 2,
            label_x + text_width + 5,
            label_y + 18
        ],
        radius=5,
        fill="#000000"
    )

    draw.text(
        [
            label_x,
            label_y
        ],
        name,
        fill="#FFFFFF",
        font=font
    )


# ============================================================
# DIALOGUE BOX
# ============================================================

def draw_dialogue_box(
    draw,
    speaker,
    text
):

    text = str(
        text or ""
    ).strip()

    if not text:
        return

    font = get_font(
        17,
        False
    )

    speaker_font = get_font(
        16,
        True
    )

    max_width = WIDTH - 65

    words = text.split()

    lines = []
    current = ""

    for word in words:

        candidate = (
            current
            + " "
            + word
        ).strip()

        bbox = draw.textbbox(
            (0, 0),
            candidate,
            font=font
        )

        if bbox[2] - bbox[0] <= max_width:

            current = candidate

        else:

            if current:
                lines.append(
                    current
                )

            current = word

    if current:
        lines.append(
            current
        )

    lines = lines[:6]

    line_height = 22

    box_height = (
        42
        + len(lines)
        * line_height
    )

    top = (
        HEIGHT
        - box_height
        - 20
    )

    draw.rounded_rectangle(
        [
            18,
            top,
            WIDTH - 18,
            HEIGHT - 18
        ],
        radius=16,
        fill="#111111",
        outline="#FFFFFF",
        width=2
    )

    draw.text(
        [
            32,
            top + 10
        ],
        speaker,
        fill="#F2C94C",
        font=speaker_font
    )

    y = top + 35

    for line in lines:

        draw.text(
            [
                32,
                y
            ],
            line,
            fill="#FFFFFF",
            font=font
        )

        y += line_height


# ============================================================
# ACTION FLAGS
# ============================================================

def action_flags(action):

    text = str(
        action or ""
    ).lower()

    return {
        "walking": any(
            x in text
            for x in [
                "walk",
                "walking",
                "enters",
                "approaches",
                "moves toward"
            ]
        ),

        "leaving": any(
            x in text
            for x in [
                "leaves",
                "exit",
                "exits"
            ]
        ),

        "sitting": any(
            x in text
            for x in [
                "sit",
                "sits",
                "sitting",
                "seated",
                "sofa",
                "couch"
            ]
        ),

        "standing": any(
            x in text
            for x in [
                "stand",
                "stands",
                "standing"
            ]
        )
    }


# ============================================================
# AUDIO
# ============================================================

def get_audio_duration(path):

    try:

        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path)
            ],
            capture_output=True,
            text=True
        )

        value = result.stdout.strip()

        duration = float(
            value
        )

        return max(
            0.1,
            duration
        )

    except Exception:
        return 1.0


def make_tts(
    text,
    language,
    output_path
):

    lang_map = {
        "english": "en",
        "spanish": "es",
        "french": "fr",
        "portuguese": "pt",
        "en": "en",
        "es": "es",
        "fr": "fr",
        "pt": "pt"
    }

    lang = lang_map.get(
        str(language).lower(),
        "en"
    )

    tts = gTTS(
        text=text,
        lang=lang,
        slow=False
    )

    tts.save(
        str(output_path)
    )

    return output_path


def combine_audio(
    audio_files,
    output_path
):

    if not audio_files:
        return None

    concat_file = (
        output_path.parent
        / "audio_concat.txt"
    )

    with open(
        concat_file,
        "w",
        encoding="utf-8"
    ) as f:

        for audio in audio_files:

            escaped = str(
                audio
            ).replace(
                "'",
                "'\\''"
            )

            f.write(
                f"file '{escaped}'\n"
            )

    command = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(output_path)
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Audio combining failed:\n"
            + result.stderr[-2500:]
        )

    return output_path


# ============================================================
# TIMELINE
# ============================================================

def create_timeline(
    story,
    work_dir
):

    timeline = []
    audio_files = []

    line_number = 0

    for scene in story.get(
        "scenes",
        []
    ):

        dialogue_lines = scene.get(
            "dialogue",
            []
        )

        for dialogue in dialogue_lines:

            line_number += 1

            text = str(
                dialogue.get(
                    "text",
                    ""
                )
            ).strip()

            if not text:
                continue

            audio_path = (
                work_dir
                / f"line_{line_number}.mp3"
            )

            make_tts(
                text,
                story.get(
                    "language",
                    "English"
                ),
                audio_path
            )

            duration = get_audio_duration(
                audio_path
            )

            timeline.append({
                "scene": scene.get(
                    "scene",
                    1
                ),
                "location": scene.get(
                    "location",
                    "open area"
                ),
                "time": scene.get(
                    "time",
                    "day"
                ),
                "camera": scene.get(
                    "camera",
                    "wide"
                ),
                "scene_action": scene.get(
                    "action",
                    ""
                ),
                "speaker": normalize_name(
                    dialogue.get(
                        "speaker",
                        ""
                    )
                ),
                "text": text,
                "action": dialogue.get(
                    "action",
                    ""
                ),
                "emotion": dialogue.get(
                    "emotion",
                    "talking"
                ),
                "facing": dialogue.get(
                    "facing",
                    ""
                ),
                "duration": max(
                    0.5,
                    duration
                ),
                "start": 0,
                "end": 0
            })

            audio_files.append(
                audio_path
            )

    current = 0

    for item in timeline:

        item["start"] = current

        current += item[
            "duration"
        ]

        item["end"] = current

    return (
        timeline,
        audio_files,
        current
    )


# ============================================================
# FRAME
# ============================================================

def draw_frame(
    story,
    profiles,
    item,
    frame_index
):

    image = Image.new(
        "RGB",
        (
            WIDTH,
            HEIGHT
        ),
        "#DDE5EA"
    )

    draw = ImageDraw.Draw(
        image
    )

    scene = {
        "location": item.get(
            "location",
            "open area"
        ),
        "time": item.get(
            "time",
            "day"
        ),
        "camera": item.get(
            "camera",
            "wide"
        )
    }

    draw_scene_background(
        draw,
        scene
    )

    # ========================================================
    # FIND SCENE
    # ========================================================

    current_scene = None

    for candidate in story.get(
        "scenes",
        []
    ):

        if int(
            candidate.get(
                "scene",
                0
            )
        ) == int(
            item.get(
                "scene",
                0
            )
        ):

            current_scene = candidate
            break

    if current_scene is None:
        current_scene = {}

    present = list(
        current_scene.get(
            "characters_present",
            []
        )
    )

    speaker = normalize_name(
        item.get(
            "speaker",
            ""
        )
    )

    if speaker and not any(
        name_key(x)
        == name_key(speaker)
        for x in present
    ):

        present.append(
            speaker
        )

    present = present[
        :MAX_VISIBLE_CHARACTERS
    ]

    if not present:
        present = [speaker] if speaker else []

    # ========================================================
    # POSITIONS
    # ========================================================

    positions = base_character_positions(
        len(present)
    )

    position_map = {}

    for index, name in enumerate(
        present
    ):

        base_x, base_y = positions[index]

        action = ""

        if name_key(name) == name_key(
            speaker
        ):

            action = item.get(
                "action",
                ""
            )

        if not action:

            for dialogue in current_scene.get(
                "dialogue",
                []
            ):

                if name_key(
                    dialogue.get(
                        "speaker",
                        ""
                    )
                ) == name_key(name):

                    action = dialogue.get(
                        "action",
                        ""
                    )

                    if action:
                        break

        x, y = calculate_position(
            base_x,
            base_y,
            action,
            index
        )

        position_map[
            name_key(name)
        ] = (
            x,
            y
        )

    # ========================================================
    # CAMERA
    # ========================================================

    camera = camera_mode(
        current_scene
    )

    if camera == "close":
        scale = 1.20

    elif camera == "medium":
        scale = 1.05

    elif camera == "two":
        scale = 0.95

    elif camera == "over":
        scale = 1.00

    else:

        scale = (
            0.72
            if len(present) >= 4
            else 0.90
        )

    # ========================================================
    # ANIMATION PHASE
    # ========================================================

    animation_phase = (
        frame_index / FPS
    )

    # ========================================================
    # DRAW CHARACTERS
    # ========================================================

    for index, name in enumerate(
        present
    ):

        profile = profiles.get(
            name_key(name)
        )

        if not profile:
            continue

        x, y = position_map[
            name_key(name)
        ]

        is_speaker = (
            name_key(name)
            == name_key(speaker)
        )

        # ----------------------------------------------------
        # Facing
        # ----------------------------------------------------

        facing_target = ""

        if is_speaker:

            facing_target = normalize_name(
                item.get(
                    "facing",
                    ""
                )
            )

            if not facing_target:

                for other in present:

                    if name_key(other) != name_key(name):

                        facing_target = other
                        break

        else:

            if speaker:
                facing_target = speaker

        facing_direction = "right"

        if facing_target:

            target_pos = position_map.get(
                name_key(
                    facing_target
                )
            )

            if target_pos:

                if target_pos[0] < x:
                    facing_direction = "left"

                else:
                    facing_direction = "right"

        # ----------------------------------------------------
        # Action
        # ----------------------------------------------------

        action_text = ""

        if is_speaker:

            action_text = item.get(
                "action",
                ""
            )

        else:

            for dialogue in current_scene.get(
                "dialogue",
                []
            ):

                if name_key(
                    dialogue.get(
                        "speaker",
                        ""
                    )
                ) == name_key(name):

                    action_text = dialogue.get(
                        "action",
                        ""
                    )

                    break

        flags = action_flags(
            action_text
        )

        # ----------------------------------------------------
        # Draw
        # ----------------------------------------------------

        draw_character(
            draw,
            profile,
            x,
            y,
            scale=scale,
            facing=facing_direction,
            talking=is_speaker,
            emotion=(
                item.get(
                    "emotion",
                    "talking"
                )
                if is_speaker
                else "neutral"
            ),
            seated=flags["sitting"],
            walking=flags["walking"],
            animation_phase=animation_phase
        )

    # ========================================================
    # ACTION CAPTION
    # ========================================================

    scene_action = str(
        item.get(
            "scene_action",
            ""
        )
    ).strip()

    if scene_action:

        font = get_font(
            13,
            False
        )

        caption = re.sub(
            r"\s+",
            " ",
            scene_action
        )

        caption = caption[:100]

        draw.rounded_rectangle(
            [
                15,
                65,
                WIDTH - 15,
                96
            ],
            radius=8,
            fill="#000000"
        )

        draw.text(
            [25, 72],
            caption,
            fill="#FFFFFF",
            font=font
        )

    # ========================================================
    # DIALOGUE
    # ========================================================

    draw_dialogue_box(
        draw,
        speaker,
        item.get(
            "text",
            ""
        )
    )

    return image


# ============================================================
# JOB HELPERS
# ============================================================

def update_job(
    job_id,
    **updates
):

    with JOBS_LOCK:

        if job_id not in JOBS:
            JOBS[job_id] = {}

        JOBS[job_id].update(
            updates
        )

        JOBS[job_id]["updated_at"] = now()


def get_job(job_id):

    with JOBS_LOCK:

        return dict(
            JOBS.get(
                job_id,
                {}
            )
        )


# ============================================================
# VIDEO CREATION
# ============================================================

def create_video(
    job_id,
    story,
    requested_duration,
    backend_url
):

    work_dir = (
        OUTPUT_DIR
        / f"job_{job_id}"
    )

    work_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    try:

        update_job(
            job_id,
            status="rendering",
            progress=5,
            message="Preparing movie..."
        )

        # ====================================================
        # PROFILES
        # ====================================================

        profiles = {}

        for index, character in enumerate(
            story.get(
                "characters",
                []
            )
        ):

            name = normalize_name(
                character.get(
                    "name",
                    ""
                )
            )

            if not name:
                continue

            profiles[
                name_key(name)
            ] = build_character_profile(
                name,
                index,
                character
            )

        # ====================================================
        # TIMELINE
        # ====================================================

        update_job(
            job_id,
            status="rendering",
            progress=10,
            message="Creating voices..."
        )

        (
            timeline,
            audio_files,
            audio_duration
        ) = create_timeline(
            story,
            work_dir
        )

        if not timeline:

            raise RuntimeError(
                "The story contains no dialogue."
            )

        # ====================================================
        # COMBINE AUDIO
        # ====================================================

        combined_audio = (
            work_dir
            / "combined.m4a"
        )

        combine_audio(
            audio_files,
            combined_audio
        )

        # ====================================================
        # DURATION
        # ====================================================

        explicit_script = bool(
            story.get(
                "_explicit_script",
                False
            )
        )

        if explicit_script:

            # Exact script = full voice duration.
            target_duration = audio_duration

        else:

            target_duration = min(
                float(requested_duration),
                audio_duration
            )

        target_duration = max(
            0.5,
            target_duration
        )

        update_job(
            job_id,
            status="rendering",
            progress=20,
            message="Rendering cartoon..."
        )

        # ====================================================
        # FRAMES
        # ====================================================

        frames_dir = (
            work_dir
            / "frames"
        )

        frames_dir.mkdir(
            exist_ok=True
        )

        total_frames = max(
            1,
            int(
                math.ceil(
                    target_duration
                    * FPS
                )
            )
        )

        timeline_index = 0

        # ====================================================
        # RENDER
        # ====================================================

        for frame_number in range(
            total_frames
        ):

            current_time = (
                frame_number
                / FPS
            )

            while (
                timeline_index
                < len(timeline) - 1
                and current_time
                >= timeline[
                    timeline_index
                ]["end"]
            ):

                timeline_index += 1

            item = timeline[
                timeline_index
            ]

            image = draw_frame(
                story,
                profiles,
                item,
                frame_number
            )

            frame_path = (
                frames_dir
                / f"frame_{frame_number:06d}.png"
            )

            image.save(
                frame_path,
                "PNG",
                optimize=True
            )

            if (
                frame_number
                % max(
                    1,
                    FPS * 2
                )
                == 0
            ):

                progress = (
                    20
                    + int(
                        (
                            frame_number
                            / total_frames
                        )
                        * 60
                    )
                )

                update_job(
                    job_id,
                    status="rendering",
                    progress=min(
                        80,
                        progress
                    ),
                    message=(
                        "Rendering scene "
                        + str(
                            item.get(
                                "scene",
                                1
                            )
                        )
                        + "..."
                    )
                )

        # ====================================================
        # OUTPUT
        # ====================================================

        output_name = (
            safe_filename(
                story.get(
                    "title",
                    "cartoon_movie"
                )
            )
            + "_"
            + job_id
            + ".mp4"
        )

        output_path = (
            OUTPUT_DIR
            / output_name
        )

        update_job(
            job_id,
            status="encoding",
            progress=82,
            message="Encoding MP4..."
        )

        # ====================================================
        # FFMPEG
        # ====================================================

        ffmpeg_command = [
            "ffmpeg",
            "-y",

            "-framerate",
            str(FPS),

            "-i",
            str(
                frames_dir
                / "frame_%06d.png"
            ),

            "-i",
            str(combined_audio),

            "-c:v",
            "libx264",

            "-preset",
            "veryfast",

            "-crf",
            "23",

            "-pix_fmt",
            "yuv420p",

            "-c:a",
            "aac",

            "-b:a",
            "128k",

            "-shortest",

            "-movflags",
            "+faststart",

            str(output_path)
        ]

        result = subprocess.run(
            ffmpeg_command,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:

            raise RuntimeError(
                "FFmpeg failed:\n"
                + result.stderr[-4000:]
            )

        if not output_path.exists():

            raise RuntimeError(
                "FFmpeg completed but "
                "the MP4 file was not created."
            )

        # ====================================================
        # URLS
        # ====================================================

        if backend_url:

            video_url = (
                backend_url
                + "/generated/"
                + output_name
            )

            download_url = (
                backend_url
                + "/download/"
                + output_name
            )

        else:

            video_url = (
                "/generated/"
                + output_name
            )

            download_url = (
                "/download/"
                + output_name
            )

        # ====================================================
        # COMPLETE
        # ====================================================

        update_job(
            job_id,
            status="complete",
            progress=100,
            message="Movie ready.",
            video_url=video_url,
            download_url=download_url,
            filename=output_name,
            duration=target_duration
        )

        # ====================================================
        # CLEAN TEMP FILES
        # ====================================================

        try:

            shutil.rmtree(
                work_dir
            )

        except Exception:
            pass

    except Exception as exc:

        update_job(
            job_id,
            status="error",
            progress=0,
            message=str(exc)
        )

        try:

            shutil.rmtree(
                work_dir
            )

        except Exception:
            pass


# ============================================================
# API GENERATE
# ============================================================

@app.route(
    "/api/generate",
    methods=["POST"]
)
def api_generate():

    try:

        data = request.get_json(
            silent=True
        ) or {}

        category = str(
            data.get(
                "category",
                "General"
            )
        ).strip()

        language = str(
            data.get(
                "language",
                "English"
            )
        ).strip()

        try:

            duration = int(
                data.get(
                    "duration",
                    60
                )
            )

        except Exception:

            duration = 60

        # ====================================================
        # SCRIPT HAS PRIORITY
        # ====================================================

        script = str(
            data.get(
                "script",
                ""
            )
        ).strip()

        topic = str(
            data.get(
                "topic",
                data.get(
                    "prompt",
                    ""
                )
            )
        ).strip()

        user_input = (
            script
            if script
            else topic
        )

        if not user_input:

            return jsonify({
                "success": False,
                "error": (
                    "Please provide a topic "
                    "or script."
                )
            }), 400

        duration = clamp(
            duration,
            10,
            3600
        )

        # ====================================================
        # EXACT SCRIPT
        # ====================================================

        explicit_story = (
            parse_explicit_script(
                user_input,
                category,
                language
            )
        )

        if explicit_story:

            story = explicit_story

        else:

            story = generate_story(
                category,
                user_input,
                duration,
                language
            )

        # ====================================================
        # JOB
        # ====================================================

        job_id = uuid.uuid4().hex[:12]

        JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "Queued.",
            "created_at": now(),
            "story": story
        }

        backend_url = (
            BASE_URL
            or request.host_url.rstrip("/")
        )

        thread = threading.Thread(
            target=create_video,
            args=(
                job_id,
                story,
                duration,
                backend_url
            ),
            daemon=True
        )

        thread.start()

        return jsonify({
            "success": True,
            "job_id": job_id,
            "status": "queued",
            "story": story
        })

    except Exception as exc:

        return jsonify({
            "success": False,
            "error": str(exc)
        }), 500


# ============================================================
# CREATE VIDEO
# ============================================================

@app.route(
    "/api/create-video",
    methods=["POST"]
)
def api_create_video():

    return api_generate()


# ============================================================
# STATUS
# ============================================================

@app.route(
    "/api/video-status/<job_id>",
    methods=["GET"]
)
def api_video_status(job_id):

    job = get_job(
        job_id
    )

    if not job:

        return jsonify({
            "success": False,
            "error": "Job not found."
        }), 404

    return jsonify({
        "success": True,
        **job
    })


@app.route(
    "/api/status/<job_id>",
    methods=["GET"]
)
def api_status(job_id):

    return api_video_status(
        job_id
    )


# ============================================================
# VIDEO PREVIEW
# ============================================================

@app.route(
    "/generated/<path:filename>",
    methods=["GET"]
)
def generated_file(filename):

    file_path = (
        OUTPUT_DIR
        / filename
    )

    if not file_path.exists():
        return jsonify({
            "success": False,
            "error": "Video file not found."
        }), 404

    return send_from_directory(
        OUTPUT_DIR,
        filename,
        as_attachment=False,
        mimetype="video/mp4"
    )


# ============================================================
# DOWNLOAD
# ============================================================

@app.route(
    "/download/<path:filename>",
    methods=["GET"]
)
def download_file(filename):

    file_path = (
        OUTPUT_DIR
        / filename
    )

    if not file_path.exists():

        return jsonify({
            "success": False,
            "error": "Video file not found."
        }), 404

    return send_from_directory(
        OUTPUT_DIR,
        filename,
        as_attachment=True,
        download_name=filename,
        mimetype="video/mp4"
    )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return jsonify({
        "name": "AfriToon Studio",
        "status": "running",
        "engine": "Movie-style 2D cartoon renderer",
        "features": [
            "Dynamic characters",
            "Dynamic character names",
            "Character consistency",
            "Room environments",
            "Walking",
            "Entering and leaving",
            "Sitting",
            "Standing",
            "Facing characters",
            "Talking animation",
            "Clothing changes",
            "Camera shots",
            "Exact dialogue scripts",
            "Natural dialogue parsing",
            "Gemini story generation",
            "gTTS voice",
            "FFmpeg MP4",
            "Video preview",
            "Phone download"
        ]
    })


# ============================================================
# HEALTH
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({
        "status": "ok",
        "gemini_configured": bool(
            GEMINI_API_KEY
        ),
        "model": GEMINI_MODEL,
        "fps": FPS,
        "resolution": (
            f"{WIDTH}x{HEIGHT}"
        )
    })


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    port = int(
        os.getenv(
            "PORT",
            "5000"
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
