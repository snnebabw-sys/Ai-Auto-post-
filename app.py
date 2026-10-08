import os
import json
import uuid
import time
import re
import subprocess
import threading
import shutil
import hashlib
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
CORS(app, origins="*")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JOBS = {}
JOBS_LOCK = threading.Lock()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
BASE_URL = os.getenv("BASE_URL", "").strip().rstrip("/")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)

MAX_SCENES = 8
MAX_DIALOGUE_LINES = 30

WIDTH = 480
HEIGHT = 854
FPS = 8


# ============================================================
# FONTS
# ============================================================

FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
]

FONT_BOLD_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
]


def get_font(size, bold=False):
    paths = FONT_BOLD_PATHS if bold else FONT_PATHS

    for path in paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass

    return ImageFont.load_default()


# ============================================================
# URL HELPERS
# ============================================================

def build_video_url(base_url, filename):
    return f"{base_url}/generated/{filename}"


def build_download_url(base_url, filename):
    return f"{base_url}/download/{filename}"


# ============================================================
# JOB HELPERS
# ============================================================

def update_job(job_id, **kwargs):
    with JOBS_LOCK:
        if job_id not in JOBS:
            JOBS[job_id] = {}

        JOBS[job_id].update(kwargs)


def get_job(job_id):
    with JOBS_LOCK:
        return dict(JOBS.get(job_id, {}))


# ============================================================
# BASIC HELPERS
# ============================================================

def clean_json(text):
    if not text:
        return ""

    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.I)
        text = re.sub(r"```$", "", text)
        text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end >= 0:
        text = text[start:end + 1]

    return text


def safe_filename(text):
    text = str(text or "video")

    text = re.sub(
        r"[^a-zA-Z0-9_\- ]+",
        "",
        text
    )

    text = re.sub(r"\s+", "_", text).strip("_")

    if not text:
        text = "video"

    return text[:80]


def normalize_name(name):
    return re.sub(
        r"\s+",
        " ",
        str(name or "").strip()
    )


def name_key(name):
    return normalize_name(name).lower()


def deterministic_number(value):
    digest = hashlib.md5(
        str(value).encode("utf-8")
    ).hexdigest()

    return int(digest[:8], 16)


# ============================================================
# CHARACTER DESIGN
# ============================================================

SKIN_TONES = [
    "#6B3E26",
    "#7B482C",
    "#8D552F",
    "#9B6238",
    "#A96B42",
    "#5A321E",
    "#704020",
    "#B8784B",
]

SHIRT_COLORS = [
    "#D94841",
    "#2563EB",
    "#15803D",
    "#9333EA",
    "#EA580C",
    "#0891B2",
    "#CA8A04",
    "#BE185D",
    "#374151",
    "#0F766E",
]

HAIR_COLORS = [
    "#111111",
    "#21140C",
    "#352015",
    "#432818",
]

PANTS_COLORS = [
    "#1F2937",
    "#334155",
    "#3F3F46",
    "#4B5563",
    "#172554",
]

HAIR_STYLES = [
    "short",
    "round",
    "afro",
    "close",
    "long",
]


def build_character_profile(name, index, raw_profile=None):
    """
    Creates a stable visual identity for each character.

    IMPORTANT:
    There are NO hard-coded names here.
    Any name from the user's script can become a character.
    """

    seed = deterministic_number(name)

    skin = SKIN_TONES[
        (seed + index) % len(SKIN_TONES)
    ]

    shirt = SHIRT_COLORS[
        (seed // 7 + index) % len(SHIRT_COLORS)
    ]

    hair = HAIR_COLORS[
        (seed // 13 + index) % len(HAIR_COLORS)
    ]

    pants = PANTS_COLORS[
        (seed // 17 + index) % len(PANTS_COLORS)
    ]

    hairstyle = HAIR_STYLES[
        (seed // 23 + index) % len(HAIR_STYLES)
    ]

    gender = ""
    age = ""

    if isinstance(raw_profile, dict):
        gender = str(raw_profile.get("gender", "") or "")
        age = str(raw_profile.get("age", "") or "")

        supplied_skin = raw_profile.get("skin_tone")
        supplied_shirt = raw_profile.get("clothing_color")

        if supplied_skin:
            skin = str(supplied_skin)

        if supplied_shirt:
            shirt = str(supplied_shirt)

    return {
        "name": name,
        "skin": skin,
        "shirt": shirt,
        "pants": pants,
        "hair": hair,
        "hairstyle": hairstyle,
        "gender": gender,
        "age": age,
    }


def build_character_profiles(characters):
    profiles = {}

    for index, character in enumerate(characters):
        if isinstance(character, str):
            name = normalize_name(character)
            raw = {}
        else:
            name = normalize_name(
                character.get("name", "")
            )
            raw = character

        if not name:
            continue

        profiles[name_key(name)] = build_character_profile(
            name,
            index,
            raw
        )

    return profiles


# ============================================================
# TEXT DRAWING
# ============================================================

def draw_centered_text(draw, text, y, font, fill):
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]

    x = (WIDTH - width) // 2

    draw.text(
        (x, y),
        text,
        font=font,
        fill=fill
    )


def wrap_text(text, font, max_width):
    words = str(text or "").split()

    lines = []
    current = ""

    for word in words:
        test = word if not current else current + " " + word

        bbox = font.getbbox(test)
        width = bbox[2] - bbox[0]

        if width <= max_width:
            current = test
        else:
            if current:
                lines.append(current)

            current = word

    if current:
        lines.append(current)

    return lines


# ============================================================
# BACKGROUND
# ============================================================

def draw_scene_background(
    draw,
    scene,
    scene_number,
    total_scenes
):
    """
    Creates a simple animated-cartoon background based on
    the actual location/action/time from the user's scene.
    """

    location = str(
        scene.get("location", "")
    ).lower()

    action = str(
        scene.get("action", "")
    ).lower()

    time_of_day = str(
        scene.get("time", "")
    ).lower()

    combined = (
        location + " " +
        action + " " +
        time_of_day
    )

    # --------------------------------------------------------
    # SKY
    # --------------------------------------------------------

    if any(word in combined for word in [
        "night",
        "midnight",
        "evening",
        "dark"
    ]):
        sky = "#172554"
        ground = "#1F2937"
    else:
        sky = "#87CEEB"
        ground = "#65A30D"

    draw.rectangle(
        [0, 0, WIDTH, HEIGHT],
        fill=sky
    )

    # Sun / moon
    if "night" in combined or "evening" in combined:
        draw.ellipse(
            [365, 70, 425, 130],
            fill="#F8FAFC"
        )
    else:
        draw.ellipse(
            [360, 60, 430, 130],
            fill="#FACC15"
        )

    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    if any(word in combined for word in [
        "market",
        "shop",
        "store"
    ]):

        draw.rectangle(
            [0, 520, WIDTH, HEIGHT],
            fill="#A16207"
        )

        for x in [35, 165, 300]:
            draw.rectangle(
                [x, 360, x + 110, 525],
                fill="#92400E"
            )

            draw.polygon(
                [
                    (x - 10, 360),
                    (x + 55, 315),
                    (x + 120, 360)
                ],
                fill="#DC2626"
            )

            draw.rectangle(
                [x + 20, 405, x + 90, 465],
                fill="#FDE68A"
            )

    elif any(word in combined for word in [
        "school",
        "classroom"
    ]):

        draw.rectangle(
            [0, 470, WIDTH, HEIGHT],
            fill="#65A30D"
        )

        draw.rectangle(
            [80, 230, 400, 520],
            fill="#F5E6C8"
        )

        draw.polygon(
            [
                (55, 230),
                (240, 120),
                (425, 230)
            ],
            fill="#991B1B"
        )

        draw.rectangle(
            [190, 370, 290, 520],
            fill="#78350F"
        )

        draw.rectangle(
            [120, 290, 175, 345],
            fill="#93C5FD"
        )

        draw.rectangle(
            [305, 290, 360, 345],
            fill="#93C5FD"
        )

    elif any(word in combined for word in [
        "house",
        "home",
        "bedroom",
        "living room"
    ]):

        draw.rectangle(
            [0, 470, WIDTH, HEIGHT],
            fill="#A16207"
        )

        draw.rectangle(
            [55, 220, 425, 520],
            fill="#F5E6C8"
        )

        draw.polygon(
            [
                (30, 220),
                (240, 80),
                (450, 220)
            ],
            fill="#7F1D1D"
        )

        draw.rectangle(
            [185, 350, 295, 520],
            fill="#78350F"
        )

        draw.rectangle(
            [95, 280, 155, 340],
            fill="#93C5FD"
        )

        draw.rectangle(
            [325, 280, 385, 340],
            fill="#93C5FD"
        )

    elif any(word in combined for word in [
        "river",
        "lake",
        "water"
    ]):

        draw.rectangle(
            [0, 460, WIDTH, HEIGHT],
            fill="#2563EB"
        )

        draw.rectangle(
            [0, 430, WIDTH, 470],
            fill="#65A30D"
        )

        for y in range(500, HEIGHT, 55):
            draw.line(
                [(30, y), (450, y)],
                fill="#93C5FD",
                width=4
            )

    elif any(word in combined for word in [
        "beach",
        "sea",
        "ocean"
    ]):

        draw.rectangle(
            [0, 450, WIDTH, HEIGHT],
            fill="#38BDF8"
        )

        draw.rectangle(
            [0, 410, WIDTH, 470],
            fill="#FDE68A"
        )

        for y in range(500, HEIGHT, 50):
            draw.line(
                [(20, y), (460, y)],
                fill="#BAE6FD",
                width=4
            )

    elif any(word in combined for word in [
        "forest",
        "jungle"
    ]):

        draw.rectangle(
            [0, 450, WIDTH, HEIGHT],
            fill="#166534"
        )

        for x in range(20, WIDTH, 70):
            draw.rectangle(
                [x + 20, 250, x + 35, 500],
                fill="#78350F"
            )

            draw.ellipse(
                [x - 15, 180, x + 70, 300],
                fill="#15803D"
            )

            draw.ellipse(
                [x - 30, 230, x + 60, 350],
                fill="#16A34A"
            )

    elif any(word in combined for word in [
        "city",
        "town",
        "street",
        "road"
    ]):

        draw.rectangle(
            [0, 460, WIDTH, HEIGHT],
            fill="#4B5563"
        )

        draw.rectangle(
            [0, 430, WIDTH, 470],
            fill="#374151"
        )

        for x, h in [
            (20, 250),
            (100, 180),
            (180, 290),
            (285, 210),
            (370, 270)
        ]:
            draw.rectangle(
                [x, 430 - h, x + 65, 430],
                fill="#64748B"
            )

            for yy in range(
                450 - h,
                420,
                45
            ):
                draw.rectangle(
                    [x + 12, yy, x + 25, yy + 18],
                    fill="#FDE68A"
                )

    elif any(word in combined for word in [
        "farm",
        "village"
    ]):

        draw.rectangle(
            [0, 470, WIDTH, HEIGHT],
            fill="#65A30D"
        )

        # huts
        for x in [45, 315]:
            draw.rectangle(
                [x, 315, x + 120, 490],
                fill="#D97706"
            )

            draw.polygon(
                [
                    (x - 15, 315),
                    (x + 60, 240),
                    (x + 135, 315)
                ],
                fill="#92400E"
            )

    else:
        # Generic African village/cartoon environment
        draw.rectangle(
            [0, 470, WIDTH, HEIGHT],
            fill=ground
        )

        draw.rectangle(
            [50, 315, 190, 470],
            fill="#D97706"
        )

        draw.polygon(
            [
                (35, 315),
                (120, 240),
                (205, 315)
            ],
            fill="#92400E"
        )

        draw.rectangle(
            [290, 335, 405, 470],
            fill="#C2410C"
        )

        draw.polygon(
            [
                (275, 335),
                (347, 270),
                (420, 335)
            ],
            fill="#7C2D12"
        )

        # Trees
        for x in [15, 430]:
            draw.rectangle(
                [x + 20, 310, x + 35, 500],
                fill="#78350F"
            )

            draw.ellipse(
                [x - 15, 240, x + 75, 350],
                fill="#15803D"
            )

    # --------------------------------------------------------
    # SCENE NUMBER
    # --------------------------------------------------------

    scene_font = get_font(18, bold=True)

    scene_text = f"SCENE {scene_number}/{total_scenes}"

    draw.rounded_rectangle(
        [12, 12, 140, 42],
        radius=8,
        fill="#111827"
    )

    draw.text(
        (20, 17),
        scene_text,
        font=scene_font,
        fill="white"
    )

    # Location label
    location_text = str(
        scene.get("location", "")
    ).strip()

    if location_text:
        loc_font = get_font(17, bold=True)

        label = location_text[:42]

        bbox = draw.textbbox(
            (0, 0),
            label,
            font=loc_font
        )

        label_width = bbox[2] - bbox[0]

        x = WIDTH - label_width - 20

        draw.rounded_rectangle(
            [x - 8, 12, WIDTH - 10, 42],
            radius=8,
            fill="#111827"
        )

        draw.text(
            (x, 17),
            label,
            font=loc_font,
            fill="white"
        )


# ============================================================
# CHARACTER DRAWING
# ============================================================

def draw_character(
    draw,
    x,
    ground_y,
    scale,
    profile,
    talking=False,
    emotion="neutral",
    action=""
):
    """
    Generic cartoon character.

    It does NOT care what the character's name is.
    """

    skin = profile["skin"]
    shirt = profile["shirt"]
    pants = profile["pants"]
    hair = profile["hair"]
    hairstyle = profile["hairstyle"]

    body_h = int(170 * scale)
    head_r = int(48 * scale)

    # Talking animation
    bob = 0

    if talking:
        bob = int(
            ((time.time() * 7) % 2) * 3
        )

    ground_y -= bob

    # Legs
    draw.rectangle(
        [
            x - int(28 * scale),
            ground_y - int(65 * scale),
            x - int(5 * scale),
            ground_y
        ],
        fill=pants
    )

    draw.rectangle(
        [
            x + int(5 * scale),
            ground_y - int(65 * scale),
            x + int(28 * scale),
            ground_y
        ],
        fill=pants
    )

    # Shoes
    draw.ellipse(
        [
            x - int(38 * scale),
            ground_y - int(8 * scale),
            x - int(3 * scale),
            ground_y + int(12 * scale)
        ],
        fill="#111827"
    )

    draw.ellipse(
        [
            x + int(3 * scale),
            ground_y - int(8 * scale),
            x + int(38 * scale),
            ground_y + int(12 * scale)
        ],
        fill="#111827"
    )

    # Body
    body_top = ground_y - body_h + int(35 * scale)

    draw.rounded_rectangle(
        [
            x - int(48 * scale),
            body_top,
            x + int(48 * scale),
            ground_y - int(50 * scale)
        ],
        radius=int(22 * scale),
        fill=shirt
    )

    # Arms
    arm_y = body_top + int(45 * scale)

    if talking:
        # One hand raised when speaking
        draw.line(
            [
                (
                    x - int(40 * scale),
                    arm_y
                ),
                (
                    x - int(80 * scale),
                    arm_y - int(40 * scale)
                )
            ],
            fill=skin,
            width=max(3, int(14 * scale))
        )

        draw.ellipse(
            [
                x - int(88 * scale),
                arm_y - int(55 * scale),
                x - int(70 * scale),
                arm_y - int(37 * scale)
            ],
            fill=skin
        )

        draw.line(
            [
                (
                    x + int(40 * scale),
                    arm_y
                ),
                (
                    x + int(70 * scale),
                    arm_y + int(20 * scale)
                )
            ],
            fill=skin,
            width=max(3, int(14 * scale))
        )

    else:
        draw.line(
            [
                (
                    x - int(40 * scale),
                    arm_y
                ),
                (
                    x - int(62 * scale),
                    arm_y + int(55 * scale)
                )
            ],
            fill=skin,
            width=max(3, int(14 * scale))
        )

        draw.line(
            [
                (
                    x + int(40 * scale),
                    arm_y
                ),
                (
                    x + int(62 * scale),
                    arm_y + int(55 * scale)
                )
            ],
            fill=skin,
            width=max(3, int(14 * scale))
        )

    # Neck
    neck_y = body_top - int(10 * scale)

    draw.rectangle(
        [
            x - int(15 * scale),
            neck_y,
            x + int(15 * scale),
            neck_y + int(30 * scale)
        ],
        fill=skin
    )

    # Head
    head_center_y = neck_y - head_r + int(5 * scale)

    draw.ellipse(
        [
            x - head_r,
            head_center_y - head_r,
            x + head_r,
            head_center_y + head_r
        ],
        fill=skin
    )

    # Hair
    hair_top = head_center_y - head_r

    if hairstyle == "afro":
        for dx, dy, r in [
            (-30, 0, 25),
            (-15, -25, 25),
            (10, -30, 27),
            (30, -5, 25),
            (0, 5, 30)
        ]:
            draw.ellipse(
                [
                    x + int((dx - r) * scale),
                    head_center_y + int((dy - r) * scale),
                    x + int((dx + r) * scale),
                    head_center_y + int((dy + r) * scale)
                ],
                fill=hair
            )

    elif hairstyle == "long":
        draw.ellipse(
            [
                x - int(55 * scale),
                hair_top - int(5 * scale),
                x + int(55 * scale),
                head_center_y + int(35 * scale)
            ],
            fill=hair
        )

    else:
        draw.arc(
            [
                x - head_r,
                hair_top - int(12 * scale),
                x + head_r,
                head_center_y + int(20 * scale)
            ],
            180,
            360,
            fill=hair,
            width=max(4, int(16 * scale))
        )

    # Eyes
    eye_y = head_center_y - int(7 * scale)

    draw.ellipse(
        [
            x - int(23 * scale),
            eye_y,
            x - int(13 * scale),
            eye_y + int(10 * scale)
        ],
        fill="#111111"
    )

    draw.ellipse(
        [
            x + int(13 * scale),
            eye_y,
            x + int(23 * scale),
            eye_y + int(10 * scale)
        ],
        fill="#111111"
    )

    # Emotion
    emotion = str(emotion or "").lower()

    if any(word in emotion for word in [
        "angry",
        "mad",
        "furious"
    ]):
        draw.line(
            [
                x - int(27 * scale),
                eye_y - int(7 * scale),
                x - int(12 * scale),
                eye_y - int(1 * scale)
            ],
            fill="#111111",
            width=max(2, int(5 * scale))
        )

        draw.line(
            [
                x + int(12 * scale),
                eye_y - int(1 * scale),
                x + int(27 * scale),
                eye_y - int(7 * scale)
            ],
            fill="#111111",
            width=max(2, int(5 * scale))
        )

    elif any(word in emotion for word in [
        "sad",
        "crying"
    ]):
        draw.arc(
            [
                x - int(25 * scale),
                head_center_y + int(5 * scale),
                x + int(25 * scale),
                head_center_y + int(32 * scale)
            ],
            20,
            160,
            fill="#111111",
            width=max(2, int(5 * scale))
        )

    else:
        # Mouth
        if talking:
            draw.ellipse(
                [
                    x - int(15 * scale),
                    head_center_y + int(15 * scale),
                    x + int(15 * scale),
                    head_center_y + int(36 * scale)
                ],
                fill="#7F1D1D"
            )
        else:
            draw.arc(
                [
                    x - int(18 * scale),
                    head_center_y + int(12 * scale),
                    x + int(18 * scale),
                    head_center_y + int(34 * scale)
                ],
                0,
                180,
                fill="#7F1D1D",
                width=max(2, int(4 * scale))
            )

    # Character name
    name_font = get_font(
        max(13, int(18 * scale)),
        bold=True
    )

    name = profile["name"]

    bbox = draw.textbbox(
        (0, 0),
        name,
        font=name_font
    )

    text_width = bbox[2] - bbox[0]

    name_y = ground_y + int(15 * scale)

    draw.rounded_rectangle(
        [
            x - text_width // 2 - 7,
            name_y,
            x + text_width // 2 + 7,
            name_y + int(27 * scale)
        ],
        radius=6,
        fill="#111827"
    )

    draw.text(
        (
            x - text_width // 2,
            name_y + int(3 * scale)
        ),
        name,
        font=name_font,
        fill="white"
    )


# ============================================================
# CHARACTER POSITIONS
# ============================================================

def character_positions(count):
    if count <= 1:
        return [WIDTH // 2]

    margin = 55

    usable = WIDTH - (margin * 2)

    return [
        int(
            margin +
            usable * i / (count - 1)
        )
        for i in range(count)
    ]


# ============================================================
# DIALOGUE BOX
# ============================================================

def draw_dialogue_box(
    draw,
    speaker,
    text,
    emotion=""
):
    if not text:
        return

    box_font = get_font(
        19,
        bold=False
    )

    name_font = get_font(
        18,
        bold=True
    )

    max_width = WIDTH - 50

    lines = wrap_text(
        text,
        box_font,
        max_width - 30
    )

    lines = lines[:4]

    line_height = 25

    box_height = (
        50 +
        len(lines) * line_height
    )

    top = HEIGHT - box_height - 18

    # Box
    draw.rounded_rectangle(
        [
            15,
            top,
            WIDTH - 15,
            HEIGHT - 18
        ],
        radius=16,
        fill="#111827",
        outline="#FFFFFF",
        width=2
    )

    # Speaker
    draw.text(
        (30, top + 12),
        str(speaker),
        font=name_font,
        fill="#FACC15"
    )

    # Dialogue
    y = top + 40

    for line in lines:
        draw.text(
            (30, y),
            line,
            font=box_font,
            fill="white"
        )

        y += line_height


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

    user_input = str(topic or "").strip()

    prompt = f"""
You are the screenplay and scene planner for AfriToon Studio.

Create a short animated African cartoon/video.

IMPORTANT:
The user's input below is the source of truth.

USER INPUT:
----------------
{user_input}
----------------

CATEGORY:
{category}

LANGUAGE:
{language}

TARGET VIDEO LENGTH:
{duration} seconds

============================================================
CRITICAL SCRIPT RULE
============================================================

If the USER INPUT contains an explicit script, scene list,
dialogue, character names, or actions:

DO NOT rewrite the story.

DO NOT replace character names.

DO NOT invent different character names.

DO NOT change the order of scenes.

DO NOT change who says a dialogue line.

DO NOT remove dialogue.

DO NOT merge different characters together.

DO NOT make one character speak another character's line.

Preserve the user's dialogue wording as closely as possible.

Your job is to STRUCTURE the user's script into JSON so the
video renderer can follow it.

If the user gives:

Amaka: Hello John.

John: Hello Amaka.

Mary: Wait for me!

Then the characters MUST be:

Amaka
John
Mary

And the dialogue MUST remain associated with those exact
characters.

============================================================
CHARACTERS
============================================================

Extract EVERY named character from the user's script.

There can be:

2 characters
3 characters
4 characters
5 characters
6 characters
or more.

Do not limit the story to Kofi, Amina, Nana, Tunde,
or any predefined names.

Use the names supplied by the user.

Every character must receive a stable visual identity.

============================================================
SCENES
============================================================

Preserve explicit scenes from the user's script.

Each scene must contain:

- scene number
- location
- time
- action
- characters_present
- dialogue

The action should describe what the characters visibly do.

Example:

"Amaka walks toward John and points at the house."

Dialogue format:

{{
  "character": "Amaka",
  "text": "John, where are you going?",
  "emotion": "curious",
  "action": "Amaka walks toward John and points at him."
}}

The "character" field MUST contain the exact character name.

============================================================
VISUAL CONSISTENCY
============================================================

Each character gets:

- gender
- approximate age
- skin tone
- hair
- clothing
- body type
- personality

Keep the same character appearance throughout every scene.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Use exactly this general structure:

{{
  "title": "string",
  "description": "string",
  "characters": [
    {{
      "name": "Amaka",
      "gender": "female",
      "age": "young adult",
      "skin_tone": "dark brown",
      "hair": "black braided hair",
      "clothing": "yellow shirt and blue skirt",
      "body_type": "average",
      "personality": "friendly"
    }}
  ],
  "scenes": [
    {{
      "scene": 1,
      "location": "village road",
      "time": "morning",
      "action": "Amaka walks along the road.",
      "characters_present": ["Amaka", "John"],
      "dialogue": [
        {{
          "character": "Amaka",
          "text": "Hello John.",
          "emotion": "happy",
          "action": "Amaka waves at John."
        }}
      ]
    }}
  ]
}}

Rules:

- Maximum {MAX_SCENES} scenes unless the user explicitly
  provided more scenes.
- Maximum {MAX_DIALOGUE_LINES} dialogue lines.
- Every dialogue speaker MUST exist in characters.
- Every character mentioned by name must be in characters.
- characters_present must contain the people visible in that
  scene.
- Keep scene order.
- Keep dialogue order.
- Return JSON only.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.15,
            response_mime_type="application/json"
        )
    )

    raw = response.text or ""

    data = json.loads(
        clean_json(raw)
    )

    return normalize_story(
        data,
        language,
        user_input
    )


# ============================================================
# STORY NORMALIZATION
# ============================================================

def normalize_story(
    story,
    language,
    original_input=""
):
    if not isinstance(story, dict):
        raise ValueError(
            "Gemini returned invalid story data."
        )

    title = normalize_name(
        story.get("title", "AfriToon Story")
    )

    description = str(
        story.get("description", "")
    ).strip()

    raw_characters = story.get(
        "characters",
        []
    )

    characters = []

    # --------------------------------------------------------
    # Collect characters from character list
    # --------------------------------------------------------

    if isinstance(raw_characters, list):
        for item in raw_characters:
            if isinstance(item, str):
                name = normalize_name(item)
                profile = {
                    "name": name
                }
            elif isinstance(item, dict):
                name = normalize_name(
                    item.get("name", "")
                )

                profile = dict(item)

            else:
                continue

            if name:
                profile["name"] = name

                if name_key(name) not in [
                    name_key(x.get("name", ""))
                    for x in characters
                ]:
                    characters.append(profile)

    # --------------------------------------------------------
    # Collect EVERY dialogue speaker
    # --------------------------------------------------------

    raw_scenes = story.get(
        "scenes",
        []
    )

    if not isinstance(raw_scenes, list):
        raw_scenes = []

    for scene in raw_scenes:
        if not isinstance(scene, dict):
            continue

        dialogue = scene.get(
            "dialogue",
            []
        )

        if not isinstance(dialogue, list):
            continue

        for line in dialogue:
            if not isinstance(line, dict):
                continue

            speaker = normalize_name(
                line.get("character")
                or line.get("speaker")
                or ""
            )

            if not speaker:
                continue

            if speaker.lower() not in [
                name_key(x.get("name", ""))
                for x in characters
            ]:
                characters.append({
                    "name": speaker
                })

    # --------------------------------------------------------
    # Normalize scenes
    # --------------------------------------------------------

    normalized_scenes = []

    total_dialogue = 0

    for index, raw_scene in enumerate(
        raw_scenes[:MAX_SCENES],
        start=1
    ):
        if not isinstance(raw_scene, dict):
            continue

        location = str(
            raw_scene.get(
                "location",
                "African village"
            )
        ).strip()

        scene_time = str(
            raw_scene.get(
                "time",
                "day"
            )
        ).strip()

        action = str(
            raw_scene.get(
                "action",
                ""
            )
        ).strip()

        # ----------------------------------------------------
        # Characters present
        # ----------------------------------------------------

        present = raw_scene.get(
            "characters_present",
            []
        )

        if not isinstance(present, list):
            present = []

        present_names = []

        for item in present:
            if isinstance(item, dict):
                n = normalize_name(
                    item.get("name", "")
                )
            else:
                n = normalize_name(item)

            if n:
                present_names.append(n)

        # ----------------------------------------------------
        # Dialogue
        # ----------------------------------------------------

        raw_dialogue = raw_scene.get(
            "dialogue",
            []
        )

        if not isinstance(raw_dialogue, list):
            raw_dialogue = []

        normalized_dialogue = []

        for line in raw_dialogue:
            if total_dialogue >= MAX_DIALOGUE_LINES:
                break

            if isinstance(line, dict):
                speaker = normalize_name(
                    line.get("character")
                    or line.get("speaker")
                    or ""
                )

                text = str(
                    line.get("text")
                    or line.get("dialogue")
                    or ""
                ).strip()

                emotion = str(
                    line.get(
                        "emotion",
                        "neutral"
                    )
                ).strip()

                line_action = str(
                    line.get(
                        "action",
                        ""
                    )
                ).strip()

            else:
                continue

            if not speaker or not text:
                continue

            # Make sure speaker exists in characters
            exists = False

            for character in characters:
                if name_key(
                    character.get("name", "")
                ) == name_key(speaker):
                    exists = True
                    break

            if not exists:
                characters.append({
                    "name": speaker
                })

            if name_key(speaker) not in [
                name_key(x)
                for x in present_names
            ]:
                present_names.append(speaker)

            normalized_dialogue.append({
                "character": speaker,
                "text": text,
                "emotion": emotion,
                "action": line_action
            })

            total_dialogue += 1

        # If there is no explicit characters_present,
        # derive it from dialogue.
        if not present_names:
            present_names = [
                line["character"]
                for line in normalized_dialogue
            ]

        normalized_scenes.append({
            "scene": index,
            "location": location,
            "time": scene_time,
            "action": action,
            "characters_present": present_names,
            "dialogue": normalized_dialogue
        })

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if not normalized_scenes:
        normalized_scenes = [{
            "scene": 1,
            "location": "African village",
            "time": "day",
            "action": "",
            "characters_present": [],
            "dialogue": []
        }]

    # --------------------------------------------------------
    # Add language
    # --------------------------------------------------------

    story_out = {
        "title": title,
        "description": description,
        "language": language,
        "characters": characters,
        "scenes": normalized_scenes
    }

    # Keep original user input for debugging/reference
    story_out["_original_input"] = original_input

    return story_out


# ============================================================
# TTS
# ============================================================

def get_tts_language(language):
    language = str(
        language or "English"
    ).lower()

    if "spanish" in language:
        return "es"

    if "french" in language:
        return "fr"

    if "portuguese" in language:
        return "pt"

    return "en"


def make_tts(
    text,
    output_path,
    language
):
    tts_language = get_tts_language(
        language
    )

    tts = gTTS(
        text=text,
        lang=tts_language,
        slow=False
    )

    tts.save(str(output_path))


def get_audio_duration(path):
    ffmpeg = shutil.which("ffprobe")

    if not ffmpeg:
        return 2.0

    try:
        result = subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path)
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        value = float(
            result.stdout.strip()
        )

        if value > 0:
            return value

    except Exception:
        pass

    return 2.0


# ============================================================
# AUDIO CONCAT
# ============================================================

def combine_audio(
    audio_files,
    output_path,
    work_dir
):
    if not audio_files:
        return None

    ffmpeg = shutil.which("ffmpeg")

    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg is not installed."
        )

    concat_file = (
        work_dir /
        "audio_concat.txt"
    )

    with open(
        concat_file,
        "w",
        encoding="utf-8"
    ) as f:
        for audio in audio_files:
            safe_path = str(
                audio
            ).replace(
                "'",
                "'\\''"
            )

            f.write(
                f"file '{safe_path}'\n"
            )

    subprocess.run(
        [
            ffmpeg,
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
        ],
        check=True,
        capture_output=True
    )

    return output_path


# ============================================================
# CREATE TIMELINE
# ============================================================

def create_timeline(
    story,
    work_dir
):
    """
    Creates one timeline item for every dialogue line.

    Each item remembers:
    - scene
    - location
    - action
    - characters
    - speaker
    - emotion
    - dialogue
    - audio duration
    """

    language = story.get(
        "language",
        "English"
    )

    timeline = []

    audio_files = []

    for scene in story.get(
        "scenes",
        []
    ):

        scene_number = scene.get(
            "scene",
            len(timeline) + 1
        )

        dialogue = scene.get(
            "dialogue",
            []
        )

        if not dialogue:
            continue

        for line_index, line in enumerate(
            dialogue
        ):
            speaker = normalize_name(
                line.get(
                    "character",
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

            audio_path = (
                work_dir /
                f"line_{len(timeline):04d}.mp3"
            )

            make_tts(
                text,
                audio_path,
                language
            )

            duration = get_audio_duration(
                audio_path
            )

            item = {
                "scene": scene_number,
                "location": scene.get(
                    "location",
                    "African village"
                ),
                "time": scene.get(
                    "time",
                    "day"
                ),
                "scene_action": scene.get(
                    "action",
                    ""
                ),
                "characters_present": scene.get(
                    "characters_present",
                    []
                ),
                "speaker": speaker,
                "text": text,
                "emotion": line.get(
                    "emotion",
                    "neutral"
                ),
                "action": line.get(
                    "action",
                    ""
                ),
                "duration": max(
                    0.8,
                    duration
                ),
                "audio": audio_path
            }

            timeline.append(item)
            audio_files.append(
                audio_path
            )

    return timeline, audio_files


# ============================================================
# DRAW FRAME
# ============================================================

def draw_frame(
    story,
    timeline_item,
    profiles,
    frame_index,
    total_frames
):
    image = Image.new(
        "RGB",
        (WIDTH, HEIGHT),
        "#87CEEB"
    )

    draw = ImageDraw.Draw(
        image
    )

    scene_number = timeline_item[
        "scene"
    ]

    total_scenes = max(
        1,
        len(story.get("scenes", []))
    )

    # --------------------------------------------------------
    # BACKGROUND
    # --------------------------------------------------------

    draw_scene_background(
        draw,
        timeline_item,
        scene_number,
        total_scenes
    )

    # --------------------------------------------------------
    # CHARACTERS IN SCENE
    # --------------------------------------------------------

    present = timeline_item.get(
        "characters_present",
        []
    )

    if not present:
        present = [
            timeline_item["speaker"]
        ]

    # Remove duplicates but preserve order
    unique_present = []

    for name in present:
        name = normalize_name(name)

        if not name:
            continue

        if name_key(name) not in [
            name_key(x)
            for x in unique_present
        ]:
            unique_present.append(name)

    speaker = normalize_name(
        timeline_item["speaker"]
    )

    # If speaker somehow isn't in present,
    # add them.
    if name_key(speaker) not in [
        name_key(x)
        for x in unique_present
    ]:
        unique_present.append(speaker)

    # Maximum reasonable number visible
    visible = unique_present[:6]

    positions = character_positions(
        len(visible)
    )

    if len(visible) >= 5:
        scale = 0.55
    elif len(visible) == 4:
        scale = 0.62
    elif len(visible) == 3:
        scale = 0.70
    elif len(visible) == 2:
        scale = 0.78
    else:
        scale = 0.88

    for index, name in enumerate(
        visible
    ):
        key = name_key(name)

        profile = profiles.get(key)

        if not profile:
            profile = build_character_profile(
                name,
                index
            )

        talking = (
            key ==
            name_key(speaker)
        )

        draw_character(
            draw=draw,
            x=positions[index],
            ground_y=620,
            scale=scale,
            profile=profile,
            talking=talking,
            emotion=(
                timeline_item.get(
                    "emotion",
                    "neutral"
                )
                if talking
                else "neutral"
            ),
            action=(
                timeline_item.get(
                    "action",
                    ""
                )
                if talking
                else ""
            )
        )

    # --------------------------------------------------------
    # ACTION CAPTION
    # --------------------------------------------------------

    action = str(
        timeline_item.get(
            "action",
            ""
        ) or ""
    ).strip()

    if action:
        action_font = get_font(
            15,
            bold=False
        )

        action_lines = wrap_text(
            action,
            action_font,
            WIDTH - 50
        )

        action_lines = action_lines[:2]

        if action_lines:
            y = 58

            for line in action_lines:
                draw.rounded_rectangle(
                    [
                        15,
                        y - 2,
                        WIDTH - 15,
                        y + 24
                    ],
                    radius=7,
                    fill="#111827"
                )

                draw_centered_text(
                    draw,
                    line,
                    y + 2,
                    action_font,
                    "#F8FAFC"
                )

                y += 27

    # --------------------------------------------------------
    # DIALOGUE
    # --------------------------------------------------------

    draw_dialogue_box(
        draw,
        speaker,
        timeline_item["text"],
        timeline_item.get(
            "emotion",
            "neutral"
        )
    )

    return image


# ============================================================
# CREATE VIDEO
# ============================================================

def create_video(
    job_id,
    story,
    requested_duration,
    backend_url
):
    work_dir = (
        BASE_DIR /
        f"work_{job_id}"
    )

    work_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    try:
        update_job(
            job_id,
            status="processing",
            progress=5,
            message="Preparing characters and scenes..."
        )

        # ----------------------------------------------------
        # CHARACTER PROFILES
        # ----------------------------------------------------

        profiles = build_character_profiles(
            story.get(
                "characters",
                []
            )
        )

        update_job(
            job_id,
            progress=10,
            message="Preparing dialogue audio..."
        )

        # ----------------------------------------------------
        # TIMELINE
        # ----------------------------------------------------

        timeline, audio_files = create_timeline(
            story,
            work_dir
        )

        if not timeline:
            raise RuntimeError(
                "The story contains no usable dialogue."
            )

        update_job(
            job_id,
            progress=25,
            message="Building movie timeline..."
        )

        # ----------------------------------------------------
        # AUDIO
        # ----------------------------------------------------

        combined_audio = (
            work_dir /
            "combined_audio.m4a"
        )

        combine_audio(
            audio_files,
            combined_audio,
            work_dir
        )

        audio_duration = sum(
            float(item["duration"])
            for item in timeline
        )

        # Requested duration should not destroy dialogue.
        target_duration = max(
            1.0,
            min(
                float(requested_duration),
                audio_duration
            )
        )

        # ----------------------------------------------------
        # FRAMES
        # ----------------------------------------------------

        frames_dir = (
            work_dir /
            "frames"
        )

        frames_dir.mkdir(
            exist_ok=True
        )

        update_job(
            job_id,
            progress=30,
            message="Animating characters..."
        )

        total_frames = int(
            target_duration * FPS
        )

        # Build timeline positions
        current_start = 0.0

        timeline_ranges = []

        for item in timeline:
            start = current_start
            end = (
                current_start +
                item["duration"]
            )

            timeline_ranges.append(
                (
                    start,
                    end,
                    item
                )
            )

            current_start = end

        for frame_index in range(
            total_frames
        ):
            current_time = (
                frame_index /
                FPS
            )

            current_item = timeline[-1]

            for start, end, item in timeline_ranges:
                if start <= current_time < end:
                    current_item = item
                    break

            image = draw_frame(
                story,
                current_item,
                profiles,
                frame_index,
                total_frames
            )

            frame_path = (
                frames_dir /
                f"frame_{frame_index:06d}.jpg"
            )

            image.save(
                frame_path,
                quality=88
            )

            if frame_index % max(
                1,
                FPS * 2
            ) == 0:

                progress = 30 + int(
                    (
                        frame_index /
                        max(1, total_frames)
                    ) * 55
                )

                update_job(
                    job_id,
                    progress=min(
                        85,
                        progress
                    ),
                    message=(
                        f"Animating frame "
                        f"{frame_index + 1}/"
                        f"{total_frames}"
                    )
                )

        # ----------------------------------------------------
        # FFMPEG
        # ----------------------------------------------------

        ffmpeg = shutil.which(
            "ffmpeg"
        )

        if not ffmpeg:
            raise RuntimeError(
                "FFmpeg is not installed."
            )

        update_job(
            job_id,
            progress=88,
            message="Rendering MP4..."
        )

        title = safe_filename(
            story.get(
                "title",
                "AfriToon"
            )
        )

        filename = (
            f"{title}_"
            f"{job_id[:8]}.mp4"
        )

        output_path = (
            OUTPUT_DIR /
            filename
        )

        command = [
            ffmpeg,
            "-y",

            "-framerate",
            str(FPS),

            "-i",
            str(
                frames_dir /
                "frame_%06d.jpg"
            ),

            "-i",
            str(combined_audio),

            "-c:v",
            "libx264",

            "-preset",
            "ultrafast",

            "-crf",
            "28",

            "-pix_fmt",
            "yuv420p",

            "-c:a",
            "aac",

            "-b:a",
            "96k",

            "-shortest",

            "-movflags",
            "+faststart",

            str(output_path)
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(
                "FFmpeg failed:\n" +
                result.stderr[-4000:]
            )

        # ----------------------------------------------------
        # COMPLETE
        # ----------------------------------------------------

        video_url = build_video_url(
            backend_url,
            filename
        )

        download_url = build_download_url(
            backend_url,
            filename
        )

        update_job(
            job_id,
            status="complete",
            progress=100,
            message="Video completed successfully.",
            filename=filename,
            video_url=video_url,
            download_url=download_url
        )

    except Exception as e:

        update_job(
            job_id,
            status="error",
            progress=0,
            message=str(e)
        )

    finally:
        # Remove temporary files
        try:
            shutil.rmtree(
                work_dir,
                ignore_errors=True
            )
        except Exception:
            pass


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def home():
    return jsonify({
        "success": True,
        "app": "AfriToon Studio",
        "status": "online"
    })


# ============================================================
# GENERATE STORY
# ============================================================

@app.post("/api/generate")
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

        topic = str(
            data.get(
                "topic",
                data.get(
                    "prompt",
                    data.get(
                        "script",
                        ""
                    )
                )
            )
        ).strip()

        language = str(
            data.get(
                "language",
                "English"
            )
        ).strip()

        duration = int(
            data.get(
                "duration",
                30
            )
        )

        if not topic:
            return jsonify({
                "success": False,
                "error": (
                    "Enter a story prompt or script."
                )
            }), 400

        if duration not in [
            30,
            60
        ]:
            duration = 30

        story = generate_story(
            category=category,
            topic=topic,
            duration=duration,
            language=language
        )

        return jsonify({
            "success": True,
            "story": story
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# CREATE VIDEO
# ============================================================

@app.post("/api/create-video")
def api_create_video():
    try:
        data = request.get_json(
            silent=True
        ) or {}

        story = data.get(
            "story"
        )

        duration = int(
            data.get(
                "duration",
                30
            )
        )

        if not isinstance(
            story,
            dict
        ):
            return jsonify({
                "success": False,
                "error": "Story is required."
            }), 400

        if duration not in [
            30,
            60
        ]:
            duration = 30

        # Re-normalize in case the frontend
        # modified the story.
        story = normalize_story(
            story,
            story.get(
                "language",
                "English"
            ),
            story.get(
                "_original_input",
                ""
            )
        )

        job_id = uuid.uuid4().hex

        backend_url = (
            BASE_URL
            or request.host_url.rstrip("/")
        )

        update_job(
            job_id,
            status="queued",
            progress=0,
            message="Video generation queued."
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
            "status": "queued"
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# VIDEO STATUS
# ============================================================

@app.get("/api/video-status/<job_id>")
def api_video_status(job_id):

    job = get_job(job_id)

    if not job:
        return jsonify({
            "success": False,
            "error": "Job not found."
        }), 404

    return jsonify({
        "success": True,
        **job
    })


# ============================================================
# SERVE VIDEO
# ============================================================

@app.get("/generated/<path:filename>")
def serve_generated(filename):
    return send_from_directory(
        OUTPUT_DIR,
        filename,
        as_attachment=False
    )


# ============================================================
# DOWNLOAD VIDEO
# ============================================================

@app.get("/download/<path:filename>")
def download_generated(filename):
    return send_from_directory(
        OUTPUT_DIR,
        filename,
        as_attachment=True,
        download_name=filename
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "ffmpeg": bool(
            shutil.which("ffmpeg")
        ),
        "gemini": bool(
            GEMINI_API_KEY
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
