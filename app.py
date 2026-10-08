import os
import json
import uuid
import re
import subprocess
import threading
import shutil
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

CORS(
    app,
    resources={
        r"/*": {
            "origins": "*"
        }
    }
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "generated"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

JOBS = {}
JOB_LOCK = threading.Lock()


# ============================================================
# ENVIRONMENT
# ============================================================

API_KEY = os.getenv("GEMINI_API_KEY")

if API_KEY:
    client = genai.Client(
        api_key=API_KEY
    )
else:
    client = None


BASE_URL = os.getenv(
    "BASE_URL",
    ""
).rstrip("/")


# ============================================================
# VIDEO SETTINGS
# ============================================================

WIDTH = 480
HEIGHT = 854

FPS = 8

MAX_SCENES = 6
MAX_DIALOGUE_LINES = 14


# ============================================================
# JOB HELPERS
# ============================================================

def update_job(
    job_id,
    status=None,
    message=None,
    progress=None,
    **extra
):

    with JOB_LOCK:

        job = JOBS.get(job_id)

        if not job:
            return

        if status is not None:
            job["status"] = status

        if message is not None:
            job["message"] = message

        if progress is not None:

            job["progress"] = max(
                0,
                min(
                    100,
                    int(progress)
                )
            )

        for key, value in extra.items():
            job[key] = value


def get_job(job_id):

    with JOB_LOCK:

        job = JOBS.get(job_id)

        if not job:
            return None

        return dict(job)


# ============================================================
# PUBLIC URLS
# ============================================================

def build_video_url(
    backend_url,
    filename
):

    backend_url = (
        backend_url or ""
    ).rstrip("/")

    return (
        f"{backend_url}"
        f"/generated/"
        f"{filename}"
    )


def build_download_url(
    backend_url,
    filename
):

    backend_url = (
        backend_url or ""
    ).rstrip("/")

    return (
        f"{backend_url}"
        f"/download/"
        f"{filename}"
    )


# ============================================================
# FFMPEG
# ============================================================

def get_ffmpeg():

    ffmpeg = shutil.which(
        "ffmpeg"
    )

    if not ffmpeg:

        raise RuntimeError(
            "FFmpeg is not installed on the Render server."
        )

    return ffmpeg


# ============================================================
# FONTS
# ============================================================

FONT_CACHE = {}


def font(
    size=30,
    bold=False
):

    key = (
        size,
        bold
    )

    if key in FONT_CACHE:
        return FONT_CACHE[key]

    if bold:

        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        ]

    else:

        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"
        ]

    for path in candidates:

        if os.path.exists(path):

            loaded = ImageFont.truetype(
                path,
                size
            )

            FONT_CACHE[key] = loaded

            return loaded

    loaded = ImageFont.load_default()

    FONT_CACHE[key] = loaded

    return loaded


# ============================================================
# JSON CLEANING
# ============================================================

def clean_json(text):

    if not text:

        raise Exception(
            "Gemini returned an empty response."
        )

    text = text.strip()

    if text.startswith("```"):

        lines = text.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines)

    return text.strip()


# ============================================================
# SAFE FILE NAME
# ============================================================

def safe_filename(name):

    cleaned = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        str(name)
    )

    cleaned = cleaned.strip("_")

    return (
        cleaned[:70]
        or "afritoon"
    )


# ============================================================
# STORY GENERATION
# ============================================================

def generate_story(
    category,
    topic,
    duration,
    language
):

    if not client:

        raise Exception(
            "GEMINI_API_KEY is missing on Render."
        )

    user_story = (
        topic.strip()
        if topic
        else
        "Create an original short African animated story."
    )

    prompt = f"""
You are the STORY DIRECTOR for AfriToon Studio.

The user wants to make a {duration}-second animated movie.

==================================================
USER'S STORY IDEA
==================================================

{user_story}

==================================================
STRICT STORY RULE
==================================================

The user's story idea above is the PRIMARY STORY.

You MUST follow it.

DO NOT replace it with an unrelated story.

DO NOT change the main subject.

DO NOT ignore important events in the user's prompt.

DO NOT randomly introduce a different plot.

Expand the user's idea into a short cinematic animated movie.

Every scene MUST directly contribute to the user's story.

If the user gives events in a particular order,
keep those events in that chronological order.

==================================================
CATEGORY
==================================================

{category}

==================================================
LANGUAGE
==================================================

{language}

==================================================
DURATION
==================================================

{duration} seconds.

==================================================
MOVIE STRUCTURE
==================================================

Create 4 to 6 scenes.

The story must have:

1. Opening
2. Development
3. Important event
4. Climax
5. Ending

The scenes must be chronological.

Do not repeat the same scene.

==================================================
CINEMATIC VISUAL STORYTELLING
==================================================

This is an animated MOVIE.

Do not make every scene look like a talking-head presentation.

Characters should physically act when appropriate:

- walk
- run
- look around
- point
- sit
- stand
- turn
- react
- smile
- become surprised
- become afraid
- interact with objects
- interact with other characters

Every scene MUST have visible action.

==================================================
SCENE LOCATIONS
==================================================

The location must match the story.

Examples:

forest
village
market
house
school
farm
river
ocean
beach
mountain
city
desert
space
heaven
garden
laboratory
hospital
road
palace

Do not use a generic village background if the story requires
another location.

==================================================
CAMERA
==================================================

For every scene specify:

camera_shot

Examples:

wide shot
extreme wide shot
medium shot
close-up
extreme close-up
over-the-shoulder

camera_movement

Examples:

slow zoom in
slow zoom out
pan left
pan right
tracking shot
camera tilt up
camera tilt down
static shot

==================================================
CHARACTERS
==================================================

Use 2 to 4 important characters maximum.

Keep their appearance consistent.

Each character must have:

name
age
appearance
clothing
personality

==================================================
DIALOGUE
==================================================

Use short natural dialogue.

Maximum 10 dialogue lines.

Every dialogue line MUST belong to the correct scene.

==================================================
RETURN ONLY JSON
==================================================

Use exactly this structure:

{{
  "title": "Movie title",

  "description": "Short movie description",

  "characters": [
    {{
      "name": "Character name",
      "age": "Age",
      "appearance": "Detailed appearance",
      "clothing": "Clothing",
      "personality": "Personality"
    }}
  ],

  "scenes": [
    {{
      "scene": 1,

      "duration": 10,

      "location": "Exact location",

      "time_of_day": "Time",

      "weather": "Weather",

      "background": "Detailed visual background",

      "action": "Detailed visible physical action",

      "camera_shot": "Camera shot",

      "camera_movement": "Camera movement",

      "mood": "Mood",

      "characters_present": [
        "Character name"
      ],

      "character_actions": [
        {{
          "character": "Character name",
          "action": "Physical action",
          "emotion": "Emotion"
        }}
      ],

      "dialogue": [
        {{
          "character": "Character name",
          "text": "Short dialogue",
          "emotion": "Emotion"
        }}
      ]
    }}
  ]
}}

IMPORTANT:

Follow the user's story.

Do not create a different story.

Make every scene visually different when the story requires it.

Return ONLY valid JSON.
"""

    response = client.models.generate_content(

        model="gemini-3.5-flash-lite",

        contents=prompt,

        config=types.GenerateContentConfig(

            response_mime_type="application/json"
        )
    )

    text = clean_json(
        response.text
    )

    story = json.loads(
        text
    )

    scenes = story.get(
        "scenes",
        []
    )

    if not scenes:

        raise Exception(
            "Gemini returned no scenes."
        )

    scenes = scenes[:MAX_SCENES]

    # --------------------------------------------------------
    # NORMALIZE SCENES
    # --------------------------------------------------------

    for index, scene in enumerate(
        scenes,
        1
    ):

        scene["scene"] = index

        scene["location"] = str(
            scene.get(
                "location",
                "village"
            )
        )

        scene["background"] = str(
            scene.get(
                "background",
                scene["location"]
            )
        )

        scene["action"] = str(
            scene.get(
                "action",
                "The characters continue the story."
            )
        )

        scene["camera_shot"] = str(
            scene.get(
                "camera_shot",
                "medium shot"
            )
        )

        scene["camera_movement"] = str(
            scene.get(
                "camera_movement",
                "slow zoom in"
            )
        )

        scene["mood"] = str(
            scene.get(
                "mood",
                "cinematic"
            )
        )

        scene["characters_present"] = (
            scene.get(
                "characters_present",
                []
            )
        )

        scene["character_actions"] = (
            scene.get(
                "character_actions",
                []
            )
        )

        scene["dialogue"] = (
            scene.get(
                "dialogue",
                []
            )
        )

        try:

            scene["duration"] = float(
                scene.get(
                    "duration",
                    0
                )
            )

        except Exception:

            scene["duration"] = 0

    # --------------------------------------------------------
    # FIX SCENE DURATIONS
    # --------------------------------------------------------

    requested = float(
        duration
    )

    scene_count = len(
        scenes
    )

    specified_total = sum(
        max(
            0,
            scene["duration"]
        )
        for scene in scenes
    )

    if specified_total <= 0:

        each = requested / scene_count

        for scene in scenes:
            scene["duration"] = each

    else:

        scale = (
            requested /
            specified_total
        )

        for scene in scenes:

            scene["duration"] = max(
                3.0,
                scene["duration"] * scale
            )

        # Re-normalize if minimum durations pushed
        # total above requested duration.

        total = sum(
            scene["duration"]
            for scene in scenes
        )

        if total > 0:

            for scene in scenes:

                scene["duration"] = (
                    scene["duration"]
                    / total
                    * requested
                )

    story["scenes"] = scenes
    story["language"] = language

    return story


# ============================================================
# COLOR HELPERS
# ============================================================

def lerp_color(
    a,
    b,
    amount
):

    amount = max(
        0,
        min(
            1,
            amount
        )
    )

    return tuple(
        int(
            a[i]
            +
            (b[i] - a[i])
            * amount
        )
        for i in range(3)
    )


# ============================================================
# BACKGROUND HELPERS
# ============================================================

def draw_sun(
    draw,
    x,
    y,
    radius
):

    draw.ellipse(
        [
            x - radius,
            y - radius,
            x + radius,
            y + radius
        ],
        fill="#FFD54A"
    )


def draw_cloud(
    draw,
    x,
    y,
    scale=1
):

    parts = [
        (-35, 10, 45),
        (0, -5, 55),
        (40, 12, 40)
    ]

    for dx, dy, r in parts:

        draw.ellipse(
            [
                x + int((dx - r) * scale),
                y + int((dy - r) * scale),
                x + int((dx + r) * scale),
                y + int((dy + r) * scale)
            ],
            fill="#F5F8FA"
        )


def draw_tree(
    draw,
    x,
    ground_y,
    scale=1
):

    trunk_w = int(
        28 * scale
    )

    trunk_h = int(
        150 * scale
    )

    draw.rectangle(
        [
            x - trunk_w,
            ground_y - trunk_h,
            x + trunk_w,
            ground_y
        ],
        fill="#684020"
    )

    for dx, dy, r in [
        (-45, -170, 65),
        (20, -190, 75),
        (70, -155, 55),
        (-5, -120, 70)
    ]:

        draw.ellipse(
            [
                x + int(dx * scale) - int(r * scale),
                ground_y + int(dy * scale) - int(r * scale),
                x + int(dx * scale) + int(r * scale),
                ground_y + int(dy * scale) + int(r * scale)
            ],
            fill="#3E7D3A"
        )


def draw_mountain(
    draw,
    x,
    ground_y,
    width,
    height
):

    draw.polygon(
        [
            (x - width, ground_y),
            (x, ground_y - height),
            (x + width, ground_y)
        ],
        fill="#765A45"
    )

    draw.polygon(
        [
            (x, ground_y - height),
            (x - int(width * 0.25), ground_y - int(height * 0.55)),
            (x + int(width * 0.20), ground_y - int(height * 0.45))
        ],
        fill="#F1F1F1"
    )


def draw_house(
    draw,
    x,
    ground_y,
    scale=1
):

    w = int(
        150 * scale
    )

    h = int(
        115 * scale
    )

    draw.rectangle(
        [
            x - w,
            ground_y - h,
            x + w,
            ground_y
        ],
        fill="#C77D45"
    )

    draw.polygon(
        [
            (x - w - 20, ground_y - h),
            (x, ground_y - h - int(90 * scale)),
            (x + w + 20, ground_y - h)
        ],
        fill="#713F25"
    )

    door_w = int(
        45 * scale
    )

    draw.rectangle(
        [
            x - door_w,
            ground_y - int(80 * scale),
            x + door_w,
            ground_y
        ],
        fill="#4B2E1E"
    )


def draw_ocean(
    draw,
    ground_y
):

    draw.rectangle(
        [
            0,
            ground_y - 100,
            WIDTH,
            HEIGHT
        ],
        fill="#2077A8"
    )

    for i in range(5):

        y = ground_y - 70 + i * 45

        draw.arc(
            [
                -80,
                y - 20,
                WIDTH + 80,
                y + 35
            ],
            180,
            360,
            fill="#A9E7F5",
            width=3
        )


def draw_stars(
    draw
):

    positions = [
        (40, 80),
        (100, 150),
        (170, 90),
        (240, 180),
        (320, 80),
        (400, 145),
        (440, 60),
        (280, 120),
        (130, 230),
        (370, 240)
    ]

    for x, y in positions:

        r = 3

        draw.ellipse(
            [
                x - r,
                y - r,
                x + r,
                y + r
            ],
            fill="white"
        )


def draw_city(
    draw,
    ground_y
):

    buildings = [
        (20, 300, 100, 260),
        (130, 220, 220, 340),
        (250, 280, 330, 300),
        (360, 180, 465, 400)
    ]

    for x1, y1, x2, y2 in buildings:

        draw.rectangle(
            [
                x1,
                y1,
                x2,
                ground_y
            ],
            fill="#566573"
        )

        for wx in range(
            x1 + 15,
            x2 - 5,
            25
        ):

            for wy in range(
                y1 + 20,
                ground_y - 20,
                35
            ):

                draw.rectangle(
                    [
                        wx,
                        wy,
                        wx + 10,
                        wy + 15
                    ],
                    fill="#F7D774"
                )


def draw_space(
    draw
):

    draw.rectangle(
        [
            0,
            0,
            WIDTH,
            HEIGHT
        ],
        fill="#050713"
    )

    draw_stars(
        draw
    )

    draw.ellipse(
        [
            120,
            250,
            360,
            490
        ],
        fill="#294E9B"
    )

    draw.ellipse(
        [
            145,
            275,
            335,
            460
        ],
        fill="#3C9A62"
    )


def draw_heaven(
    draw
):

    draw.rectangle(
        [
            0,
            0,
            WIDTH,
            HEIGHT
        ],
        fill="#F6E9B8"
    )

    draw.ellipse(
        [
            -100,
            250,
            250,
            700
        ],
        fill="#FFF8E1"
    )

    draw.ellipse(
        [
            220,
            200,
            650,
            720
        ],
        fill="#FFFFFF"
    )

    draw_sun(
        draw,
        390,
        120,
        55
    )


# ============================================================
# SCENE LOCATION DETECTION
# ============================================================

def detect_location(
    scene
):

    text = " ".join([
        str(scene.get("location", "")),
        str(scene.get("background", "")),
        str(scene.get("action", ""))
    ]).lower()

    if any(
        word in text
        for word in [
            "space",
            "galaxy",
            "planet",
            "cosmos",
            "universe"
        ]
    ):
        return "space"

    if any(
        word in text
        for word in [
            "heaven",
            "heavenly",
            "divine",
            "paradise"
        ]
    ):
        return "heaven"

    if any(
        word in text
        for word in [
            "ocean",
            "sea",
            "underwater"
        ]
    ):
        return "ocean"

    if any(
        word in text
        for word in [
            "beach",
            "coast",
            "shore"
        ]
    ):
        return "beach"

    if any(
        word in text
        for word in [
            "mountain",
            "hill",
            "highland"
        ]
    ):
        return "mountain"

    if any(
        word in text
        for word in [
            "city",
            "town",
            "downtown"
        ]
    ):
        return "city"

    if any(
        word in text
        for word in [
            "forest",
            "jungle",
            "woods"
        ]
    ):
        return "forest"

    if any(
        word in text
        for word in [
            "farm",
            "field",
            "plantation"
        ]
    ):
        return "farm"

    if any(
        word in text
        for word in [
            "house",
            "home",
            "room",
            "bedroom",
            "kitchen"
        ]
    ):
        return "house"

    if any(
        word in text
        for word in [
            "desert",
            "sand dunes"
        ]
    ):
        return "desert"

    if any(
        word in text
        for word in [
            "river",
            "lake",
            "waterfall"
        ]
    ):
        return "river"

    if any(
        word in text
        for word in [
            "market",
            "marketplace"
        ]
    ):
        return "market"

    return "village"


# ============================================================
# BACKGROUND RENDERER
# ============================================================

def draw_background(
    draw,
    scene,
    frame_progress=0.0
):

    location = detect_location(
        scene
    )

    time_of_day = str(
        scene.get(
            "time_of_day",
            ""
        )
    ).lower()

    night = any(
        word in time_of_day
        for word in [
            "night",
            "midnight",
            "evening"
        ]
    )

    # --------------------------------------------------------
    # SPACE
    # --------------------------------------------------------

    if location == "space":

        draw_space(
            draw
        )

        return

    # --------------------------------------------------------
    # HEAVEN
    # --------------------------------------------------------

    if location == "heaven":

        draw_heaven(
            draw
        )

        return

    # --------------------------------------------------------
    # NIGHT SKY
    # --------------------------------------------------------

    if night:

        sky = "#101B3A"

    else:

        sky = "#87CEEB"

    draw.rectangle(
        [
            0,
            0,
            WIDTH,
            560
        ],
        fill=sky
    )

    # --------------------------------------------------------
    # SUN / MOON
    # --------------------------------------------------------

    if night:

        draw.ellipse(
            [
                360,
                60,
                430,
                130
            ],
            fill="#FFF3B0"
        )

        draw_stars(
            draw
        )

    else:

        draw_sun(
            draw,
            400,
            100,
            42
        )

        draw_cloud(
            draw,
            120,
            130,
            0.8
        )

        draw_cloud(
            draw,
            320,
            200,
            0.6
        )

    # --------------------------------------------------------
    # OCEAN
    # --------------------------------------------------------

    if location == "ocean":

        draw.rectangle(
            [
                0,
                350,
                WIDTH,
                HEIGHT
            ],
            fill="#1676A6"
        )

        for i in range(7):

            y = 380 + i * 55

            offset = int(
                math.sin(
                    frame_progress * math.pi * 2
                    + i
                ) * 20
            )

            draw.arc(
                [
                    -80 + offset,
                    y,
                    WIDTH + 80 + offset,
                    y + 55
                ],
                180,
                360,
                fill="#B9EDF8",
                width=3
            )

        return

    # --------------------------------------------------------
    # BEACH
    # --------------------------------------------------------

    if location == "beach":

        draw.rectangle(
            [
                0,
                350,
                WIDTH,
                530
            ],
            fill="#2082B1"
        )

        draw.rectangle(
            [
                0,
                530,
                WIDTH,
                HEIGHT
            ],
            fill="#E5C17C"
        )

        draw_tree(
            draw,
            90,
            680,
            0.8
        )

        return

    # --------------------------------------------------------
    # CITY
    # --------------------------------------------------------

    if location == "city":

        draw.rectangle(
            [
                0,
                500,
                WIDTH,
                HEIGHT
            ],
            fill="#444444"
        )

        draw_city(
            draw,
            560
        )

        return

    # --------------------------------------------------------
    # MOUNTAIN
    # --------------------------------------------------------

    if location == "mountain":

        draw.rectangle(
            [
                0,
                450,
                WIDTH,
                HEIGHT
            ],
            fill="#789B57"
        )

        draw_mountain(
            draw,
            180,
            540,
            210,
            330
        )

        draw_mountain(
            draw,
            410,
            560,
            180,
            270
        )

        return

    # --------------------------------------------------------
    # DESERT
    # --------------------------------------------------------

    if location == "desert":

        draw.rectangle(
            [
                0,
                500,
                WIDTH,
                HEIGHT
            ],
            fill="#D7A75D"
        )

        draw.arc(
            [
                -100,
                470,
                500,
                850
            ],
            180,
            360,
            fill="#E8C27C",
            width=20
        )

        return

    # --------------------------------------------------------
    # FOREST / JUNGLE
    # --------------------------------------------------------

    if location == "forest":

        draw.rectangle(
            [
                0,
                500,
                WIDTH,
                HEIGHT
            ],
            fill="#426B36"
        )

        for x in [
            50,
            170,
            300,
            430
        ]:

            draw_tree(
                draw,
                x,
                650,
                0.8
            )

        return

    # --------------------------------------------------------
    # FARM
    # --------------------------------------------------------

    if location == "farm":

        draw.rectangle(
            [
                0,
                480,
                WIDTH,
                HEIGHT
            ],
            fill="#6C9E45"
        )

        for x in range(
            20,
            WIDTH,
            45
        ):

            draw.line(
                [
                    x,
                    520,
                    x - 15,
                    800
                ],
                fill="#3F762F",
                width=4
            )

        draw_house(
            draw,
            350,
            520,
            0.7
        )

        return

    # --------------------------------------------------------
    # HOUSE
    # --------------------------------------------------------

    if location == "house":

        draw.rectangle(
            [
                0,
                0,
                WIDTH,
                HEIGHT
            ],
            fill="#D9B382"
        )

        draw.rectangle(
            [
                0,
                540,
                WIDTH,
                HEIGHT
            ],
            fill="#A9784E"
        )

        draw_house(
            draw,
            250,
            540,
            0.8
        )

        return

    # --------------------------------------------------------
    # RIVER
    # --------------------------------------------------------

    if location == "river":

        draw.rectangle(
            [
                0,
                470,
                WIDTH,
                HEIGHT
            ],
            fill="#4F9CC2"
        )

        for i in range(6):

            y = 500 + i * 50

            draw.arc(
                [
                    -50,
                    y,
                    WIDTH + 50,
                    y + 30
                ],
                180,
                360,
                fill="#BDE8F5",
                width=2
            )

        draw_tree(
            draw,
            80,
            550,
            0.7
        )

        draw_tree(
            draw,
            420,
            550,
            0.7
        )

        return

    # --------------------------------------------------------
    # MARKET
    # --------------------------------------------------------

    if location == "market":

        draw.rectangle(
            [
                0,
                500,
                WIDTH,
                HEIGHT
            ],
            fill="#B87945"
        )

        for x in [
            50,
            180,
            310
        ]:

            draw.polygon(
                [
                    (x, 300),
                    (x + 120, 300),
                    (x + 100, 500),
                    (x + 20, 500)
                ],
                fill="#E4A64E"
            )

        return

    # --------------------------------------------------------
    # VILLAGE DEFAULT
    # --------------------------------------------------------

    draw.rectangle(
        [
            0,
            500,
            WIDTH,
            HEIGHT
        ],
        fill="#C99758"
    )

    draw_tree(
        draw,
        75,
        560,
        0.8
    )

    draw_house(
        draw,
        350,
        550,
        0.7
    )


# ============================================================
# CHARACTER COLORS
# ============================================================

CHARACTER_COLORS = [
    "#8B5A2B",
    "#6B3E26",
    "#704020",
    "#925C35",
    "#5A3825"
]


# ============================================================
# CHARACTER DRAWING
# ============================================================

def get_character_color(
    character_index
):

    return CHARACTER_COLORS[
        character_index
        %
        len(CHARACTER_COLORS)
    ]


def draw_character(
    draw,
    x,
    y,
    scale,
    name,
    emotion="neutral",
    talking=False,
    movement=0,
    character_index=0
):

    skin = get_character_color(
        character_index
    )

    head_r = int(
        55 * scale
    )

    # --------------------------------------------------------
    # BODY
    # --------------------------------------------------------

    body_color = [
        "#315C3A",
        "#6A3D7A",
        "#2E5E7E",
        "#8A4B2B"
    ][
        character_index % 4
    ]

    body_top = y + int(
        45 * scale
    )

    draw.ellipse(
        [
            x - int(60 * scale),
            body_top,
            x + int(60 * scale),
            y + int(190 * scale)
        ],
        fill=body_color
    )

    # --------------------------------------------------------
    # HEAD
    # --------------------------------------------------------

    draw.ellipse(
        [
            x - head_r,
            y - head_r,
            x + head_r,
            y + head_r
        ],
        fill=skin,
        outline="#111111",
        width=2
    )

    # --------------------------------------------------------
    # HAIR
    # --------------------------------------------------------

    draw.arc(
        [
            x - head_r,
            y - head_r,
            x + head_r,
            y + head_r
        ],
        180,
        360,
        fill="#151515",
        width=8
    )

    # --------------------------------------------------------
    # EYES
    # --------------------------------------------------------

    eye_y = y - 10

    for eye_x in [
        x - 20,
        x + 20
    ]:

        draw.ellipse(
            [
                eye_x - 9,
                eye_y - 8,
                eye_x + 9,
                eye_y + 8
            ],
            fill="white"
        )

        draw.ellipse(
            [
                eye_x - 4,
                eye_y - 4,
                eye_x + 4,
                eye_y + 4
            ],
            fill="black"
        )

    # --------------------------------------------------------
    # EYEBROWS
    # --------------------------------------------------------

    if emotion.lower() in [
        "angry",
        "furious",
        "grumpy",
        "worried"
    ]:

        draw.line(
            [
                x - 32,
                y - 30,
                x - 10,
                y - 22
            ],
            fill="black",
            width=4
        )

        draw.line(
            [
                x + 10,
                y - 22,
                x + 32,
                y - 30
            ],
            fill="black",
            width=4
        )

    # --------------------------------------------------------
    # MOUTH
    # --------------------------------------------------------

    mouth_y = y + 25

    if talking:

        draw.ellipse(
            [
                x - 17,
                mouth_y - 3,
                x + 17,
                mouth_y + 24
            ],
            fill="#280909"
        )

    elif emotion.lower() in [
        "happy",
        "excited",
        "laughing",
        "joyful"
    ]:

        draw.arc(
            [
                x - 22,
                mouth_y - 5,
                x + 22,
                mouth_y + 28
            ],
            0,
            180,
            fill="#280909",
            width=4
        )

    else:

        draw.arc(
            [
                x - 18,
                mouth_y,
                x + 18,
                mouth_y + 20
            ],
            0,
            180,
            fill="#280909",
            width=3
        )

    # --------------------------------------------------------
    # ARMS WITH MOVEMENT
    # --------------------------------------------------------

    arm_move = int(
        math.sin(
            movement * math.pi * 2
        )
        * 15
    )

    draw.line(
        [
            x - int(50 * scale),
            y + int(85 * scale),
            x - int(95 * scale),
            y + int(
                125 * scale
            ) + arm_move
        ],
        fill=body_color,
        width=max(
            6,
            int(10 * scale)
        )
    )

    draw.line(
        [
            x + int(50 * scale),
            y + int(85 * scale),
            x + int(95 * scale),
            y + int(
                125 * scale
            ) - arm_move
        ],
        fill=body_color,
        width=max(
            6,
            int(10 * scale)
        )


# ============================================================
# CHARACTER NAME INDEX
# ============================================================

def character_index_map(
    story
):

    result = {}

    for index, character in enumerate(
        story.get(
            "characters",
            []
        )
    ):

        name = str(
            character.get(
                "name",
                f"Character {index + 1}"
            )
        )

        result[name] = index

    return result


# ============================================================
# GET SCENE CHARACTER ACTION
# ============================================================

def get_scene_character_state(
    scene,
    dialogue_character=None
):

    states = {}

    for item in scene.get(
        "character_actions",
        []
    ):

        name = str(
            item.get(
                "character",
                ""
            )
        )

        if not name:
            continue

        states[name] = {
            "action": str(
                item.get(
                    "action",
                    ""
                )
            ),
            "emotion": str(
                item.get(
                    "emotion",
                    "neutral"
                )
            )
        }

    if dialogue_character:

        if dialogue_character not in states:

            states[
                dialogue_character
            ] = {
                "action": "speaking",
                "emotion": "talking"
            }

    return states


# ============================================================
# SCENE TEXT
# ============================================================

def draw_scene_title(
    draw,
    scene_number,
    location
):

    title = (
        f"SCENE {scene_number} • "
        f"{location}"
    )

    draw.rounded_rectangle(
        [
            12,
            15,
            WIDTH - 12,
            58
        ],
        radius=12,
        fill="#111827"
    )

    draw.text(
        (
            25,
            23
        ),
        title[:42],
        fill="white",
        font=font(
            18,
            True
        )
    )


# ============================================================
# TEXT WRAPPING
# ============================================================

def wrap_text(
    draw,
    text,
    font_obj,
    max_width
):

    words = str(
        text
    ).split()

    lines = []

    current = ""

    for word in words:

        test = (
            current
            + " "
            + word
        ).strip()

        bbox = draw.textbbox(
            (
                0,
                0
            ),
            test,
            font=font_obj
        )

        if bbox[2] <= max_width:

            current = test

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

    return lines


# ============================================================
# DIALOGUE BOX
# ============================================================

def draw_dialogue_box(
    draw,
    character,
    text
):

    box_top = 635

    draw.rounded_rectangle(
        [
            12,
            box_top,
            WIDTH - 12,
            840
        ],
        radius=18,
        fill="#111827"
    )

    draw.text(
        (
            28,
            box_top + 15
        ),
        str(character)[:25],
        fill="#FFD166",
        font=font(
            21,
            True
        )
    )

    lines = wrap_text(
        draw,
        text,
        font(
            20
        ),
        WIDTH - 55
    )

    y = box_top + 52

    for line in lines[:5]:

        draw.text(
            (
                28,
                y
            ),
            line,
            fill="white",
            font=font(
                20
            )
        )

        y += 28


# ============================================================
# VOICE
# ============================================================

def create_voice(
    text,
    filename,
    language="English"
):

    lang_map = {

        "English": "en",

        "Simple English": "en",

        "Nigerian Pidgin": "en",

        "Spanish": "es",

        "French": "fr",

        "Portuguese": "pt"
    }

    lang = lang_map.get(
        language,
        "en"
    )

    tts = gTTS(
        text=text,
        lang=lang,
        slow=False
    )

    tts.save(
        str(filename)
    )


# ============================================================
# AUDIO LENGTH
# ============================================================

def get_audio_duration(
    filename
):

    ffmpeg = get_ffmpeg()

    result = subprocess.run(
        [
            ffmpeg,
            "-i",
            str(filename)
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True
    )

    match = re.search(
        r"Duration:\s*(\d+):(\d+):([\d.]+)",
        result.stderr
    )

    if not match:
        return 2.0

    hours = int(
        match.group(1)
    )

    minutes = int(
        match.group(2)
    )

    seconds = float(
        match.group(3)
    )

    return (
        hours * 3600
        +
        minutes * 60
        +
        seconds
    )


# ============================================================
# BUILD DIALOGUE AUDIO
# ============================================================

def build_audio(
    job_id,
    story,
    scenes,
    work_dir,
    language
):

    audio_dir = (
        work_dir /
        "audio"
    )

    audio_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    dialogue_items = []

    total_dialogue = 0

    for scene_index, scene in enumerate(
        scenes
    ):

        for line in scene.get(
            "dialogue",
            []
        ):

            if total_dialogue >= MAX_DIALOGUE_LINES:
                break

            text = str(
                line.get(
                    "text",
                    line.get(
                        "line",
                        ""
                    )
                )
            ).strip()

            if not text:
                continue

            dialogue_items.append({
                "scene_index": scene_index,
                "character": str(
                    line.get(
                        "character",
                        "Narrator"
                    )
                ),
                "emotion": str(
                    line.get(
                        "emotion",
                        "neutral"
                    )
                ),
                "text": text
            })

            total_dialogue += 1

    if not dialogue_items:

        raise Exception(
            "The story contains no dialogue."
        )

    update_job(
        job_id,
        status="creating_audio",
        message="Creating voices...",
        progress=5
    )

    for index, item in enumerate(
        dialogue_items
    ):

        filename = (
            audio_dir /
            f"voice_{index}.mp3"
        )

        create_voice(
            item["text"],
            filename,
            language
        )

        item["audio"] = filename

        item["duration"] = get_audio_duration(
            filename
        )

        update_job(
            job_id,
            message=(
                f"Creating voices "
                f"({index + 1}/"
                f"{len(dialogue_items)})..."
            ),
            progress=5 + int(
                ((index + 1) /
                 len(dialogue_items))
                * 20
            )
        )

    # --------------------------------------------------------
    # CONCATENATE
    # --------------------------------------------------------

    update_job(
        job_id,
        status="building_audio",
        message="Combining voices...",
        progress=27
    )

    ffmpeg = get_ffmpeg()

    concat_file = (
        work_dir /
        "audio_concat.txt"
    )

    with open(
        concat_file,
        "w",
        encoding="utf-8"
    ) as f:

        for item in dialogue_items:

            path = str(
                item["audio"]
            ).replace(
                "'",
                "'\\''"
            )

            f.write(
                f"file '{path}'\n"
            )

    combined_audio = (
        work_dir /
        "dialogue.mp3"
    )

    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-vn",
            "-c:a",
            "libmp3lame",
            "-q:a",
            "6",
            str(combined_audio)
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Could not combine voice audio."
        )

    return (
        dialogue_items,
        combined_audio
    )


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
        OUTPUT_DIR /
        job_id
    )

    frames_dir = (
        work_dir /
        "frames"
    )

    work_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    frames_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    try:

        # ====================================================
        # SCENES
        # ====================================================

        scenes = story.get(
            "scenes",
            []
        )[:MAX_SCENES]

        if not scenes:

            raise Exception(
                "No scenes were generated."
            )

        # ====================================================
        # AUDIO
        # ====================================================

        dialogue_items, combined_audio = build_audio(
            job_id,
            story,
            scenes,
            work_dir,
            story.get(
                "language",
                "English"
            )
        )

        # ====================================================
        # CREATE SCENE TIMELINE
        # ====================================================

        target_duration = float(
            requested_duration
        )

        scene_timeline = []

        current_time = 0.0

        for scene in scenes:

            scene_duration = float(
                scene.get(
                    "duration",
                    1
                )
            )

            start = current_time

            end = (
                current_time
                +
                scene_duration
            )

            scene_timeline.append({
                "scene": scene,
                "start": start,
                "end": end
            })

            current_time = end

        # Normalize timeline exactly to requested duration

        if current_time > 0:

            factor = (
                target_duration
                /
                current_time
            )

            current_time = 0

            for item in scene_timeline:

                old_duration = (
                    item["end"]
                    -
                    item["start"]
                )

                new_duration = (
                    old_duration
                    *
                    factor
                )

                item["start"] = current_time

                item["end"] = (
                    current_time
                    +
                    new_duration
                )

                current_time = item["end"]

        # ====================================================
        # RENDER
        # ====================================================

        update_job(
            job_id,
            status="rendering",
            message="Rendering movie scenes...",
            progress=30
        )

        total_frames = max(
            1,
            int(
                target_duration
                *
                FPS
            )
        )

        character_map = (
            character_index_map(
                story
            )
        )

        frame_number = 0

        # ----------------------------------------------------
        # RENDER EACH FRAME
        # ----------------------------------------------------

        for frame_index in range(
            total_frames
        ):

            video_time = (
                frame_index
                /
                FPS
            )

            current_timeline = (
                scene_timeline[-1]
            )

            for item in scene_timeline:

                if (
                    item["start"]
                    <=
                    video_time
                    <
                    item["end"]
                ):

                    current_timeline = item
                    break

            scene = current_timeline[
                "scene"
            ]

            scene_start = current_timeline[
                "start"
            ]

            scene_end = current_timeline[
                "end"
            ]

            scene_duration = max(
                0.1,
                scene_end - scene_start
            )

            scene_progress = (
                video_time - scene_start
            ) / scene_duration

            scene_progress = max(
                0,
                min(
                    1,
                    scene_progress
                )
            )

            # =================================================
            # IMAGE
            # =================================================

            img = Image.new(
                "RGB",
                (
                    WIDTH,
                    HEIGHT
                ),
                "#111111"
            )

            draw = ImageDraw.Draw(
                img
            )

            # =================================================
            # BACKGROUND
            # =================================================

            draw_background(
                draw,
                scene,
                scene_progress
            )

            # =================================================
            # CAMERA EFFECT
            # =================================================

            camera = str(
                scene.get(
                    "camera_movement",
                    ""
                )
            ).lower()

            zoom = 1.0

            if "zoom in" in camera:

                zoom = (
                    1.0
                    +
                    0.08
                    *
                    scene_progress
                )

            elif "zoom out" in camera:

                zoom = (
                    1.08
                    -
                    0.08
                    *
                    scene_progress
                )

            # Camera pan is represented by character
            # position movement.

            pan_offset = 0

            if "pan left" in camera:

                pan_offset = int(
                    -35
                    *
                    scene_progress
                )

            elif "pan right" in camera:

                pan_offset = int(
                    35
                    *
                    scene_progress
                )

            # =================================================
            # SCENE TITLE
            # =================================================

            draw_scene_title(
                draw,
                scene.get(
                    "scene",
                    1
                ),
                scene.get(
                    "location",
                    "Scene"
                )
            )

            # =================================================
            # CHARACTER STATES
            # =================================================

            states = get_scene_character_state(
                scene
            )

            present_characters = (
                scene.get(
                    "characters_present",
                    []
                )
            )

            # If Gemini did not explicitly provide characters,
            # use the character actions.

            if not present_characters:

                present_characters = list(
                    states.keys()
                )

            # If still empty, use story characters.

            if not present_characters:

                present_characters = [
                    c.get(
                        "name",
                        f"Character {i + 1}"
                    )
                    for i, c in enumerate(
                        story.get(
                            "characters",
                            []
                        )[:3]
                    )
                ]

            present_characters = (
                present_characters[:3]
            )

            # =================================================
            # DIALOGUE FOR THIS SCENE
            # =================================================

            scene_dialogue = [
                item
                for item in dialogue_items
                if item["scene_index"]
                ==
                scene.get(
                    "scene",
                    1
                ) - 1
            ]

            active_dialogue = None

            if scene_dialogue:

                # Find which dialogue line should be active
                # based on scene progress.

                dialogue_total = sum(
                    max(
                        0.5,
                        float(
                            item.get(
                                "duration",
                                1
                            )
                        )
                    )
                    for item in scene_dialogue
                )

                dialogue_cursor = (
                    scene_progress
                    *
                    dialogue_total
                )

                running = 0

                for item in scene_dialogue:

                    running += max(
                        0.5,
                        float(
                            item.get(
                                "duration",
                                1
                            )
                        )
                    )

                    if dialogue_cursor <= running:

                        active_dialogue = item
                        break

                if not active_dialogue:

                    active_dialogue = (
                        scene_dialogue[-1]
                    )

            # =================================================
            # DRAW CHARACTERS
            # =================================================

            count = len(
                present_characters
            )

            positions = []

            if count == 1:

                positions = [
                    WIDTH // 2
                ]

            elif count == 2:

                positions = [
                    145,
                    335
                ]

            else:

                positions = [
                    100,
                    240,
                    380
                ]

            for index, name in enumerate(
                present_characters
            ):

                state = states.get(
                    name,
                    {}
                )

                emotion = state.get(
                    "emotion",
                    "neutral"
                )

                action = state.get(
                    "action",
                    ""
                ).lower()

                is_talking = (
                    active_dialogue
                    is not None
                    and
                    active_dialogue[
                        "character"
                    ]
                    ==
                    name
                )

                # Walking / running movement

                movement_amount = (
                    math.sin(
                        video_time
                        *
                        5
                    )
                )

                y_offset = 0

                if (
                    "jump" in action
                    or
                    "jumping" in action
                ):

                    y_offset = int(
                        abs(
                            math.sin(
                                video_time
                                *
                                4
                            )
                        )
                        *
                        -30
                    )

                elif (
                    "run" in action
                    or
                    "running" in action
                ):

                    y_offset = int(
                        movement_amount
                        *
                        8
                    )

                x = (
                    positions[index]
                    +
                    pan_offset
                )

                # Small cinematic scale variation

                scale = (
                    0.78
                    *
                    zoom
                )

                draw_character(
                    draw,
                    x,
                    535 + y_offset,
                    scale,
                    name,
                    emotion,
                    is_talking,
                    video_time,
                    character_map.get(
                        name,
                        index
                    )
                )

            # =================================================
            # ACTION TEXT FOR VISUAL CONTEXT
            # =================================================

            action_text = str(
                scene.get(
                    "action",
                    ""
                )
            )

            if action_text:

                short_action = (
                    action_text[:100]
                )

                draw.rounded_rectangle(
                    [
                        18,
                        570,
                        WIDTH - 18,
                        615
                    ],
                    radius=10,
                    fill="#000000"
                )

                draw.text(
                    (
                        28,
                        582
                    ),
                    short_action,
                    fill="#FFFFFF",
                    font=font(
                        15
                    )
                )

            # =================================================
            # DIALOGUE
            # =================================================

            if active_dialogue:

                draw_dialogue_box(
                    draw,
                    active_dialogue[
                        "character"
                    ],
                    active_dialogue[
                        "text"
                    ]
                )

            else:

                # Narration/action caption when there
                # is no dialogue.

                caption = str(
                    scene.get(
                        "action",
                        ""
                    )
                )

                if caption:

                    draw.rounded_rectangle(
                        [
                            18,
                            665,
                            WIDTH - 18,
                            810
                        ],
                        radius=15,
                        fill="#111827"
                    )

                    lines = wrap_text(
                        draw,
                        caption,
                        font(
                            18
                        ),
                        WIDTH - 55
                    )

                    y = 690

                    for line in lines[:4]:

                        draw.text(
                            (
                                28,
                                y
                            ),
                            line,
                            fill="white",
                            font=font(
                                18
                            )
                        )

                        y += 27

            # =================================================
            # SAVE FRAME
            # =================================================

            frame_path = (
                frames_dir /
                f"frame_{frame_number:06d}.jpg"
            )

            img.save(
                frame_path,
                "JPEG",
                quality=72
            )

            frame_number += 1

            if frame_index % FPS == 0:

                progress = (
                    30
                    +
                    int(
                        (
                            frame_index
                            /
                            total_frames
                        )
                        *
                        55
                    )
                )

                update_job(
                    job_id,
                    message=(
                        "Rendering movie "
                        f"({frame_index + 1}/"
                        f"{total_frames})..."
                    ),
                    progress=progress
                )

        # ====================================================
        # CREATE MP4
        # ====================================================

        update_job(
            job_id,
            status="creating_mp4",
            message="Encoding final movie...",
            progress=88
        )

        ffmpeg = get_ffmpeg()

        output_name = (
            safe_filename(
                story.get(
                    "title",
                    "afritoon"
                )
            )
            + "_"
            + job_id[:8]
            + ".mp4"
        )

        output_file = (
            OUTPUT_DIR /
            output_name
        )

        video_input = (
            frames_dir /
            "frame_%06d.jpg"
        )

        result = subprocess.run(
            [
                ffmpeg,
                "-y",

                "-framerate",
                str(FPS),

                "-i",
                str(video_input),

                "-i",
                str(combined_audio),

                "-t",
                str(target_duration),

                "-c:v",
                "libx264",

                "-preset",
                "ultrafast",

                "-crf",
                "27",

                "-pix_fmt",
                "yuv420p",

                "-c:a",
                "aac",

                "-b:a",
                "128k",

                "-shortest",

                "-movflags",
                "+faststart",

                str(output_file)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:

            print(
                "FFMPEG ERROR:",
                result.stderr
            )

            raise RuntimeError(
                "FFmpeg failed to create the MP4."
            )

        # ====================================================
        # VERIFY
        # ====================================================

        if not output_file.exists():

            raise RuntimeError(
                "MP4 was not created."
            )

        file_size = (
            output_file.stat().st_size
        )

        if file_size <= 0:

            raise RuntimeError(
                "MP4 file is empty."
            )

        # ====================================================
        # URLS
        # ====================================================

        video_url = build_video_url(
            backend_url,
            output_file.name
        )

        download_url = build_download_url(
            backend_url,
            output_file.name
        )

        # ====================================================
        # COMPLETE
        # ====================================================

        update_job(
            job_id,

            status="complete",

            message=(
                "Movie created successfully."
            ),

            progress=100,

            video_url=video_url,

            download_url=download_url,

            filename=output_file.name,

            file_size=file_size
        )

        print(
            "===================================="
        )

        print(
            "VIDEO CREATED"
        )

        print(
            f"FILE: {output_file}"
        )

        print(
            f"SIZE: {file_size} bytes"
        )

        print(
            f"VIDEO URL: {video_url}"
        )

        print(
            f"DOWNLOAD URL: {download_url}"
        )

        print(
            "===================================="
        )

        # ====================================================
        # CLEAN TEMPORARY FILES
        # ====================================================

        try:

            shutil.rmtree(
                work_dir
            )

        except Exception as cleanup_error:

            print(
                "Cleanup warning:",
                cleanup_error
            )

    except Exception as e:

        print(
            f"VIDEO ERROR [{job_id}]:",
            repr(e)
        )

        update_job(
            job_id,

            status="error",

            message=str(e),

            progress=0
        )

        try:

            shutil.rmtree(
                work_dir
            )

        except Exception:
            pass


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return jsonify({

        "status": "online",

        "app": "AfriToon Studio",

        "message":
            "Cinematic cartoon backend is running",

        "video_storage":
            str(OUTPUT_DIR)

    })


# ============================================================
# HEALTH
# ============================================================

@app.route("/health")
def health():

    return jsonify({

        "status": "healthy",

        "ffmpeg":
            bool(
                shutil.which(
                    "ffmpeg"
                )
            ),

        "gemini":
            bool(
                API_KEY
            ),

        "base_url":
            BASE_URL or "automatic"

    })


# ============================================================
# GENERATE STORY
# ============================================================

@app.route(
    "/api/generate",
    methods=["POST"]
)
def api_generate():

    try:

        data = (
            request.get_json()
            or {}
        )

        category = str(
            data.get(
                "category",
                "surprise"
            )
        )

        topic = str(
            data.get(
                "topic",
                ""
            )
        )

        duration = int(
            data.get(
                "duration",
                30
            )
        )

        language = str(
            data.get(
                "language",
                "English"
            )
        )

        if duration not in [
            30,
            60
        ]:

            return jsonify({

                "success": False,

                "error":
                    "Duration must be 30 or 60 seconds."

            }), 400

        story = generate_story(
            category,
            topic,
            duration,
            language
        )

        return jsonify({

            "success": True,

            "story": story

        })

    except Exception as e:

        print(
            "STORY ERROR:",
            repr(e)
        )

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# CREATE VIDEO
# ============================================================

@app.route(
    "/api/create-video",
    methods=["POST"]
)
def api_create_video():

    try:

        data = (
            request.get_json()
            or {}
        )

        story = data.get(
            "story"
        )

        duration = int(
            data.get(
                "duration",
                30
            )
        )

        if not story:

            return jsonify({

                "success": False,

                "error":
                    "Story is required."

            }), 400

        if duration not in [
            30,
            60
        ]:

            return jsonify({

                "success": False,

                "error":
                    "Duration must be 30 or 60 seconds."

            }), 400

        job_id = uuid.uuid4().hex

        backend_url = (
            BASE_URL
            or request.host_url.rstrip("/")
        )

        with JOB_LOCK:

            JOBS[job_id] = {

                "status":
                    "starting",

                "message":
                    "Starting movie generation...",

                "progress":
                    0,

                "video_url":
                    None,

                "download_url":
                    None,

                "filename":
                    None,

                "file_size":
                    None
            }

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

            "job_id":
                job_id,

            "status":
                "starting"

        })

    except Exception as e:

        print(
            "VIDEO START ERROR:",
            repr(e)
        )

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# VIDEO STATUS
# ============================================================

@app.route(
    "/api/video-status/<job_id>",
    methods=["GET"]
)
def video_status(job_id):

    job = get_job(
        job_id
    )

    if not job:

        return jsonify({

            "success": False,

            "error":
                "Job not found."

        }), 404

    return jsonify({

        "success": True,

        "status":
            job.get(
                "status"
            ),

        "message":
            job.get(
                "message"
            ),

        "progress":
            job.get(
                "progress",
                0
            ),

        "video_url":
            job.get(
                "video_url"
            ),

        "download_url":
            job.get(
                "download_url"
            ),

        "filename":
            job.get(
                "filename"
            ),

        "file_size":
            job.get(
                "file_size"
            ),

        "job":
            job

    })


# ============================================================
# VIDEO PREVIEW
# ============================================================

@app.route(
    "/generated/<path:filename>",
    methods=["GET"]
)
def generated_file(filename):

    file_path = (
        OUTPUT_DIR /
        filename
    )

    if not file_path.is_file():

        return jsonify({

            "success": False,

            "error":
                "Video file not found."

        }), 404

    return send_from_directory(

        OUTPUT_DIR,

        filename,

        as_attachment=False,

        mimetype="video/mp4",

        max_age=0
    )


# ============================================================
# VIDEO DOWNLOAD
# ============================================================

@app.route(
    "/download/<path:filename>",
    methods=["GET"]
)
def download_file(filename):

    file_path = (
        OUTPUT_DIR /
        filename
    )

    if not file_path.is_file():

        return jsonify({

            "success": False,

            "error":
                "Video file not found."

        }), 404

    return send_from_directory(

        OUTPUT_DIR,

        filename,

        as_attachment=True,

        download_name=Path(
            filename
        ).name,

        mimetype="video/mp4",

        max_age=0
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    print(
        "======================================"
    )

    print(
        "AFRITOON STUDIO"
    )

    print(
        "Cinematic Scene Renderer"
    )

    print(
        f"Port: {port}"
    )

    print(
        f"Base URL: "
        f"{BASE_URL or 'AUTOMATIC'}"
    )

    print(
        f"FFmpeg: "
        f"{shutil.which('ffmpeg')}"
    )

    print(
        f"Gemini: "
        f"{'YES' if API_KEY else 'NO'}"
    )

    print(
        "======================================"
    )

    app.run(
        host="0.0.0.0",
        port=port,
        threaded=True
    )
