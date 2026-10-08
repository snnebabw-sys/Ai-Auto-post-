import os
import json
import uuid
import re
import subprocess
import threading
import shutil
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


# IMPORTANT:
# Set BASE_URL on Render to your actual Render backend URL.
#
# Example:
# BASE_URL=https://afritoon-backend.onrender.com
#
# If BASE_URL is not set, the code will still create
# relative URLs as a fallback.

BASE_URL = os.getenv(
    "BASE_URL",
    ""
).rstrip("/")


# ============================================================
# SETTINGS
# ============================================================

WIDTH = 480
HEIGHT = 854

FPS = 8

MAX_DIALOGUE_LINES = 14

MAX_SCENES = 6


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
# BUILD PUBLIC VIDEO URL
# ============================================================

def public_video_url(filename):

    path = (
        f"/generated/"
        f"{filename}"
    )

    if BASE_URL:

        return (
            f"{BASE_URL}"
            f"{path}"
        )

    return path


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
# FONT CACHE
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

    candidates = []

    if bold:

        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        ])

    else:

        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"
        ])

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

        if (
            lines
            and
            lines[0].startswith("```")
        ):

            lines = lines[1:]

        if (
            lines
            and
            lines[-1].strip() == "```"
        ):

            lines = lines[:-1]

        text = "\n".join(
            lines
        )

    return text.strip()


# ============================================================
# SAFE FILE NAME
# ============================================================

def safe_filename(name):

    cleaned = "".join(
        c
        if c.isalnum()
        or c in "-_"
        else "_"
        for c in str(name)
    )

    return (
        cleaned[:70]
        or
        "afritoon"
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

    prompt = f"""
Create an ORIGINAL short African cartoon episode.

Category:
{category}

Topic:
{topic or "Create a funny original African story."}

Language:
{language}

Target duration:
{duration} seconds.

IMPORTANT:
Keep the story short enough for a simple animated video.

Use exactly 2 or 3 main characters.

Use 3 to 5 scenes.

Use a maximum of 10 dialogue lines.

Each dialogue line should normally be short.

The story must contain:

- beginning
- middle
- ending
- funny or entertaining moments
- clear visual actions
- facial emotions
- body movements
- scene changes

Do not copy existing cartoons,
movies, characters,
or copyrighted stories.

Return ONLY valid JSON.

Use exactly this structure:

{{
  "title": "Story title",
  "description": "Short description",

  "characters": [
    {{
      "name": "Character name",
      "personality": "Personality",
      "voice": "Voice description"
    }}
  ],

  "scenes": [
    {{
      "scene": 1,
      "location": "Location",
      "action": "What is happening",

      "dialogue": [
        {{
          "character": "Character name",
          "text": "Short dialogue",
          "emotion": "emotion"
        }}
      ]
    }}
  ]
}}
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

    return story


# ============================================================
# CHARACTER DRAWING
# ============================================================

def draw_character(
    draw,
    x,
    y,
    scale,
    name,
    emotion,
    talking=False
):

    colors = {

        "Kofi": "#8B5A2B",

        "Amina": "#6B3E26",

        "Nana": "#7A4A2B",

        "Tunde": "#704020"
    }

    skin = colors.get(
        name,
        "#8B5A2B"
    )

    head_r = int(
        58 * scale
    )

    # BODY

    draw.ellipse(
        [
            x - int(60 * scale),
            y + int(45 * scale),
            x + int(60 * scale),
            y + int(190 * scale)
        ],
        fill="#315C3A"
    )

    # HEAD

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

    # HAIR

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
        width=7
    )

    # EYES

    eye_y = y - 12

    draw.ellipse(
        [
            x - 27,
            eye_y - 8,
            x - 12,
            eye_y + 8
        ],
        fill="white"
    )

    draw.ellipse(
        [
            x + 12,
            eye_y - 8,
            x + 27,
            eye_y + 8
        ],
        fill="white"
    )

    draw.ellipse(
        [
            x - 22,
            eye_y - 4,
            x - 16,
            eye_y + 4
        ],
        fill="black"
    )

    draw.ellipse(
        [
            x + 16,
            eye_y - 4,
            x + 22,
            eye_y + 4
        ],
        fill="black"
    )

    # EMOTION

    if emotion in [
        "angry",
        "grumpy",
        "bossy"
    ]:

        draw.line(
            [
                x - 30,
                y - 30,
                x - 12,
                y - 24
            ],
            fill="black",
            width=3
        )

        draw.line(
            [
                x + 12,
                y - 24,
                x + 30,
                y - 30
            ],
            fill="black",
            width=3
        )

    # MOUTH

    mouth_y = y + 23

    if talking:

        draw.ellipse(
            [
                x - 17,
                mouth_y - 5,
                x + 17,
                mouth_y + 22
            ],
            fill="#250909"
        )

    else:

        draw.arc(
            [
                x - 20,
                mouth_y - 3,
                x + 20,
                mouth_y + 20
            ],
            0,
            180,
            fill="#250909",
            width=3
        )

    # ARMS

    movement = (
        8
        if talking
        else 0
    )

    draw.line(
        [
            x - 55,
            y + 80,
            x - 100,
            y + 130 + movement
        ],
        fill="#315C3A",
        width=10
    )

    draw.line(
        [
            x + 55,
            y + 80,
            x + 100,
            y + 130 - movement
        ],
        fill="#315C3A",
        width=10
    )


# ============================================================
# CALABASH
# ============================================================

def draw_calabash(
    draw,
    x,
    y,
    scale,
    talking=False
):

    width = int(
        90 * scale
    )

    height = int(
        75 * scale
    )

    draw.ellipse(
        [
            x - width,
            y - height,
            x + width,
            y + height
        ],
        fill="#B87333",
        outline="#4A2A10",
        width=3
    )

    # OPENING

    draw.ellipse(
        [
            x - 42,
            y - 20,
            x + 42,
            y + 4
        ],
        fill="#321B0B"
    )

    # EYES

    draw.ellipse(
        [
            x - 32,
            y + 12,
            x - 15,
            y + 29
        ],
        fill="white"
    )

    draw.ellipse(
        [
            x + 15,
            y + 12,
            x + 32,
            y + 29
        ],
        fill="white"
    )

    # MOUTH

    if talking:

        draw.ellipse(
            [
                x - 20,
                y + 36,
                x + 20,
                y + 60
            ],
            fill="#160909"
        )

    else:

        draw.arc(
            [
                x - 20,
                y + 30,
                x + 20,
                y + 55
            ],
            0,
            180,
            fill="#160909",
            width=3
        )


# ============================================================
# BACKGROUND
# ============================================================

def draw_background(
    draw,
    scene_number
):

    # SKY

    draw.rectangle(
        [0, 0, WIDTH, 560],
        fill="#87CEEB"
    )

    # GROUND

    draw.rectangle(
        [0, 560, WIDTH, HEIGHT],
        fill="#D19A5A"
    )

    # SUN

    draw.ellipse(
        [370, 55, 435, 120],
        fill="#FFD54A"
    )

    # TREE TRUNK

    draw.rectangle(
        [55, 280, 85, 560],
        fill="#5C3A21"
    )

    # TREE CROWN

    draw.ellipse(
        [10, 190, 130, 350],
        fill="#3F7D3A"
    )

    # HOUSE

    draw.polygon(
        [
            (290, 390),
            (420, 285),
            (480, 390)
        ],
        fill="#9A5A32"
    )

    draw.rectangle(
        [305, 390, 470, 555],
        fill="#C77D45"
    )

    # SCENE LABEL

    draw.text(
        (20, 20),
        f"Scene {scene_number}",
        fill="black",
        font=font(
            23,
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
            (0, 0),
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
# AUDIO
# ============================================================

def create_voice(
    text,
    filename,
    language="English"
):

    # gTTS language mapping

    lang_map = {

        "English": "en",

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

    command = [
        ffmpeg,
        "-i",
        str(filename)
    ]

    result = subprocess.run(
        command,
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
        + minutes * 60
        + seconds
    )


# ============================================================
# CREATE VIDEO
# ============================================================

def create_video(
    job_id,
    story,
    requested_duration
):

    work_dir = (
        OUTPUT_DIR /
        job_id
    )

    audio_dir = (
        work_dir /
        "audio"
    )

    frames_dir = (
        work_dir /
        "frames"
    )

    work_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    audio_dir.mkdir(
        exist_ok=True
    )

    frames_dir.mkdir(
        exist_ok=True
    )

    try:

        # ====================================================
        # COLLECT DIALOGUE
        # ====================================================

        dialogue_items = []

        scenes = story.get(
            "scenes",
            []
        )[:MAX_SCENES]

        for scene_index, scene in enumerate(
            scenes
        ):

            for line in scene.get(
                "dialogue",
                []
            ):

                if len(dialogue_items) >= MAX_DIALOGUE_LINES:

                    break

                text = str(
                    line.get(
                        "text",
                        ""
                    )
                ).strip()

                if not text:

                    continue

                dialogue_items.append({

                    "scene": scene_index,

                    "character":
                        line.get(
                            "character",
                            "Character"
                        ),

                    "emotion":
                        line.get(
                            "emotion",
                            "neutral"
                        ),

                    "text": text
                })

        if not dialogue_items:

            raise Exception(
                "The story contains no dialogue."
            )

        # ====================================================
        # CREATE AUDIO
        # ====================================================

        update_job(
            job_id,
            status="creating_audio",
            message="Creating voices...",
            progress=5
        )

        audio_files = []

        total_lines = len(
            dialogue_items
        )

        story_language = story.get(
            "language",
            "English"
        )

        for index, item in enumerate(
            dialogue_items
        ):

            audio_file = (
                audio_dir /
                f"voice_{index}.mp3"
            )

            create_voice(
                item["text"],
                audio_file,
                story_language
            )

            duration = get_audio_duration(
                audio_file
            )

            item["audio"] = audio_file

            item["duration"] = duration

            audio_files.append(
                audio_file
            )

            progress = (
                5
                +
                int(
                    (
                        (index + 1)
                        /
                        total_lines
                    )
                    * 20
                )
            )

            update_job(
                job_id,
                message=(
                    f"Creating voices "
                    f"({index + 1}/"
                    f"{total_lines})..."
                ),
                progress=progress
            )

        # ====================================================
        # COMBINE AUDIO
        # ====================================================

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

            for audio_file in audio_files:

                safe_path = str(
                    audio_file
                ).replace(
                    "'",
                    "'\\''"
                )

                f.write(
                    f"file '{safe_path}'\n"
                )

        combined_audio = (
            work_dir /
            "dialogue.mp3"
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
                str(
                    concat_file
                ),

                "-vn",

                "-c:a",
                "libmp3lame",

                "-q:a",
                "6",

                str(
                    combined_audio
                )
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )

        # ====================================================
        # TARGET DURATION
        # ====================================================

        total_audio_duration = sum(
            item["duration"]
            for item in dialogue_items
        )

        target_duration = min(
            float(
                requested_duration
            ),
            max(
                8.0,
                total_audio_duration
            )
        )

        # ====================================================
        # RENDER FRAMES
        # ====================================================

        update_job(
            job_id,
            status="rendering",
            message="Rendering animation...",
            progress=30
        )

        total_frames = int(
            target_duration
            * FPS
        )

        total_frames = max(
            1,
            total_frames
        )

        frame_number = 0

        # ----------------------------------------------------
        # IMPORTANT:
        # Calculate cumulative dialogue times.
        # This fixes dialogue timing.
        # ----------------------------------------------------

        current_dialogue_index = 0

        current_item = dialogue_items[0]

        elapsed_in_item = 0.0

        frame_time = (
            1.0 / FPS
        )

        # ====================================================
        # FRAME LOOP
        # ====================================================

        for frame_index in range(
            total_frames
        ):

            # Move to next dialogue item
            # when current audio finishes.

            while (
                current_dialogue_index
                <
                len(dialogue_items) - 1
                and
                elapsed_in_item
                >=
                current_item["duration"]
            ):

                elapsed_in_item -= (
                    current_item[
                        "duration"
                    ]
                )

                current_dialogue_index += 1

                current_item = (
                    dialogue_items[
                        current_dialogue_index
                    ]
                )

            # IMAGE

            img = Image.new(
                "RGB",
                (
                    WIDTH,
                    HEIGHT
                ),
                "#F5DFA5"
            )

            draw = ImageDraw.Draw(
                img
            )

            # BACKGROUND

            draw_background(
                draw,
                current_item[
                    "scene"
                ] + 1
            )

            # TALKING

            talking = (
                frame_index % 8
            ) < 5

            character = (
                current_item[
                    "character"
                ]
            )

            emotion = (
                current_item[
                    "emotion"
                ]
            )

            # =================================================
            # CHARACTERS
            # =================================================

            if character == "Kofi":

                draw_character(
                    draw,
                    160,
                    530,
                    0.82,
                    "Kofi",
                    emotion,
                    talking
                )

                draw_character(
                    draw,
                    350,
                    530,
                    0.82,
                    "Amina",
                    "neutral",
                    False
                )

            elif character == "Amina":

                draw_character(
                    draw,
                    160,
                    530,
                    0.82,
                    "Kofi",
                    "neutral",
                    False
                )

                draw_character(
                    draw,
                    350,
                    530,
                    0.82,
                    "Amina",
                    emotion,
                    talking
                )

            elif character == "Calabash":

                draw_character(
                    draw,
                    150,
                    530,
                    0.78,
                    "Kofi",
                    "neutral",
                    False
                )

                draw_calabash(
                    draw,
                    330,
                    540,
                    0.8,
                    talking
                )

            else:

                draw_character(
                    draw,
                    160,
                    530,
                    0.82,
                    "Kofi",
                    "neutral",
                    False
                )

                draw_character(
                    draw,
                    350,
                    530,
                    0.82,
                    "Amina",
                    emotion,
                    talking
                )

            # =================================================
            # DIALOGUE BOX
            # =================================================

            box_top = 620

            draw.rounded_rectangle(
                [
                    15,
                    box_top,
                    WIDTH - 15,
                    825
                ],
                radius=18,
                fill="#111827"
            )

            speaker_font = font(
                22,
                True
            )

            text_font = font(
                22,
                False
            )

            draw.text(
                (
                    32,
                    box_top + 18
                ),
                str(
                    character
                ),
                fill="#FFD166",
                font=speaker_font
            )

            lines = wrap_text(
                draw,
                current_item[
                    "text"
                ],
                text_font,
                WIDTH - 60
            )

            y = (
                box_top
                + 55
            )

            for line in lines[:4]:

                draw.text(
                    (
                        32,
                        y
                    ),
                    line,
                    fill="white",
                    font=text_font
                )

                y += 30

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
                quality=70,
                optimize=False
            )

            frame_number += 1

            elapsed_in_item += (
                frame_time
            )

            # =================================================
            # PROGRESS
            # =================================================

            if (
                frame_index % FPS == 0
            ):

                render_progress = (
                    30
                    +
                    int(
                        (
                            (frame_index + 1)
                            /
                            total_frames
                        )
                        * 55
                    )
                )

                update_job(
                    job_id,
                    message=(
                        "Rendering animation "
                        f"({frame_index + 1}/"
                        f"{total_frames})..."
                    ),
                    progress=render_progress
                )

        # ====================================================
        # CREATE MP4
        # ====================================================

        update_job(
            job_id,
            status="creating_mp4",
            message="Encoding final video...",
            progress=88
        )

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

        # ====================================================
        # FFmpeg
        # ====================================================

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
        # VERIFY FILE
        # ====================================================

        if not output_file.exists():

            raise RuntimeError(
                "MP4 was not created."
            )

        if output_file.stat().st_size <= 0:

            raise RuntimeError(
                "MP4 file is empty."
            )

        # ====================================================
        # PUBLIC URL
        # ====================================================

        video_url = public_video_url(
            output_file.name
        )

        # ====================================================
        # COMPLETE
        # ====================================================

        update_job(
            job_id,

            status="complete",

            message=(
                "Video created successfully."
            ),

            progress=100,

            video_url=video_url,

            download_url=video_url,

            filename=output_file.name,

            file_size=output_file.stat().st_size
        )

        print(
            "VIDEO CREATED:",
            str(output_file)
        )

        print(
            "VIDEO URL:",
            video_url
        )

        # ====================================================
        # REMOVE TEMPORARY FILES
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
            "Cartoon backend is running",

        "video_storage":
            str(OUTPUT_DIR)
    })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({

        "status": "healthy",

        "ffmpeg":
            bool(
                shutil.which("ffmpeg")
            ),

        "gemini":
            bool(
                API_KEY
            )
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

        category = data.get(
            "category",
            "surprise"
        )

        topic = data.get(
            "topic",
            ""
        )

        duration = int(
            data.get(
                "duration",
                30
            )
        )

        language = data.get(
            "language",
            "English"
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

        # Store language inside story
        # so the video creator can use it.

        story["language"] = language

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
# START VIDEO JOB
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

        with JOB_LOCK:

            JOBS[job_id] = {

                "status":
                    "starting",

                "message":
                    "Starting video generation...",

                "progress":
                    0,

                "video_url":
                    None,

                "download_url":
                    None
            }

        thread = threading.Thread(

            target=create_video,

            args=(
                job_id,
                story,
                duration
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
    "/api/video-status/<job_id>"
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

        "job":
            job
    })


# ============================================================
# GENERATED MP4 FILES
# ============================================================

@app.route(
    "/generated/<path:filename>"
)
def generated_file(filename):

    file_path = (
        OUTPUT_DIR /
        filename
    )

    if not file_path.exists():

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
        "===================================="
    )

    print(
        "AfriToon Studio Backend"
    )

    print(
        f"Port: {port}"
    )

    print(
        f"Base URL: {BASE_URL or 'NOT SET'}"
    )

    print(
        f"FFmpeg: {shutil.which('ffmpeg')}"
    )

    print(
        f"Gemini: {'YES' if API_KEY else 'NO'}"
    )

    print(
        "===================================="
    )

    app.run(
        host="0.0.0.0",
        port=port,
        threaded=True
    )
