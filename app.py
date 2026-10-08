import os
import json
import uuid
import time
import html
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
import imageio_ffmpeg


# ============================================================
# APP
# ============================================================

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "generated"
OUTPUT_DIR.mkdir(exist_ok=True)

JOBS = {}

API_KEY = os.getenv("GEMINI_API_KEY")

if API_KEY:
    client = genai.Client(api_key=API_KEY)
else:
    client = None


# ============================================================
# HELPERS
# ============================================================

def get_ffmpeg():
    return imageio_ffmpeg.get_ffmpeg_exe()


def font(size=40, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"
    ]

    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)

    return ImageFont.load_default()


def clean_json(text):
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines)

    return text.strip()


def safe_filename(name):
    return "".join(
        c if c.isalnum() or c in "-_" else "_"
        for c in name
    )[:80]


# ============================================================
# STORY GENERATION
# ============================================================

def generate_story(category, topic, duration, language):

    if not client:
        raise Exception("GEMINI_API_KEY is missing.")

    prompt = f"""
Create an ORIGINAL short African cartoon episode.

Category:
{category}

Topic:
{topic or "Surprise me with a funny original African story."}

Language:
{language}

Target duration:
{duration} seconds.

Create a story suitable for a simple 2D talking cartoon.

Use 2 to 4 characters.

The story must contain:
- interesting characters
- dialogue
- emotions
- funny or entertaining moments
- clear scene changes
- visual actions
- facial reactions
- body movements
- a beginning
- a middle
- an ending

Do not copy existing cartoons, movies, characters or copyrighted stories.

Return ONLY valid JSON in this exact structure:

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
      "location": "Location description",
      "action": "What the characters are doing",
      "dialogue": [
        {{
          "character": "Character name",
          "text": "Dialogue",
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

    text = clean_json(response.text)

    return json.loads(text)


# ============================================================
# DRAWING
# ============================================================

def draw_character(draw, x, y, scale, name, emotion, talking=False):
    """
    Simple 2D cartoon character.

    Each character gets:
    - head
    - body
    - eyes
    - mouth
    - arms

    Mouth changes when talking.
    """

    colors = {
        "Kofi": "#8B5A2B",
        "Amina": "#6B3E26",
        "Calabash": "#C58B35"
    }

    skin = colors.get(name, "#8B5A2B")

    head_r = int(75 * scale)

    # body
    draw.ellipse(
        [
            x - int(80 * scale),
            y + int(60 * scale),
            x + int(80 * scale),
            y + int(250 * scale)
        ],
        fill="#315C3A",
        outline="#111111",
        width=max(2, int(4 * scale))
    )

    # head
    draw.ellipse(
        [
            x - head_r,
            y - head_r,
            x + head_r,
            y + head_r
        ],
        fill=skin,
        outline="#111111",
        width=max(2, int(4 * scale))
    )

    # hair
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
        width=max(5, int(12 * scale))
    )

    # eyes
    eye_y = y - int(15 * scale)

    draw.ellipse(
        [
            x - int(35 * scale),
            eye_y - int(10 * scale),
            x - int(15 * scale),
            eye_y + int(10 * scale)
        ],
        fill="white"
    )

    draw.ellipse(
        [
            x + int(15 * scale),
            eye_y - int(10 * scale),
            x + int(35 * scale),
            eye_y + int(10 * scale)
        ],
        fill="white"
    )

    draw.ellipse(
        [
            x - int(29 * scale),
            eye_y - int(5 * scale),
            x - int(21 * scale),
            eye_y + int(5 * scale)
        ],
        fill="black"
    )

    draw.ellipse(
        [
            x + int(21 * scale),
            eye_y - int(5 * scale),
            x + int(29 * scale),
            eye_y + int(5 * scale)
        ],
        fill="black"
    )

    # eyebrows for emotion
    if emotion in ["angry", "grumpy", "bossy"]:
        draw.line(
            [
                x - int(40 * scale),
                y - int(40 * scale),
                x - int(15 * scale),
                y - int(30 * scale)
            ],
            fill="black",
            width=max(2, int(6 * scale))
        )

        draw.line(
            [
                x + int(15 * scale),
                y - int(30 * scale),
                x + int(40 * scale),
                y - int(40 * scale)
            ],
            fill="black",
            width=max(2, int(6 * scale))
        )

    # mouth
    mouth_y = y + int(30 * scale)

    if talking:
        draw.ellipse(
            [
                x - int(22 * scale),
                mouth_y - int(8 * scale),
                x + int(22 * scale),
                mouth_y + int(25 * scale)
            ],
            fill="#250909"
        )
    else:
        draw.arc(
            [
                x - int(25 * scale),
                mouth_y - int(5 * scale),
                x + int(25 * scale),
                mouth_y + int(25 * scale)
            ],
            0,
            180,
            fill="#250909",
            width=max(2, int(5 * scale))
        )

    # arms
    movement = int(10 * scale) if talking else 0

    draw.line(
        [
            x - int(70 * scale),
            y + int(100 * scale),
            x - int(140 * scale),
            y + int(160 * scale) + movement
        ],
        fill="#315C3A",
        width=max(5, int(15 * scale))
    )

    draw.line(
        [
            x + int(70 * scale),
            y + int(100 * scale),
            x + int(140 * scale),
            y + int(160 * scale) - movement
        ],
        fill="#315C3A",
        width=max(5, int(15 * scale))
    )


def draw_calabash(draw, x, y, scale, talking=False):
    width = int(150 * scale)
    height = int(130 * scale)

    draw.ellipse(
        [
            x - width,
            y - height,
            x + width,
            y + height
        ],
        fill="#B87333",
        outline="#4A2A10",
        width=max(3, int(6 * scale))
    )

    # opening
    draw.ellipse(
        [
            x - int(70 * scale),
            y - int(35 * scale),
            x + int(70 * scale),
            y + int(5 * scale)
        ],
        fill="#321B0B"
    )

    # eyes
    draw.ellipse(
        [
            x - int(55 * scale),
            y + int(15 * scale),
            x - int(25 * scale),
            y + int(45 * scale)
        ],
        fill="white"
    )

    draw.ellipse(
        [
            x + int(25 * scale),
            y + int(15 * scale),
            x + int(55 * scale),
            y + int(45 * scale)
        ],
        fill="white"
    )

    # mouth
    if talking:
        draw.ellipse(
            [
                x - int(35 * scale),
                y + int(55 * scale),
                x + int(35 * scale),
                y + int(95 * scale)
            ],
            fill="#160909"
        )
    else:
        draw.arc(
            [
                x - int(35 * scale),
                y + int(45 * scale),
                x + int(35 * scale),
                y + int(90 * scale)
            ],
            0,
            180,
            fill="#160909",
            width=max(3, int(5 * scale))
        )


# ============================================================
# AUDIO
# ============================================================

def create_voice(text, filename, character):
    """
    gTTS is used for the first version.

    Different speaking rates/pitches are not directly controlled
    here, but each character can later be upgraded to a dedicated
    voice system.
    """

    # Simple English voice for maximum compatibility.
    language = "en"

    tts = gTTS(
        text=text,
        lang=language,
        slow=False
    )

    tts.save(str(filename))


def ffprobe_duration(filename):
    ffmpeg = get_ffmpeg()

    command = [
        ffmpeg,
        "-i",
        str(filename)
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    output = result.stderr

    import re

    match = re.search(
        r"Duration:\s*(\d+):(\d+):([\d.]+)",
        output
    )

    if not match:
        return 2.0

    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))

    return hours * 3600 + minutes * 60 + seconds


# ============================================================
# VIDEO CREATION
# ============================================================

def create_video(job_id, story, requested_duration):
    job = JOBS[job_id]

    work_dir = OUTPUT_DIR / job_id
    frames_dir = work_dir / "frames"
    audio_dir = work_dir / "audio"

    work_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(exist_ok=True)
    audio_dir.mkdir(exist_ok=True)

    try:
        job["status"] = "creating_audio"
        job["message"] = "Generating character voices..."

        dialogue_items = []

        for scene_index, scene in enumerate(story.get("scenes", [])):

            for dialogue_index, line in enumerate(
                scene.get("dialogue", [])
            ):

                text = line.get("text", "").strip()

                if not text:
                    continue

                character = line.get(
                    "character",
                    "Character"
                )

                emotion = line.get(
                    "emotion",
                    "neutral"
                )

                audio_file = (
                    audio_dir /
                    f"{scene_index}_{dialogue_index}.mp3"
                )

                create_voice(
                    text,
                    audio_file,
                    character
                )

                duration = ffprobe_duration(audio_file)

                dialogue_items.append({
                    "scene": scene_index,
                    "character": character,
                    "emotion": emotion,
                    "text": text,
                    "audio": audio_file,
                    "duration": duration
                })

        if not dialogue_items:
            raise Exception("The story contains no dialogue.")

        # --------------------------------------------------------
        # Build audio sequence
        # --------------------------------------------------------

        job["status"] = "building_audio"
        job["message"] = "Combining character voices..."

        concat_file = work_dir / "audio_concat.txt"

        with open(concat_file, "w", encoding="utf-8") as f:
            for item in dialogue_items:
                f.write(
                    "file '{}'\n".format(
                        str(item["audio"]).replace("'", "'\\''")
                    )
                )

        combined_audio = work_dir / "dialogue.mp3"

        ffmpeg = get_ffmpeg()

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
                "-c",
                "copy",
                str(combined_audio)
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # --------------------------------------------------------
        # Render cartoon frames
        # --------------------------------------------------------

        job["status"] = "rendering"
        job["message"] = "Animating characters..."

        WIDTH = 720
        HEIGHT = 1280
        FPS = 12

        frame_number = 0

        # Track how long each dialogue lasts.
        for scene_index, scene in enumerate(
            story.get("scenes", [])
        ):

            scene_dialogue = [
                x for x in dialogue_items
                if x["scene"] == scene_index
            ]

            for item in scene_dialogue:

                seconds = max(
                    1.0,
                    item["duration"]
                )

                frames = int(seconds * FPS)

                for i in range(frames):

                    img = Image.new(
                        "RGB",
                        (WIDTH, HEIGHT),
                        "#F5DFA5"
                    )

                    draw = ImageDraw.Draw(img)

                    # Sky
                    draw.rectangle(
                        [0, 0, WIDTH, 700],
                        fill="#87CEEB"
                    )

                    # Ground
                    draw.rectangle(
                        [0, 700, WIDTH, HEIGHT],
                        fill="#D19A5A"
                    )

                    # Sun
                    draw.ellipse(
                        [560, 80, 660, 180],
                        fill="#FFD54A"
                    )

                    # Tree
                    draw.rectangle(
                        [80, 350, 125, 700],
                        fill="#5C3A21"
                    )

                    draw.ellipse(
                        [20, 250, 190, 450],
                        fill="#3F7D3A"
                    )

                    # Village house
                    draw.polygon(
                        [
                            (450, 500),
                            (650, 350),
                            (850, 500)
                        ],
                        fill="#9A5A32"
                    )

                    draw.rectangle(
                        [480, 500, 820, 720],
                        fill="#C77D45"
                    )

                    # Title/location
                    draw.text(
                        (30, 30),
                        f"Scene {scene_index + 1}",
                        fill="black",
                        font=font(34, True)
                    )

                    # Determine character placement
                    current_character = item["character"]

                    # Kofi
                    if current_character == "Kofi":
                        draw_character(
                            draw,
                            220,
                            650,
                            1.0,
                            "Kofi",
                            item["emotion"],
                            talking=(i % 8 < 5)
                        )

                    # Amina
                    if current_character == "Amina":
                        draw_character(
                            draw,
                            500,
                            650,
                            1.0,
                            "Amina",
                            item["emotion"],
                            talking=(i % 8 < 5)
                        )

                    # If another character is talking,
                    # show the other characters too.
                    if current_character not in ["Kofi", "Amina"]:
                        draw_character(
                            draw,
                            220,
                            650,
                            1.0,
                            "Kofi",
                            "neutral",
                            talking=False
                        )

                        draw_character(
                            draw,
                            500,
                            650,
                            1.0,
                            "Amina",
                            "neutral",
                            talking=False
                        )

                    # Calabash
                    if current_character == "Calabash":
                        draw_calabash(
                            draw,
                            360,
                            650,
                            1.0,
                            talking=(i % 8 < 5)
                        )

                    # Dialogue box
                    box_top = 880

                    draw.rounded_rectangle(
                        [
                            25,
                            box_top,
                            WIDTH - 25,
                            1190
                        ],
                        radius=25,
                        fill="#111827"
                    )

                    speaker = item["character"]

                    draw.text(
                        (50, box_top + 25),
                        speaker,
                        fill="#FFD166",
                        font=font(32, True)
                    )

                    # Wrap text
                    words = item["text"].split()

                    lines = []
                    current_line = ""

                    for word in words:
                        test = (
                            current_line + " " + word
                        ).strip()

                        bbox = draw.textbbox(
                            (0, 0),
                            test,
                            font=font(30)
                        )

                        if bbox[2] <= WIDTH - 100:
                            current_line = test
                        else:
                            if current_line:
                                lines.append(
                                    current_line
                                )

                            current_line = word

                    if current_line:
                        lines.append(
                            current_line
                        )

                    y = box_top + 75

                    for line in lines[:5]:

                        draw.text(
                            (50, y),
                            line,
                            fill="white",
                            font=font(30)
                        )

                        y += 42

                    frame_path = (
                        frames_dir /
                        f"frame_{frame_number:06d}.jpg"
                    )

                    img.save(
                        frame_path,
                        quality=85
                    )

                    frame_number += 1

        # --------------------------------------------------------
        # Create MP4
        # --------------------------------------------------------

        job["status"] = "creating_mp4"
        job["message"] = "Creating final MP4..."

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

        output_file = OUTPUT_DIR / output_name

        video_input = frames_dir / "frame_%06d.jpg"

        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-framerate",
                str(FPS),
                "-i",
                str(video_input),
                "-i",
                str(combined_audio),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                "-movflags",
                "+faststart",
                str(output_file)
            ],
            check=True
        )

        job["status"] = "complete"
        job["message"] = "Cartoon video created successfully."
        job["video_url"] = (
            f"/generated/{output_file.name}"
        )

        # Cleanup large frame files
        try:
            shutil.rmtree(work_dir)
        except Exception:
            pass

    except Exception as e:

        job["status"] = "error"
        job["message"] = str(e)

        try:
            shutil.rmtree(work_dir)
        except Exception:
            pass


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():

    return jsonify({
        "status": "online",
        "app": "AfriToon AI",
        "message": "Cartoon backend is running"
    })


@app.route("/api/generate", methods=["POST"])
def api_generate():

    try:

        data = request.get_json() or {}

        category = data.get(
            "category",
            "Surprise Me"
        )

        topic = data.get(
            "topic",
            ""
        )

        duration = int(
            data.get(
                "duration",
                60
            )
        )

        language = data.get(
            "language",
            "English"
        )

        if duration not in [30, 60]:
            return jsonify({
                "success": False,
                "error": "Duration must be 30 or 60 seconds."
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

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/create-video", methods=["POST"])
def api_create_video():

    try:

        data = request.get_json() or {}

        story = data.get("story")

        duration = int(
            data.get(
                "duration",
                60
            )
        )

        if not story:
            return jsonify({
                "success": False,
                "error": "Story is required."
            }), 400

        job_id = uuid.uuid4().hex

        JOBS[job_id] = {
            "status": "starting",
            "message": "Starting video generation...",
            "video_url": None
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
            "job_id": job_id
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/video-status/<job_id>")
def video_status(job_id):

    job = JOBS.get(job_id)

    if not job:
        return jsonify({
            "success": False,
            "error": "Job not found."
        }), 404

    return jsonify({
        "success": True,
        "job": job
    })


@app.route("/generated/<path:filename>")
def generated_file(filename):

    return send_from_directory(
        OUTPUT_DIR,
        filename
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
