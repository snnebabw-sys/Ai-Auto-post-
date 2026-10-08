import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = Flask(__name__)

# Only allow your GitHub Pages website later.
CORS(app)

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing")

client = genai.Client(api_key=API_KEY)


@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "app": "AfriToon AI",
        "message": "Cartoon backend is running"
    })


@app.route("/api/generate", methods=["POST"])
def generate_story():

    data = request.get_json(silent=True) or {}

    category = data.get("category", "surprise")
    topic = data.get("topic", "")
    duration = data.get("duration", "30")
    language = data.get("language", "en")
    style = data.get("style", "funny")

    # Safety limits for the first version
    allowed_durations = ["30", "60"]

    if duration not in allowed_durations:
        return jsonify({
            "error": "For the free test version, use 30 or 60 seconds."
        }), 400

    prompt = f"""
You are the story director for an original African cartoon series.

Create a completely ORIGINAL short cartoon episode.

Category: {category}
Topic: {topic or "Create an interesting original topic"}
Duration: {duration} seconds
Language: {language}
Style: {style}

The cartoon must contain:

- 2 to 4 original characters
- Character personalities
- Natural conversation
- Funny or emotional reactions
- A clear beginning
- A problem/conflict
- Dialogue between characters
- A climax
- A satisfying ending
- Scene changes
- Visual actions
- Facial expressions
- Body movements
- African setting where appropriate

IMPORTANT:

The characters must actually TALK to each other.

Do NOT make this a slideshow.

Do NOT copy existing cartoons, characters, stories or scripts.

If the category is African History, separate historical facts
from fictional dialogue and do not invent historical facts.

Return ONLY valid JSON.

Use this exact structure:

{{
  "title": "",
  "description": "",
  "characters": [
    {{
      "name": "",
      "gender": "",
      "personality": "",
      "voice": ""
    }}
  ],
  "scenes": [
    {{
      "scene": 1,
      "location": "",
      "visual_action": "",
      "dialogue": [
        {{
          "character": "",
          "line": "",
          "emotion": ""
        }}
      ]
    }}
  ]
}}
"""

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )

        text = response.text.strip()

        # Remove markdown JSON fences if Gemini adds them
        if text.startswith("```"):
            text = text.replace("```json", "")
            text = text.replace("```", "")
            text = text.strip()

        story = json.loads(text)

        return jsonify({
            "success": True,
            "story": story
        })

    except json.JSONDecodeError:
        return jsonify({
            "success": False,
            "error": "AI returned invalid story data."
        }), 500

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
