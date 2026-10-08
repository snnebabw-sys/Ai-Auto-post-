import os
import json
import random

from google import genai


API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing")


client = genai.Client(api_key=API_KEY)


categories = [
    "Love",
    "Comedy",
    "Education",
    "African History",
    "Family",
    "Animal Adventure",
    "Love + Comedy",
    "Education + Comedy",
    "History + Comedy"
]


category = os.environ.get(
    "CATEGORY",
    random.choice(categories)
)

duration = os.environ.get(
    "DURATION",
    "180"
)

topic = os.environ.get(
    "TOPIC",
    ""
)


prompt = f"""
You are the head writer for an original African cartoon
YouTube series.

Create a completely ORIGINAL story.

Category:
{category}

Approximate video duration:
{duration} seconds

Optional idea:
{topic}

The story must be:

- entertaining
- funny when appropriate
- emotional when appropriate
- family friendly
- culturally respectful
- original
- suitable for YouTube
- designed for characters who actually TALK to one another

Create 2 to 5 main characters.

Every character should have:
- name
- personality
- age group
- role in the story

The story must contain:
- opening hook
- conflict
- character dialogue
- funny or emotional moments
- climax
- satisfying ending

For African history stories:
Do not invent historical facts.
Clearly separate fictional dialogue from historical facts.

Return JSON only:

{
  "title": "",
  "description": "",
  "characters": [],
  "scenes": [
    {
      "scene": 1,
      "location": "",
      "visual": "",
      "characters": [],
      "dialogue": [
        {
          "character": "",
          "line": ""
        }
      ],
      "narration": "",
      "duration": 8
    }
  ]
}
"""


response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)


text = response.text.strip()

if text.startswith("```"):
    text = text.split("```", 2)[1]
    text = text.replace("json", "", 1).strip()


story = json.loads(text)


os.makedirs("output", exist_ok=True)

with open(
    "output/story.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        story,
        file,
        ensure_ascii=False,
        indent=2
    )


print("Story created successfully.")

print(
    "Title:",
    story["title"]
)
