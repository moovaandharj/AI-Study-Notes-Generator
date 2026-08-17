from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import json
from groq import Groq

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Get Groq API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing from .env file")

# Create Groq client
client = Groq(api_key=api_key)


# ---------------- HOME PAGE ----------------

@app.route("/")
def home():
    return render_template("index.html")


# ---------------- AI GENERATION ----------------

@app.route("/api/generate", methods=["POST"])
def generate_notes():

    try:

        # Get data from frontend
        data = request.get_json()

        topic = data.get("topic", "").strip()
        material = data.get("material", "").strip()

        # Check empty input
        if not topic or not material:
            return jsonify({
                "success": False,
                "message": "Please enter both topic and study material."
            })

        # Check very short study material
        if len(material) < 20:
            return jsonify({
                "success": False,
                "message": "Please enter meaningful study material. At least 20 characters are required."
            })

        # Check for meaningless/random input
        words = material.split()

        if len(words) < 4:
            return jsonify({
                "success": False,
                "message": "Please enter meaningful study material with at least 4 words."
            })

        # ---------------- AI PROMPT ----------------

        prompt = f"""
You are an AI Study Notes Generator.

Topic:
{topic}

Study Material:
{material}

Create six DIFFERENT study outputs based ONLY on the study material.

1. SIMPLE EXPLANATION
Explain the topic in very simple English.
Use easy words so a student can understand it.

2. IMPORTANT POINTS
Give 5 important points from the study material.
Each point should be short and useful for revision.

3. SHORT NOTES
Create short revision notes.
Keep them concise and easy to remember.

4. KEY TERMS AND DEFINITIONS
Find important technical terms from the study material.
Give a simple definition for each term.
Give at least 3 terms.

5. QUIZ QUESTIONS
Create 5 questions based on the study material.
Include the correct answer for every question.

6. SUMMARY
Give a short summary of the study material.
The summary MUST be different from the simple explanation.
The summary must be shorter than the simple explanation.

IMPORTANT RULES:

- Use ONLY the provided study material.
- Do not add unrelated information.
- Keep the language simple.
- Simple Explanation and Summary MUST be different.
- Return ONLY valid JSON.
- Do not use markdown.
- Do not use ```json.
- Do not write anything before or after the JSON.

Return exactly this structure:

{{
    "simple_explanation": "Simple explanation here.",

    "important_points": [
        "Important point 1",
        "Important point 2",
        "Important point 3",
        "Important point 4",
        "Important point 5"
    ],

    "short_notes": "Short revision notes here.",

    "key_terms": [
        {{
            "term": "Term 1",
            "definition": "Definition of term 1"
        }},
        {{
            "term": "Term 2",
            "definition": "Definition of term 2"
        }},
        {{
            "term": "Term 3",
            "definition": "Definition of term 3"
        }}
    ],

    "quiz": [
        {{
            "question": "Question 1",
            "answer": "Answer 1"
        }},
        {{
            "question": "Question 2",
            "answer": "Answer 2"
        }},
        {{
            "question": "Question 3",
            "answer": "Answer 3"
        }},
        {{
            "question": "Question 4",
            "answer": "Answer 4"
        }},
        {{
            "question": "Question 5",
            "answer": "Answer 5"
        }}
    ],

    "summary": "Short summary here."
}}
"""

        # ---------------- SEND REQUEST TO GROQ ----------------

        response = client.chat.completions.create(

            model="llama-3.3-70b-versatile",

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI study assistant. "
                        "Always return valid JSON and follow the requested structure exactly."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.3
        )

        # ---------------- GET AI RESPONSE ----------------

        ai_response = response.choices[0].message.content.strip()

        print("\n========== GROQ RESPONSE ==========")
        print(ai_response)
        print("====================================")


        # ---------------- CLEAN JSON ----------------

        # Remove ```json if Groq returns it
        if ai_response.startswith("```json"):
            ai_response = ai_response[7:]

        if ai_response.startswith("```"):
            ai_response = ai_response[3:]

        if ai_response.endswith("```"):
            ai_response = ai_response[:-3]

        ai_response = ai_response.strip()


        # ---------------- CONVERT JSON ----------------

        result = json.loads(ai_response)


        # ---------------- GET RESULTS ----------------

        simple_explanation = result.get(
            "simple_explanation",
            "No explanation generated."
        )

        important_points = result.get(
            "important_points",
            []
        )

        short_notes = result.get(
            "short_notes",
            "No short notes generated."
        )

        key_terms = result.get(
            "key_terms",
            []
        )

        quiz = result.get(
            "quiz",
            []
        )

        summary = result.get(
            "summary",
            "No summary generated."
        )


        # ---------------- SEND TO FRONTEND ----------------

        return jsonify({

            "success": True,

            "simple_explanation": simple_explanation,

            "important_points": important_points,

            "short_notes": short_notes,

            "key_terms": key_terms,

            "quiz": quiz,

            "summary": summary

        })


    # ---------------- JSON ERROR ----------------

    except json.JSONDecodeError:

        print("ERROR: Groq did not return valid JSON.")

        return jsonify({
            "success": False,
            "message": "AI returned an invalid response format."
        })


    # ---------------- OTHER ERROR ----------------

    except Exception as e:

        print("ERROR:", str(e))

        return jsonify({
            "success": False,
            "message": str(e)
        })


# ---------------- RUN FLASK ----------------

if __name__ == "__main__":
    app.run(debug=True)