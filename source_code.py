import json
import re

import pandas as pd
import time
from tqdm import tqdm
from google import genai
from google.genai import types

# ========== CONFIGURATION ==========
API_KEY = "***********"  # Use your actual API key
INPUT_CSV = "twitter_training.csv"
OUTPUT_EXCEL = "gemini_sentiment_output.xlsx"
NUM_SAMPLES = 15
RANDOM_SEED = 30
LOG_FILE = "gemini_api_logs.txt"
SLEEP_TIME = 0.5
# ===================================

# Set up Gemini client
client = genai.Client(api_key=API_KEY)


# === Logging helper ===
def log_message(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{msg}\n")


# === Gemini Analysis Function ===
def analyze_feedback_combined(feedback_text, given_sentiment):
    try:
        prompt = f"""
        A user left the following feedback:
        "{feedback_text}"

        The originally assigned sentiment was: '{given_sentiment}'.

        Your task is to independently evaluate the actual content of the feedback and determine what the sentiment should be, regardless of the assigned label.

        Return a JSON object with:
        - sentiment: Positive, Negative, or Neutral
        - emotion: Joy, Anger, or Sadness
        - category: Suggestion, Complaint, or Praise

        Only return a valid JSON object like:
        {{
          "sentiment": "Negative",
          "emotion": "Anger",
          "category": "Complaint"
        }}
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            ),
        )

        raw_text = response.text.strip()
        log_message(f"\n---\nFeedback: {feedback_text}\nPrompt Sentiment: {given_sentiment}\nResponse:\n{raw_text}")

        # Extract and parse JSON from Gemini output
        json_match = re.search(r'\{.*?\}', raw_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())

            sentiment = parsed.get("sentiment", "").title()
            emotion = parsed.get("emotion", "").title()
            category = parsed.get("category", "").title()

            if sentiment in {"Positive", "Negative", "Neutral"}:
                return {
                    "sentiment": sentiment,
                    "emotion": emotion if emotion in {"Joy", "Anger", "Sadness"} else "Unknown",
                    "category": category if category in {"Suggestion", "Complaint", "Praise"} else "Unknown"
                }

        # fallback if invalid response
        return {
            "sentiment": "Unknown",
            "emotion": "Unknown",
            "category": "Unknown"
        }

    except Exception as e:
        log_message(f"❌ Error: {e} for feedback: {feedback_text[:100]}")
        return {
            "sentiment": "Unknown",
            "emotion": "Unknown",
            "category": "Unknown"
        }


# === Main execution ===
def main():
    try:
        df = pd.read_csv(INPUT_CSV, dtype={"id": int, "title": str, "sentiment": str, "text": str})
    except Exception as e:
        raise ValueError(f"❌ Failed to read CSV file: {e}")

    if not {'id', 'title', 'sentiment', 'text'}.issubset(df.columns):
        raise ValueError("❌ CSV must have columns: id, title, sentiment, text")

    print(f"✅ Loaded {len(df)} rows")
    sampled_df = df.sample(n=min(NUM_SAMPLES, len(df)), random_state=RANDOM_SEED).reset_index(drop=True)

    results = []

    print("🔍 Analyzing feedback using Gemini 2.5 Flash...")
    for _, row in tqdm(sampled_df.iterrows(), total=len(sampled_df)):
        feedback_text = str(row['text'])
        given_sentiment = str(row['sentiment'])

        ai_result = analyze_feedback_combined(feedback_text, given_sentiment)

        results.append({
            "Feedback": feedback_text,
            "Actual Sentiment": given_sentiment,
            "Generated Sentiment": ai_result['sentiment'],
            "Emotion": ai_result['emotion'],
            "Category": ai_result['category']
        })

        time.sleep(SLEEP_TIME)

    final_df = pd.DataFrame(results)

    # Accuracy calculation
    matches = final_df["Actual Sentiment"].str.lower() == final_df["Generated Sentiment"].str.lower()
    accuracy = (matches.sum() / len(final_df)) * 100
    print(f"\n✅ Sentiment Accuracy: {accuracy:.2f}%")

    # Save to Excel
    final_df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"📁 Output saved to: {OUTPUT_EXCEL}")
    log_message(f"\n✅ Final Sentiment Accuracy: {accuracy:.2f}%\nSaved to {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()
