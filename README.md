# 🧠 Gemini Sentiment & Emotion Analysis

This project uses Google's Gemini 2.5 Flash API to analyze sentiment, emotion, and category for user feedback data. It processes a dataset (e.g., Twitter Sentiment Analysis), classifies entries, and compares AI-generated sentiment with actual labels to compute accuracy.

---

## 📦 Features

- Uses Gemini AI to extract:
  - **Sentiment**: Positive, Negative, Neutral
  - **Emotion**: Joy, Anger, Sadness
  - **Category**: Suggestion, Complaint, Praise
- Compares Gemini-generated sentiment with actual labels
- Outputs results to Excel
- Logs all API responses and errors
- Handles failures gracefully (fallback to "Unknown")

---

## 🧰 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
