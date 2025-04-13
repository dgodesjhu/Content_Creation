import streamlit as st
import openai
import pandas as pd
import numpy as np
import time
from sentence_transformers import SentenceTransformer

# ---------------- Page Config ----------------
st.set_page_config(page_title="Tweet Generator & Evaluator", layout="wide")

# ---------------- Sidebar: API Key ----------------
st.sidebar.title("🔐 API Key Setup")
user_api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
if user_api_key:
    openai.api_key = user_api_key

# ---------------- Header ----------------
st.title("AI-Powered Tweet Generator & Evaluator")
st.markdown("""
Upload two CSVs and enter a message. The app will use examples of high-engagement tweets and style inspiration tweets to generate 10 optimized tweets that communicate your message.
""")

# ---------------- File Upload ----------------
col1, col2 = st.columns(2)
with col1:
    high_file = st.file_uploader("📈 High-Engagement Tweets CSV", type="csv", key="high")
with col2:
    style_file = st.file_uploader("🎨 Style Inspiration Tweets CSV", type="csv", key="style")

# ---------------- Message ----------------
message = st.text_area("💬 What message do you want to communicate?", placeholder="E.g. AI helps marketers work faster")

# ---------------- Embedding Model ----------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")
embed_model = load_embedder()

def color_score(val):
    try:
        score = float(val)
        if score >= 4.5:
            color = "#4CAF50"  # dark green
        elif score >= 3.5:
            color = "#8BC34A"
        elif score >= 2.5:
            color = "#FFEB3B"
        elif score >= 1.5:
            color = "#FFC107"
        else:
            color = "#F44336"
        return f"background-color: {color}"
    except:
        return ""

def evaluate_tweet(tweet, style_examples):
    style_sample = " | ".join(style_examples)
    prompt = f"""
You are a tweet evaluator. Evaluate the following tweet on a scale of 1 to 5 (1 = poor, 5 = excellent) for these categories:
- Readability
- Clarity
- Persuasiveness
- Humor
- Edginess
- Style Match (based on these examples: {style_sample})

Tweet: "{tweet}"

Respond in this format:
Readability: #
Clarity: #
Persuasiveness: #
Humor: #
Edginess: #
Style Match: #
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=200
        )
        raw = response["choices"][0]["message"]["content"]
        lines = raw.splitlines()
        scores = {}
        for line in lines:
            if ":" in line:
                key, val = line.split(":", 1)
                try:
                    scores[key.strip()] = float(val.strip())
                except:
                    scores[key.strip()] = np.nan
        return scores
    except Exception as e:
        return {}

def generate_tweets(message, high_examples, style_examples):
    high = "\\n".join(high_examples[:5])
    style = "\\n".join(style_examples[:5])
    prompt = f"""
You are helping a student write tweets to communicate this message:

"{message}"

Here are examples of tweets with high engagement:
{high}

Here are examples of tweets that reflect the desired tone:
{style}

Based on the examples, write 10 short, engaging tweets under 280 characters. Do not number them. Keep each tweet on its own line.
"""
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=800
    )
    content = response["choices"][0]["message"]["content"]
    tweets = [line.strip() for line in content.splitlines() if line.strip()]
    return tweets[:10]

# ---------------- Run Button ----------------
if st.button("🚀 Generate Tweets"):
    if not (high_file and style_file and message and user_api_key):
        st.warning("Please upload both CSVs, enter a message, and provide your API key.")
    else:
        try:
            high_df = pd.read_csv(high_file, encoding="utf-8", errors="replace")
            style_df = pd.read_csv(style_file, encoding="utf-8", errors="replace")

            if "tweet_text" not in high_df.columns or "tweet_text" not in style_df.columns:
                st.error("Each file must contain a 'tweet_text' column.")
            else:
                high_texts = high_df["tweet_text"].dropna().tolist()
                style_texts = style_df["tweet_text"].dropna().tolist()

                with st.spinner("Generating and evaluating tweets..."):
                    tweets = generate_tweets(message, high_texts, style_texts)

                    all_evals = []
                    for tweet in tweets:
                        scores = evaluate_tweet(tweet, style_texts[:5])
                        scores["Tweet"] = tweet
                        all_evals.append(scores)
                        time.sleep(1.5)  # to avoid rate limit

                    df_out = pd.DataFrame(all_evals).set_index("Tweet")
                    st.markdown("### ✅ Results: Color-coded Evaluation")
                    st.dataframe(df_out.style.applymap(color_score), height=600)

        except Exception as e:
            st.error(f"Something went wrong: {e}")
