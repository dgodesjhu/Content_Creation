import streamlit as st
import pandas as pd
import numpy as np
import time
import re
from sentence_transformers import SentenceTransformer
from openai import OpenAI

st.set_page_config(page_title="Tweet Generator & Evaluator", layout="wide")

# Sidebar for API key
st.sidebar.title("API Key Setup")
user_api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
if not user_api_key:
    st.warning("Please enter your API key to continue.")
    st.stop()

client = OpenAI(api_key=user_api_key)

st.title("AI-Powered Tweet Generator & Evaluator")
st.markdown("Upload two CSVs and a message. Get 10 AI-generated tweets, evaluated for effectiveness.")

# Upload
col1, col2 = st.columns(2)
with col1:
    high_file = st.file_uploader("High-Engagement Tweets CSV", type="csv")
with col2:
    style_file = st.file_uploader("Style Tweets CSV", type="csv")

message = st.text_area("What message should the tweets communicate?", placeholder="E.g. AI helps marketers work faster")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def remove_non_ascii(text):
    return text.encode("ascii", errors="ignore").decode()

def clean_text(text):
    text = text.replace("’", "'").replace("“", "\"").replace("”", "\"").replace("–", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def evaluate_tweet(tweet, style_examples, client):
    style_sample = " | ".join(style_examples)
    prompt = f"""
You are a tweet evaluator. Rate the tweet below on a scale of 1–5 (1 = poor, 5 = excellent):
- Readability
- Clarity
- Persuasiveness
- Humor
- Edginess
- Style Match (based on: {style_sample})

Tweet: "{tweet}"

Respond as:
Readability: #
Clarity: #
Persuasiveness: #
Humor: #
Edginess: #
Style Match: #
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=200
        )
        raw = response.choices[0].message.content
        lines = raw.splitlines()
        scores = {}
        for line in lines:
            if ":" in line:
                key, val = line.split(":", 1)
                scores[key.strip()] = int(float(val.strip()))
        return scores
    except:
        return {}

def generate_tweets(message, high_examples, style_examples, client):
    high = "\n".join(high_examples[:5])
    style = "\n".join(style_examples[:5])
    prompt = f"""
You are helping a student write tweets about:

"{message}"

Here are high-engagement examples:
{high}

Here are stylistic inspirations:
{style}

Generate 10 short, engaging tweets (<280 characters). Don't number them. Put each on a new line.
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=800
    )
    content = response.choices[0].message.content
    tweets = [clean_text(line.strip()) for line in content.splitlines() if line.strip()]
    return tweets[:10]

def regenerate_tweet(original, feedback, client):
    prompt = f"""
You are rewriting a tweet to match the following feedback:

Tweet: "{original}"

Feedback: "{feedback}"

Please rewrite the tweet accordingly. Keep it under 280 characters.
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=200
    )
    return clean_text(response.choices[0].message.content.strip())

# Initialize session state
if "tweets" not in st.session_state:
    st.session_state.tweets = []
if "evaluations" not in st.session_state:
    st.session_state.evaluations = []

# Generate
if st.button("Generate Tweets"):
    if not (high_file and style_file and message):
        st.warning("Please upload both CSVs and a message.")
    else:
        try:
            high_df = pd.read_csv(high_file, encoding="latin1")
            style_df = pd.read_csv(style_file, encoding="latin1")

            if "tweet_text" not in high_df.columns or "tweet_text" not in style_df.columns:
                st.error("Each file must contain a column named 'tweet_text'.")
            else:
                high_texts = high_df["tweet_text"].dropna().tolist()
                style_texts = style_df["tweet_text"].dropna().tolist()

                with st.spinner("Generating and evaluating tweets..."):
                    tweets = generate_tweets(message, high_texts, style_texts, client)
                    st.session_state.tweets = tweets
                    all_evals = []
                    for tweet in tweets:
                        scores = evaluate_tweet(tweet, style_texts[:5], client)
                        scores["Tweet"] = remove_non_ascii(tweet)
                        all_evals.append(scores)
                        time.sleep(1.5)
                    st.session_state.evaluations = all_evals
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# Display results and regenerate options
if st.session_state.evaluations:
    st.markdown("### Evaluation Results")
    selected = []

    for i, entry in enumerate(st.session_state.evaluations):
        with st.container():
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                st.markdown(f"**Tweet {i+1}:** {entry['Tweet']}")
                st.markdown(
                    f"Readability: {entry['Readability']} | "
                    f"Clarity: {entry['Clarity']} | "
                    f"Persuasiveness: {entry['Persuasiveness']} | "
                    f"Humor: {entry['Humor']} | "
                    f"Edginess: {entry['Edginess']} | "
                    f"Style Match: {entry['Style Match']}"
                )
            with col2:
                if st.checkbox("Regenerate", key=f"select_{i}"):
                    selected.append(i)

    if selected:
        st.markdown("---")
        feedback = st.text_input("How would you like to revise the selected tweets?")
        if st.button("Regenerate Selected"):
            regenerated = []
            for i in selected:
                original = st.session_state.evaluations[i]["Tweet"]
                new_version = regenerate_tweet(original, feedback, client)
                regenerated.append({
                    "Original Tweet": original,
                    "Regenerated Tweet": new_version
                })
            st.markdown("### Regenerated Tweets")
            st.table(pd.DataFrame(regenerated))
