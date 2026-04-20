"""
ai_checker.py
-------------
Handles three core tasks:
  1. ML signal  → detect fake news writing STYLE using model.pkl (secondary signal)
  2. News search → fetch live news articles for context (NewsAPI → GNews fallback)
  3. AI verdict  → Groq/Llama-3 is the PRIMARY fact-checker (returns verdict + confidence + explanation)
"""

import os
import json
import pickle
import logging
import requests
from groq import Groq

# Terminal logger for debugging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ─── API Keys (supports both local .env and Streamlit Cloud Secrets) ──────────
def _get_secret(key, fallback=""):
    """Read from Streamlit secrets first (cloud), then environment variables (local)."""
    try:
        import streamlit as st
        return st.secrets.get(key, os.getenv(key, fallback))
    except Exception:
        return os.getenv(key, fallback)

NEWS_API_KEY  = _get_secret("NEWS_API_KEY")
GNEWS_API_KEY = _get_secret("GNEWS_API_KEY")
GROQ_API_KEY  = _get_secret("GROQ_API_KEY")




# ─── Load ML Model Once on Startup ────────────────────────────────────────────
ml_model      = None
ml_vectorizer = None

if os.path.exists("model.pkl") and os.path.exists("vectorizer.pkl"):
    try:
        with open("model.pkl", "rb") as f:
            ml_model = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f:
            ml_vectorizer = pickle.load(f)
        log.info("ML model loaded.")
    except Exception as e:
        log.error(f"Could not load ML model: {e}")


# ─── ML Style Prediction (Secondary Signal) ───────────────────────────────────

def get_ml_prediction(text):
    """
    Predict REAL/FAKE based on writing STYLE using the trained TF-IDF model.

    NOTE: This model was trained on political news writing patterns.
    It is a SECONDARY signal only — it cannot verify real-world facts.
    Groq AI is the primary verdict source.

    Returns:
        tuple: (verdict: str, confidence: int)
    """
    if ml_model is None or ml_vectorizer is None:
        return "UNCERTAIN", 50

    try:
        vec        = ml_vectorizer.transform([text])
        prediction = ml_model.predict(vec)[0]         # 0 = FAKE, 1 = REAL
        proba      = ml_model.predict_proba(vec)[0]

        verdict    = "REAL" if prediction == 1 else "FAKE"
        confidence = int(max(proba) * 100)
        return verdict, confidence

    except Exception as e:
        log.error(f"ML prediction error: {e}")
        return "UNCERTAIN", 50


# ─── Live News Search ─────────────────────────────────────────────────────────

def _search_newsapi(query):
    """Primary news source: NewsAPI."""
    try:
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": query, "language": "en", "sortBy": "relevancy", "pageSize": 5, "apiKey": NEWS_API_KEY},
            timeout=8
        )
        r.raise_for_status()
        return [
            {"title": a.get("title",""), "url": a.get("url","#"), "source": a.get("source",{}).get("name","Unknown")}
            for a in r.json().get("articles", [])
        ]
    except Exception as e:
        log.warning(f"NewsAPI failed: {e}")
        return []


def _search_gnews(query):
    """Fallback news source: GNews."""
    try:
        r = requests.get(
            "https://gnews.io/api/v4/search",
            params={"q": query, "lang": "en", "max": 5, "apikey": GNEWS_API_KEY},
            timeout=8
        )
        r.raise_for_status()
        return [
            {"title": a.get("title",""), "url": a.get("url","#"), "source": a.get("source",{}).get("name","Unknown")}
            for a in r.json().get("articles", [])
        ]
    except Exception as e:
        log.warning(f"GNews failed: {e}")
        return []


def _extract_keywords(user_text):
    """
    Use Groq to extract the best 4-6 search keywords from the claim.
    This gives much better news search results than taking first N words.
    Falls back to simple extraction if Groq fails.
    """
    if not GROQ_API_KEY:
        return " ".join(user_text.strip().split()[:10])
    try:
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",   # Fast lightweight model for keyword task
            messages=[{
                "role": "user",
                "content": (
                    f"Extract exactly 2 or 3 core search keywords (proper nouns/events) from this news claim. "
                    f"Return ONLY the keywords separated by spaces, no punctuation.\n\n"
                    f"Claim: {user_text}"
                )
            }],
            temperature=0.0,
            max_tokens=30
        )
        keywords = resp.choices[0].message.content.strip()
        log.info(f"Smart keywords: '{keywords}'")
        return keywords
    except Exception:
        # Simple fallback: first 3 words
        return " ".join(user_text.strip().split()[:3])


def _search_duckduckgo(query):
    """Third fallback: DuckDuckGo HTML search."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query + " news"},
            headers=headers,
            timeout=6
        )
        results = []
        # Simple parse of DuckDuckGo results
        from html.parser import HTMLParser

        class DDGParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.results = []
                self._in_result = False
                self._current_title = ""

            def handle_starttag(self, tag, attrs):
                attrs_dict = dict(attrs)
                if tag == "a" and "result__a" in attrs_dict.get("class", ""):
                    self._in_result = True
                    self.results.append({
                        "title": "",
                        "url": attrs_dict.get("href", "#"),
                        "source": "DuckDuckGo"
                    })

            def handle_data(self, data):
                if self._in_result and self.results:
                    self.results[-1]["title"] += data
                    if len(self.results[-1]["title"]) > 10:
                        self._in_result = False

        parser = DDGParser()
        parser.feed(r.text)
        return parser.results[:5]
    except Exception as e:
        log.warning(f"DuckDuckGo failed: {e}")
        return []


def search_news(user_text):
    """
    Search for live news using smart keyword extraction.
    Pipeline: Extract keywords → NewsAPI → GNews → DuckDuckGo
    """
    # Step 1: Extract smart keywords from the claim
    query = _extract_keywords(user_text)
    log.info(f"Searching with: '{query}'")

    # Step 2: Try NewsAPI
    results = _search_newsapi(query)

    # Step 3: Try GNews if empty
    if not results:
        log.info("NewsAPI empty — trying GNews...")
        results = _search_gnews(query)

    # Step 4: If still empty, the query might be too restrictive. Try simpler fallback.
    if not results:
        fallback_query = " ".join(user_text.strip().split()[:3])
        if fallback_query and fallback_query.lower() != query.lower():
            log.info(f"Results empty. Retrying with simple fallback query: '{fallback_query}'")
            results = _search_newsapi(fallback_query)
            if not results:
                results = _search_gnews(fallback_query)

    # Step 5: Try DuckDuckGo if still empty
    if not results:
        log.info("Still empty — trying DuckDuckGo...")
        results = _search_duckduckgo(query)

    log.info(f"{len(results)} articles found.")
    return results


# ─── PRIMARY: AI Verdict via Groq/Llama-3 ────────────────────────────────────

def analyze_news(user_text, news_articles, ml_verdict):
    """
    Use Groq's Llama-3 model as the PRIMARY fact-checker.

    The AI:
      - Checks the claim against live news articles
      - Uses its own knowledge for well-known facts
      - Returns a structured verdict: REAL, FAKE, or UNCERTAIN
      - Provides confidence + explanation

    The ML verdict is passed as a HINT (secondary signal) — the AI makes the final call.

    Args:
        user_text     (str):  The claim entered by the user.
        news_articles (list): Live news articles from search_news().
        ml_verdict    (str):  ML model's style-based prediction (hint only).

    Returns:
        dict with keys: 'verdict', 'confidence', 'explanation'
    """
    # Fallback if Groq is unavailable
    if not GROQ_API_KEY:
        return {
            "verdict":     ml_verdict,
            "confidence":  50,
            "explanation": "Groq API key missing. Showing ML model result only."
        }

    # Format the news articles into a readable list
    if news_articles:
        news_context = "\n".join(
            f"- {a['title']} ({a['source']})" for a in news_articles
        )
        has_context = True
    else:
        news_context = "No live news found for this query."
        has_context = False

    # Smart prompt that handles 3 types of claims correctly:
    # 1. Known facts (history, science) → REAL/FAKE from training knowledge
    # 2. Recent events with news articles → cross-check articles
    # 3. Recent events with NO articles → UNCERTAIN (not FAKE!)
    prompt = f"""You are a professional AI fact-checker. Today's date is April 2026.

CLAIM TO VERIFY: "{user_text}"

LIVE NEWS ARTICLES FOUND:
{news_context}

YOUR DECISION PROCESS (follow in order):

STEP 1 - Is this a well-known established fact?
  (history, science, geography, famous people, countries, sports records)
  → If YES: Answer REAL or FAKE based on your knowledge. High confidence (85-99%).

STEP 2 - Is this about a recent or current event?
  → If news articles CONFIRM it: Answer REAL (75-95% confidence)
  → If news articles CONTRADICT it: Answer FAKE (80-95% confidence)
  → If news articles exist but are UNRELATED: Answer UNCERTAIN (65-75%)
  → If NO news articles found AND it could be a recent event: Answer UNCERTAIN (65%)
     ⚠️ IMPORTANT: Do NOT mark as FAKE just because you don't know about it.
     Recent breaking news may not be in your training data.

STEP 3 - Is it clearly absurd or impossible?
  → Answer FAKE (90%+ confidence)

KEY RULES:
- "USA Iran ceasefire", "peace deal", recent political events = could be REAL if news confirms it
- NEVER say FAKE for recent news just because you are unsure — say UNCERTAIN instead
- Only say FAKE when you have clear evidence it is false

Respond ONLY with valid JSON:
{{
    "verdict": "REAL" or "FAKE" or "UNCERTAIN",
    "confidence": <integer 60-99>,
    "explanation": "<2 clear sentences explaining the verdict and what evidence was used>"
}}"""

    try:
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful, accurate fact-checker. You respond ONLY with valid JSON. You NEVER say FAKE for recent events just because you don't know about them — you say UNCERTAIN instead."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=300
        )

        raw = completion.choices[0].message.content
        result = json.loads(raw)

        # Validate verdict
        if result.get("verdict") not in ("REAL", "FAKE", "UNCERTAIN"):
            result["verdict"] = "UNCERTAIN"

        # Ensure confidence is in valid range
        conf = result.get("confidence", 75)
        result["confidence"] = max(60, min(99, int(conf)))

        # Ensure explanation exists
        if not result.get("explanation"):
            result["explanation"] = "Verdict based on available evidence and world knowledge."

        return result

    except json.JSONDecodeError as e:
        log.error(f"JSON parse error: {e}")
        return {"verdict": ml_verdict, "confidence": 65, "explanation": "AI response parsing failed. Showing ML result."}

    except Exception as e:
        log.error(f"Groq API error: {e}")
        return {"verdict": ml_verdict, "confidence": 65, "explanation": f"AI service temporarily unavailable. Showing ML model result only."}

