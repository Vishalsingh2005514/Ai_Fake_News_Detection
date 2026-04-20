# AI Fake vs Real News Detection System

A highly robust, completely deterministic Streamlit web application designed manually to analyze news facts using a robust 3-layer deterministic search API and Groq AI reasoning system.

## Setup Instructions

### 1. Prerequisites
Ensure you have Python 3.9+ installed. You also need API keys from:
- [NewsAPI](https://newsapi.org/)
- [GNews](https://gnews.io/)
- [Groq](https://groq.com/)

### 2. Environment Variables
You must set your environment variables for local development. Create a file named `.env` in the root folder (`e:\Ai fake new detection system`) with the following contents:

```env
NEWS_API_KEY="your_news_api_key_here"
GNEWS_API_KEY="your_gnews_api_key_here"
GROQ_API_KEY="your_groq_api_key_here"
```

### 3. Installation
Open your terminal inside this folder and run:

```bash
pip install -r requirements.txt
```

### 4. Running the App
To start the Streamlit application, run the executable command:

```bash
streamlit run app.py
```

## Deployment on Streamlit Cloud

1. Push this folder to a GitHub repository.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and select "New app".
3. Connect your GitHub repository and point to `app.py`.
4. Before clicking "Deploy", go to **Advanced settings** and add your Secrets (the same as your `.env` contents):
   ```
   NEWS_API_KEY="..."
   GNEWS_API_KEY="..."
   GROQ_API_KEY="..."
   ```
5. Click **Deploy**. The application is configured to handle missing data gracefully without ever crashing, ensuring perfect uptime.
