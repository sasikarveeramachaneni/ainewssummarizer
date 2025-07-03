import os
import time
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from crewai import Agent, Crew, Task
from dotenv import load_dotenv

# Load API Key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- Agent 1: News Fetcher ---
class NewsFetcher:
    def fetch_news(self, url):
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")

            title_tag = soup.find("h1") or soup.find("title")
            article_title = title_tag.get_text().strip() if title_tag else "Untitled Article"

            paragraphs = soup.find_all("p")
            article_text = " ".join([p.get_text() for p in paragraphs])

            return article_title, article_text if article_text else "No content found."
        except Exception as e:
            return "Error fetching article", f"Error: {str(e)}"

fetcher_agent = Agent(
    role="News Fetcher",
    backstory="Scrapes and extracts article text and heading.",
    goal="Fetch article heading and text.",
)

# --- Retry wrapper for model calls ---
def generate_with_retry(model, prompt, role="Agent", retries=3):
    delay = 30
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            return response.text.strip() if response else "No response."
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                print(f"[{role}] Quota exceeded. Retrying in {delay}s... ({attempt + 1}/{retries})")
                time.sleep(delay)
                delay *= 2  # exponential backoff
            else:
                return f"[{role}] Error: {str(e)}"
    return f"[{role}] Failed after {retries} retries."

# --- Agent 2: Categorizer ---
def categorize_news(article_text):
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    trimmed_text = article_text[:3000]  # avoid input token overflow
    prompt = f"Categorize this news article: '{trimmed_text}' into a category like Technology, Politics, Sports, etc."
    print("Fetching Category...")
    return generate_with_retry(model, prompt, role="Categorizer")

categorizer_agent = Agent(
    role="Categorizer",
    backstory="Classifies news articles.",
    goal="Categorize articles.",
)

# --- Agent 3: Summarizer ---
def summarize_article(article_title, article_text):
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    trimmed_text = article_text[:7000]  # leave space for title and prompt
    prompt = (f"Here is a news article titled '{article_title}'.\n\n"
              f"Summarize this article in a detailed way with a minimum of 200 words:\n\n{trimmed_text}")
    print("Generating Summary...")
    return generate_with_retry(model, prompt, role="Summarizer")

summarizer_agent = Agent(
    role="Summarizer",
    backstory="Generates detailed article summaries of at least 200 words.",
    goal="Summarize articles in a detailed way.",
)

# --- Crew Setup ---
crew = Crew(agents=[fetcher_agent, categorizer_agent, summarizer_agent])

# --- CLI Interaction ---
def main():
    url = input("Enter the news article URL: ")

    fetcher = NewsFetcher()
    article_title, article_text = fetcher.fetch_news(url)

    if "Error" in article_text:
        print(article_text)
        return

    category = categorize_news(article_text)
    summary = summarize_article(article_title, article_text)

    print("\n--- Article Analysis ---")
    print(f"\n📌 Title: {article_title}")
    print(f"\n📂 Category: {category}")
    print(f"\n📝 Summary:\n{summary}")

if __name__ == "__main__":
    main()
