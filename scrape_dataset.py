import os
import feedparser
import requests
import fitz  # PyMuPDF
import re
import json
from tqdm import tqdm

TAGS = [
    "llm%20prompt%20optimization"
]

MAX_RESULTS = 100000
BASE_URL = "http://export.arxiv.org/api/query?search_query="
PDF_DIR = "pdfs"

os.makedirs(PDF_DIR, exist_ok=True)

def build_query(tag):
    query = f"all:{tag}"
    return f"{BASE_URL}{query}&start=0&max_results={MAX_RESULTS}"


def fetch_papers(tag, total_results=100000, batch_size=1000):
    all_papers = []
    for start in range(0, total_results, batch_size):
        print(f"Fetching {tag}: {start}–{start+batch_size}")
        query = f"{BASE_URL}all:{tag}&start={start}&max_results={batch_size}"
        print(query)
        feed = feedparser.parse(query)
        for entry in feed.entries:
            pdf_link = entry.link.replace("abs", "pdf") + ".pdf"
            paper_id = entry.id.split('/')[-1]
            all_papers.append({
                "tag": tag,
                "title": entry.title,
                "authors": ", ".join(a.name for a in entry.authors),
                "published": entry.published,
                "summary": entry.summary.replace("\n", " "),
                "link": entry.link,
                "pdf_link": pdf_link,
                "id": paper_id
            })
        if len(feed.entries) < batch_size:
            break  # End of results
    
    return all_papers





def main():
    papers = []
    for tag in TAGS:
        print(f"\n Fetching papers for tag: {tag}")
        temp_papers = fetch_papers(tag)
        
        progress_bar = tqdm(range(len(temp_papers)))
        
        for i, p in enumerate(temp_papers):
            papers.append({'paper_id': i, **p})
            # download_pdf(p)
            progress_bar.update(1)
    
    print(f"\n Parsed {len(papers)} papers.")
    return papers

if __name__ == "__main__":
    data = main()
    
    filename = TAGS[0].replace("%20", "_")
    json.dump(data, open(f"{filename}.json", "w+"))
