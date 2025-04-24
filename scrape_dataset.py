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

# def fetch_papers(tag):
#     feed = feedparser.parse(build_query(tag))
#     papers = []
#     for entry in tqdm(feed.entries):
#         pdf_link = entry.link.replace("abs", "pdf") + ".pdf"
#         paper_id = entry.id.split('/')[-1]
#         papers.append({
#             "tag": tag,
#             "title": entry.title,
#             "authors": ", ".join(a.name for a in entry.authors),
#             "published": entry.published,
#             "summary": entry.summary.replace("\n", " "),
#             "link": entry.link,
#             "pdf_link": pdf_link,
#             "id": paper_id
#         })
#     return papers

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


def download_pdf(paper):
    path = os.path.join(PDF_DIR, f"{paper['id']}.pdf")
    if not os.path.exists(path):
        r = requests.get(paper["pdf_link"])
        with open(path, "wb") as f:
            f.write(r.content)
    return path

def extract_sections(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()

    # Very simple section splitting
    sections = {}
    patterns = ["introduction", "method", "methodology", "experiment", "results", "conclusion"]
    for i in range(len(patterns)):
        pattern = patterns[i]
        next_pattern = patterns[i + 1] if i + 1 < len(patterns) else None
        start = re.search(rf"\b{pattern}\b", text, re.IGNORECASE)
        end = re.search(rf"\b{next_pattern}\b", text, re.IGNORECASE) if next_pattern else None
        if start:
            start_idx = start.start()
            end_idx = end.start() if end else len(text)
            sections[pattern] = text[start_idx:end_idx].strip()
    return sections

def main():
    papers = []
    for tag in TAGS:
        print(f"\n🔍 Fetching papers for tag: {tag}")
        temp_papers = fetch_papers(tag)
        
        print("Downloading papers... ")
        
        progress_bar = tqdm(range(len(temp_papers)))
        
        for i, p in enumerate(temp_papers):
            papers.append({'paper_id': i, **p})
            # download_pdf(p)
            progress_bar.update(1)
    
    print(f"\n✅ Parsed {len(papers)} papers.")
    return papers

if __name__ == "__main__":
    data = main()
    
    filename = TAGS[0].replace("%20", "_")
    json.dump(data, open(f"{filename}.json", "w+"))
