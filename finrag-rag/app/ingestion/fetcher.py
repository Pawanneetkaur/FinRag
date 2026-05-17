"""
fetcher.py — Downloads financial documents from SEC EDGAR and NewsAPI.

Two public functions you'll use from pipeline.py:
    fetch_10k(ticker)          -> list of Document
    fetch_news(company_name)   -> list of Document

A Document is a LlamaIndex object: it holds the raw text plus metadata
(company, year, source, etc.) that travels with the text all the way into
Qdrant. This metadata is what lets us filter by company at query time.
"""

import time
import requests
from datetime import datetime, timedelta
from llama_index.core import Document
from config import settings   # we'll create config.py next


# ── EDGAR helpers ─────────────────────────────────────────────────────────────

# EDGAR requires a descriptive User-Agent header or it will block requests.
# Format: "YourAppName your@email.com"
EDGAR_HEADERS = {"User-Agent": settings.EDGAR_USER_AGENT}
EDGAR_BASE    = "https://data.sec.gov"


def _get_cik(ticker: str) -> str:
    """
    Convert a stock ticker (e.g. "AAPL") to an EDGAR CIK number.

    EDGAR's company_tickers.json is a public lookup table — no auth needed.
    CIK numbers are zero-padded to 10 digits in API URLs.
    """
    url  = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=EDGAR_HEADERS)
    resp.raise_for_status()

    # The JSON is a dict of dicts: {"0": {"cik_str": 320193, "ticker": "AAPL", ...}, ...}
    data = resp.json()
    ticker_upper = ticker.upper()

    for entry in data.values():
        if entry["ticker"].upper() == ticker_upper:
            # Zero-pad to 10 digits — EDGAR requires this format
            return str(entry["cik_str"]).zfill(10)

    raise ValueError(f"Ticker '{ticker}' not found in EDGAR. Check the symbol.")


def _get_latest_10k_url(cik: str) -> tuple[str, int]:
    """
    Find the URL of the latest 10-K filing for this CIK.

    Returns: (filing_url, filing_year)

    EDGAR's submissions API returns a list of all filings for a company.
    We walk through them to find the most recent form type "10-K".
    """
    url  = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=EDGAR_HEADERS)
    resp.raise_for_status()

    filings = resp.json()["filings"]["recent"]

    # filings is a dict of parallel arrays: forms[i] matches dates[i] matches accessionNumbers[i]
    forms      = filings["form"]
    dates      = filings["filingDate"]
    accessions = filings["accessionNumber"]

    for i, form in enumerate(forms):
        if form == "10-K":
            # Accession number format: "0000320193-23-000106" → remove dashes for URL
            accession_clean = accessions[i].replace("-", "")
            filing_year     = int(dates[i][:4])
            filing_url      = (
                f"https://www.sec.gov/Archives/edgar/full-index/"
                f"{filing_year}/QTR{_quarter(dates[i])}/{accession_clean}-index.htm"
            )
            return filing_url, filing_year

    raise ValueError(f"No 10-K found for CIK {cik}")


def _quarter(date_str: str) -> int:
    """Return 1-4 quarter number from a YYYY-MM-DD date string."""
    month = int(date_str[5:7])
    return (month - 1) // 3 + 1


def _download_filing_text(cik: str, accession_number: str) -> str:
    """
    Download the plain-text version of a filing from EDGAR.

    EDGAR stores each filing as a folder of files. The main document
    is usually a .htm or .txt file. We grab the filing index to find it.
    """
    accession_clean = accession_number.replace("-", "")
    index_url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession_clean}/{accession_number}-index.json"
    )

    resp = requests.get(index_url, headers=EDGAR_HEADERS)
    resp.raise_for_status()
    files = resp.json().get("documents", [])

    # Find the primary document (the actual 10-K, not exhibits)
    for f in files:
        if f.get("type") == "10-K":
            doc_url  = f"https://www.sec.gov{f['documentUrl']}"
            doc_resp = requests.get(doc_url, headers=EDGAR_HEADERS)
            doc_resp.raise_for_status()
            return doc_resp.text

    raise ValueError("Could not find primary 10-K document in filing index.")


# ── Public fetch functions ────────────────────────────────────────────────────

def fetch_10k(ticker: str) -> list[Document]:
    """
    Fetch the latest 10-K filing for a ticker and return it as a
    LlamaIndex Document with metadata.

    Usage:
        docs = fetch_10k("AAPL")
        # docs[0].text       -> the full 10-K text
        # docs[0].metadata   -> {"company": "AAPL", "doc_type": "10-K", "year": 2023}
    """
    print(f"[EDGAR] Looking up CIK for {ticker}...")
    cik = _get_cik(ticker)

    print(f"[EDGAR] Found CIK {cik}. Fetching latest 10-K filing index...")
    # Re-fetch submissions to get accession number (reuse the API we already know)
    url  = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=EDGAR_HEADERS)
    resp.raise_for_status()
    filings    = resp.json()["filings"]["recent"]
    forms      = filings["form"]
    dates      = filings["filingDate"]
    accessions = filings["accessionNumber"]

    accession_number = None
    filing_year      = None
    for i, form in enumerate(forms):
        if form == "10-K":
            accession_number = accessions[i]
            filing_year      = int(dates[i][:4])
            break

    if not accession_number:
        raise ValueError(f"No 10-K found for {ticker}")

    print(f"[EDGAR] Downloading 10-K text (year={filing_year})...")
    text = _download_filing_text(cik, accession_number)

    # EDGAR rate limit: 10 requests/second. Sleep to be polite.
    time.sleep(0.2)

    # Wrap in a LlamaIndex Document — metadata rides along into Qdrant
    doc = Document(
        text=text,
        metadata={
            "company":   ticker.upper(),
            "doc_type":  "10-K",
            "year":      filing_year,
            "source":    "SEC EDGAR",
            "accession": accession_number,
        }
    )
    print(f"[EDGAR] Done. Document length: {len(text):,} chars.")
    return [doc]


def fetch_news(company_name: str, ticker: str, days_back: int = 90) -> list[Document]:
    """
    Fetch recent news articles about a company using NewsAPI.

    Returns one Document per article (up to 100 articles).
    Each Document contains the article text and metadata.

    Requires: settings.NEWS_API_KEY (free tier at newsapi.org)
    """
    if not settings.NEWS_API_KEY:
        print("[NewsAPI] No API key set — skipping news fetch.")
        return []

    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    to_date   = datetime.now().strftime("%Y-%m-%d")

    print(f"[NewsAPI] Fetching news for '{company_name}' from {from_date} to {to_date}...")

    resp = requests.get(
        "https://newsapi.org/v2/everything",
        params={
            "q":        f'"{company_name}" earnings OR revenue OR financial',
            "from":     from_date,
            "to":       to_date,
            "language": "en",
            "sortBy":   "relevancy",
            "pageSize": 100,
        },
        headers={"X-Api-Key": settings.NEWS_API_KEY},
    )
    resp.raise_for_status()
    articles = resp.json().get("articles", [])
    print(f"[NewsAPI] Retrieved {len(articles)} articles.")

    documents = []
    for article in articles:
        # Combine title + description + content for richer context
        text = " ".join(filter(None, [
            article.get("title", ""),
            article.get("description", ""),
            article.get("content", ""),
        ]))

        if len(text.strip()) < 100:
            # Skip very short or empty articles — not worth embedding
            continue

        doc = Document(
            text=text,
            metadata={
                "company":      ticker.upper(),
                "doc_type":     "news",
                "source":       article.get("source", {}).get("name", "unknown"),
                "published_at": article.get("publishedAt", ""),
                "url":          article.get("url", ""),
            }
        )
        documents.append(doc)

    print(f"[NewsAPI] Kept {len(documents)} articles after filtering.")
    return documents
