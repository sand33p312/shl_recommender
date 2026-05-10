"""
SHL Catalog Scraper v2
Uses httpx with proper headers to scrape all Individual Test Solutions.
32 pages x 12 items = ~384 assessments.
"""

import httpx
import json
import time
import re
from bs4 import BeautifulSoup

BASE_URL = "https://www.shl.com"
CATALOG_BASE = "https://www.shl.com/products/product-catalog/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
    "Referer": "https://www.google.com/",
}

TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behaviour",
    "S": "Simulations",
}


def parse_catalog_table(html: str, source_url: str) -> list[dict]:
    """Extract Individual Test Solutions rows from a catalog page."""
    soup = BeautifulSoup(html, "html.parser")
    assessments = []

    # Find the Individual Test Solutions table
    # It's the second table on the page (first is Pre-packaged Job Solutions)
    tables = soup.find_all("table")
    
    individual_table = None
    for table in tables:
        # Check if previous heading says "Individual Test Solutions"
        prev = table.find_previous(["h2", "h3", "caption", "th"])
        table_text = table.get_text()
        if "Individual Test Solutions" in table_text or (prev and "Individual" in prev.get_text()):
            individual_table = table
            break
    
    if not individual_table and tables:
        # Fallback: use second table if two exist
        if len(tables) >= 2:
            individual_table = tables[1]
        else:
            individual_table = tables[0]

    if not individual_table:
        print(f"  WARNING: No table found on {source_url}")
        return assessments

    rows = individual_table.find_all("tr")
    # Skip header row
    for row in rows[1:]:
        cells = row.find_all("td")
        if not cells:
            continue

        # Cell 0: Name + URL
        name_cell = cells[0]
        link = name_cell.find("a")
        if not link:
            continue

        name = link.get_text(strip=True)
        href = link.get("href", "")
        if href and not href.startswith("http"):
            href = BASE_URL + href

        if not name:
            continue

        # Cell 1: Remote Testing (checkmark = True)
        remote = False
        if len(cells) > 1:
            remote_cell = cells[1]
            # Look for checkmark image or non-empty content
            remote = bool(remote_cell.find("img")) or bool(remote_cell.get_text(strip=True))

        # Cell 2: Adaptive/IRT
        adaptive = False
        if len(cells) > 2:
            adaptive_cell = cells[2]
            adaptive = bool(adaptive_cell.find("img")) or bool(adaptive_cell.get_text(strip=True))

        # Cell 3: Test Types (letters)
        test_types = []
        if len(cells) > 3:
            type_cell = cells[3]
            # Each letter is usually in a span or just text
            type_text = type_cell.get_text(strip=True)
            for char in type_text:
                if char.upper() in TEST_TYPE_MAP:
                    test_types.append(char.upper())
            # Also check spans
            for span in type_cell.find_all("span"):
                letter = span.get_text(strip=True).upper()
                if letter in TEST_TYPE_MAP and letter not in test_types:
                    test_types.append(letter)

        assessments.append({
            "name": name,
            "url": href,
            "test_types": test_types,
            "test_type_labels": [TEST_TYPE_MAP.get(t, t) for t in test_types],
            "remote_testing": remote,
            "adaptive_irt": adaptive,
            "description": "",  # filled by detail scrape
        })

    return assessments


def scrape_all_pages(total_pages: int = 32, page_size: int = 12) -> list[dict]:
    """Scrape all pages of Individual Test Solutions catalog."""
    all_assessments = []
    seen_urls = set()

    with httpx.Client(
        headers=HEADERS,
        follow_redirects=True,
        timeout=30,
    ) as client:
        for page_num in range(total_pages):
            start = page_num * page_size
            url = f"{CATALOG_BASE}?start={start}&type=1"
            print(f"Fetching page {page_num + 1}/{total_pages}: {url}")

            try:
                resp = client.get(url)
                if resp.status_code != 200:
                    print(f"  HTTP {resp.status_code} — skipping")
                    continue
            except Exception as e:
                print(f"  Error: {e}")
                continue

            items = parse_catalog_table(resp.text, url)
            new_items = 0
            for item in items:
                if item["url"] not in seen_urls:
                    seen_urls.add(item["url"])
                    all_assessments.append(item)
                    new_items += 1

            print(f"  +{new_items} new items (total: {len(all_assessments)})")

            if not items:
                print("  Empty page — stopping early")
                break

            time.sleep(0.4)  # polite rate limit

    return all_assessments


def scrape_detail_page(client: httpx.Client, url: str) -> str:
    """Get description from a product detail page."""
    if not url or "product-catalog" not in url:
        return ""
    try:
        resp = client.get(url, timeout=20)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Try various content areas
        for selector in [
            ".product-catalogue__description",
            ".description",
            "article p",
            "main p",
            "[class*='description']",
            "[class*='overview']",
            "meta[name='description']",
        ]:
            if selector.startswith("meta"):
                el = soup.find("meta", {"name": "description"})
                if el:
                    return el.get("content", "")[:400]
            else:
                el = soup.select_one(selector)
                if el:
                    text = el.get_text(separator=" ", strip=True)
                    if len(text) > 30:
                        return text[:400]
        
        # Last resort: first meaningful paragraph
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if len(text) > 50:
                return text[:400]

        return ""
    except Exception:
        return ""


def enrich_with_details(assessments: list[dict], max_workers: int = 1) -> list[dict]:
    """Enrich assessments with descriptions from detail pages."""
    print(f"\nEnriching {len(assessments)} assessments with detail pages...")
    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=20) as client:
        for i, a in enumerate(assessments):
            if i % 10 == 0:
                print(f"  [{i}/{len(assessments)}]...")
            desc = scrape_detail_page(client, a["url"])
            if desc:
                a["description"] = desc
            time.sleep(0.15)
    return assessments


def main(output: str = "/home/claude/shl_recommender/catalog/catalog.json",
         skip_details: bool = False):
    assessments = scrape_all_pages(total_pages=32)
    
    if not skip_details:
        assessments = enrich_with_details(assessments)

    catalog = {
        "source": CATALOG_BASE,
        "scope": "Individual Test Solutions only (type=1)",
        "total": len(assessments),
        "test_type_legend": TEST_TYPE_MAP,
        "assessments": assessments,
    }

    with open(output, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(assessments)} assessments saved to {output}")
    return assessments


if __name__ == "__main__":
    import sys
    skip = "--no-detail" in sys.argv
    main(skip_details=skip)
