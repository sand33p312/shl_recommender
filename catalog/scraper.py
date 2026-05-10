"""
SHL Catalog Scraper
Scrapes Individual Test Solutions from https://www.shl.com/solutions/products/product-catalog/
Outputs: catalog.json with structured assessment data
"""

import httpx
import json
import time
import re
from bs4 import BeautifulSoup
from pathlib import Path

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# SHL test type codes
TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "M": "Motivation",
    "P": "Personality & Behaviour",
    "S": "Simulations",
}


def scrape_catalog_page(client: httpx.Client, url: str) -> list[dict]:
    """Scrape a single catalog page, returning list of assessment dicts."""
    assessments = []
    try:
        resp = client.get(url, headers=HEADERS, timeout=30, follow_redirects=True)
        resp.raise_for_status()
    except Exception as e:
        print(f"  ERROR fetching {url}: {e}")
        return assessments

    soup = BeautifulSoup(resp.text, "html.parser")

    # Find the catalog table — SHL uses a custom catalogue component
    # Try multiple selectors as the site may vary
    rows = soup.select("tr.catalogue__row") or soup.select("tr[data-course-id]")
    
    if not rows:
        # Try finding table rows in the product catalogue section
        table = soup.find("table", class_=re.compile(r"catalogue", re.I))
        if table:
            rows = table.find_all("tr")[1:]  # skip header

    if not rows:
        # Fallback: look for product cards / list items
        rows = soup.select(".product-catalogue__item, .catalogue-item, [class*='catalogue__row']")

    print(f"  Found {len(rows)} rows on {url}")

    for row in rows:
        assessment = parse_row(row)
        if assessment:
            assessments.append(assessment)

    return assessments


def parse_row(row) -> dict | None:
    """Parse a catalog table row into a structured dict."""
    try:
        # Name + URL
        name_el = (
            row.find("a")
            or row.find("td", class_=re.compile(r"name|title|product", re.I))
        )
        if not name_el:
            return None

        name = name_el.get_text(strip=True)
        if not name:
            return None

        href = name_el.get("href", "") if name_el.name == "a" else ""
        if href and not href.startswith("http"):
            href = BASE_URL + href

        # Test types — look for single-letter badges or data attrs
        test_types = []
        type_els = row.select(".product-catalogue__key, [class*='key'], .test-type, td[data-type]")
        for el in type_els:
            letter = el.get_text(strip=True).upper()
            if letter in TEST_TYPE_MAP:
                test_types.append(letter)

        # Also check data attributes
        data_type = row.get("data-type", "") or row.get("data-test-type", "")
        if data_type:
            for letter in data_type.upper().split(","):
                letter = letter.strip()
                if letter in TEST_TYPE_MAP and letter not in test_types:
                    test_types.append(letter)

        # Remote / adaptive / timed flags from checkmarks
        remote = bool(row.select("[class*='remote'], [data-remote='true']"))
        adaptive = bool(row.select("[class*='adaptive'], [data-adaptive='true']"))

        # Duration — look for time text
        duration_text = ""
        for td in row.find_all("td"):
            text = td.get_text(strip=True)
            if re.search(r"\d+\s*(min|minute)", text, re.I):
                duration_text = text
                break

        # Description from title attr or next sibling
        description = row.get("title", "") or name_el.get("title", "")

        return {
            "name": name,
            "url": href or CATALOG_URL,
            "test_types": test_types,
            "test_type_labels": [TEST_TYPE_MAP.get(t, t) for t in test_types],
            "remote": remote,
            "adaptive": adaptive,
            "duration": duration_text,
            "description": description,
        }
    except Exception as e:
        print(f"  Row parse error: {e}")
        return None


def scrape_product_detail(client: httpx.Client, url: str) -> dict:
    """Scrape individual product page for richer description."""
    if not url or url == CATALOG_URL:
        return {}
    try:
        resp = client.get(url, headers=HEADERS, timeout=20, follow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Try common description containers
        desc_el = (
            soup.find("div", class_=re.compile(r"description|overview|intro", re.I))
            or soup.find("section", class_=re.compile(r"description|overview", re.I))
            or soup.find("meta", {"name": "description"})
        )

        description = ""
        if desc_el:
            if desc_el.name == "meta":
                description = desc_el.get("content", "")
            else:
                description = desc_el.get_text(separator=" ", strip=True)[:500]

        # Key facts table
        facts = {}
        for row in soup.select("table tr, .key-facts tr, dl dt"):
            cells = row.find_all(["td", "th", "dd"])
            if len(cells) >= 2:
                key = cells[0].get_text(strip=True)
                val = cells[1].get_text(strip=True)
                if key and val:
                    facts[key] = val

        return {"description": description, "facts": facts}
    except Exception:
        return {}


def paginate_catalog(client: httpx.Client) -> list[str]:
    """Get all paginated catalog URLs for Individual Test Solutions."""
    urls = []
    # SHL catalog uses ?start=0&type=1 style pagination
    # type=1 = Individual Test Solutions
    page = 0
    page_size = 12  # SHL default

    while True:
        url = f"{CATALOG_URL}?start={page}&type=1&reloaded=true"
        try:
            resp = client.get(url, headers=HEADERS, timeout=30, follow_redirects=True)
            resp.raise_for_status()
        except Exception as e:
            print(f"Pagination error at page {page}: {e}")
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        rows = (
            soup.select("tr.catalogue__row")
            or soup.select("[class*='catalogue__row']")
            or soup.select("tr[data-entity-id]")
        )

        if not rows:
            print(f"No rows found at page {page}, stopping pagination.")
            break

        urls.append(url)
        print(f"Page start={page}: {len(rows)} rows")

        # Check if there's a next page
        next_btn = soup.find("a", class_=re.compile(r"next|pagination", re.I))
        if not next_btn or len(rows) < page_size:
            break

        page += page_size
        time.sleep(0.5)  # polite delay

    return urls


def run_scraper(output_path: str = "catalog.json", detail_scrape: bool = False):
    """Main scraper entry point."""
    print("Starting SHL catalog scrape...")
    all_assessments = []
    seen_names = set()

    with httpx.Client(timeout=30, follow_redirects=True) as client:
        # First try the main catalog with pagination
        page = 0
        page_size = 12
        consecutive_empty = 0

        while consecutive_empty < 2:
            url = f"{CATALOG_URL}?start={page}&type=1"
            print(f"\nFetching: {url}")

            try:
                resp = client.get(url, headers=HEADERS, timeout=30)
                resp.raise_for_status()
            except Exception as e:
                print(f"  Error: {e}")
                break

            soup = BeautifulSoup(resp.text, "html.parser")

            # Find catalogue rows - try multiple patterns
            rows = []
            for selector in [
                "tr.catalogue__row",
                "[class*='catalogue__row']",
                "tr[data-entity-id]",
                ".js-catalogue-row",
                "tbody tr",
            ]:
                rows = soup.select(selector)
                if rows:
                    print(f"  Matched selector: {selector}")
                    break

            if not rows:
                print(f"  No rows found at page {page}")
                consecutive_empty += 1
                if page == 0:
                    # Try scraping without pagination params
                    print("  Trying base URL without pagination params...")
                    try:
                        resp2 = client.get(CATALOG_URL, headers=HEADERS, timeout=30)
                        soup2 = BeautifulSoup(resp2.text, "html.parser")
                        # Dump structure for debugging
                        tables = soup2.find_all("table")
                        print(f"  Found {len(tables)} tables on base page")
                        all_links = soup2.find_all("a", href=re.compile(r"/solutions/products/"))
                        print(f"  Found {len(all_links)} product links")
                        # Save HTML snippet for inspection
                        with open("/home/claude/shl_recommender/catalog/page_sample.html", "w") as f:
                            f.write(resp2.text[:50000])
                        print("  Saved HTML sample to page_sample.html")
                    except Exception as e2:
                        print(f"  Base URL error: {e2}")
                page += page_size
                continue

            consecutive_empty = 0

            for row in rows:
                assessment = parse_row(row)
                if assessment and assessment["name"] not in seen_names:
                    seen_names.add(assessment["name"])
                    all_assessments.append(assessment)
                    print(f"  + {assessment['name']}")

            # Check if more pages exist
            total_count_el = soup.find(class_=re.compile(r"total|count|showing", re.I))
            if total_count_el:
                print(f"  Count indicator: {total_count_el.get_text(strip=True)}")

            next_exists = soup.select_one(
                "a[class*='next'], button[class*='next'], [aria-label='Next']"
            )
            if not next_exists and len(rows) < page_size:
                print("  No next page, stopping.")
                break

            page += page_size
            time.sleep(0.3)

        # If we got nothing from table rows, try extracting all product links
        if not all_assessments:
            print("\nFalling back to link extraction method...")
            all_assessments = extract_via_links(client)

    # Optional: enrich with detail pages
    if detail_scrape and all_assessments:
        print(f"\nEnriching {len(all_assessments)} assessments with detail pages...")
        with httpx.Client(timeout=20, follow_redirects=True) as client:
            for i, assessment in enumerate(all_assessments):
                if assessment["url"] and assessment["url"] != CATALOG_URL:
                    print(f"  [{i+1}/{len(all_assessments)}] {assessment['name']}")
                    detail = scrape_product_detail(client, assessment["url"])
                    if detail.get("description") and not assessment.get("description"):
                        assessment["description"] = detail["description"]
                    if detail.get("facts"):
                        assessment["facts"] = detail["facts"]
                    time.sleep(0.2)

    # Save
    output = {
        "source": CATALOG_URL,
        "scope": "Individual Test Solutions only",
        "total": len(all_assessments),
        "assessments": all_assessments,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_assessments)} assessments to {output_path}")
    return all_assessments


def extract_via_links(client: httpx.Client) -> list[dict]:
    """Fallback: extract assessments from all product links on catalog page."""
    assessments = []
    seen = set()
    
    try:
        resp = client.get(CATALOG_URL, headers=HEADERS, timeout=30)
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Find all links pointing to product detail pages
        product_links = soup.find_all(
            "a",
            href=re.compile(r"/solutions/products/product-catalog/view/", re.I)
        )
        
        print(f"  Found {len(product_links)} product detail links")
        
        for link in product_links:
            name = link.get_text(strip=True)
            href = link.get("href", "")
            if not href.startswith("http"):
                href = BASE_URL + href
            
            if name and name not in seen:
                seen.add(name)
                # Infer test type from URL or surrounding context
                parent_row = link.find_parent("tr") or link.find_parent("li") or link.parent
                test_types = []
                if parent_row:
                    for el in parent_row.find_all(text=re.compile(r'^[ABCDEKMP]$')):
                        letter = el.strip()
                        if letter in TEST_TYPE_MAP:
                            test_types.append(letter)
                
                assessments.append({
                    "name": name,
                    "url": href,
                    "test_types": test_types,
                    "test_type_labels": [TEST_TYPE_MAP.get(t, t) for t in test_types],
                    "remote": False,
                    "adaptive": False,
                    "duration": "",
                    "description": "",
                })
                print(f"  + {name}")
    
    except Exception as e:
        print(f"Link extraction error: {e}")
    
    return assessments


if __name__ == "__main__":
    import sys
    detail = "--detail" in sys.argv
    results = run_scraper(
        output_path="/home/claude/shl_recommender/catalog/catalog.json",
        detail_scrape=detail
    )
    print(f"\nTotal assessments scraped: {len(results)}")
