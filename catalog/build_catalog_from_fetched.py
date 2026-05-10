"""
Build SHL catalog from already-fetched HTML.
The web_fetch tool can bypass the 403, so we use a Python
script that reads pre-fetched HTML files to build the catalog.
"""
import json
import re
from bs4 import BeautifulSoup
from pathlib import Path

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

def parse_html(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    assessments = []
    tables = soup.find_all("table")
    
    # Find individual test solutions table
    individual_table = None
    for i, t in enumerate(tables):
        headers = [th.get_text(strip=True) for th in t.find_all("th")]
        text_before = ""
        prev = t.find_previous_sibling()
        if prev:
            text_before = prev.get_text()
        
        # Check if this table header row contains "Individual Test Solutions"  
        first_row = t.find("tr")
        if first_row:
            row_text = first_row.get_text()
            if "Individual Test Solutions" in row_text:
                individual_table = t
                break
    
    if not individual_table and len(tables) >= 2:
        individual_table = tables[-1]  # Last table is usually individual
    elif not individual_table and tables:
        individual_table = tables[0]
    
    if not individual_table:
        return assessments
    
    rows = individual_table.find_all("tr")
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        
        link = cells[0].find("a")
        if not link:
            continue
        
        name = link.get_text(strip=True)
        href = link.get("href", "")
        if href and not href.startswith("http"):
            href = "https://www.shl.com" + href
        
        if not name or not href:
            continue
        
        # Check for checkmarks in remote/adaptive columns
        def has_checkmark(cell):
            return bool(cell.find("img")) or cell.get_text(strip=True) not in ("", " ")
        
        remote = has_checkmark(cells[1]) if len(cells) > 1 else False
        adaptive = has_checkmark(cells[2]) if len(cells) > 2 else False
        
        # Test types
        test_types = []
        if len(cells) > 3:
            type_text = cells[3].get_text(strip=True)
            for ch in type_text:
                if ch.upper() in TEST_TYPE_MAP and ch.upper() not in test_types:
                    test_types.append(ch.upper())
        
        assessments.append({
            "name": name,
            "url": href,
            "test_types": test_types,
            "test_type_labels": [TEST_TYPE_MAP[t] for t in test_types],
            "remote_testing": remote,
            "adaptive_irt": adaptive,
            "description": generate_description(name, test_types),
        })
    
    return assessments

def generate_description(name: str, test_types: list[str]) -> str:
    """Generate a reasonable description based on name and type."""
    type_descs = {
        "A": "measures cognitive ability and aptitude",
        "B": "evaluates behavioral preferences and situational judgment",
        "C": "assesses competencies and work-related behaviors",
        "D": "supports development planning and 360-degree feedback",
        "E": "uses interactive assessment exercises",
        "K": "tests knowledge and technical skills",
        "P": "measures personality traits and behavioral style",
        "S": "provides realistic job simulations",
    }
    parts = [type_descs[t] for t in test_types if t in type_descs]
    if parts:
        return f"{name}: {'; '.join(parts)}."
    return f"{name}: SHL assessment tool."

if __name__ == "__main__":
    # Read any pre-saved HTML files
    html_dir = Path("/home/claude/shl_recommender/catalog/pages")
    if html_dir.exists():
        all_assessments = []
        seen = set()
        for html_file in sorted(html_dir.glob("*.html")):
            items = parse_html(html_file.read_text(encoding="utf-8", errors="replace"))
            for item in items:
                if item["url"] not in seen:
                    seen.add(item["url"])
                    all_assessments.append(item)
            print(f"{html_file.name}: {len(items)} items")
        
        catalog = {
            "source": "https://www.shl.com/products/product-catalog/",
            "scope": "Individual Test Solutions (type=1)",
            "total": len(all_assessments),
            "test_type_legend": TEST_TYPE_MAP,
            "assessments": all_assessments,
        }
        out = "/home/claude/shl_recommender/catalog/catalog.json"
        with open(out, "w") as f:
            json.dump(catalog, f, indent=2)
        print(f"\nTotal: {len(all_assessments)} saved to {out}")
    else:
        print("No pages directory found. HTML pages need to be saved first.")
