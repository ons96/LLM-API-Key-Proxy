import requests
from bs4 import BeautifulSoup
import re
import json


def debug_lm_arena():
    url = "https://lmarena.ai/leaderboard/webdev"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    }
    resp = requests.get(url, headers=headers, timeout=15)
    print(f"Status: {resp.status_code}")

    soup = BeautifulSoup(resp.text, "html.parser")

    # Check for structured data first
    json_match = re.search(
        r"window\.leaderboard\s*=\s*(\[.*?\]);", resp.text, re.DOTALL
    )
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            print(f"Found window.leaderboard with {len(data)} entries")
            for i, item in enumerate(data[:5]):
                print(
                    f"  {i + 1}. {item.get('model', 'N/A')}: {item.get('score', item.get('elo', 'N/A'))}"
                )
            return data
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")

    # Check for NEXT_DATA
    next_match = re.search(
        r"window\.__NEXT_DATA__[^>]*>(.*?)</script>", resp.text, re.DOTALL
    )
    if next_match:
        try:
            data = json.loads(next_match.group(1))
            print("Found __NEXT_DATA__")
            props = data.get("props", {})
            page_props = props.get("pageProps", {})

            # Try different paths to find leaderboard data
            leaderboard = page_props.get("leaderboard", [])
            if not leaderboard:
                leaderboard = page_props.get("models", [])
            if not leaderboard:
                # Check for embedded JSON in pageContent
                page_content = page_props.get("pageContent", "")
                embedded = re.search(r'"leaderboard"\s*:\s*(\[.*?\])', page_content)
                if embedded:
                    leaderboard = json.loads(embedded.group(1))

            print(f"Found {len(leaderboard)} models in NEXT_DATA")
            for i, item in enumerate(leaderboard[:5]):
                print(f"  {i + 1}. {item}")
            return leaderboard
        except (json.JSONDecodeError, TypeError) as e:
            print(f"NEXT_DATA parse error: {e}")

    print("No structured data found, checking table...")
    table = soup.find("table")
    if table:
        tbody = table.find("tbody")
        if tbody:
            rows = tbody.find_all("tr")
            print(f"Found table with {len(rows)} rows")

            results = []
            for row in rows[:10]:  # Show first 10
                cols = row.find_all(["td", "th"])
                if len(cols) >= 2:
                    # Model name is usually in first column
                    model_link = cols[0].find("a")
                    model_name = (
                        model_link.get_text(strip=True)
                        if model_link
                        else cols[0].get_text(strip=True)
                    )
                    model_name = re.sub(
                        r"^\d+\s*", "", model_name
                    )  # Remove leading rank

                    # Score is in second column
                    score_text = cols[1].get_text(strip=True)
                    score_match = re.search(r"(\d+\.?\d*)", score_text)
                    score = float(score_match.group(1)) if score_match else 0

                    print(f"  Model: {model_name}, ELO: {score}")
                    results.append({"model": model_name, "elo": score})

            print(f"\nTotal parsed from table: {len(results)}")
            return results
    else:
        print("No table found")

    return []


if __name__ == "__main__":
    debug_lm_arena()
