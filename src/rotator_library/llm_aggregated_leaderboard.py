# llm_aggregated_leaderboard.py
import requests
import os
from bs4 import BeautifulSoup
import json
import time
import pandas as pd
import numpy as np
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import cloudscraper
from retrying import retry
import warnings
from datetime import datetime
import re
import undetected_chromedriver as uc
import argparse
from fuzzywuzzy import fuzz, process

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
MAX_RETRIES = 3
SELENIUM_TIMEOUT = 30
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
]

# --- Helper Functions ---


@retry(
    stop_max_attempt_number=MAX_RETRIES,
    wait_exponential_multiplier=1000,
    wait_exponential_max=10000,
)
def fetch_with_retry(url, scraper=None, use_selenium=False, wait_for_element=None):
    """Fetches URL content with retries."""
    headers = {"User-Agent": np.random.choice(USER_AGENTS)}
    driver = None

    try:
        if use_selenium:
            options = uc.ChromeOptions()
            options.add_argument(f"user-agent={headers['User-Agent']}")
            driver = uc.Chrome(options=options, headless=True, use_subprocess=False)

            print(f"Fetching (Selenium): {url}")
            driver.get(url)

            if wait_for_element:
                try:
                    WebDriverWait(driver, SELENIUM_TIMEOUT).until(
                        EC.presence_of_element_located(wait_for_element)
                    )
                    time.sleep(5)  # Extra wait for JS
                except TimeoutException:
                    print(f"Timeout waiting for {wait_for_element} on {url}")

            return driver.page_source

        elif scraper:
            print(f"Fetching (Cloudscraper): {url}")
            response = scraper.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.text

        else:
            print(f"Fetching (Requests): {url}")
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.text

    except Exception as e:
        print(f"Request failed for {url}: {e}")
        raise
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass


def safe_float_convert(text):
    if not text:
        return None
    try:
        text = str(text).strip().replace("%", "").replace(",", "").replace("$", "")
        return float(re.sub(r"[^\d.eE+-]", "", text))
    except:
        return None


# --- Scrapers ---


def scrape_livebench_leaderboard():
    """Scrapes LiveBench."""
    url = "https://livebench.ai/#/"
    data = []
    try:
        html = fetch_with_retry(
            url, use_selenium=True, wait_for_element=(By.CSS_SELECTOR, "table")
        )
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if not table:
            return []

        # Simple extraction strategy: assume standard columns if headers match
        thead = table.find("thead")
        if not thead:
            return []
        headers = [th.get_text(strip=True).lower() for th in thead.find_all("th")]

        # Find indices
        try:
            model_idx = next(i for i, h in enumerate(headers) if "model" in h)
            score_idx = next(
                i
                for i, h in enumerate(headers)
                if "global average" in h or "average" in h
            )
            coding_idx = next((i for i, h in enumerate(headers) if "coding" in h), None)
        except StopIteration:
            print("Could not map LiveBench headers")
            return []

        tbody = table.find("tbody")
        rows = tbody.find_all("tr") if tbody else table.find_all("tr")[1:]

        for row in rows:
            cols = row.find_all("td")
            if len(cols) <= max(model_idx, score_idx):
                continue

            model = cols[model_idx].get_text(strip=True)
            # Prioritize coding score if available, else global
            score_val = (
                cols[coding_idx].get_text(strip=True)
                if coding_idx
                else cols[score_idx].get_text(strip=True)
            )
            score = safe_float_convert(score_val)

            if model and score:
                data.append(
                    (
                        model,
                        score,
                        "LiveBench Coding" if coding_idx else "LiveBench Global",
                        "livebench",
                    )
                )

        print(f"Scraped {len(data)} from LiveBench")
        return data
    except Exception as e:
        print(f"LiveBench scrape error: {e}")
        return []


def scrape_aider_leaderboard():
    """Scrapes Aider."""
    url = "https://aider.chat/docs/leaderboards/"
    data = []
    try:
        scraper = cloudscraper.create_scraper()
        html = fetch_with_retry(url, scraper=scraper)
        soup = BeautifulSoup(html, "html.parser")

        # Look for the main results table
        tables = soup.find_all("table")
        target_table = tables[0] if tables else None

        if not target_table:
            return []

        rows = target_table.find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            if not cols:
                continue

            # Aider usually has Model in col 0 or 1, and Score in col 1 or 2
            # Heuristic: Find the text column and the number column
            if len(cols) < 2:
                continue
            model = cols[0].get_text(strip=True)
            score_text = cols[1].get_text(strip=True)

            # Swap if model looks like a number (unlikely but safe)
            if safe_float_convert(model) and not safe_float_convert(score_text):
                model, score_text = score_text, model

            score = safe_float_convert(score_text)

            if model and score:
                data.append((model, score, "Aider Score", "aider"))

        print(f"Scraped {len(data)} from Aider")
        return data
    except Exception as e:
        print(f"Aider scrape error: {e}")
        return []


def scrape_artificial_analysis():
    """Scrapes Artificial Analysis (Models Page)."""
    url = "https://artificialanalysis.ai/leaderboards/models"
    data = []
    try:
        scraper = cloudscraper.create_scraper()
        html = fetch_with_retry(url, scraper=scraper)
        soup = BeautifulSoup(html, "html.parser")

        table = soup.find("table")
        if not table:
            return []

        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]

        try:
            model_idx = next(i for i, h in enumerate(headers) if "model" in h)
            # Look for coding index or overall
            coding_idx = next((i for i, h in enumerate(headers) if "coding" in h), None)
            score_idx = next(
                (
                    i
                    for i, h in enumerate(headers)
                    if "intelligence" in h or "score" in h
                ),
                -1,
            )

            # Look for TPS
            tps_idx = next(
                (
                    i
                    for i, h in enumerate(headers)
                    if "tokens per second" in h or "output speed" in h
                ),
                None,
            )

            idx_to_use = coding_idx if coding_idx is not None else score_idx
            if idx_to_use == -1:
                return []

        except StopIteration:
            return []

        rows = table.find_all("tr")[1:]
        for row in rows:
            cols = row.find_all("td")
            if len(cols) <= max(model_idx, idx_to_use):
                continue

            model = cols[model_idx].get_text(strip=True)
            score = safe_float_convert(cols[idx_to_use].get_text(strip=True))

            tps = 0
            if tps_idx and len(cols) > tps_idx:
                tps = safe_float_convert(cols[tps_idx].get_text(strip=True)) or 0

            if model and score:
                # Store TPS in context if available
                context = f"TPS:{tps}" if tps else ""
                data.append((model, score, "AA Coding", "artificial_analysis", context))

        print(f"Scraped {len(data)} from Artificial Analysis")
        return data
    except Exception as e:
        print(f"AA scrape error: {e}")
        return []


def scrape_swe_rebench():
    """Scrapes SWE-rebench."""
    url = "https://swe-rebench.com/"
    data = []
    try:
        html = fetch_with_retry(url)
        soup = BeautifulSoup(html, "html.parser")

        table = soup.find("table")
        if not table:
            return []

        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]

        try:
            model_idx = next(i for i, h in enumerate(headers) if "model" in h)
            score_idx = next(i for i, h in enumerate(headers) if "resolved rate" in h)
        except StopIteration:
            print("Could not map SWE-rebench headers")
            return []

        rows = table.find_all("tr")[1:]
        for row in rows:
            cols = row.find_all("td")
            if len(cols) <= max(model_idx, score_idx):
                continue

            model = cols[model_idx].get_text(strip=True)
            score_text = cols[score_idx].get_text(strip=True)
            score = safe_float_convert(score_text)

            if model and score is not None:
                data.append((model, score, "SWE-rebench Resolved", "swe-rebench"))

        print(f"Scraped {len(data)} from SWE-rebench")
        return data
    except Exception as e:
        print(f"SWE-rebench scrape error: {e}")
        return []


def scrape_swe_bench_bash():
    """Scrapes SWE-bench Bash Only."""
    url = "https://swebench.com/bash-only.html"
    data = []
    try:
        # Use cloudscraper via fetch_with_retry if needed, but requests might work
        scraper = cloudscraper.create_scraper()
        html = fetch_with_retry(url, scraper=scraper)
        soup = BeautifulSoup(html, "html.parser")

        script = soup.find("script", {"id": "leaderboard-data"})
        if not script or not script.string:
            print("SWE-bench Bash: No data script found")
            return []

        json_data = json.loads(str(script.string))
        if not json_data or not isinstance(json_data, list):
            return []

        # Access the first item's results (usually one item for the leaderboard)
        results = json_data[0].get("results", [])

        for item in results:
            model = item.get("name")
            per_instance = item.get("per_instance_details", {})
            if not per_instance:
                continue

            total = len(per_instance)
            resolved = sum(1 for v in per_instance.values() if v.get("resolved"))

            if total > 0:
                score = (resolved / total) * 100.0
                # Round to 2 decimals
                score = round(score, 2)
                data.append((model, score, "SWE-bench Bash", "swe-bench"))

        print(f"Scraped {len(data)} from SWE-bench Bash")
        return data
    except Exception as e:
        print(f"SWE-bench Bash scrape error: {e}")
        return []


def scrape_swe_bench_verified():
    url = "https://www.swebench.com/"
    data = []
    try:
        scraper = cloudscraper.create_scraper()
        html = fetch_with_retry(url, scraper=scraper)
        soup = BeautifulSoup(html, "html.parser")

        script = soup.find("script", {"id": "leaderboard-data"})
        if not script or not script.string:
            print("SWE-bench Verified: No data script found")
            return []

        json_data = json.loads(str(script.string))
        if not json_data or not isinstance(json_data, list):
            return []

        verified_board = next(
            (item for item in json_data if item.get("name") == "Verified"), None
        )
        if not verified_board:
            print("SWE-bench Verified: 'Verified' leaderboard not found in JSON")
            return []

        results = verified_board.get("results", [])

        for item in results:
            model = item.get("name")
            per_instance = item.get("per_instance_details", {})
            if not per_instance:
                continue

            total = len(per_instance)
            resolved = sum(1 for v in per_instance.values() if v.get("resolved"))

            if total > 0:
                score = (resolved / total) * 100.0
                score = round(score, 2)
                data.append((model, score, "SWE-bench Verified", "swe-bench"))

        print(f"Scraped {len(data)} from SWE-bench Verified")
        return data
    except Exception as e:
        print(f"SWE-bench Verified scrape error: {e}")
        return []


def scrape_ts_bench():
    """Scrapes TS Bench from GitHub README."""
    url = "https://raw.githubusercontent.com/laiso/ts-bench/main/README.md"
    data = []
    try:
        text = fetch_with_retry(url)
        # Extract leaderboard section
        start_marker = "<!-- BEGIN_LEADERBOARD -->"
        end_marker = "<!-- END_LEADERBOARD -->"
        start = text.find(start_marker)
        end = text.find(end_marker)

        if start == -1 or end == -1:
            print("TS Bench: Could not find leaderboard markers")
            return []

        table_text = text[start + len(start_marker) : end].strip()

        # Parse markdown table
        lines = table_text.split("\n")
        # Skip header and separator (usually first 2 lines)
        data_lines = [
            l
            for l in lines
            if l.strip().startswith("|") and "---" not in l and "Rank" not in l
        ]

        for line in data_lines:
            # Format: | Rank | Agent | Model | Success Rate | ...
            parts = [p.strip() for p in line.split("|")]
            # parts[0] is empty, parts[1] is Rank, parts[2] is Agent, parts[3] is Model, parts[4] is Success Rate
            if len(parts) >= 5:
                agent = parts[2]
                model_name = parts[3]
                success_rate_str = parts[4].replace("*", "").replace("%", "")

                full_name = f"{model_name} ({agent})"
                score = safe_float_convert(success_rate_str)

                if score is not None:
                    data.append((full_name, score, "TS Bench Success Rate", "ts-bench"))

        print(f"Scraped {len(data)} from TS Bench")
        return data
    except Exception as e:
        print(f"TS Bench scrape error: {e}")
        return []


def scrape_gso_bench():
    """Scrapes GSO Bench / LiveCodeBench (GSO)."""
    url = "https://gso-bench.github.io/assets/leaderboard.json"
    data = []
    try:
        # Fetch the JSON data directly
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        json_data = response.json()

        # Structure: {"models": [...]}
        models = json_data.get("models", [])

        for item in models:
            model_name = item.get("name")
            score = item.get("score")

            if model_name and score is not None:
                # Convert to percentage if it's 0-1
                if 0 <= float(score) <= 1.0:
                    score = float(score) * 100.0

                data.append(
                    (model_name, round(float(score), 2), "GSO Opt@1", "gso-bench")
                )

        print(f"Scraped {len(data)} from GSO Bench")
        return data
    except Exception as e:
        print(f"GSO Bench scrape error: {e}")
        return []


def scrape_vals_benchmarks():
    """Scrapes Vals.ai benchmarks (SWE-bench, Terminal, LCB) using Selenium."""
    benchmarks = [
        ("https://www.vals.ai/benchmarks/swebench", "Vals SWE-bench"),
        ("https://www.vals.ai/benchmarks/terminal-bench", "Vals Terminal"),
        ("https://www.vals.ai/benchmarks/lcb", "Vals LCB"),
    ]
    data = []

    for url, source_name in benchmarks:
        try:
            print(f"Scraping Vals.ai: {url}")
            # Use Selenium to wait for the model links to render
            html_content = fetch_with_retry(
                url,
                use_selenium=True,
                wait_for_element=(By.CSS_SELECTOR, 'a[href*="/models/"]'),
            )

            soup = BeautifulSoup(html_content, "html.parser")
            # The model entries are contained within link elements
            links = soup.select('a[href*="/models/"]')

            for link in links:
                # The text inside the link contains rank, model name, and scores on separate lines
                text = link.get_text(separator="\n").strip()
                lines = [line.strip() for line in text.split("\n") if line.strip()]

                # Robust parsing: Look for a pattern like [Rank, Model Name, Score, ...]
                # Usually: "1", "Model X", "75.0", "%", ...
                if len(lines) >= 3:
                    model = ""
                    score = None

                    # BDD-style check: find the first line that looks like a score (percentage)
                    # and assume the line immediately preceding it is the model name.
                    for i in range(1, len(lines)):
                        current_val = safe_float_convert(lines[i])
                        if current_val is not None:
                            # Verify if it's followed by '%' or looks like a primary score
                            model = lines[i - 1]
                            score = current_val
                            break

                    if (
                        model
                        and score is not None
                        and not model.isdigit()
                        and model != "Models"
                    ):
                        data.append((model, score, source_name, "vals_ai"))

        except Exception as e:
            print(f"Vals.ai scrape error for {url}: {e}")

    print(f"Scraped {len(data)} from Vals.ai")
    return data


def scrape_lm_arena():
    """
    Scrapes LM Arena (Web Development category).
    URL: https://lmarena.ai/leaderboard/webdev
    """
    url = "https://lmarena.ai/leaderboard/webdev"
    data = []

    try:
        print(f"Fetching LM Arena: {url}")
        # LM Arena often needs Selenium for JS rendering
        html = fetch_with_retry(
            url, use_selenium=True, wait_for_element=(By.CSS_SELECTOR, "table")
        )
        soup = BeautifulSoup(html, "html.parser")

        # Look for table structure
        table = soup.find("table")
        if table:
            # Try to parse table rows
            tbody = table.find("tbody")
            rows = tbody.find_all("tr") if tbody else table.find_all("tr")[1:]
            for row in rows:
                cols = row.find_all("td")
                if len(cols) >= 2:
                    model = cols[0].get_text(strip=True)
                    score_text = cols[1].get_text(strip=True)

                    # Try to extract numeric score
                    score_match = re.search(r"(\d+\.?\d*)", score_text)
                    if score_match:
                        score = float(score_match.group(1))

                        if model and score > 0:
                            # Normalize to 0-100 scale if needed (LM Arena ELO is usually around 1000-1200)
                            # For consistency with other benchmarks, we can normalize to 0-100
                            normalized_score = (
                                score / 1200
                            ) * 100  # Rough normalization
                            normalized_score = max(0, min(100, normalized_score))

                            data.append(
                                (
                                    model,
                                    round(normalized_score, 2),
                                    "LM Arena ELO",
                                    "lm-arena",
                                )
                            )

            print(f"Scraped {len(data)} from LM Arena table")
        else:
            # Check for __NEXT_DATA__ as fallback
            script = soup.find("script", {"id": "__NEXT_DATA__"})
            if script and script.string:
                arena_data = json.loads(script.string)
                # This part is highly dependent on their internal structure which changes
                # but let's try a generic search
                pass

    except Exception as e:
        print(f"LM Arena scrape error: {e}")

    return data


def normalize_model_name(name):
    """Normalize model name for matching."""
    if not name:
        return ""
    # Remove version numbers, dates, and special characters
    name = name.lower()
    name = re.sub(r"[^a-z0-9]", "", name)
    return name


def match_models(target_names, source_names, threshold=85):
    """Match model names across sources using fuzzy matching."""
    mapping = {}
    normalized_source = {normalize_model_name(name): name for name in source_names}
    source_keys = list(normalized_source.keys())

    for target in target_names:
        norm_target = normalize_model_name(target)
        if norm_target in normalized_source:
            mapping[target] = normalized_source[norm_target]
            continue

        result = process.extractOne(norm_target, source_keys, scorer=fuzz.ratio)

        if result and result[1] >= threshold:
            mapping[target] = normalized_source[result[0]]
        else:
            mapping[target] = None

    return mapping


def calculate_aggregate_scores(df):
    """Calculate aggregate and composite scores."""
    print("\nCalculating aggregate scores...")

    model_groups = df.groupby("Model")
    agg_df = model_groups.agg(
        {
            "Source": lambda x: ", ".join(sorted(set(x))),
            "Header": "count",
            "Context": lambda x: " | ".join(sorted(set(filter(None, x)))),
        }
    ).rename(columns={"Header": "Source_Count"})

    verified_scores = (
        df[df["Header"] == "SWE-bench Verified"].groupby("Model")["Score"].mean()
    )
    agg_df["swe_bench_verified"] = verified_scores

    def get_weighted_score(row):
        m_name = row.name
        m_data = df[df["Model"] == m_name]

        verified = m_data[m_data["Header"] == "SWE-bench Verified"]["Score"]
        others = m_data[m_data["Header"] != "SWE-bench Verified"]["Score"]

        if not verified.empty:
            v_val = verified.mean()
            if not others.empty:
                o_val = others.mean()
                return (v_val * 2 + o_val) / 3
            return v_val
        return others.mean() if not others.empty else np.nan

    agg_df["Quality_Score"] = agg_df.apply(get_weighted_score, axis=1)

    # 2. Extract TPS if available in Context
    def extract_tps(context):
        matches = re.findall(r"TPS:(\d+\.?\d*)", context)
        if matches:
            return np.mean([float(m) for m in matches])
        return np.nan

    agg_df["TPS"] = agg_df["Context"].apply(extract_tps)

    # 3. Normalize Quality Score (0-1)
    q_min, q_max = agg_df["Quality_Score"].min(), agg_df["Quality_Score"].max()
    if q_max > q_min:
        agg_df["Quality_Normalized"] = (agg_df["Quality_Score"] - q_min) / (
            q_max - q_min
        )
    else:
        agg_df["Quality_Normalized"] = 1.0

    # 4. Normalize TPS (0-1)
    tps_valid = agg_df["TPS"].dropna()
    if not tps_valid.empty:
        t_min, t_max = tps_valid.min(), tps_valid.max()
        if t_max > t_min:
            agg_df["TPS_Normalized"] = (agg_df["TPS"] - t_min) / (t_max - t_min)
        else:
            agg_df["TPS_Normalized"] = 1.0
    else:
        agg_df["TPS_Normalized"] = 0.0

    agg_df["TPS_Normalized"] = agg_df["TPS_Normalized"].fillna(0)

    # 5. Composite Score (70% Quality, 30% Speed)
    agg_df["Composite_Score"] = (
        0.7 * agg_df["Quality_Normalized"] + 0.3 * agg_df["TPS_Normalized"]
    )

    # Sort by Composite Score descending
    agg_df = agg_df.sort_values("Composite_Score", ascending=False)

    return agg_df.reset_index()


def main():
    parser = argparse.ArgumentParser(description="LLM Aggregated Leaderboard Scraper")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of entries per scraper"
    )
    parser.add_argument(
        "--include-arena", action="store_true", help="Include LM Arena scraper (slow)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="llm_aggregated_leaderboard.csv",
        help="Output CSV filename",
    )
    args = parser.parse_args()

    print("Starting Optimized Aggregated Leaderboard Scraper...")
    all_data = []

    # Run scrapers
    scrapers = [
        scrape_livebench_leaderboard,
        scrape_aider_leaderboard,
        scrape_artificial_analysis,
        scrape_swe_rebench,
        scrape_swe_bench_bash,
        scrape_swe_bench_verified,
        scrape_ts_bench,
        scrape_gso_bench,
        scrape_vals_benchmarks,
    ]

    if args.include_arena:
        scrapers.append(scrape_lm_arena)

    for scraper_func in scrapers:
        try:
            data = scraper_func()
            if args.limit:
                data = data[: args.limit]
            all_data.extend(data)
        except Exception as e:
            print(f"Error running {scraper_func.__name__}: {e}")

    print(f"\nTotal raw entries collected: {len(all_data)}")

    if not all_data:
        print("No data collected.")
        return

    # Normalize tuples (some have 4 items, some 5)
    normalized = []
    for item in all_data:
        if len(item) == 4:
            normalized.append(item + ("",))  # Add empty context
        else:
            normalized.append(item)

    df_raw = pd.DataFrame(
        normalized,
        columns=["Model", "Score", "Header", "Source", "Context"],  # type: ignore
    )

    # Calculate aggregate scores
    agg_df = calculate_aggregate_scores(df_raw)

    # Save to CSV
    agg_df.to_csv(args.output, index=False)
    print(f"Saved aggregated data to {args.output}")

    # Display Top 10
    print("\nTop 10 Models by Composite Score:")
    cols_to_show = ["Model", "Quality_Score"]
    if "swe_bench_verified" in agg_df.columns:
        cols_to_show.append("swe_bench_verified")
    cols_to_show.extend(["TPS", "Composite_Score"])

    print(agg_df[cols_to_show].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
