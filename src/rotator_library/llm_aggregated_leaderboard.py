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
    except (ValueError, TypeError):
        return None


# --- Scrapers ---


def scrape_livebench_leaderboard():
    """Scrapes LiveBench.

    Extracts multiple scores per model:
    - 'agentic coding average' column → LiveBench Agentic Coding
    - 'coding average' column → LiveBench Coding
    - 'global average' column → LiveBench Global
    - 'reasoning average' column → LiveBench Reasoning
    """
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

        # Find indices for model name and all relevant score columns
        try:
            model_idx = next(i for i, h in enumerate(headers) if "model" in h)
        except StopIteration:
            print("Could not find model column in LiveBench headers")
            return []

        # Map column names to our score categories
        score_columns = {}
        for i, h in enumerate(headers):
            if "agentic" in h and "coding" in h:
                score_columns["agentic_coding"] = i
            elif "coding" in h and "agentic" not in h:
                score_columns["coding"] = i
            elif "global" in h and "average" in h:
                score_columns["global"] = i
            elif "reasoning" in h and "average" in h:
                score_columns["reasoning"] = i
            elif h in ("average", "overall"):
                score_columns.setdefault("global", i)

        if not score_columns:
            print(f"Could not find score columns in LiveBench headers: {headers}")
            return []

        tbody = table.find("tbody")
        rows = tbody.find_all("tr") if tbody else table.find_all("tr")[1:]

        for row in rows:
            cols = row.find_all("td")
            if len(cols) <= model_idx:
                continue

            model = cols[model_idx].get_text(strip=True)
            if not model:
                continue

            # Extract each available score as a separate entry
            for score_type, col_idx in score_columns.items():
                if col_idx < len(cols):
                    score = safe_float_convert(cols[col_idx].get_text(strip=True))
                    if score is not None and score > 0:
                        label_map = {
                            "agentic_coding": "LiveBench Agentic Coding",
                            "coding": "LiveBench Coding",
                            "global": "LiveBench Global",
                            "reasoning": "LiveBench Reasoning",
                        }
                        data.append(
                            (
                                model,
                                score,
                                label_map.get(score_type, f"LiveBench {score_type}"),
                                "livebench",
                            )
                        )

        print(f"Scraped {len(data)} entries from LiveBench")
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
    """Scrapes Artificial Analysis (Models Page).

    Extracts:
    - Intelligence/coding score
    - TPS (output speed)
    - Agentic coding index (average of GDPval-AA and τ²-Bench Telecom
      columns, handling cases where a model has only one of the two)
    """
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

            # Look for agentic coding index components
            gdpval_idx = next(
                (i for i, h in enumerate(headers) if "gdpval" in h or "gdp" in h),
                None,
            )
            tau2_idx = next(
                (
                    i
                    for i, h in enumerate(headers)
                    if "τ²" in h or "tau2" in h or "t2-bench" in h or "telecom" in h
                ),
                None,
            )
            # Also look for a direct "agentic" column
            agentic_idx = next(
                (i for i, h in enumerate(headers) if "agentic" in h), None
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

            # Calculate agentic coding index from GDPval-AA and τ²-Bench Telecom
            agentic_score = None
            if agentic_idx and len(cols) > agentic_idx:
                agentic_score = safe_float_convert(
                    cols[agentic_idx].get_text(strip=True)
                )

            if agentic_score is None and (gdpval_idx or tau2_idx):
                gdpval = None
                tau2 = None
                if gdpval_idx and len(cols) > gdpval_idx:
                    gdpval = safe_float_convert(cols[gdpval_idx].get_text(strip=True))
                if tau2_idx and len(cols) > tau2_idx:
                    tau2 = safe_float_convert(cols[tau2_idx].get_text(strip=True))

                # Average of available scores (handle models missing one)
                components = [s for s in [gdpval, tau2] if s is not None]
                if components:
                    agentic_score = sum(components) / len(components)

            if model and score:
                # Store TPS and agentic score in context if available
                context_parts = []
                if tps:
                    context_parts.append(f"TPS:{tps}")
                if agentic_score is not None:
                    context_parts.append(f"AGENTIC:{agentic_score:.1f}")
                context = " ".join(context_parts)
                data.append((model, score, "AA Coding", "artificial_analysis", context))

                # Also emit agentic coding as a separate entry if found
                if agentic_score is not None:
                    data.append(
                        (
                            model,
                            agentic_score,
                            "AA Agentic Coding Index",
                            "artificial_analysis",
                            context,
                        )
                    )

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
    Scrapes LM Arena (Code category).
    URL: https://arena.ai/leaderboard/code
    (Formerly: https://lmarena.ai/leaderboard/webdev)
    """
    url = "https://arena.ai/leaderboard/code"
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
    """Normalize model name for matching.

    Handles naming variants like:
    - claude-opus-4-6 vs claude-4.6-opus
    - gpt-5.2 vs gpt-5-2
    - Gemini 2.5 Pro vs gemini-2-5-pro
    """
    if not name:
        return ""
    name = name.lower().strip()
    # Normalize separators: dots, underscores, spaces → hyphens
    name = re.sub(r"[._\s]+", "-", name)
    # Remove parenthetical suffixes like "(high)", "(low)", "(free)"
    name = re.sub(r"\([^)]*\)", "", name)
    # Remove non-alphanumeric except hyphens
    name = re.sub(r"[^a-z0-9\-]", "", name)
    # Collapse multiple hyphens
    name = re.sub(r"-+", "-", name).strip("-")
    return name


def _extract_model_tokens(name):
    """Extract canonical tokens from a normalized model name for order-insensitive matching.

    Splits on hyphens, groups version-number tokens together, and sorts
    alphabetic tokens so that 'claude-opus-4-6' and 'claude-4-6-opus'
    produce the same canonical set.
    """
    parts = name.split("-")
    alpha_tokens = sorted(p for p in parts if not p.isdigit())
    # Preserve numeric tokens in order (version numbers)
    num_tokens = [p for p in parts if p.isdigit()]
    return alpha_tokens, num_tokens


# Variant suffixes that distinguish fundamentally different model configurations
_VARIANT_SUFFIXES = {
    "thinking",
    "reasoning",
    "extended",
    "mini",
    "nano",
    "micro",
    "max",
}


def _has_variant_mismatch(name_a, name_b):
    """Check if two model names differ by a variant suffix.

    Returns True if one has a variant suffix the other lacks,
    which means they should NOT be matched together.
    E.g., 'claude-opus-4-5' vs 'claude-opus-4-5-thinking' → True (mismatch)
    """
    parts_a = set(name_a.split("-"))
    parts_b = set(name_b.split("-"))
    for suffix in _VARIANT_SUFFIXES:
        if (suffix in parts_a) != (suffix in parts_b):
            return True
    return False


def match_models(target_names, source_names, threshold=80):
    """Match model names across sources using fuzzy matching.

    Uses token_sort_ratio to handle reordered name components
    (e.g., claude-opus-4-6 vs claude-4.6-opus).
    Falls back to exact canonical-token matching before fuzzy.

    Reasoning/thinking variants are treated as distinct models
    and will NOT be matched to their non-reasoning counterpart.
    """
    mapping = {}
    normalized_source = {normalize_model_name(name): name for name in source_names}
    source_keys = list(normalized_source.keys())

    # Build canonical token index for exact structural matches
    canonical_index = {}
    for norm_name in source_keys:
        alpha, nums = _extract_model_tokens(norm_name)
        key = (tuple(alpha), tuple(nums))
        canonical_index.setdefault(key, []).append(norm_name)

    for target in target_names:
        norm_target = normalize_model_name(target)

        # 1. Exact normalized match
        if norm_target in normalized_source:
            mapping[target] = normalized_source[norm_target]
            continue

        # 2. Canonical token match (order-insensitive)
        t_alpha, t_nums = _extract_model_tokens(norm_target)
        t_key = (tuple(t_alpha), tuple(t_nums))
        if t_key in canonical_index:
            mapping[target] = normalized_source[canonical_index[t_key][0]]
            continue

        # 3. Fuzzy match with token_sort_ratio (handles reordering)
        result = process.extractOne(
            norm_target, source_keys, scorer=fuzz.token_sort_ratio
        )

        if result and result[1] >= threshold:
            # Guard: reject match if variant suffix mismatch
            if _has_variant_mismatch(norm_target, result[0]):
                mapping[target] = None
            else:
                mapping[target] = normalized_source[result[0]]
        else:
            mapping[target] = None

    return mapping


def calculate_aggregate_scores(df):
    """Calculate aggregate and composite scores.

    Handles missing data properly:
    - NaN scores are excluded from weighted averages (not treated as 0)
    - Models with no valid scores are dropped
    - Normalization uses only valid (non-NaN) values
    """
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

        # Drop NaN values before computing means
        verified = verified.dropna()
        others = others.dropna()

        if not verified.empty:
            v_val = verified.mean()
            if not others.empty:
                o_val = others.mean()
                return (v_val * 2 + o_val) / 3
            return v_val
        return others.mean() if not others.empty else np.nan

    agg_df["Quality_Score"] = agg_df.apply(get_weighted_score, axis=1)

    # Drop models with no valid scores
    agg_df = agg_df.dropna(subset=["Quality_Score"])

    if agg_df.empty:
        print("Warning: No models with valid quality scores")
        return agg_df.reset_index()

    # 2. Extract TPS if available in Context
    def extract_tps(context):
        matches = re.findall(r"TPS:(\d+\.?\d*)", context)
        if matches:
            return np.mean([float(m) for m in matches])
        return np.nan

    agg_df["TPS"] = agg_df["Context"].apply(extract_tps)

    # 3. Normalize Quality Score (0-1), using only valid values
    valid_quality = agg_df["Quality_Score"].dropna()
    q_min, q_max = valid_quality.min(), valid_quality.max()
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
