"""
Europeana Image Dataset Downloader (API-based)
===============================================
Downloads images from the Europeana photography collection using the official
Search API with cursor-based pagination. Much faster and more reliable than
web scraping, with access to full-resolution images.

Usage:
    1. Set your API key below (get one at https://pro.europeana.eu/page/get-api)
    2. Adjust SEARCH_QUERIES and MAX_IMAGES as needed
    3. Run:  python Dataset_maker.py
"""

import requests
import os
import time
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

API_KEY = "ppybradmonce"  # <-- Replace with your personal Europeana API key

# Output directory for downloaded images
OUTPUT_DIR = "europeana_images"

# Maximum images to download per search query (set to None for unlimited)
MAX_IMAGES = 5000

# Download full-resolution images (True) or 400px thumbnails (False)
DOWNLOAD_FULL_RES = False

# Number of concurrent image download threads
MAX_WORKERS = 10

# Delay between API search requests in seconds (courteous rate limiting)
API_DELAY = 0.15  # 150ms — safe and fast

# Results per API page (max allowed is 100)
ROWS_PER_PAGE = 100

# Timeout for downloading individual images (seconds)
DOWNLOAD_TIMEOUT = 15

# Max retries for failed image downloads
MAX_RETRIES = 2

# Search queries — photography collection with open reusability
SEARCH_QUERIES = [
    {
        "query": "*",  # All images
        "filters": [
            "collection:photography",
            'TYPE:"IMAGE"',
            'MIME_TYPE:image/jpeg',
            'contentTier:"4"',  # Highest quality tier
            # Public domain only (PDM 1.0 or CC0)
            'RIGHTS:"http://creativecommons.org/publicdomain/mark/1.0/" OR RIGHTS:"http://creativecommons.org/publicdomain/zero/1.0/"',
        ],
        "label": "photography",
    },
]

# Europeana Search API base URL
API_BASE = "https://api.europeana.eu/record/v2/search.json"


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def fetch_search_page(query: str, filters: list, cursor: str = "*") -> dict:
    """
    Fetch a single page of search results from the Europeana API.

    Args:
        query: Search query string
        filters: List of qf (query filter) values
        cursor: Cursor string for pagination ("*" for first page)

    Returns:
        JSON response dict with 'items', 'nextCursor', 'totalResults'
    """
    params = {
        "wskey": API_KEY,
        "query": query,
        "rows": ROWS_PER_PAGE,
        "cursor": cursor,
        "profile": "rich",       # Includes full media URLs
        "reusability": "open",   # Only openly licensed content
    }

    # Add query filters
    qf_list = filters if filters else []
    if qf_list:
        params["qf"] = qf_list

    response = requests.get(API_BASE, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def get_image_url(item: dict) -> str | None:
    """
    Extract the best image URL from an API result item.

    Args:
        item: A single item dict from the API response

    Returns:
        Image URL string, or None if no suitable URL found
    """
    if DOWNLOAD_FULL_RES:
        # Try full-resolution sources first
        for field in ("edmIsShownBy", "edmHasView"):
            urls = item.get(field)
            if urls:
                url = urls[0] if isinstance(urls, list) else urls
                if url and url.startswith("http"):
                    return url

    # Fall back to thumbnail preview
    previews = item.get("edmPreview")
    if previews:
        url = previews[0] if isinstance(previews, list) else previews
        if url and url.startswith("http"):
            return url

    return None


def download_image(url: str, filepath: str) -> bool:
    """
    Download a single image with retry logic.

    Args:
        url: Image URL to download
        filepath: Local path to save the image

    Returns:
        True if download succeeded, False otherwise
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
            resp.raise_for_status()

            # Verify it looks like an image
            content_type = resp.headers.get("content-type", "")
            if "image" not in content_type and "octet-stream" not in content_type:
                return False

            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True

        except (requests.RequestException, IOError) as e:
            if attempt < MAX_RETRIES:
                time.sleep(0.5 * (attempt + 1))  # Backoff
            else:
                print(f"  ✗ Failed ({attempt + 1} tries): {url[:80]}... — {e}")
                return False

    return False


def scrape_category(query: str, filters: list, label: str, max_images: int | None):
    """
    Download images for a single search query/category.

    Args:
        query: Search query string
        filters: List of qf filter values
        label: Category label for file naming
        max_images: Maximum images to download (None = unlimited)
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Category: {label.upper()}")
    print(f"  Query: {query}")
    print(f"  Filters: {filters}")
    print(f"  Max images: {max_images or 'unlimited'}")
    print(f"{'=' * 60}")

    # First request to get total count
    cursor = "*"
    total_downloaded = 0
    total_failed = 0
    total_results = None
    page_num = 0

    while True:
        page_num += 1

        # Check if we've hit our max
        if max_images and total_downloaded >= max_images:
            print(f"\n  ✓ Reached max of {max_images} images.")
            break

        # Fetch a page of results
        try:
            data = fetch_search_page(query, filters, cursor)
        except requests.RequestException as e:
            print(f"\n  ✗ API error on page {page_num}: {e}")
            break

        # On first page, show total results
        if total_results is None:
            total_results = data.get("totalResults", 0)
            effective_max = min(max_images, total_results) if max_images else total_results
            print(f"  Total matching results: {total_results:,}")
            print(f"  Will download up to: {effective_max:,}")
            print()

        items = data.get("items", [])
        if not items:
            print(f"\n  No more results on page {page_num}.")
            break

        # Prepare download tasks: (url, filepath) tuples
        download_tasks = []
        for item in items:
            if max_images and total_downloaded + len(download_tasks) >= max_images:
                break

            img_url = get_image_url(item)
            if not img_url:
                continue

            idx = total_downloaded + len(download_tasks) + 1
            ext = "jpg"  # Default; Europeana photography is mostly JPEG
            filename = f"{label}_{idx:05d}.{ext}"
            filepath = os.path.join(OUTPUT_DIR, filename)

            # Skip if already exists
            if os.path.exists(filepath):
                continue

            download_tasks.append((img_url, filepath))

        # Download concurrently
        if download_tasks:
            batch_ok = 0
            batch_fail = 0

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(download_image, url, path): (url, path)
                    for url, path in download_tasks
                }
                for future in as_completed(futures):
                    if future.result():
                        batch_ok += 1
                    else:
                        batch_fail += 1

            total_downloaded += batch_ok
            total_failed += batch_fail

            effective_max = min(max_images, total_results) if max_images else total_results
            print(
                f"  Page {page_num:>4} | "
                f"Batch: +{batch_ok} downloaded, {batch_fail} failed | "
                f"Total: {total_downloaded:,}/{effective_max:,}"
            )

        # Get next cursor for pagination
        next_cursor = data.get("nextCursor")
        if not next_cursor:
            print(f"\n  ✓ No more pages (end of results).")
            break
        cursor = next_cursor

        # Courteous delay between API requests
        time.sleep(API_DELAY)

    print(f"\n  DONE: {total_downloaded:,} downloaded, {total_failed:,} failed")
    return total_downloaded


# ============================================================================
# MAIN
# ============================================================================

def main():
    if API_KEY == "YOUR_API_KEY":
        print("ERROR: Please set your Europeana API key in the API_KEY variable.")
        print("Get one at: https://pro.europeana.eu/page/get-api")
        sys.exit(1)

    print("Europeana Image Dataset Downloader")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Download mode: {'Full resolution' if DOWNLOAD_FULL_RES else 'Thumbnails (400px)'}")
    print(f"Workers: {MAX_WORKERS} | API delay: {API_DELAY}s | Rows/page: {ROWS_PER_PAGE}")

    start_time = time.time()
    grand_total = 0

    for search in SEARCH_QUERIES:
        count = scrape_category(
            query=search["query"],
            filters=search.get("filters", []),
            label=search["label"],
            max_images=MAX_IMAGES,
        )
        grand_total += count

    elapsed = time.time() - start_time
    mins, secs = divmod(int(elapsed), 60)

    print(f"\n{'=' * 60}")
    print(f"  ALL DONE!")
    print(f"  Total images: {grand_total:,}")
    print(f"  Time: {mins}m {secs}s")
    print(f"  Saved to: {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
