"""
usgsUtils.py — USGS Bee Lab image scraper.

Parallel to apiUtils.py (which targets GBIF). USGS does not expose a JSON API
for the Bee Lab image gallery, so this module scrapes the public search pages.

Workflow per species:
  1. Hit the search URL with `search_api_fulltext=<species>` and walk paginated
     listing pages until a page returns no results.
  2. From each listing page, extract `/media/images/<slug>` links along with
     the title text used as the link.
  3. For each detail page, fetch it once and pull out the `Original` full-size
     image URL (the big S3 link without the `?itok=...` style suffix).
  4. (Optional) Filter results so only those whose page title starts with the
     species name are kept — this trims out incidental matches from the search.

Two functions are exported:
  - get_bee_images(species_name)        — synchronous (debug / one-off use)
  - get_bee_images_async(species_name)  — async (used by main.py)
"""

import asyncio
import re
from typing import Dict, List, Optional
from urllib.parse import quote_plus, urljoin

import aiohttp
import requests
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = "https://www.usgs.gov"

# Image search endpoint. The EESC path covers the USGS Bee Lab images. Switching
# to e.g. /labs/biml/multimedia/images would scope the search differently.
SEARCH_PATH = "/centers/eesc/multimedia/images"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Strict matching — only keep images whose detail-page title starts with the
# species name. Set False to keep every image the search returns.
# "fallback" means: try strict first, and if it yields zero results, accept
# everything the search returned for that species (loose mode).
STRICT_MATCH = True
STRICT_FALLBACK_TO_LOOSE = True

# How many detail pages to fetch concurrently per species
DETAIL_CONCURRENCY = 10

# Max listing pages to walk per species. The site shows ~10 results per page,
# so 50 pages = up to ~500 results. Bumped high as a safety stop.
MAX_LISTING_PAGES = 50

# Substrings that, if found in an image URL, mean it's not a bee photo
# (org charts, logos, banners, generic page assets, etc.). All matches are
# case-insensitive. Add new entries here as you spot more junk.
URL_DENYLIST_SUBSTRINGS = [
    "org%20chart",
    "org_chart",
    "orgchart",
    "organizational%20chart",
    "organizational_chart",
    "logo",
    "banner",
    "letterhead",
    "infographic",
    "screenshot",
    "icon",
    "diagram",
]


def is_denylisted_url(url: str) -> bool:
    """True if the URL looks like a non-photo asset (org charts, logos, etc.)."""
    u = url.lower()
    return any(token in u for token in URL_DENYLIST_SUBSTRINGS)


# ---------------------------------------------------------------------------
# URL / HTML helpers
# ---------------------------------------------------------------------------

def build_search_url(species_name: str, page: int = 0) -> str:
    """Build the species-search URL for a given page number (0-indexed)."""
    qs = (
        f"media_image_type=All"
        f"&media_states_1="
        f"&media_release_date="
        f"&search_api_fulltext={quote_plus(species_name)}"
        f"&page={page}"
    )
    return f"{BASE_URL}{SEARCH_PATH}?{qs}"


def parse_listing_page(html: str) -> List[Dict[str, str]]:
    """
    Extract image entries from a search listing page.

    Returns a list of dicts: {"detail_url": "...", "title": "..."}.
    Each /media/images/<slug> link on the page is treated as one result, and
    duplicates within the same page (each card renders the same link multiple
    times for different breakpoints) are de-duped.
    """
    soup = BeautifulSoup(html, "html.parser")
    seen = set()
    results = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/media/images/" not in href:
            continue

        # Normalize to absolute URL and strip any fragment / trailing slash
        detail_url = urljoin(BASE_URL, href).split("#")[0].rstrip("/")
        if detail_url in seen:
            continue
        seen.add(detail_url)

        # The same link appears multiple times per card — pick the variant whose
        # text looks like a real title (longest non-empty text wins).
        title = (a.get_text(strip=True) or "").strip()
        results.append({"detail_url": detail_url, "title": title})

    # Merge titles: collapse entries that share a URL but pick the longest title
    merged: Dict[str, str] = {}
    for entry in results:
        url = entry["detail_url"]
        existing = merged.get(url, "")
        if len(entry["title"]) > len(existing):
            merged[url] = entry["title"]

    return [{"detail_url": url, "title": title} for url, title in merged.items()]


def extract_original_image_url(detail_html: str) -> Optional[str]:
    """
    Pull the 'Original' full-resolution image URL from a media detail page.

    On USGS detail pages there's a row of links labeled Original / Thumbnail /
    Medium below the preview. We want the Original.
    """
    soup = BeautifulSoup(detail_html, "html.parser")

    # Preferred: the explicit "Original" anchor
    for a in soup.find_all("a", href=True):
        if a.get_text(strip=True).lower() == "original":
            return a["href"]

    # Fallback: derive from the preview <img> by stripping the style folder
    # and the ?itok=... query string. e.g.
    # .../styles/full_width/public/foo.jpg?itok=ABC  ->  .../public/foo.jpg
    for img in soup.find_all("img", src=True):
        src = img["src"]
        if "d9-wret.s3" in src and "s3fs-public" in src:
            cleaned = re.sub(r"/styles/[^/]+/public/", "/public/", src)
            cleaned = cleaned.split("?")[0]
            return cleaned

    return None


def title_matches_species(title: str, species_name: str) -> bool:
    """
    Strict match: the title must begin with the species name (case-insensitive),
    optionally followed by a comma/space/colon/parenthesis. This catches titles
    like 'Andrena miserabilis, F, face, MD' but rejects titles where the
    species name only appears mid-string.
    """
    title_norm = re.sub(r"\s+", " ", title.strip().lower())
    species_norm = species_name.strip().lower()
    if not title_norm.startswith(species_norm):
        return False
    tail = title_norm[len(species_norm):]
    return tail == "" or tail[0] in ",.;: ()-/"


# ---------------------------------------------------------------------------
# Synchronous version (for quick testing / debugging)
# ---------------------------------------------------------------------------

def get_bee_images(species_name: str) -> List[Dict[str, str]]:
    """Synchronous scrape — handy for testing one species at a time."""
    session = requests.Session()
    session.headers.update(HEADERS)

    listings: List[Dict[str, str]] = []
    for page in range(MAX_LISTING_PAGES):
        url = build_search_url(species_name, page)
        print(f"Fetching listing: {url}")
        resp = session.get(url, timeout=30)
        if resp.status_code != 200:
            print(f"  Status {resp.status_code} — stopping pagination")
            break

        page_results = parse_listing_page(resp.text)
        if not page_results:
            print(f"  No results on page {page} — stopping pagination")
            break

        listings.extend(page_results)
        print(f"  +{len(page_results)} (running total {len(listings)})")

    results: List[Dict[str, str]] = []
    for entry in listings:
        if STRICT_MATCH and not title_matches_species(entry["title"], species_name):
            continue
        try:
            r = session.get(entry["detail_url"], timeout=30)
            if r.status_code != 200:
                continue
            img_url = extract_original_image_url(r.text)
            if img_url and not is_denylisted_url(img_url):
                results.append({"species": species_name, "image_url": img_url})
        except requests.RequestException as e:
            print(f"  Error on {entry['detail_url']}: {e}")

    if STRICT_MATCH and STRICT_FALLBACK_TO_LOOSE and not results:
        print(f"  Strict match returned 0 — falling back to loose for '{species_name}'")
        for entry in listings:
            try:
                r = session.get(entry["detail_url"], timeout=30)
                if r.status_code != 200:
                    continue
                img_url = extract_original_image_url(r.text)
                if img_url and not is_denylisted_url(img_url):
                    results.append({"species": species_name, "image_url": img_url})
            except requests.RequestException:
                continue

    print(f"Retrieved {len(results)} images for {species_name}\n")
    return results


# ---------------------------------------------------------------------------
# Async version (used by main.py)
# ---------------------------------------------------------------------------

async def _fetch_text(session: aiohttp.ClientSession, url: str) -> Optional[str]:
    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                return None
            return await resp.text()
    except Exception as e:
        print(f"  Fetch error {url}: {e}")
        return None


async def _fetch_detail(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    entry: Dict[str, str],
    species_name: str,
) -> Optional[Dict[str, str]]:
    async with semaphore:
        html = await _fetch_text(session, entry["detail_url"])
    if not html:
        return None
    img_url = extract_original_image_url(html)
    if not img_url:
        return None
    if is_denylisted_url(img_url):
        # Junk URL (org chart, logo, banner, etc.) — skip silently
        return None
    return {"species": species_name, "image_url": img_url, "title": entry["title"]}


async def get_bee_images_async(species_name: str) -> List[Dict[str, str]]:
    """
    Async scrape — drop-in replacement for the GBIF version in main.py.
    Returns dicts with 'species' and 'image_url' keys (matching the GBIF schema).
    """
    print(f"\n=== USGS search: {species_name} ===")

    connector = aiohttp.TCPConnector(limit=20)
    timeout = aiohttp.ClientTimeout(total=60)

    async with aiohttp.ClientSession(
        connector=connector, timeout=timeout, headers=HEADERS
    ) as session:

        # 1. Walk listing pages until a page is empty
        listings: List[Dict[str, str]] = []
        for page in range(MAX_LISTING_PAGES):
            search_url = build_search_url(species_name, page)
            print(f"  Listing page {page}: {search_url}")
            html = await _fetch_text(session, search_url)
            if html is None:
                break
            page_results = parse_listing_page(html)
            if not page_results:
                break

            # If results overlap with what we already have, the site has wrapped
            # around (i.e. page beyond the last real one). Stop.
            new = [r for r in page_results if r["detail_url"] not in {x["detail_url"] for x in listings}]
            if not new:
                break

            listings.extend(new)
            print(f"    +{len(new)} (total {len(listings)})")

        if not listings:
            print(f"  No listings found for {species_name}")
            return []

        # 2. Optionally apply strict species-name match
        if STRICT_MATCH:
            strict_listings = [l for l in listings if title_matches_species(l["title"], species_name)]
        else:
            strict_listings = listings

        target_listings = strict_listings
        if STRICT_MATCH and STRICT_FALLBACK_TO_LOOSE and not strict_listings:
            print(f"  Strict match: 0/{len(listings)} — falling back to loose")
            target_listings = listings

        # 3. Fetch each detail page concurrently and extract the Original URL
        sem = asyncio.Semaphore(DETAIL_CONCURRENCY)
        tasks = [_fetch_detail(session, sem, entry, species_name) for entry in target_listings]
        detail_results = await asyncio.gather(*tasks)

    results = [r for r in detail_results if r is not None]

    # Drop the 'title' key before returning, to match the GBIF output schema.
    cleaned = [{"species": r["species"], "image_url": r["image_url"]} for r in results]

    print(f"  ✓ {species_name}: {len(cleaned)} image URLs extracted "
          f"({len(target_listings)} listing matches, {len(listings)} total search hits)")
    return cleaned