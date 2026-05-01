"""
Smoke tests for usgsUtils — runs the parsers against captured HTML so we
can verify them without internet access in the sandbox.
"""

from usgsUtils import (
    parse_listing_page,
    extract_original_image_url,
    title_matches_species,
    build_search_url,
)

# Trimmed snippet from a real listing page (the one Claude fetched earlier).
LISTING_SNIPPET = """
<a href="https://www.usgs.gov/media/images/agapostemon-angelicus-m-face-pennington-county-sd">
  Agapostemon angelicus, M, face, Pennington County, SD
</a>
<a href="https://www.usgs.gov/media/images/agapostemon-femoratusm-side-white-pine-conv">
  Agapostemon femoratus,M, Side, White Pine Co,NV
</a>
<a href="https://www.usgs.gov/media/images/acmaeodera-virgo-u-back-kruger-national-park-south-africa-0">
  Acmaeodera virgo, u, back, Kruger National Park, South Africa
</a>
<!-- Same link appearing twice, as the site does for responsive cards -->
<a href="https://www.usgs.gov/media/images/agapostemon-angelicus-m-face-pennington-county-sd">
  <img src="...">
</a>
"""

DETAIL_SNIPPET = """
<img src="https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/styles/full_width/public/Agapostemon%20angelicus%2C%20M%2C%20face%2C%20Pennington%20County%2C%20SD_2012-11-13-10.39.30%20ZS%20PMax.jpg?itok=GQoDSZZJ">
<a href="https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/Agapostemon%20angelicus%2C%20M%2C%20face%2C%20Pennington%20County%2C%20SD_2012-11-13-10.39.30%20ZS%20PMax.jpg">Original</a>
<a href="https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/styles/thumbnail/public/Agapostemon%20angelicus%2C%20M%2C%20face%2C%20Pennington%20County%2C%20SD_2012-11-13-10.39.30%20ZS%20PMax.jpg?itok=l4am1TQa">Thumbnail</a>
<a href="https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/styles/medium/public/Agapostemon%20angelicus%2C%20M%2C%20face%2C%20Pennington%20County%2C%20SD_2012-11-13-10.39.30%20ZS%20PMax.jpg?itok=ozn3Y0V7">Medium</a>
"""

# Detail page that's missing the "Original" anchor — should fall back to deriving
# from the full-width <img>.
DETAIL_SNIPPET_NO_ORIGINAL = """
<img src="https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/styles/full_width/public/Andrena_miserabilis_F_face_MD.jpg?itok=ABC123">
"""


def test_search_url():
    url = build_search_url("Andrena miserabilis", 2)
    assert "search_api_fulltext=Andrena+miserabilis" in url
    assert "page=2" in url
    print("✓ build_search_url")


def test_listing_parser():
    results = parse_listing_page(LISTING_SNIPPET)
    urls = {r["detail_url"] for r in results}
    assert len(results) == 3, f"Expected 3 unique entries, got {len(results)}"
    assert "https://www.usgs.gov/media/images/agapostemon-angelicus-m-face-pennington-county-sd" in urls
    titles = {r["title"] for r in results}
    assert any("Agapostemon angelicus" in t for t in titles)
    print(f"✓ parse_listing_page returned {len(results)} entries:")
    for r in results:
        print(f"    {r['title'][:60]!r:<65} -> {r['detail_url'][-60:]}")


def test_detail_extractor_with_original():
    url = extract_original_image_url(DETAIL_SNIPPET)
    assert url is not None
    assert "?itok=" not in url, "Original URL should not have a token suffix"
    assert "/styles/" not in url, "Original URL should not be a styled variant"
    assert url.endswith(".jpg")
    print(f"✓ extract_original_image_url (with Original anchor): {url[:80]}...")


def test_detail_extractor_fallback():
    url = extract_original_image_url(DETAIL_SNIPPET_NO_ORIGINAL)
    assert url is not None
    assert "?itok=" not in url
    assert "/styles/" not in url
    assert "/public/Andrena_miserabilis_F_face_MD.jpg" in url
    print(f"✓ extract_original_image_url (fallback): {url}")


def test_strict_match():
    assert title_matches_species("Andrena miserabilis, F, face, MD", "Andrena miserabilis")
    assert title_matches_species("andrena miserabilis", "Andrena miserabilis")
    assert title_matches_species("Andrena miserabilis (face)", "Andrena miserabilis")
    assert not title_matches_species("Andrena miserabilisalbinose, F", "Andrena miserabilis"), \
        "Should reject mid-word matches"
    assert not title_matches_species("On Andrena miserabilis", "Andrena miserabilis"), \
        "Should reject names that don't start at position 0"
    print("✓ title_matches_species")


if __name__ == "__main__":
    test_search_url()
    test_listing_parser()
    test_detail_extractor_with_original()
    test_detail_extractor_fallback()
    test_strict_match()
    print("\nAll smoke tests passed ✓")
