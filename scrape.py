from playwright.async_api import Page
from utils import *
from browser_utils import *
import asyncio


async def scrape_species(page: Page, species: str) -> str:
    '''Returns all src fields of all species images across paginated pages'''

    # Skip if we already have results for this species (resume support)
    out_path = Path('image_urls') / f'{species}.txt'
    if out_path.exists():
        print(f'{species} - already done, skipping')
        with open(out_path) as f:
            return species, [line.strip() for line in f if line.strip()]

    base_url = page.url

    all_srcs = []
    page_num = 1

    while True:
        # Replace page parameter
        paginated_url = re.sub(r'page=\d+', f'page={page_num}', base_url)

        # Retry up to 5 times on network errors
        last_err = None
        for attempt in range(5):
            try:
                await page.goto(paginated_url, wait_until="domcontentloaded", timeout=60000)
                last_err = None
                break
            except Exception as e:
                last_err = e
                wait = 2 ** attempt  # 1s, 2s, 4s, 8s, 16s
                print(f'{species} p.{page_num} attempt {attempt+1} failed: {e.__class__.__name__}, retrying in {wait}s')
                await asyncio.sleep(wait)

        if last_err is not None:
            print(f'{species} p.{page_num} GIVING UP after 5 attempts')
            break

        print(f'{species} - p. {page_num}')

        # Check for "no results" message
        no_results = page.locator(
            "h3:has-text('Your query did not return any results')"
        )

        if await no_results.first.is_visible():  # If no results, exit loop
            break

        # Wait for images to exist (important for dynamic pages)
        imgs = page.locator('img[alt="Image Associated With the Occurrence"]')

        if not await imgs.first.is_visible():
            break  # Exit if no images as fallback

        srcs = await imgs.evaluate_all(
            "imgs => imgs.map(img => img.src)"
        )

        all_srcs.extend(srcs)
        page_num += 1

    # Write this species' results immediately so progress isn't lost on crash
    if all_srcs:
        os.makedirs('image_urls', exist_ok=True)
        with open(out_path, 'w') as f:
            f.write('\n'.join(all_srcs))

    return species, all_srcs


async def main():
    with open('urls.txt') as f:
        urls = [line.strip() for line in f.readlines()]

    species = get_species()

    scraper = PlaywrightScraperPool(scrape_species)

    await scraper.start()

    url_args_pairs = [(url, (sp,)) for url, sp in zip(urls, species)]

    try:
        sp_srcs_pairs: List[Tuple[str, List[str]]] = await scraper.run(url_args_pairs)
    finally:
        await scraper.close()  # Always close scraper

    print('Done')


if __name__ == "__main__":
    asyncio.run(main())