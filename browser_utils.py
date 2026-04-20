import asyncio
from playwright.async_api import async_playwright
from typing import List, Tuple, Callable


class PlaywrightScraperPool:
    def __init__(self, scrape_action: Callable, max_concurrency=10):
        self.max_concurrency = max_concurrency
        self.scrape_action = scrape_action
        self.semaphore = asyncio.Semaphore(max_concurrency)


    async def start(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)


    async def close(self):
        await self.browser.close()
        await self.playwright.stop()


    async def scrape(self, url, scrape_args):
        async with self.semaphore:
            # Each task gets its own context (important for isolation)
            context = await self.browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            )
            page = await context.new_page()

            try:
                await page.goto(url, timeout=60000)
                return await self.scrape_action(page, *scrape_args)

            finally:
                await context.close()


    async def run(self, url_args_pairs: List[Tuple]):
        tasks = [asyncio.create_task(self.scrape(url, args)) for url, args in url_args_pairs]
        return await asyncio.gather(*tasks)