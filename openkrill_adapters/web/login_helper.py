"""Login Helper — opens a real browser for you to log in, then saves cookies.

Usage:
    python -m openkrill_adapters.web.login_helper --site chatgpt
    python -m openkrill_adapters.web.login_helper --site gemini
    python -m openkrill_adapters.web.login_helper --site deepseek

After you log in manually, press Enter in the terminal. The script saves
cookies to a JSON file and prints them for pasting into OpenKrill.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from playwright.async_api import async_playwright

# Ensure site drivers are registered
import openkrill_adapters.web.sites  # noqa: F401
from openkrill_adapters.web.site_driver import SiteDriverRegistry

COOKIES_DIR = Path.home() / ".openkrill" / "cookies"


async def run_login(site_name: str, output: str | None) -> None:
    driver_class = SiteDriverRegistry.get(site_name)
    driver = driver_class()

    print(f"\n  Opening {driver.base_url} in browser...")
    print("  Please log in manually. When you're done, come back here and press Enter.\n")

    pw = await async_playwright().start()
    browser = await pw.chromium.launch(headless=False)
    context = await browser.new_context()
    page = await context.new_page()

    await page.goto(driver.base_url, wait_until="domcontentloaded")

    # Wait for user to finish logging in
    input("  >>> Press Enter after you've logged in... ")

    # Check login status
    logged_in = await driver.is_logged_in(page)
    if not logged_in:
        print("  Warning: login check failed — saving cookies anyway (they might still work).\n")

    # Save cookies
    cookies = await context.cookies()
    cookies_json = json.dumps(cookies, indent=2, ensure_ascii=False)

    # Determine output path
    if output:
        out_path = Path(output)
    else:
        COOKIES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = COOKIES_DIR / f"{site_name}.json"

    out_path.write_text(cookies_json)
    print(f"  Cookies saved to: {out_path}")
    print(f"  ({len(cookies)} cookies captured)\n")

    # Also print compact JSON for pasting into OpenKrill
    compact = json.dumps(cookies, ensure_ascii=False)
    print("  --- Copy the line below into OpenKrill agent config 'cookies' field ---")
    print(compact)
    print("  --- End ---\n")

    await browser.close()
    await pw.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Log in to an AI chat site and save cookies")
    parser.add_argument(
        "--site",
        required=True,
        choices=SiteDriverRegistry.available(),
        help="Site to log in to",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: ~/.openkrill/cookies/<site>.json)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(run_login(args.site, args.output))
    except KeyboardInterrupt:
        print("\n  Cancelled.")
        sys.exit(1)


if __name__ == "__main__":
    main()
