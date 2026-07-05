"""Capture the README hero from the live app via Playwright.

Sends one message that adds a source and asks about it, so the shot shows the
agent's tool cards and a grounded, cited answer. Run:

    .venv/Scripts/python.exe scripts/capture_hero.py [url] [out]
"""

import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "https://embedchain-fastdash.fly.dev/"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/hero.png"
# Two turns — add, then ask — so the transcript is clean (no wasted first search).
ADD = (
    "Add this note: The James Webb Space Telescope launched on December 25, 2021 "
    "and observes the universe in infrared light."
)
ASK = "When did it launch, and what does it observe?"


def send(page, text, wait_ms):
    page.fill("#chat-input", text)
    page.press("#chat-input", "Enter")
    page.wait_for_timeout(wait_ms)


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(URL, wait_until="load", timeout=60000)
    page.wait_for_selector("#chat-input", timeout=30000)
    send(page, ADD, 9000)          # ingest
    send(page, ASK, 14000)         # retrieve + answer
    page.screenshot(path=OUT)
    browser.close()

print("saved", OUT)
