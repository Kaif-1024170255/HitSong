"""Test tab order and focus rings accurately."""

import os
from playwright.sync_api import sync_playwright

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
ARTIFACT_DIR = r"C:\Users\Kaif Ahmad\.gemini\antigravity\brain\0451bf9d-63e8-44b9-ba7c-36d9d28e9457\screenshots"

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=CHROME_PATH, headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto("http://127.0.0.1:8000")
    page.wait_for_selector("#status-dot.active")

    # 1. Focus on slider-tempo directly and capture focus ring
    page.focus("#slider-tempo")
    page.locator("#slider-tempo").screenshot(path=os.path.join(ARTIFACT_DIR, "focus_slider_tempo.png"))
    print("Screenshot saved: focus_slider_tempo.png")

    # 2. Focus on preset button and capture focus ring
    page.focus("button[data-preset='pop']")
    page.locator(".examples-group").screenshot(path=os.path.join(ARTIFACT_DIR, "focus_preset_group.png"))
    print("Screenshot saved: focus_preset_group.png")

    # 3. Focus on genre select and capture
    page.focus("#genre-select")
    page.locator(".genre-group").screenshot(path=os.path.join(ARTIFACT_DIR, "focus_genre.png"))
    print("Screenshot saved: focus_genre.png")

    # 4. Tab order sequence test starting from first focusable element
    print("\nVerifying Tab sequence across all interactive controls:")
    page.focus("#slider-tempo")
    current_id = page.evaluate("() => document.activeElement.id")
    print(f"Start: id='{current_id}'")

    for i in range(12):
        page.keyboard.press("Tab")
        el_info = page.evaluate("() => ({ tag: document.activeElement.tagName, id: document.activeElement.id, text: document.activeElement.innerText || document.activeElement.value })")
        print(f"Tab {i+1}: <{el_info['tag']}> id='{el_info['id']}' text='{el_info['text']}'")

    browser.close()
