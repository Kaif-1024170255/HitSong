"""Comprehensive Playwright verification script for Hit Song Predictor frontend."""

import os
import time
from playwright.sync_api import sync_playwright

ARTIFACT_DIR = r"C:\Users\Kaif Ahmad\.gemini\antigravity\brain\0451bf9d-63e8-44b9-ba7c-36d9d28e9457\screenshots"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
BASE_URL = "http://127.0.0.1:8000"


def run_verification():
    print("=== Starting Frontend UI Verification with Playwright ===")
    results = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME_PATH, headless=True)

        # ----------------------------------------------------
        # 1. Desktop Initial Load (1440px)
        # ----------------------------------------------------
        print("\n--- 1. Testing Initial Load at 1440px ---")
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(BASE_URL)
        page.wait_for_selector("#status-dot.active", timeout=10000)
        page.wait_for_selector(".final-model", timeout=10000)
        # Wait for numeral animation to finish
        time.sleep(1.0)

        initial_prob_1440 = page.inner_text("#prob-numeral")
        initial_verdict_1440 = page.inner_text("#verdict-text")
        print(f"1440px Initial Load: Probability = {initial_prob_1440}, Verdict = {initial_verdict_1440}")

        path_1440 = os.path.join(ARTIFACT_DIR, "initial_load_1440.png")
        page.screenshot(path=path_1440, full_page=True)
        print(f"Saved: {path_1440}")

        # ----------------------------------------------------
        # 2. Mobile Initial Load (390px)
        # ----------------------------------------------------
        print("\n--- 2. Testing Mobile Initial Load at 390px ---")
        page_mobile = browser.new_page(viewport={"width": 390, "height": 844})
        page_mobile.goto(BASE_URL)
        page_mobile.wait_for_selector("#status-dot.active", timeout=10000)
        time.sleep(1.0)

        path_390 = os.path.join(ARTIFACT_DIR, "initial_load_390.png")
        page_mobile.screenshot(path=path_390, full_page=True)
        print(f"Saved: {path_390}")
        page_mobile.close()

        # ----------------------------------------------------
        # 3. Move Three Sliders & Compare Before / After
        # ----------------------------------------------------
        print("\n--- 3. Testing Slider Adjustments ---")
        path_before = os.path.join(ARTIFACT_DIR, "sliders_before.png")
        page.locator(".content-grid").screenshot(path=path_before)
        prob_before = page.inner_text("#prob-numeral")
        dance_before = page.inner_text("#val-danceability")
        loud_before = page.inner_text("#val-loudness")
        inst_before = page.inner_text("#val-instrumentalness")
        print(f"Before move: Prob = {prob_before} (danceability: {dance_before}, loudness: {loud_before}, instrumentalness: {inst_before})")

        # Move 3 sliders: danceability, loudness, instrumentalness
        page.evaluate("""() => {
            const setSlider = (name, val) => {
                const inp = document.getElementById('slider-' + name);
                inp.value = val;
                inp.dispatchEvent(new Event('input', { bubbles: true }));
            };
            setSlider('danceability', 0.25);
            setSlider('loudness', -20.0);
            setSlider('instrumentalness', 0.85);
        }""")

        # Wait for debounced prediction (250ms + network)
        time.sleep(0.8)

        prob_after = page.inner_text("#prob-numeral")
        verdict_after = page.inner_text("#verdict-text")
        dance_after = page.inner_text("#val-danceability")
        loud_after = page.inner_text("#val-loudness")
        inst_after = page.inner_text("#val-instrumentalness")
        print(f"After move: Prob = {prob_after}, Verdict = {verdict_after} (danceability: {dance_after}, loudness: {loud_after}, instrumentalness: {inst_after})")

        path_after = os.path.join(ARTIFACT_DIR, "sliders_after.png")
        page.locator(".content-grid").screenshot(path=path_after)
        print(f"Saved: {path_after}")

        assert prob_before != prob_after, "Probability did not change after moving sliders!"
        results["slider_change"] = {
            "before": {"prob": prob_before, "danceability": dance_before, "loudness": loud_before, "instrumentalness": inst_before},
            "after": {"prob": prob_after, "danceability": dance_after, "loudness": loud_after, "instrumentalness": inst_after},
        }

        # ----------------------------------------------------
        # 4. Load Presets and Sanity-Check
        # ----------------------------------------------------
        print("\n--- 4. Testing Presets ---")
        preset_data = {}
        for p_name in ["pop", "ballad", "ambient"]:
            page.locator(f"button[data-preset='{p_name}']").click()
            time.sleep(0.8)
            prob_p = page.inner_text("#prob-numeral")
            verd_p = page.inner_text("#verdict-text")
            preset_data[p_name] = {"prob_str": prob_p, "prob": float(prob_p.replace("%", "")), "verdict": verd_p}
            path_p = os.path.join(ARTIFACT_DIR, f"preset_{p_name}.png")
            page.locator(".content-grid").screenshot(path=path_p)
            print(f"Preset '{p_name}': Prob = {prob_p}, Verdict = {verd_p}")

        results["presets"] = preset_data

        # Sanity check: Pop should have highest probability, Ambient lowest
        pop_p = preset_data["pop"]["prob"]
        ballad_p = preset_data["ballad"]["prob"]
        ambient_p = preset_data["ambient"]["prob"]
        print(f"Sanity Check comparison: Pop ({pop_p}%) vs Ballad ({ballad_p}%) vs Ambient ({ambient_p}%)")
        assert pop_p > ballad_p > ambient_p, (
            f"Preset ordering unexpected! Pop={pop_p}%, Ballad={ballad_p}%, Ambient={ambient_p}%"
        )
        print("Sanity check PASSED: Pop single > Acoustic ballad > Ambient instrumental.")

        # ----------------------------------------------------
        # 5. Out-of-Range Value via Console & UI Error
        # ----------------------------------------------------
        print("\n--- 5. Testing Out-of-Range Console Submission ---")
        page.evaluate("""() => {
            const badPayload = {
                tempo: 999.0, // max is 217.282
                energy: 0.8,
                danceability: 0.75,
                valence: 0.65,
                loudness: -5.0,
                acousticness: 0.15,
                instrumentalness: 0.0,
                speechiness: 0.05,
                duration_sec: 210.0,
                genre: 'pop'
            };
            window.fetchPrediction(badPayload);
        }""")
        time.sleep(0.5)

        err_visible = page.is_visible("#result-error")
        err_text = page.inner_text("#result-error")
        slider_err_text = page.inner_text("#slider-tempo ~ .slider-error-msg") if page.is_visible("#slider-tempo ~ .slider-error-msg") else "None"
        print(f"Error panel visible: {err_visible}")
        print(f"Error instructions text: {err_text}")
        print(f"Under-slider error text: {slider_err_text}")

        path_error = os.path.join(ARTIFACT_DIR, "error_state.png")
        page.locator(".content-grid").screenshot(path=path_error)
        print(f"Saved: {path_error}")

        assert err_visible, "Result error container was not displayed on 422 error!"
        results["error_state"] = {"panel_error": err_text, "slider_error": slider_err_text}

        # ----------------------------------------------------
        # 6. Tab Navigation & Focus Ring State
        # ----------------------------------------------------
        print("\n--- 6. Testing Keyboard Navigation & Focus Rings ---")
        # Reset back to valid pop preset
        page.locator("button[data-preset='pop']").click()
        time.sleep(0.5)

        # Tab through the page
        page.focus("button[data-preset='pop']")
        focused_tag = page.evaluate("() => document.activeElement.tagName")
        focused_text = page.evaluate("() => document.activeElement.innerText")
        print(f"Focused element on preset button: <{focused_tag}> '{focused_text}'")

        # Tab into slider
        page.keyboard.press("Tab") # ballad
        page.keyboard.press("Tab") # ambient
        page.keyboard.press("Tab") # first slider (tempo)
        focused_id = page.evaluate("() => document.activeElement.id")
        print(f"Focused element on slider: id='{focused_id}'")

        path_focus = os.path.join(ARTIFACT_DIR, "focus_state.png")
        page.locator(".predictor-panel").screenshot(path=path_focus)
        print(f"Saved: {path_focus}")

        browser.close()

    print("\n=== Verification Completed Successfully! ===")
    return results


if __name__ == "__main__":
    run_verification()
