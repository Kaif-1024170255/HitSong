/**
 * Hit Song Predictor — Frontend Application
 * Vanilla JS interacting with FastAPI backend
 */

(function () {
  'use strict';

  // Application State
  let schema = null;
  let currentThreshold = 0.48;
  let debounceTimer = null;
  let isInitialLoad = true;
  let currentProb = 0.0;
  let animationFrameId = null;

  // Preset Configurations
  const PRESETS = {
    pop: {
      tempo: 122.0,
      energy: 0.80,
      danceability: 0.75,
      valence: 0.65,
      loudness: -5.0,
      acousticness: 0.12,
      instrumentalness: 0.0,
      speechiness: 0.06,
      duration_sec: 205.0,
      genre: 'pop',
    },
    ballad: {
      tempo: 85.0,
      energy: 0.35,
      danceability: 0.45,
      valence: 0.30,
      loudness: -10.5,
      acousticness: 0.78,
      instrumentalness: 0.0,
      speechiness: 0.03,
      duration_sec: 240.0,
      genre: 'singer-songwriter',
    },
    ambient: {
      tempo: 95.0,
      energy: 0.18,
      danceability: 0.22,
      valence: 0.15,
      loudness: -18.0,
      acousticness: 0.85,
      instrumentalness: 0.82,
      speechiness: 0.04,
      duration_sec: 320.0,
      genre: 'ambient',
    },
  };

  // Feature display labels and units
  const FEATURE_METADATA = {
    tempo: { label: 'Tempo', unit: ' BPM' },
    energy: { label: 'Energy', unit: '' },
    danceability: { label: 'Danceability', unit: '' },
    valence: { label: 'Valence', unit: '' },
    loudness: { label: 'Loudness', unit: ' dB' },
    acousticness: { label: 'Acousticness', unit: '' },
    instrumentalness: { label: 'Instrumentalness', unit: '' },
    speechiness: { label: 'Speechiness', unit: '' },
    duration_sec: { label: 'Duration', unit: ' s' },
    genre: { label: 'Genre', unit: '' },
  };

  const MODEL_NAMES = {
    logistic_regression: 'Logistic Regression (Final)',
    svm_rbf: 'SVM (RBF)',
    svm_linear: 'Linear SVM',
    random_forest: 'Random Forest',
    knn: 'K-Nearest Neighbors',
    naive_bayes: 'Gaussian Naive Bayes',
    tree_entropy: 'Decision Tree (Entropy)',
    tree_gini: 'Decision Tree (Gini)',
  };

  // DOM Elements
  const statusDot = document.getElementById('status-dot');
  const statusText = document.getElementById('status-text');
  const slidersContainer = document.getElementById('sliders-container');
  const genreSelect = document.getElementById('genre-select');
  const probNumeral = document.getElementById('prob-numeral');
  const verdictText = document.getElementById('verdict-text');
  const verdictCaption = document.getElementById('verdict-caption');
  const meterFill = document.getElementById('meter-fill');
  const meterKnob = document.getElementById('meter-knob');
  const thresholdMarker = document.getElementById('threshold-marker');
  const thresholdLabel = document.getElementById('threshold-label');
  const ariaMeter = document.getElementById('aria-meter');
  const contributionsList = document.getElementById('contributions-list');
  const resultContent = document.getElementById('result-content');
  const resultError = document.getElementById('result-error');
  const comparisonTbody = document.getElementById('comparison-tbody');
  const presetButtons = document.querySelectorAll('.btn-preset');

  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /** Check API health and update masthead status indicator */
  async function checkHealth() {
    try {
      const res = await fetch('/api/health');
      if (!res.ok) throw new Error('Health check failed');
      statusDot.className = 'status-dot active';
      statusText.textContent = 'model ok';
    } catch {
      statusDot.className = 'status-dot error';
      statusText.textContent = 'model offline';
      showErrorState();
    }
  }

  /** Display clear instructions in result panel when API is unreachable */
  function showErrorState() {
    resultContent.style.display = 'none';
    resultError.style.display = 'block';
  }

  /** Format numerical slider values for display */
  function formatVal(feat, val) {
    const num = parseFloat(val);
    if (feat === 'tempo' || feat === 'duration_sec') {
      return Math.round(num);
    }
    return num.toFixed(2);
  }

  /** Update range slider filled background gradient */
  function updateSliderFill(input) {
    const min = parseFloat(input.min);
    const max = parseFloat(input.max);
    const val = parseFloat(input.value);
    const pct = ((val - min) / (max - min)) * 100;
    input.style.background = `linear-gradient(to right, var(--indigo) 0%, var(--indigo) ${pct}%, var(--rule) ${pct}%, var(--rule) 100%)`;
  }

  /** Render dynamic sliders from schema definitions */
  function buildSliders(slidersMeta, initialValues) {
    slidersContainer.innerHTML = '';
    Object.keys(slidersMeta).forEach(feat => {
      const b = slidersMeta[feat];
      const meta = FEATURE_METADATA[feat] || { label: feat, unit: '' };
      const currentVal = initialValues[feat] !== undefined ? initialValues[feat] : b.default;

      const row = document.createElement('div');
      row.className = 'slider-row';

      const header = document.createElement('div');
      header.className = 'slider-header';

      const label = document.createElement('label');
      label.setAttribute('for', `slider-${feat}`);
      label.className = 'control-label';
      label.textContent = meta.label;

      const valSpan = document.createElement('span');
      valSpan.id = `val-${feat}`;
      valSpan.className = 'slider-val';
      valSpan.textContent = `${formatVal(feat, currentVal)}${meta.unit}`;

      header.appendChild(label);
      header.appendChild(valSpan);

      const input = document.createElement('input');
      input.type = 'range';
      input.id = `slider-${feat}`;
      input.className = 'slider-input';
      input.name = feat;
      input.min = b.min;
      input.max = b.max;
      input.step = b.step;
      input.value = currentVal;
      input.setAttribute('aria-label', meta.label);
      input.setAttribute('aria-valuetext', `${formatVal(feat, currentVal)}${meta.unit}`);

      const bounds = document.createElement('div');
      bounds.className = 'slider-bounds caption';
      bounds.innerHTML = `<span>${formatVal(feat, b.min)}${meta.unit}</span><span>${formatVal(feat, b.max)}${meta.unit}</span>`;

      input.addEventListener('input', () => {
        valSpan.textContent = `${formatVal(feat, input.value)}${meta.unit}`;
        input.setAttribute('aria-valuetext', `${formatVal(feat, input.value)}${meta.unit}`);
        updateSliderFill(input);
        clearActivePreset();
        schedulePrediction();
      });

      row.appendChild(header);
      row.appendChild(input);
      row.appendChild(bounds);
      slidersContainer.appendChild(row);

      updateSliderFill(input);
    });
  }

  /** Populate genre options */
  function populateGenres(genres, defaultGenre) {
    genreSelect.innerHTML = '';
    genres.forEach(g => {
      const opt = document.createElement('option');
      opt.value = g;
      opt.textContent = g;
      if (g === defaultGenre) opt.selected = true;
      genreSelect.appendChild(opt);
    });

    genreSelect.addEventListener('change', () => {
      clearActivePreset();
      schedulePrediction();
    });
  }

  /** Collect current form payload */
  function getPayload() {
    const payload = {};
    const inputs = slidersContainer.querySelectorAll('.slider-input');
    inputs.forEach(inp => {
      payload[inp.name] = parseFloat(inp.value);
    });
    payload.genre = genreSelect.value;
    return payload;
  }

  /** Schedule debounced prediction call (250ms) */
  function schedulePrediction() {
    probNumeral.classList.add('loading');
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      fetchPrediction(getPayload());
    }, 250);
  }

  /** Display clear instructions in result panel when API is unreachable or validation fails */
  function showErrorState(message) {
    resultContent.style.display = 'none';
    resultError.style.display = 'block';
    const errText = resultError.querySelector('.error-instructions');
    if (errText) {
      if (message) {
        errText.innerHTML = message;
      } else {
        errText.innerHTML = "The model isn't responding. Start the server with <code>uvicorn app.main:app --reload</code> and reload this page.";
      }
    }
  }

  function clearSliderErrors() {
    document.querySelectorAll('.slider-error-msg').forEach(el => el.remove());
  }

  function showSliderError(field, message) {
    const inp = document.getElementById(`slider-${field}`);
    if (inp && inp.parentElement) {
      const existing = inp.parentElement.querySelector('.slider-error-msg');
      if (existing) existing.remove();
      const msgEl = document.createElement('p');
      msgEl.className = 'slider-error-msg caption';
      msgEl.style.color = 'var(--ink-2)';
      msgEl.style.marginTop = '4px';
      msgEl.textContent = message;
      inp.parentElement.appendChild(msgEl);
    }
  }

  /** Fetch prediction from API */
  async function fetchPrediction(payload) {
    try {
      clearSliderErrors();
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        if (res.status === 422) {
          const errData = await res.json();
          const details = errData.detail || [];
          const firstErr = details[0] || {};
          const field = firstErr.field || 'input';
          const msg = firstErr.message || 'Invalid value';
          showSliderError(field, msg);
          showErrorState(`Input validation error: ${msg}`);
          return;
        }
        throw new Error('Prediction error');
      }

      const data = await res.json();
      resultContent.style.display = 'flex';
      resultError.style.display = 'none';
      renderPrediction(data);
    } catch {
      showErrorState();
    } finally {
      probNumeral.classList.remove('loading');
    }
  }

  window.fetchPrediction = fetchPrediction;

  /** Animate probability numeral count-up */
  function animateNumeral(fromVal, toVal, durationMs) {
    if (prefersReducedMotion || durationMs <= 0) {
      probNumeral.textContent = `${(toVal * 100).toFixed(1)}%`;
      currentProb = toVal;
      return;
    }

    if (animationFrameId) cancelAnimationFrame(animationFrameId);
    const startTime = performance.now();

    function frame(now) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / durationMs, 1.0);
      // cubic-bezier(.2, .7, .2, 1) approximation
      const eased = 1 - Math.pow(1 - progress, 3);
      const val = fromVal + (toVal - fromVal) * eased;
      probNumeral.textContent = `${(val * 100).toFixed(1)}%`;

      if (progress < 1.0) {
        animationFrameId = requestAnimationFrame(frame);
      } else {
        probNumeral.textContent = `${(toVal * 100).toFixed(1)}%`;
        currentProb = toVal;
      }
    }

    animationFrameId = requestAnimationFrame(frame);
  }

  /** Render prediction result to UI */
  function renderPrediction(data) {
    const prob = data.probability;
    const th = data.threshold !== undefined ? data.threshold : currentThreshold;
    currentThreshold = th;

    // 1. Numeral Animation
    const duration = isInitialLoad ? 600 : 200;
    animateNumeral(currentProb, prob, duration);
    isInitialLoad = false;

    // 2. Verdict Text
    const isLikely = prob >= th;
    verdictText.textContent = isLikely ? 'Likely to chart' : 'Unlikely to chart';
    verdictText.className = `verdict-text ${isLikely ? 'likely' : 'unlikely'}`;
    verdictCaption.textContent = isLikely
      ? `Above the ${th.toFixed(2)} decision threshold.`
      : `Below the ${th.toFixed(2)} decision threshold.`;

    // 3. Meter Fill & Knob
    const pct = Math.min(Math.max(prob * 100, 0), 100);
    meterFill.style.width = `${pct}%`;
    meterKnob.style.left = `${pct}%`;
    ariaMeter.setAttribute('aria-valuenow', prob.toFixed(4));

    // 4. Threshold Marker
    thresholdMarker.style.left = `${th * 100}%`;
    thresholdLabel.textContent = `threshold ${th.toFixed(2)}`;

    // 5. Diverging Contribution Bars
    renderContributions(data.contributions || []);
  }

  /** Render top diverging feature contribution bars */
  function renderContributions(contributions) {
    contributionsList.innerHTML = '';
    const topContribs = contributions.slice(0, 5);
    const maxAbs = Math.max(...topContribs.map(c => Math.abs(c.effect)), 0.001);

    topContribs.forEach(c => {
      const row = document.createElement('div');
      row.className = 'contrib-row';

      const featName = FEATURE_METADATA[c.feature] ? FEATURE_METADATA[c.feature].label : c.feature;

      const nameEl = document.createElement('span');
      nameEl.className = 'contrib-name';
      nameEl.textContent = featName;

      const axisBox = document.createElement('div');
      axisBox.className = 'contrib-axis-container';

      const axisLine = document.createElement('div');
      axisLine.className = 'contrib-axis';

      const bar = document.createElement('div');
      const barPct = (Math.abs(c.effect) / maxAbs) * 48; // up to 48% of container width

      if (c.effect >= 0) {
        bar.className = 'contrib-bar positive';
        bar.style.left = '50%';
        bar.style.width = `${barPct}%`;
      } else {
        bar.className = 'contrib-bar negative';
        bar.style.left = `${50 - barPct}%`;
        bar.style.width = `${barPct}%`;
      }

      axisBox.appendChild(axisLine);
      axisBox.appendChild(bar);

      const valEl = document.createElement('span');
      valEl.className = 'contrib-val';
      const sign = c.effect > 0 ? '+' : '';
      valEl.textContent = `${sign}${c.effect.toFixed(2)}`;

      row.appendChild(nameEl);
      row.appendChild(axisBox);
      row.appendChild(valEl);

      contributionsList.appendChild(row);
    });
  }

  /** Apply a preset to all controls */
  function applyPreset(key) {
    const p = PRESETS[key];
    if (!p) return;

    presetButtons.forEach(btn => {
      const match = btn.dataset.preset === key;
      btn.classList.toggle('active', match);
      btn.setAttribute('aria-pressed', match ? 'true' : 'false');
    });

    Object.keys(p).forEach(k => {
      if (k === 'genre') {
        genreSelect.value = p[k];
      } else {
        const inp = document.getElementById(`slider-${k}`);
        const valSpan = document.getElementById(`val-${k}`);
        const meta = FEATURE_METADATA[k] || { unit: '' };
        if (inp && valSpan) {
          inp.value = p[k];
          valSpan.textContent = `${formatVal(k, p[k])}${meta.unit}`;
          inp.setAttribute('aria-valuetext', `${formatVal(k, p[k])}${meta.unit}`);
          updateSliderFill(inp);
        }
      }
    });

    schedulePrediction();
  }

  /** Clear preset button highlight on manual tweak */
  function clearActivePreset() {
    presetButtons.forEach(btn => {
      btn.classList.remove('active');
      btn.setAttribute('aria-pressed', 'false');
    });
  }

  /** Fetch and render insights table */
  async function loadInsights() {
    try {
      const res = await fetch('/api/insights');
      if (!res.ok) return;
      const data = await res.json();
      renderComparisonTable(data.model_comparison || []);
    } catch {
      // Quiet fallback
    }
  }

  /** Render model comparison table */
  function renderComparisonTable(models) {
    if (!models.length) return;
    comparisonTbody.innerHTML = '';

    models.forEach(m => {
      const tr = document.createElement('tr');
      const isFinal = m.model === 'logistic_regression';
      if (isFinal) tr.className = 'final-model';

      const displayName = MODEL_NAMES[m.model] || m.model;

      tr.innerHTML = `
        <td>${displayName}</td>
        <td class="num-col">${parseFloat(m['ROC-AUC']).toFixed(3)}</td>
        <td class="num-col">${parseFloat(m['PR-AUC']).toFixed(3)}</td>
        <td class="num-col">${parseFloat(m.Recall).toFixed(3)}</td>
        <td class="num-col">${parseFloat(m.Specificity).toFixed(3)}</td>
        <td class="num-col">${parseFloat(m.Accuracy).toFixed(3)}</td>
        <td class="num-col">${parseFloat(m['F1-Score']).toFixed(3)}</td>
      `;
      comparisonTbody.appendChild(tr);
    });
  }

  /** Initialize application */
  async function init() {
    await checkHealth();

    // Attach preset button listeners
    presetButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        applyPreset(btn.dataset.preset);
      });
    });

    // Fetch schema and construct form
    try {
      const res = await fetch('/api/schema');
      if (!res.ok) throw new Error('Schema fetch failed');
      schema = await res.json();
      currentThreshold = schema.threshold || 0.48;

      buildSliders(schema.sliders || {}, PRESETS.pop);
      populateGenres(schema.genres || [], PRESETS.pop.genre);

      // Trigger initial prediction
      fetchPrediction(getPayload());
      loadInsights();
    } catch {
      showErrorState();
    }
  }

  // Bootstrap when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
