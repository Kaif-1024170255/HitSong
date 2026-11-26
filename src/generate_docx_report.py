"""University-level DOCX report generator for Hit Song Predictor.

Produces a comprehensive Word document (.docx) thoroughly answering every
university report template question across Dataset, EDA, Preprocessing,
Model Execution, Evaluation, GUI, Deployment, and Additional Components.

Runs standalone from root: python -m src.generate_docx_report
"""

import os
import sys
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn


def set_cell_background(cell, fill_hex):
    """Set background shading color for a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=70, bottom=70, left=110, right=110):
    """Set inner cell padding in twips."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def add_page_number_footer(section):
    """Add page numbering to the document footer."""
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run_text = p.add_run("Hit Song Predictor — University ML Project Report | Page ")
    run_text.font.size = Pt(8.5)
    run_text.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    fld1 = parse_xml(r'<w:fldChar %s w:fldCharType="begin"/>' % nsdecls('w'))
    instr = parse_xml(r'<w:instrText %s xml:space="preserve"> PAGE </w:instrText>' % nsdecls('w'))
    fld2 = parse_xml(r'<w:fldChar %s w:fldCharType="end"/>' % nsdecls('w'))
    run_pg = p.add_run()
    run_pg.font.size = Pt(8.5)
    run_pg.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
    run_pg._r.append(fld1)
    run_pg._r.append(instr)
    run_pg._r.append(fld2)


def configure_styles(doc):
    """Configure base typography and heading styles for the DOCX document."""
    style_norm = doc.styles['Normal']
    style_norm.font.name = 'Calibri'
    style_norm.font.size = Pt(10.5)
    style_norm.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    style_norm.paragraph_format.space_after = Pt(5.5)
    style_norm.paragraph_format.line_spacing = 1.15

    for level, size in [(1, 16), (2, 12.5), (3, 11)]:
        h = doc.styles[f'Heading {level}']
        h.font.name = 'Calibri'
        h.font.bold = True
        h.font.size = Pt(size)
        h.font.color.rgb = RGBColor(0x1A, 0x2B, 0x4C)
        h.paragraph_format.space_before = Pt(12 if level == 1 else 8)
        h.paragraph_format.space_after = Pt(4)


def add_styled_table(doc, headers, rows, col_widths=None):
    """Add a formatted university-report style table with header shading."""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    hdr_cells = table.rows[0].cells
    for i, text in enumerate(headers):
        hdr_cells[i].text = ""
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(text)
        r.bold = True
        r.font.size = Pt(9)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_background(hdr_cells[i], "1A2B4C")
        set_cell_margins(hdr_cells[i])

    for r_idx, row_data in enumerate(rows):
        row_cells = table.rows[r_idx + 1].cells
        bg_hex = "F4F7FA" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = ""
            p = row_cells[c_idx].paragraphs[0]
            r = p.add_run(str(val))
            r.font.size = Pt(8.5)
            set_cell_background(row_cells[c_idx], bg_hex)
            set_cell_margins(row_cells[c_idx], top=55, bottom=55, left=75, right=75)

    if col_widths:
        for row in table.rows:
            for idx, width in enumerate(col_widths):
                row.cells[idx].width = Inches(width)
    doc.add_paragraph()


def add_callout(doc, title, text, border_color="1A2B4C", bg_color="F0F4F8"):
    """Create a highlighted callout box using a single-cell shaded table."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.rows[0].cells[0]
    cell.width = Inches(6.5)
    set_cell_background(cell, bg_color)
    set_cell_margins(cell, top=110, bottom=110, left=150, right=150)

    tcBorders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="{border_color}"/>'
        f'<w:top w:val="none"/><w:right w:val="none"/><w:bottom w:val="none"/>'
        f'</w:tcBorders>'
    )
    cell._tc.get_or_add_tcPr().append(tcBorders)

    p = cell.paragraphs[0]
    r_title = p.add_run(f"{title}\n")
    r_title.bold = True
    r_title.font.size = Pt(9.5)
    r_title.font.color.rgb = RGBColor(0x1A, 0x2B, 0x4C)
    r_body = p.add_run(text)
    r_body.font.size = Pt(9)
    doc.add_paragraph()


def add_figure(doc, rel_path, width_in, caption):
    """Safely embed a PNG image with centered caption."""
    if not os.path.exists(rel_path):
        doc.add_paragraph(f"[Missing figure: {rel_path}]")
        return
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_after = Pt(2)
    run_img = p_img.add_run()
    run_img.add_picture(rel_path, width=Inches(width_in))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_after = Pt(8)
    r_cap = p_cap.add_run(caption)
    r_cap.italic = True
    r_cap.font.size = Pt(8.5)
    r_cap.font.color.rgb = RGBColor(0x55, 0x55, 0x55)


def add_bold_lead(doc, lead, body):
    """Add a paragraph starting with a bold lead-in phrase."""
    p = doc.add_paragraph()
    r1 = p.add_run(lead)
    r1.bold = True
    p.add_run(body)


def build_cover_page(doc):
    """Build standard university report cover page."""
    doc.add_paragraph("\n\n")
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run("HIT SONG PREDICTOR\n")
    r_title.bold = True
    r_title.font.size = Pt(28)
    r_title.font.color.rgb = RGBColor(0x1A, 0x2B, 0x4C)
    r_sub = p_title.add_run(
        "Predicting Billboard Hot 100 Commercial Success from Spotify Audio Features:\n"
        "A Rigorous Machine Learning, Explainability, and Web Deployment Study"
    )
    r_sub.font.size = Pt(13.5)
    r_sub.font.color.rgb = RGBColor(0x4A, 0x55, 0x68)
    doc.add_paragraph()
    add_callout(
        doc, "ABSTRACT & EXECUTIVE SUMMARY",
        "Commercial record labels evaluate thousands of unreleased songs weekly, yet fewer than 1% achieve "
        "Billboard Hot 100 success. This academic project investigates whether quantitative acoustic properties "
        "contain predictive signals for charting. Using 7,403 curated songs (2000–2023) with strict zero-artist-overlap "
        "partitioning, we benchmarked 8 classifier families. Regularized Logistic Regression (C=1.0, balanced weights) "
        "achieved the top held-out test ROC-AUC of 0.929 (PR-AUC = 0.606). Using leakage-safe Out-of-Fold Youden's J "
        "thresholding (\u03c4 = 0.48), the system captures 85.2% of true hits (115/135) while rejecting 85.7% of non-hits "
        "(1,153/1,345). The model is deployed via a FastAPI REST backend and an accessible manuscript-style web interface, "
        "alongside an experimental local MP3 signal extraction module.",
        border_color="1A2B4C", bg_color="F0F4F8"
    )
    meta_lines = [
        ("Domain:", "Music Information Retrieval (MIR) & Supervised Machine Learning"),
        ("Task Type:", "Binary Supervised Classification (Billboard Year-End Hot 100, 2000–2023)"),
        ("Dataset:", "7,403 songs (10:1 controlled ratio; 5,923 train / 1,480 test; 0 artist overlap)"),
        ("Production Model:", "Regularized Logistic Regression (C=1.0, class_weight='balanced')"),
        ("Frozen Cutoff:", "\u03c4 = 0.48 (Out-of-Fold Youden's J maximizing; Test Recall = 85.19%)"),
        ("Tech Stack:", "Python 3.11+, scikit-learn, FastAPI, Uvicorn, Vanilla HTML/CSS/JS, Librosa"),
        ("Test Coverage:", "52 automated unit & integration tests passing (pytest)"),
    ]
    add_styled_table(doc, ["Project Attribute", "Specification"], meta_lines, col_widths=[2.0, 4.5])
    doc.add_page_break()


def add_dataset_schema_table(doc):
    """Add schema table for Section 1.1."""
    schema_rows = [
        ["tempo", "Continuous (BPM)", "30.2 – 217.3 (med: 122.0)", "Estimated beats per minute"],
        ["energy", "Continuous [0, 1]", "0.002 – 1.000 (med: 0.688)", "Perceptual intensity, loudness & activity"],
        ["danceability", "Continuous [0, 1]", "0.061 – 0.980 (med: 0.586)", "Beat regularity and rhythmic stability"],
        ["valence", "Continuous [0, 1]", "0.000 – 0.992 (med: 0.454)", "Musical positiveness / cheerfulness"],
        ["loudness", "Continuous (dB)", "-38.6 – +0.91 (med: -6.94)", "Integrated master loudness in decibels"],
        ["acousticness", "Continuous [0, 1]", "0.000 – 0.996 (med: 0.170)", "Confidence that track is acoustic"],
        ["instrumentalness", "Continuous [0, 1]", "0.000 – 0.999 (med: 0.000)", "Predicts absence of human vocals"],
        ["speechiness", "Continuous [0, 1]", "0.023 – 0.959 (med: 0.049)", "Proportion of spoken words vs singing"],
        ["duration_sec", "Continuous (sec)", "60.7 – 555.4 (med: 215.2)", "Track length converted from duration_ms"],
        ["genre", "Categorical (114)", "114 unique subgenres", "Spotify track genre classification"],
        ["charted (Target)", "Binary {0, 1}", "673 hits (9.09%)", "Appeared on Billboard Year-End Hot 100"],
    ]
    add_styled_table(doc, ["Feature", "Type / Scale", "Range / Median", "Acoustic & Domain Meaning"], schema_rows, [1.2, 1.1, 1.8, 2.4])


def add_dataset_biases(doc):
    """Subsection 1.2 detailing dataset challenges, biases, and mitigations."""
    doc.add_heading("1.2 Challenges & Biases in the Dataset and How They Were Addressed", level=2)
    add_bold_lead(
        doc, "1. Severe Natural Class Imbalance (0.84% Positive Rate): ",
        "In the raw candidate pool of 80,237 tracks, only 673 charted (0.84%), while 79,564 were non-hits (99.16%). "
        "Training directly on a 118:1 ratio causes standard classifiers to collapse toward predicting the majority class "
        "(99.16% accuracy, 0% recall). We addressed this via seeded 10:1 negative downsampling (random_state=42), "
        "yielding 7,403 modeling records (673 charted / 9.09%, 6,730 non-charted / 90.91%)—preserving realistic skew "
        "while providing enough positive density for reliable gradient optimization."
    )
    add_bold_lead(
        doc, "2. Ambiguous Record Matching: ",
        "Linking Billboard song titles/artists to Spotify IDs yielded 11 ambiguous tracks with conflicting chart statuses "
        "(e.g., remixes or covers sharing identical titles). These 11 records were audited (reports/tables/ambiguous_excluded.csv) "
        "and permanently excluded to prevent label noise."
    )
    add_bold_lead(
        doc, "3. Artist Identity Leakage Bias: ",
        "Superstar artists (e.g., Drake, Taylor Swift) possess massive label marketing budgets and radio distribution. "
        "If songs by the same artist appear in both training and test splits, models memorize artist-specific production "
        "signatures rather than generalizable hit acoustics. We normalized primary_artist (4,581 unique artists) and "
        "quarantined it strictly as a grouping key for StratifiedGroupKFold, achieving zero artist overlap."
    )
    add_bold_lead(
        doc, "4. Temporal Selection Bias: ",
        "While Billboard hits carry an exact chart_year (2000–2023), non-charting catalog tracks in dataset.csv lack "
        "verified release timestamps. Using a chronological split or including chart_year as a predictor would introduce "
        "fatal temporal leakage; thus, chart_year was quarantined solely for post-hoc coverage audits."
    )


def build_section_dataset(doc):
    """Section 1: Dataset Description."""
    doc.add_heading("1. Dataset Description", level=1)
    doc.add_heading("1.1 Detailed Overview of the Dataset", level=2)
    doc.add_paragraph(
        "The Hit Song Predictor models commercial chart success using historical Billboard Year-End Hot 100 records "
        "(2000–2023) linked to Spotify Web API audio engineering descriptors. Two untouched raw CSV files in data/raw/ "
        "form the empirical foundation: billboard_24years_lyrics_spotify.csv (3,397 Year-End chart entries) and "
        "dataset.csv (114,000 Spotify tracks across 114 subgenres). After deterministic matching, deduplication, and "
        "controlled 10:1 negative downsampling (random_state=42), the canonical table (data/processed/songs.parquet) "
        "contains 7,403 tracks with zero missing values across 10 predictive features."
    )
    add_dataset_schema_table(doc)
    add_dataset_biases(doc)
    add_figure(doc, "reports/figures/class_balance_modeling.png", 4.6, "Figure 1: Controlled 10:1 class balance in the 7,403-song modeling dataset (9.09% hits).")


def add_eda_visual_insights(doc):
    """Subsection 2.2 detailing insights from visualizations."""
    doc.add_heading("2.2 Insights from Visualizations & Distributional Plots", level=2)
    add_bold_lead(
        doc, "• Mastering Loudness Separation: ",
        "Boxplots and density plots show that charted tracks cluster tightly at higher commercial mastering volumes "
        "(median -5.3 dB vs. -7.2 dB for non-hits), reflecting the modern 'Loudness War' compression standard."
    )
    add_bold_lead(
        doc, "• Vocal Necessity & Acoustic Filtering: ",
        "The correlation heatmap highlights strong multicollinearity between loudness and energy (r = +0.76), and a "
        "sharp negative relationship between acousticness and energy (r = -0.73). Purely instrumental tracks "
        "(instrumentalness > 0.5) are almost completely absent from Year-End Hot 100 rankings."
    )
    add_bold_lead(
        doc, "• Tempo Invariance & Sweet Spot: ",
        "Tempo distributions for hits (median 120.0 BPM) and non-hits (median 122.0 BPM) overlap heavily, proving "
        "that tempo alone does not dictate hit status. However, danceability vs. energy scatter plots show hits "
        "concentrate in the upper-right quadrant (danceability > 0.55, energy > 0.50)."
    )
    add_figure(doc, "reports/figures/correlation_heatmap.png", 4.6, "Figure 2: Pearson correlation heatmap across all 9 continuous features and target.")
    add_figure(doc, "reports/figures/danceability_vs_energy.png", 4.5, "Figure 3: Joint scatter plot of danceability vs. energy segmented by chart status.")


def build_section_eda(doc):
    """Section 2: Exploratory Data Analysis."""
    doc.add_heading("2. Exploratory Data Analysis (EDA)", level=1)
    doc.add_heading("2.1 Statistical Summaries & Target Correlations", level=2)
    doc.add_paragraph(
        "Statistical summaries and two-tailed Pearson correlation tests were computed across all 7,403 tracks "
        "(reports/tables/target_correlations.csv) to quantify linear associations with Billboard Hot 100 entry:"
    )
    corr_rows = [
        ["loudness", "+0.168", "7.08e-48", "Strongest positive driver; commercial hits are mastered louder."],
        ["instrumentalness", "-0.164", "9.92e-46", "Strong negative driver; Hot 100 hits overwhelmingly require vocals."],
        ["acousticness", "-0.139", "3.08e-33", "Strong negative driver; electronic/produced tracks dominate charts."],
        ["danceability", "+0.134", "3.71e-31", "Significant positive driver; steady 4/4 grooves support radio play."],
        ["energy", "+0.062", "8.13e-08", "Moderate positive correlation with perceptual intensity."],
        ["valence", "+0.048", "3.85e-05", "Mild positive correlation toward upbeat emotional tone."],
        ["duration_sec", "-0.035", "2.77e-03", "Mild negative; radio programming favors concise 3–4 min cuts."],
        ["speechiness", "-0.016", "0.178", "Statistically non-significant linear correlation."],
        ["tempo", "-0.015", "0.191", "Statistically non-significant; hits occur across all BPM bands."],
    ]
    add_styled_table(doc, ["Feature", "Pearson r", "p-value", "Statistical Interpretation"], corr_rows, [1.2, 0.8, 0.9, 3.6])
    add_eda_visual_insights(doc)


def add_feature_engineering_subsection(doc):
    """Subsection 3.2 explaining feature engineering choices."""
    doc.add_heading("3.2 Feature Engineering & Rationale", level=2)
    add_bold_lead(
        doc, "1. Unit Scaling for Human Interpretability (duration_ms \u2192 duration_sec): ",
        "Raw Spotify metadata stores song length in milliseconds (e.g., 215,226 ms). Converting this into seconds "
        "(duration_sec = duration_ms / 1000.0) improves numerical conditioning, aligns with web UI sliders, and makes "
        "model coefficients directly interpretable in human terms."
    )
    add_bold_lead(
        doc, "2. Artist Name Normalization for Leakage-Free Grouping: ",
        "The helper normalise_artist() applies lowercasing, whitespace stripping, and regex collapsing to primary_artist. "
        "This prevents duplicate artist variations (e.g., 'Elton John ' vs 'elton john') from splitting across train/test."
    )
    add_bold_lead(
        doc, "3. Deliberate Omission of Polynomial / Complex Interaction Terms: ",
        "We intentionally avoided black-box polynomial expansions or PCA feature replacements in the supervised pipeline "
        "so that every standardized Logistic Regression coefficient maps 1-to-1 onto a physical musical attribute."
    )


def build_section_preprocessing(doc):
    """Section 3: Data Pre-processing."""
    doc.add_heading("3. Data Pre-processing", level=1)
    doc.add_heading("3.1 Data Pre-processing Techniques & Leakage Prevention", level=2)
    doc.add_paragraph(
        "All preprocessing steps are encapsulated in src/preprocess.py using deterministic scikit-learn ColumnTransformer "
        "pipelines to guarantee zero data leakage between training and held-out evaluation:"
    )
    add_bold_lead(
        doc, "• Artist-Grouped Outer Split: ",
        "Using StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) grouped on primary_artist, the dataset "
        "was partitioned into 5,923 training tracks (538 hits / 9.08%; 3,657 artists) and 1,480 held-out test tracks "
        "(135 hits / 9.12%; 924 artists). Artist overlap between train and test is exactly 0, and class balance differs "
        "by just 0.04 percentage points."
    )
    add_bold_lead(
        doc, "• Continuous Feature Standardization: ",
        "StandardScaler() centers and scales all 9 continuous features (z = (x - \u03bc) / \u03c3). Crucially, \u03bc and \u03c3 are "
        "fit exclusively on training folds inside the Pipeline and applied unchanged to validation and test splits."
    )
    add_bold_lead(
        doc, "• Categorical Genre Encoding: ",
        "OneHotEncoder(drop='first', handle_unknown='ignore') converts the 114 genre categories into sparse binary "
        "dummies, dropping the reference level to prevent multicollinearity while gracefully ignoring unseen categories."
    )
    add_feature_engineering_subsection(doc)


def add_training_challenges_subsection(doc):
    """Subsection 4.2 explaining training challenges and solutions."""
    doc.add_heading("4.2 Challenges Encountered During Model Training & Solutions", level=2)
    add_bold_lead(
        doc, "1. Feature Scale Disparity & Optimizer Non-Convergence: ",
        "Raw continuous features span four orders of magnitude (duration_sec ~ 220 vs. speechiness ~ 0.05). Without "
        "scaling, gradient-based Logistic Regression and kernel SVMs suffered from ill-conditioned Hessians and slow "
        "convergence. Wrapping StandardScaler inside the scikit-learn Pipeline normalized all features to zero mean "
        "and unit variance, enabling Logistic Regression to converge cleanly in 6.10 seconds."
    )
    add_bold_lead(
        doc, "2. Minority Class Suppression (10:1 Skew): ",
        "Under standard cross-entropy loss, models prioritized the 90.9% majority class, yielding high specificity but "
        "dismal recall (< 35%). We configured class_weight='balanced' on Logistic Regression, SVMs, and tree models, "
        "weighting minority hit samples inversely proportional to class frequency."
    )
    add_bold_lead(
        doc, "3. High-Dimensional One-Hot Genre Sparsity: ",
        "Expanding 114 genres into 113 dummy columns created sparse subspaces prone to overfitting rare subgenres. "
        "We countered this via L2 regularization (C=1.0) in linear models and max_features='log2' / min_samples_leaf=20 "
        "constraints in tree models."
    )


def add_hyperopt_subsection(doc):
    """Subsection 4.3 explaining hyperparameter tuning and CV setup."""
    doc.add_heading("4.3 Hyperparameter Optimization & Ensuring Convergence", level=2)
    doc.add_paragraph(
        "Hyperparameters were tuned systematically via GridSearchCV in src/train.py using an inner 5-fold "
        "StratifiedGroupKFold (grouped by primary_artist, random_state=42) optimizing ROC-AUC. Every estimator ran "
        "with sufficient iteration budgets (max_iter=2000 for Logistic Regression, probability=True for SVMs):"
    )
    train_rows = [
        ["Logistic Regression", "{'model__C': 1.0}, class_weight='balanced'", "0.9469", "\u00b1 0.0086", "6.10s"],
        ["SVM (RBF Kernel)", "{'model__C': 10.0, 'model__gamma': 'auto'}", "0.9463", "\u00b1 0.0080", "72.12s"],
        ["Linear SVM", "{'model__C': 1.0}, class_weight='balanced'", "0.9447", "\u00b1 0.0085", "36.64s"],
        ["Random Forest", "{'max_depth': None, 'max_features': 'log2'}", "0.9424", "\u00b1 0.0099", "46.67s"],
        ["K-Nearest Neighbors", "{'model__n_neighbors': 15}", "0.8950", "\u00b1 0.0133", "10.03s"],
        ["Gaussian Naive Bayes", "{'model__var_smoothing': 1e-07}", "0.8933", "\u00b1 0.0116", "1.59s"],
        ["Decision Tree (Gini)", "{'max_depth': 12, 'min_samples_leaf': 20}", "0.8585", "\u00b1 0.0271", "2.77s"],
        ["Decision Tree (Entropy)", "{'max_depth': 8, 'min_samples_leaf': 20}", "0.8503", "\u00b1 0.0266", "2.86s"],
    ]
    add_styled_table(doc, ["Model Architecture", "Optimal Hyperparameters", "CV ROC-AUC", "CV Std", "Train Time"], train_rows, [1.4, 2.3, 0.9, 0.9, 1.0])


def build_section_model_execution(doc):
    """Section 4: Model Execution."""
    doc.add_heading("4. Model Execution", level=1)
    doc.add_heading("4.1 Machine Learning Algorithms Experimented With", level=2)
    doc.add_paragraph(
        "To evaluate linear separability, non-linear boundaries, ensemble variance reduction, and probabilistic baselines, "
        "eight distinct classifier families were implemented and compared in src/train.py:"
    )
    doc.add_paragraph(
        "• Linear Models: Logistic Regression (L2-penalized log-odds classifier) and Linear SVM (hinge-loss margin classifier).\n"
        "• Non-Linear Kernel Method: Support Vector Machine with Radial Basis Function (RBF) kernel.\n"
        "• Tree & Ensemble Models: Random Forest (300-tree bagging ensemble) and single Decision Trees (Gini & Entropy criteria).\n"
        "• Instance-Based & Probabilistic: K-Nearest Neighbors (distance-weighted local neighborhoods) and Gaussian Naive Bayes."
    )
    add_training_challenges_subsection(doc)
    add_hyperopt_subsection(doc)


def add_eval_metrics_subsection(doc):
    """Subsection 5.1 explaining evaluation metrics."""
    doc.add_heading("5.1 Evaluation Metrics Used & Commercial Justification", level=2)
    doc.add_paragraph(
        "On an imbalanced 10:1 test set (90.88% non-hits), raw Accuracy is misleading because a trivial majority classifier "
        "scores 90.88% while capturing zero hits. Therefore, src/evaluate.py evaluates all models across 7 complementary metrics:"
    )
    doc.add_paragraph(
        "• ROC-AUC (Area Under Receiver Operating Characteristic Curve): Threshold-independent ranking discrimination.\n"
        "• PR-AUC (Precision-Recall AUC): Sensitivity to minority-class precision across probability thresholds.\n"
        "• Recall (Sensitivity = TP / (TP + FN)): Fraction of true Billboard hits captured (critical for A&R scouting).\n"
        "• Specificity (= TN / (TN + FP)): Fraction of non-hits correctly filtered out to prevent analyst overload.\n"
        "• Precision (= TP / (TP + FP)) & F1-Score: Harmonic trade-off between hit precision and recall."
    )


def add_cv_impact_subsection(doc):
    """Subsection 5.2 explaining cross-validation impact and threshold tuning."""
    doc.add_heading("5.2 Impact of Cross-Validation & Leakage-Safe Thresholding", level=2)
    doc.add_paragraph(
        "Conducting 5-fold StratifiedGroupKFold cross-validation impacted evaluation in two critical ways: "
        "(1) It eliminated ~2–3% optimistic ROC-AUC inflation caused by artist memorization in naive random CV, proving "
        "that Logistic Regression generalizes robustly (0.9469 CV ROC-AUC vs 0.929 Test ROC-AUC). "
        "(2) It enabled leakage-safe threshold selection (src/threshold_analysis.py): candidate cutoffs were derived "
        "strictly from 5-fold Out-of-Fold (OOF) training probabilities—never on test data—yielding Youden's J optimal "
        "threshold \u03c4 = 0.48 (Recall: 85.19%, Specificity: 85.72%) and F1-Max \u03c4 = 0.77 (Precision: 49.46%, Recall: 67.41%)."
    )
    th_rows = [
        ["Youden-J (Deployed)", "0.48", "0.8568", "0.3746", "0.8519", "0.8572", "0.5204", "115", "1,153", "192", "20"],
        ["F1-Max Selected", "0.77", "0.9074", "0.4946", "0.6741", "0.9309", "0.5705", "91", "1,252", "93", "44"],
    ]
    add_styled_table(
        doc, ["Method", "Cutoff", "Acc", "Prec", "Rec", "Spec", "F1", "TP", "TN", "FP", "FN"],
        th_rows, [1.3, 0.5, 0.55, 0.55, 0.55, 0.55, 0.55, 0.4, 0.5, 0.45, 0.4]
    )
    add_figure(doc, "reports/figures/threshold_tradeoff_cv.png", 4.7, "Figure 6: Out-of-Fold (OOF) threshold trade-off curve selecting \u03c4 = 0.48.")


def build_section_evaluation(doc):
    """Section 5: Model Evaluation."""
    doc.add_heading("5. Model Evaluation", level=1)
    add_eval_metrics_subsection(doc)

    eval_rows = [
        ["Logistic Regression", "0.929", "0.606", "0.844", "0.862", "0.380", "0.860", "0.524", "114", "1,159", "186", "21"],
        ["SVM (RBF Kernel)", "0.924", "0.595", "0.770", "0.865", "0.364", "0.856", "0.494", "104", "1,163", "182", "31"],
        ["Linear SVM", "0.924", "0.602", "0.793", "0.852", "0.350", "0.847", "0.485", "107", "1,146", "199", "28"],
        ["Random Forest", "0.918", "0.619", "0.296", "0.993", "0.816", "0.930", "0.435", "40", "1,336", "9", "95"],
        ["K-Nearest Neighbors", "0.877", "0.509", "0.370", "0.987", "0.735", "0.930", "0.493", "50", "1,327", "18", "85"],
        ["Gaussian Naive Bayes", "0.875", "0.305", "0.911", "0.787", "0.300", "0.798", "0.451", "123", "1,058", "287", "12"],
        ["Decision Tree (Entropy)", "0.821", "0.443", "0.341", "0.989", "0.754", "0.930", "0.469", "46", "1,330", "15", "89"],
        ["Decision Tree (Gini)", "0.821", "0.513", "0.363", "0.982", "0.671", "0.926", "0.471", "49", "1,321", "24", "86"],
    ]
    add_styled_table(
        doc, ["Model", "ROC", "PR", "Rec", "Spec", "Prec", "Acc", "F1", "TP", "TN", "FP", "FN"],
        eval_rows, [1.3, 0.45, 0.45, 0.45, 0.48, 0.45, 0.45, 0.45, 0.4, 0.5, 0.45, 0.4]
    )
    add_figure(doc, "reports/figures/roc_curves.png", 4.5, "Figure 4: Held-out test ROC curves across all 8 models (Logistic Regression = 0.929).")
    add_figure(doc, "reports/figures/confusion_matrices.png", 6.0, "Figure 5: Confusion matrices grid on the 1,480 held-out test songs.")
    add_cv_impact_subsection(doc)


def build_section_gui(doc):
    """Section 6: GUI Design and Development."""
    doc.add_heading("6. GUI Design and Development", level=1)
    add_bold_lead(
        doc, "6.1 Design & Development Process: ",
        "The web interface (web/) was built with zero-dependency vanilla HTML5, CSS3, and JavaScript adhering strictly "
        "to design.md. Rather than generic dark dashboards, we crafted an editorial 'manuscript paper' aesthetic "
        "(--paper: #FAF8F5, --card: #FFFFFF, --indigo: #2E4057, --brass: #B8860B) paired with Fraunces serif headings "
        "and 5-line musical staff SVG dividers."
    )
    add_bold_lead(
        doc, "6.2 Features & Functionalities Incorporated: ",
        "(1) Schema-Driven Dynamic Controls: Upon load, the UI fetches /api/schema to generate 9 range-bounded sliders "
        "and a 114-genre dropdown automatically. (2) Curated Presets: One-click buttons ('Pop single', 'Acoustic ballad', "
        "'Ambient instrumental') demonstrate contrasting acoustic profiles. (3) Live Probability Meter & Contribution Bars: "
        "Displays real-time charting probability against the brass 0.48 threshold marker, plus diverging horizontal bars "
        "showing each feature's exact log-odds push. (4) Optional MP3 Upload Panel: Gated by ENABLE_AUDIO_UPLOAD, allowing "
        "drag-and-drop MP3 estimation with explicit 'Estimated' / 'Manual' badges and an experimental disclaimer."
    )
    add_bold_lead(
        doc, "6.3 Enhancing User Experience & Model Interaction: ",
        "Debounced live inference updates predictions smoothly as users drag sliders without page reloads, turning the "
        "classifier into an interactive 'what-if' simulator. Full keyboard Tab navigation, 2px indigo focus rings, and "
        "ARIA live regions ensure accessibility, while explicit error banners handle invalid inputs gracefully."
    )
    add_figure(doc, "reports/figures/audio_upload_verified.png", 4.5, "Figure 7: Live browser interface showing sliders, probability meter, contributions, and upload block.")


def build_section_deployment(doc):
    """Section 7: Model Deployment."""
    doc.add_heading("7. Model Deployment", level=1)
    add_bold_lead(
        doc, "7.1 Technologies & Platforms Used: ",
        "The inference service (app/main.py) is powered by FastAPI and served via Uvicorn ASGI on Python 3.11+. "
        "The frozen pipeline (models/final/model.joblib) and metadata schema (models/final/metadata.json) are loaded "
        "once at startup using an asynchronous lifespan context manager into app.state, providing sub-millisecond REST "
        "predictions via GET /api/health, GET /api/schema, GET /api/insights, POST /api/predict, and POST /api/extract."
    )
    add_bold_lead(
        doc, "7.2 Deployment Challenges (Scalability, Security & Reliability): ",
        "(1) Input Boundary Security: Pydantic v2 SongFeatures validates every numeric input against real training min/max "
        "bounds; out-of-range payloads return informative HTTP 422 JSON errors rather than silently extrapolating. "
        "(2) Upload DoS & Privacy Protection: The MP3 upload endpoint enforces a strict 20 MB size limit, validates MP3 "
        "extensions, processes audio in temporary files deleted immediately inside a finally block, and stays disabled "
        "by default via ENABLE_AUDIO_UPLOAD=false. (3) Scalability: Stateless FastAPI workers and preloaded joblib artifacts "
        "allow effortless horizontal scaling behind Docker/NGINX."
    )
    add_bold_lead(
        doc, "7.3 Monitoring & Maintenance Plan: ",
        "Operational readiness is monitored via /api/health probes returning model identifiers and training timestamps. "
        "Codebase integrity is protected by a 52-test automated pytest suite (tests/) covering split leakage, API 422 "
        "contracts, and synthetic audio extraction. Because music production trends evolve, periodic retraining on "
        "post-2023 Billboard charts and tracking of feature distribution drift (e.g., loudness shifts) are scheduled."
    )


def add_additional_explainability(doc):
    """Subsection 8.1 covering Explainability and Error Analysis."""
    doc.add_heading("8.1 Model Explainability, Hypotheses & Error Analysis", level=2)
    doc.add_paragraph(
        "To correct Gini impurity bias toward high-cardinality features, Random Forest impurity importances (genre: 0.326, "
        "loudness: 0.109, instrumentalness: 0.107) were contrasted with held-out test permutation importance (n=10 repeats; "
        "genre: 0.1345, loudness: 0.0267, instrumentalness: 0.0238). Standardized Logistic Regression coefficients (\u03b2) "
        "confirm that instrumentalness (\u03b2 = -1.868) and loudness (\u03b2 = +1.349) dominate audio prediction, followed "
        "by duration_sec (\u03b2 = -0.803) and danceability (\u03b2 = +0.552). Error analysis on the test set explains real-world "
        "anomalies: 'Animals' by Martin Garrix (True Hit, predicted 0.0082) failed because it is an instrumental EDM "
        "anthem, violating the dataset's vocal-presence rule."
    )
    add_figure(doc, "reports/figures/feature_importance_rf.png", 5.6, "Figure 8: Random Forest Gini Impurity vs. Held-Out Test Permutation Importance.")
    add_figure(doc, "reports/figures/logreg_coefficients.png", 5.0, "Figure 9: Standardized Logistic Regression coefficients (\u03b2).")


def add_additional_clustering_audio(doc):
    """Subsections 8.2 and 8.3 covering Clustering and Audio DSP."""
    doc.add_heading("8.2 Unsupervised Audio Clustering (K-Means & PCA)", level=2)
    doc.add_paragraph(
        "K-Means clustering (src/clustering.py) on the 9 standardized audio features identified k=4 natural acoustic "
        "archetypes (inertia: 42,067.74, silhouette: 0.1794; 2D PCA explains 48.7% variance). Crucially, Cluster 0 "
        "('Upbeat Danceable / Pop', 3,348 tracks) captures 466 of 673 total hits (13.92% hit rate) and cuts across "
        "dance, country, salsa, and pop—proving that commercial production parameters unite hits across genres."
    )
    add_figure(doc, "reports/figures/clusters_pca.png", 4.5, "Figure 10: 2D PCA projection of K-Means acoustic clusters.")

    doc.add_heading("8.3 Experimental Audio Signal Processing & Validation Audit", level=2)
    doc.add_paragraph(
        "An optional local MP3 extractor (src/audio_extractor.py) uses librosa, soundfile, and pyloudnorm to estimate "
        "duration_sec, tempo (beat tracking + octave fold), loudness (ITU-R BS.1770-4 LUFS), energy (RMS proxy), and "
        "danceability (non-saturating autocorrelation pulse clarity). Subjective features (valence, acousticness, "
        "instrumentalness, speechiness, genre) remain manual. In an honest 20-track audit (reports/audio_validation.csv), "
        "only duration_sec passed \u03c1 \u2265 0.60 (\u03c1 = 0.7188), label agreement was 55.0%, and mean |\u0394p| was 0.3451 "
        "(due to broadcast LUFS vs. Spotify master loudness offsets and track edit differences). Hence, the feature is "
        "strictly gated behind ENABLE_AUDIO_UPLOAD=false with an explicit experimental UI disclaimer."
    )


def build_section_additional(doc):
    """Section 8: Additional Components."""
    doc.add_heading("8. Additional Components", level=1)
    add_additional_explainability(doc)
    add_additional_clustering_audio(doc)


def generate_docx(output_path="reports/HitSong_University_Report.docx"):
    """Compile and save the complete university-level DOCX report."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = Document()
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)
        add_page_number_footer(section)

    configure_styles(doc)
    build_cover_page(doc)
    build_section_dataset(doc)
    doc.add_page_break()
    build_section_eda(doc)
    doc.add_page_break()
    build_section_preprocessing(doc)
    build_section_model_execution(doc)
    doc.add_page_break()
    build_section_evaluation(doc)
    doc.add_page_break()
    build_section_gui(doc)
    build_section_deployment(doc)
    doc.add_page_break()
    build_section_additional(doc)

    doc.save(output_path)
    print(f"DOCX Report successfully generated: {output_path}")


if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "reports/HitSong_University_Report.docx"
    generate_docx(out_file)
