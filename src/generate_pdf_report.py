"""Comprehensive PDF report generation module for Hit Song Predictor.

Produces a publication-quality, complete academic & technical PDF report
including ML modeling, statistical analysis, audio extraction, and system design.
Runs standalone from root: python -m src.generate_pdf_report
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and draw running page totals."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        """Draw running header and footer on pages beyond the cover."""
        if self._pageNumber > 1:
            self.saveState()
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#555555"))
            # Running header
            self.drawString(54, 750, "Hit Song Predictor — Comprehensive Technical & Research Report")
            self.setStrokeColor(colors.HexColor("#D0D0D0"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)
            # Running footer
            self.line(54, 45, 558, 45)
            self.drawString(54, 32, "Confidential & Academic — Machine Learning Project")
            self.drawRightString(558, 32, f"Page {self._pageNumber} of {page_count}")
            self.restoreState()


def get_text_styles(styles):
    """Add core typographic styles for document generation."""
    primary = colors.HexColor("#1A2B4C")
    body_col = colors.HexColor("#222222")
    styles.add(ParagraphStyle(
        "CoverTitle", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=26, leading=32, textColor=primary
    ))
    styles.add(ParagraphStyle(
        "CoverSubtitle", parent=styles["Normal"], fontName="Helvetica",
        fontSize=13, leading=18, textColor=colors.HexColor("#4A5568")
    ))
    styles.add(ParagraphStyle(
        "CoverMeta", parent=styles["Normal"], fontName="Helvetica",
        fontSize=9, leading=14, textColor=colors.HexColor("#718096")
    ))
    styles.add(ParagraphStyle(
        "SecHeading", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=15, leading=19, textColor=primary, spaceBefore=14, spaceAfter=6,
        keepWithNext=True
    ))
    styles.add(ParagraphStyle(
        "Body", parent=styles["Normal"], fontName="Helvetica",
        fontSize=9, leading=13.5, textColor=body_col, spaceAfter=6
    ))


def get_report_styles():
    """Return all configured paragraph styles for document generation."""
    styles = getSampleStyleSheet()
    get_text_styles(styles)
    body_col = colors.HexColor("#222222")
    styles.add(ParagraphStyle(
        "TableHead", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=8, leading=10, textColor=colors.white, alignment=1
    ))
    styles.add(ParagraphStyle(
        "TableCell", parent=styles["Normal"], fontName="Helvetica",
        fontSize=8, leading=10.5, textColor=body_col
    ))
    styles.add(ParagraphStyle(
        "FigCaption", parent=styles["Normal"], fontName="Helvetica-Oblique",
        fontSize=8, leading=11, textColor=colors.HexColor("#555555"),
        alignment=1, spaceBefore=4, spaceAfter=10
    ))
    styles.add(ParagraphStyle(
        "CalloutText", parent=styles["Normal"], fontName="Helvetica",
        fontSize=8.5, leading=12.5, textColor=colors.HexColor("#1A202C")
    ))
    return styles


def sec_header(title, styles):
    """Generate section heading with divider line."""
    return [
        Paragraph(title, styles["SecHeading"]),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E0"), spaceAfter=8)
    ]


def make_callout(text, title="KEY FINDING", border_hex="#2B6CB0", bg_hex="#EDF2F7", styles=None):
    """Create a formatted callout box table with colored left border."""
    content = [
        Paragraph(f"<b>{title}</b>", styles["CalloutText"]),
        Spacer(1, 2),
        Paragraph(text, styles["CalloutText"]),
    ]
    t = Table([[content]], colWidths=[496])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg_hex)),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("LINEBEFORE", (0, 0), (0, -1), 3.5, colors.HexColor(border_hex)),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    return t


def make_table(headers, rows, col_widths, styles, align_right_cols=None):
    """Construct a clean, publication-grade ReportLab table."""
    align_right = align_right_cols or []
    data = []
    head_row = [Paragraph(h, styles["TableHead"]) for h in headers]
    data.append(head_row)
    for r in rows:
        row_cells = []
        for idx, c in enumerate(r):
            style = styles["TableCell"]
            if idx in align_right:
                style = ParagraphStyle("Right", parent=style, alignment=2)
            row_cells.append(Paragraph(str(c), style))
        data.append(row_cells)

    t = Table(data, colWidths=col_widths)
    t_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A2B4C")),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D2D6DC")),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            t_style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F8FAFC")))
    t.setStyle(TableStyle(t_style))
    return t


def make_figure(rel_path, width, height, caption, styles):
    """Embed an image safely with a formatted caption."""
    p = os.path.join(".", rel_path)
    if not os.path.exists(p):
        return [Paragraph(f"[Missing Figure: {rel_path}]", styles["Body"])]
    img = Image(p, width=width, height=height)
    return [
        Spacer(1, 4),
        img,
        Paragraph(caption, styles["FigCaption"]),
        Spacer(1, 4),
    ]


def build_cover(styles):
    """Generate professional title page flowables."""
    flow = [
        Spacer(1, 40),
        Paragraph("HIT SONG PREDICTOR", styles["CoverTitle"]),
        Spacer(1, 8),
        Paragraph(
            "An End-to-End Academic Machine Learning Investigation into Binary Billboard Hot 100 "
            "Success Prediction, Multi-Criteria Evaluation, Interpretability, and Audio Signal Processing",
            styles["CoverSubtitle"]
        ),
        Spacer(1, 24),
        HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1A2B4C"), spaceAfter=18),
        Paragraph("<b>Project Scope:</b> Graded Academic Machine Learning Capstone / Research Project", styles["CoverMeta"]),
        Paragraph("<b>Dataset Horizon:</b> Billboard Hot 100 Year-End Charts (2000–2023) & Spotify Audio Features", styles["CoverMeta"]),
        Paragraph("<b>Final Production Model:</b> Regularized Logistic Regression (C=1.0, balanced weights)", styles["CoverMeta"]),
        Paragraph("<b>Selected Decision Threshold:</b> \u03c4 = 0.48 (Youden's J maximizing on 5-fold training OOF)", styles["CoverMeta"]),
        Paragraph("<b>Target Metric Outcomes:</b> Test ROC-AUC = 0.929 | PR-AUC = 0.606 | Hit Recall = 85.2%", styles["CoverMeta"]),
        Paragraph("<b>Audio Signal Processing:</b> Tier 1 Local Extraction (Librosa, SoundFile, PyLoudNorm)", styles["CoverMeta"]),
        Paragraph("<b>Reproducibility Standard:</b> Pinned seed (random_state=42), 0 artist overlap, zero data leakage", styles["CoverMeta"]),
        Spacer(1, 30),
        make_callout(
            "This report documents the architectural, theoretical, mathematical, and practical software engineering "
            "foundations of the Hit Song Predictor. It covers data acquisition, artist-leakage prevention, model training, "
            "hyperparameter tuning, explainability analysis, out-of-fold thresholding, unsupervised clustering, production "
            "FastAPI/Vanilla JS deployment, and empirical audio feature extraction validation.",
            title="EXECUTIVE OVERVIEW", border_hex="#1A2B4C", bg_hex="#F0F4F8", styles=styles
        ),
        PageBreak()
    ]
    return flow


def build_exec_summary(styles):
    """Construct Executive Summary and Problem Statement."""
    flow = []
    flow.extend(sec_header("1. Executive Summary & Problem Statement", styles))
    p1 = (
        "<b>The Commercial Music Problem:</b> Commercial record labels and Artists & Repertoire (A&R) teams evaluate "
        "thousands of unreleased and newly recorded tracks each week. The commercial risk is extreme: signing, recording, "
        "and promoting an artist requires substantial capital outlay, yet fewer than 1% of commercially released songs "
        "ever achieve mainstream hit status on the Billboard Hot 100. Historically, A&R relied purely on subjective auditions."
    )
    p2 = (
        "<b>Project Objective:</b> Determine whether quantitative audio engineering properties (tempo, energy, danceability, "
        "loudness, valence, acousticness, instrumentalness, speechiness, duration) combined with genre categorization contain "
        "statistically reliable signals to predict commercial chart breakthrough. The system is formulated as binary classification: "
        "predicting whether a track will chart on the Billboard Year-End Hot 100 (y = 1) versus failing to chart (y = 0)."
    )
    p3 = (
        "<b>Asymmetric Commercial Costs:</b> In talent discovery, a <b>False Negative</b> (failing to identify a future chart hit) "
        "is catastrophic: the label forfeits streaming royalties, touring equity, and publishing rights. Conversely, a <b>False Positive</b> "
        "(auditing a track that ultimately fails to chart) incurs a modest evaluation overhead. Hence, models must not be selected "
        "on overall accuracy alone; operational hit capture (Recall) and false alarm control (Specificity) are paramount."
    )
    flow.extend([Paragraph(p1, styles["Body"]), Paragraph(p2, styles["Body"]), Paragraph(p3, styles["Body"])])
    flow.append(make_callout(
        "Commercial hit prediction is not about raw accuracy. On our imbalanced dataset (9.09% hits), a trivial majority "
        "classifier achieves 90.91% accuracy while discovering zero hits. Optimization must prioritize ROC-AUC, PR-AUC, and "
        "calibrated operational recall.",
        title="CORE MACHINE LEARNING PRINCIPLE", border_hex="#C53030", bg_hex="#FFF5F5", styles=styles
    ))
    flow.append(Spacer(1, 10))
    return flow


def build_dataset_section(styles):
    """Construct Dataset Architecture and Leakage Prevention section."""
    flow = []
    flow.extend(sec_header("2. Dataset Architecture & Leakage-Free Hygiene", styles))
    p1 = (
        "<b>Source Data & Downsampling:</b> The raw pool contains 80,237 candidate songs (673 Billboard Year-End Hot 100 hits "
        "and 79,564 non-charting tracks from 2000–2023). In the natural distribution, positives represent only 0.84% of records. "
        "To provide a learnable decision surface without distorting rank fidelity, negative tracks were downsampled 10:1 relative "
        "to positives (random_state=42), yielding <b>7,403 total records</b> (673 charted, 6,730 non-charted; 9.09% positive rate). "
        "Eleven ambiguous tracks with conflicting chart metadata were permanently excluded."
    )
    flow.append(Paragraph(p1, styles["Body"]))

    # Table of Dataset Breakdown
    headers = ["Dataset Partition", "Total Tracks", "Charted (y=1)", "Non-Charted (y=0)", "Hit Share", "Unique Artists", "Artist Overlap"]
    rows = [
        ["Raw Candidate Pool", "80,237", "673", "79,564", "0.84%", "24,812", "N/A"],
        ["Final Processed Set", "7,403", "673", "6,730", "9.09%", "4,581", "N/A"],
        ["Training Split (80%)", "5,923", "538", "5,385", "9.08%", "3,657", "—"],
        ["Held-Out Test (20%)", "1,480", "135", "1,345", "9.12%", "924", "0 (Zero Overlap)"],
    ]
    flow.append(make_table(headers, rows, [110, 60, 65, 75, 55, 65, 66], styles, align_right_cols=[1, 2, 3, 4, 5, 6]))
    flow.append(Spacer(1, 8))

    p2 = (
        "<b>Artist-Grouped Outer Split (Zero Artist Leakage):</b> Superstar artists (e.g., Drake, Taylor Swift) possess "
        "massive institutional marketing budgets and radio pluggers. If songs from the same artist appear in both train and test sets, "
        "the model inadvertently memorizes artist identity rather than generalizable acoustic signatures. To prevent this, the 80/20 "
        "outer partition was created using <b>StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)</b> grouped on "
        "normalized primary_artist. <b>Artist overlap between train and test is exactly ZERO</b>."
    )
    flow.append(Paragraph(p2, styles["Body"]))
    flow.extend(make_figure(
        "reports/figures/class_balance_modeling.png", 380, 260,
        "Figure 1: Class balance in the modeling dataset (10:1 ratio, 9.09% charted).", styles
    ))
    return flow


def build_eda_section(styles):
    """Construct Exploratory Data Analysis section."""
    flow = []
    flow.extend(sec_header("3. Exploratory Data Analysis & Acoustic Correlations", styles))
    p1 = (
        "The model operates on 10 predictive features: 9 continuous audio features and 1 categorical genre descriptor. "
        "Linear Pearson correlations against the binary charted target reveal decisive acoustic patterns:"
    )
    flow.append(Paragraph(p1, styles["Body"]))

    headers = ["Feature", "Type / Scale", "Pearson r", "p-value", "Statistical Finding & Interpretation"]
    rows = [
        ["loudness", "Continuous (dB)", "+0.168", "7.08e-48", "Strong positive correlation; commercial mastering drives loudness."],
        ["instrumentalness", "Continuous [0, 1]", "-0.164", "9.92e-46", "Strong negative; commercial hits almost universally feature vocals."],
        ["acousticness", "Continuous [0, 1]", "-0.139", "3.08e-33", "Strong negative; pure acoustic songs rarely chart on modern Hot 100."],
        ["danceability", "Continuous [0, 1]", "+0.134", "3.71e-31", "Significant positive; regular 4/4 dance pulses facilitate radio/club play."],
        ["energy", "Continuous [0, 1]", "+0.062", "8.13e-08", "Moderate positive; perceptual intensity correlates with chart success."],
        ["valence", "Continuous [0, 1]", "+0.048", "3.85e-05", "Mild positive; slight preference for emotionally positive tracks."],
        ["duration_sec", "Continuous (sec)", "-0.035", "2.77e-03", "Mild negative; radio programming penalizes excessive song length."],
        ["speechiness", "Continuous [0, 1]", "-0.016", "0.178", "Not statistically significant; speechiness alone does not determine hits."],
        ["tempo", "Continuous (BPM)", "-0.015", "0.191", "Not statistically significant; hit songs exist across all tempo bands."],
    ]
    flow.append(make_table(headers, rows, [80, 85, 55, 60, 216], styles, align_right_cols=[2, 3]))
    flow.append(Spacer(1, 8))

    flow.extend(make_figure(
        "reports/figures/correlation_heatmap.png", 390, 300,
        "Figure 2: Correlation heatmap across all 9 continuous audio features and chart target.", styles
    ))
    return flow


def build_modeling_section(styles):
    """Construct Modeling and Hyperparameter Tuning section."""
    flow = []
    flow.extend(sec_header("4. Machine Learning Pipeline: 8 Architectures Benchmarked", styles))
    p1 = (
        "<b>Model Training & Inner Cross-Validation:</b> To ensure rigorous comparison across algorithmic paradigms, "
        "eight distinct classifier families were benchmarked: Logistic Regression, Support Vector Machines (Linear and RBF), "
        "Random Forest, K-Nearest Neighbors, Gaussian Naive Bayes, and Decision Trees (Gini and Entropy). "
        "Hyperparameters were tuned using an inner <b>5-fold StratifiedGroupKFold cross-validation</b> on the training set, "
        "optimizing for ROC-AUC to balance sensitivity across operational thresholds."
    )
    p2 = (
        "<b>Strict Transformer Isolation:</b> Feature transformers were wrapped in scikit-learn Pipelines. "
        "StandardScaler was fit exclusively on training data (mean, std), and OneHotEncoder(drop='first', handle_unknown='ignore') "
        "encoded the 114 genres. Unseen categories at inference time are mapped to zero-vectors without causing dimensional faults."
    )
    flow.extend([Paragraph(p1, styles["Body"]), Paragraph(p2, styles["Body"])])

    headers = ["Model Architecture", "Tuned Hyperparameters", "Best CV ROC-AUC", "CV Std Dev", "Train Time"]
    rows = [
        ["Logistic Regression", "model__C: 1.0, class_weight: 'balanced'", "0.9469", "\u00b1 0.0086", "6.10s"],
        ["SVM (RBF Kernel)", "model__C: 10.0, model__gamma: 'auto'", "0.9463", "\u00b1 0.0080", "72.12s"],
        ["Linear SVM", "model__C: 1.0, class_weight: 'balanced'", "0.9447", "\u00b1 0.0085", "36.64s"],
        ["Random Forest", "model__max_depth: None, max_features: 'log2'", "0.9424", "\u00b1 0.0099", "46.67s"],
        ["K-Nearest Neighbors", "model__n_neighbors: 15", "0.8950", "\u00b1 0.0133", "10.03s"],
        ["Gaussian Naive Bayes", "model__var_smoothing: 1e-07", "0.8933", "\u00b1 0.0116", "1.59s"],
        ["Decision Tree (Gini)", "model__max_depth: 12, min_samples_leaf: 20", "0.8585", "\u00b1 0.0271", "2.77s"],
        ["Decision Tree (Entropy)", "model__max_depth: 8, min_samples_leaf: 20", "0.8503", "\u00b1 0.0266", "2.86s"],
    ]
    flow.append(make_table(headers, rows, [110, 180, 75, 65, 66], styles, align_right_cols=[2, 3, 4]))
    flow.append(Spacer(1, 10))
    return flow


def build_evaluation_section(styles):
    """Construct Held-Out Test Evaluation section."""
    flow = []
    flow.extend(sec_header("5. Held-Out Test Evaluation & Benchmark Comparison", styles))
    p1 = (
        "Each tuned model was frozen and evaluated on the held-out test split (1,480 tracks: 135 charted, 1,345 non-charted; "
        "0 artist overlap with train). Specificity was explicitly calculated via TN / (TN + FP) to measure true negative retention."
    )
    flow.append(Paragraph(p1, styles["Body"]))

    headers = ["Model", "ROC-AUC", "PR-AUC", "Recall", "Specificity", "Precision", "Accuracy", "F1", "TP", "TN", "FP", "FN"]
    rows = [
        ["Logistic Regression", "0.929", "0.606", "0.844", "0.862", "0.380", "0.860", "0.524", "114", "1159", "186", "21"],
        ["SVM (RBF Kernel)", "0.924", "0.595", "0.770", "0.865", "0.364", "0.856", "0.494", "104", "1163", "182", "31"],
        ["Linear SVM", "0.924", "0.602", "0.793", "0.852", "0.350", "0.847", "0.485", "107", "1146", "199", "28"],
        ["Random Forest", "0.918", "0.619", "0.296", "0.993", "0.816", "0.930", "0.435", "40", "1336", "9", "95"],
        ["K-Nearest Neighbors", "0.877", "0.509", "0.370", "0.987", "0.735", "0.930", "0.493", "50", "1327", "18", "85"],
        ["Gaussian Naive Bayes", "0.875", "0.305", "0.911", "0.787", "0.300", "0.798", "0.451", "123", "1058", "287", "12"],
        ["Decision Tree (Entropy)", "0.821", "0.443", "0.341", "0.989", "0.754", "0.930", "0.469", "46", "1330", "15", "89"],
        ["Decision Tree (Gini)", "0.821", "0.513", "0.363", "0.982", "0.671", "0.926", "0.471", "49", "1321", "24", "86"],
    ]
    flow.append(make_table(headers, rows, [96, 42, 40, 36, 42, 40, 42, 32, 28, 36, 32, 30], styles, align_right_cols=list(range(1, 12))))
    flow.append(Spacer(1, 8))

    flow.extend(make_figure(
        "reports/figures/roc_curves.png", 380, 290,
        "Figure 3: Held-out test ROC curves. Logistic Regression achieves highest ROC-AUC (0.929).", styles
    ))
    flow.extend(make_figure(
        "reports/figures/confusion_matrices.png", 470, 235,
        "Figure 4: Confusion matrices grid across all 8 models at default decision cutoffs.", styles
    ))
    return flow


def build_selection_section(styles):
    """Construct Model Selection Rationale section."""
    flow = []
    flow.extend(sec_header("6. Model Selection Rationale: Why Logistic Regression Won", styles))
    p1 = (
        "<b>Multi-Criteria Deployment Decision:</b> Selecting the production classifier required balancing statistical performance, "
        "operational hit capture, explainability, computational efficiency, and maintainability:"
    )
    p2 = (
        "<b>1. Accuracy Deception:</b> Ensembles (Random Forest, Decision Trees) exhibit ~93.0% accuracy on test data. "
        "However, a dummy majority classifier predicting 'never charts' achieves 90.9% accuracy. Random Forest's high accuracy "
        "masks an unacceptably low recall of <b>29.6% (missing 95 of 135 true chart hits)</b>, which is commercially fatal."
    )
    p3 = (
        "<b>2. PR-AUC Trade-off:</b> Random Forest slightly edged Logistic Regression in PR-AUC (0.619 vs 0.606), demonstrating "
        "excellent precision at extreme probability tails. But in initial screening, hit discovery (Recall) takes priority."
    )
    p4 = (
        "<b>3. Transparency & Latency:</b> Random Forest is a 300-tree black-box ensemble. Logistic Regression provides "
        "direct, interpretable log-odds coefficients (\u03b2) that can be audited by A&R executives and served with sub-millisecond latency."
    )
    flow.extend([Paragraph(p1, styles["Body"]), Paragraph(p2, styles["Body"]), Paragraph(p3, styles["Body"]), Paragraph(p4, styles["Body"])])
    flow.append(make_callout(
        "<b>Production Choice:</b> Regularized Logistic Regression (C=1.0, balanced weights) is the superior deployment choice. "
        "It achieves the top ROC-AUC (0.929), high recall (84.4%), stable calibrated probabilities, and full explainability.",
        title="DEPLOYMENT DECISION", border_hex="#2B6CB0", bg_hex="#EDF2F7", styles=styles
    ))
    flow.append(Spacer(1, 10))
    return flow


def build_explainability_section(styles):
    """Construct Explainability and Hypotheses Testing section."""
    flow = []
    flow.extend(sec_header("7. Explainability & Standardized Coefficients", styles))
    p1 = (
        "<b>Permutation Importance vs. Gini Impurity:</b> Tree impurity importances are notoriously biased toward high-cardinality "
        "features. To correct for this, Random Forest Gini importances were contrasted with held-out test permutation importances "
        "(n=10 repeats, random_state=42). While genre dominates tree splits, acoustic properties drive generalizable ROC-AUC drops:"
    )
    flow.append(Paragraph(p1, styles["Body"]))

    headers = ["Feature", "Gini Importance", "Permutation Drop (ROC-AUC)", "Std Dev", "Empirical Interpretation"]
    rows = [
        ["genre", "0.312", "0.1345", "\u00b1 0.0100", "Broad stylistic categorization is the strongest macro-level separator."],
        ["loudness", "0.098", "0.0267", "\u00b1 0.0069", "Commercial mastering loudness is the top predictive audio feature."],
        ["instrumentalness", "0.088", "0.0238", "\u00b1 0.0085", "Absence of vocals is heavily penalized on the Hot 100."],
        ["duration_sec", "0.081", "0.0151", "\u00b1 0.0026", "Length constraints reflect commercial radio programming formats."],
        ["acousticness", "0.071", "0.0147", "\u00b1 0.0059", "Acoustic instrumentation negatively associates with chart hits."],
        ["danceability", "0.053", "0.0052", "\u00b1 0.0052", "Regular 4/4 danceability provides positive commercial lift."],
        ["energy", "0.051", "0.0019", "\u00b1 0.0038", "Energy shows minor positive effect once loudness is accounted for."],
        ["valence", "0.046", "0.0012", "\u00b1 0.0030", "Musical positivity has negligible independent predictive power."],
    ]
    flow.append(make_table(headers, rows, [80, 70, 95, 60, 191], styles, align_right_cols=[1, 2, 3]))
    flow.append(Spacer(1, 8))

    flow.extend(make_figure(
        "reports/figures/logreg_coefficients.png", 440, 260,
        "Figure 5: Standardized Logistic Regression coefficients (\u03b2). Standardized features enable direct log-odds comparison.", styles
    ))
    return flow


def build_threshold_section(styles):
    """Construct Operational Thresholding section."""
    flow = []
    flow.extend(sec_header("8. Leakage-Safe Operational Thresholding", styles))
    p1 = (
        "<b>Avoiding Test-Set Overfitting:</b> Tuning decision thresholds directly on test data inflates performance estimates. "
        "Candidate thresholds were therefore identified exclusively from <b>5-fold training out-of-fold (OOF) predictions</b> "
        "and frozen before single evaluation on the held-out test split:"
    )
    flow.append(Paragraph(p1, styles["Body"]))

    headers = ["Threshold Strategy", "Selected \u03c4", "Accuracy", "Precision", "Recall", "Specificity", "F1", "TP", "TN", "FP", "FN"]
    rows = [
        ["Youden-J (Deployment)", "0.48", "0.8568", "0.3746", "0.8519", "0.8572", "0.5204", "115", "1153", "192", "20"],
        ["F1-Score Maximizing", "0.77", "0.9074", "0.4946", "0.6741", "0.9309", "0.5705", "91", "1252", "93", "44"],
    ]
    flow.append(make_table(headers, rows, [116, 45, 45, 45, 40, 45, 36, 30, 34, 30, 30], styles, align_right_cols=list(range(1, 11))))
    flow.append(Spacer(1, 8))

    flow.extend(make_figure(
        "reports/figures/threshold_tradeoff_cv.png", 440, 260,
        "Figure 6: Cross-validation Out-of-Fold (OOF) threshold trade-off curve across candidate operational operating points.", styles
    ))

    p2 = (
        "<b>Commercial Recommendation:</b> The Youden-J threshold (\u03c4 = 0.48) captures <b>85.2% of all true hits (115/135)</b> "
        "with only 20 false negatives, while rejecting 85.7% of non-hits. In contrast, \u03c4 = 0.77 misses 44 hits. "
        "For scouting applications, minimizing missed hits is the primary objective, making \u03c4 = 0.48 the optimal operating point."
    )
    flow.append(Paragraph(p2, styles["Body"]))
    return flow


def build_clustering_section(styles):
    """Construct Unsupervised Clustering section."""
    flow = []
    flow.extend(sec_header("9. Unsupervised Audio Feature Clustering", styles))
    p1 = (
        "<b>K-Means Clustering (k=4):</b> To investigate whether commercial hits form natural acoustic archetypes independent of "
        "marketing and genre labels, K-Means was fit on the 9 standardized audio features. Model selection leveled at k=4 "
        "(inertia: 42,067.74; peak silhouette: 0.1794). A 2D PCA projection explains 48.7% of feature variance."
    )
    flow.append(Paragraph(p1, styles["Body"]))

    headers = ["Cluster ID", "Archetype Label", "Track Count (% Total)", "Charted Hit Rate", "Representative Subgenres"]
    rows = [
        ["Cluster 0", "Upbeat Danceable / Pop", "3,348 (45.2%)", "13.92% (466 hits)", "dance (107), country (77), salsa (76), pagode (72)"],
        ["Cluster 1", "Acoustic / Downtempo / Ambient", "1,492 (20.2%)", "1.94% (29 hits)", "romance (69), sleep (63), new-age (62), tango (60)"],
        ["Cluster 2", "Intense / Fast / Rock & Metal", "2,315 (31.3%)", "6.83% (158 hits)", "grindcore (96), black-metal (79), hardstyle (75)"],
        ["Cluster 3", "High-Speechiness / Spoken Word", "248 (3.3%)", "8.06% (20 hits)", "comedy (76), j-dance (15), hardcore (12)"],
    ]
    flow.append(make_table(headers, rows, [60, 115, 85, 80, 156], styles, align_right_cols=[2, 3]))
    flow.append(Spacer(1, 8))

    flow.extend(make_figure(
        "reports/figures/clusters_pca.png", 390, 290,
        "Figure 7: 2D PCA projection of K-Means clusters (PC1: 30.9% variance, PC2: 17.8% variance).", styles
    ))
    return flow


def build_architecture_section(styles):
    """Construct Software Architecture and Web Deployment section."""
    flow = []
    flow.extend(sec_header("10. System Architecture & Web Deployment", styles))
    p1 = (
        "<b>FastAPI Production Backend (app/main.py):</b> Operates with an asynchronous lifespan context manager that loads the "
        "trained model bundle (models/final/model.joblib) and metadata schema (models/final/metadata.json) once at startup. "
        "Pydantic input models strictly validate numeric values against training domain bounds. Out-of-bounds features return "
        "transparent 422 JSON errors specifying offending fields. The /api/predict endpoint returns the hit probability, binary label, "
        "decision threshold, and local linear feature contributions (effect_i = x_standardized,i * \u03b2_i)."
    )
    p2 = (
        "<b>Vanilla Frontend (web/):</b> Engineered without heavy JS frameworks (no React, no build step, no Tailwind) using pure "
        "HTML/CSS/Vanilla JS adhering to design.md. Features a manuscript paper aesthetic (--paper, --card, --rule, --indigo), "
        "custom styled sliders, preset buttons, horizontal meter gauge with threshold indicator, and diverging contribution bars."
    )
    flow.extend([Paragraph(p1, styles["Body"]), Paragraph(p2, styles["Body"])])

    flow.extend(make_figure(
        "reports/figures/audio_upload_verified.png", 360, 308,
        "Figure 8: Browser-verified production UI showing feature sliders, prediction gauge, and experimental upload block.", styles
    ))
    return flow


def build_audio_extraction_section(styles):
    """Construct Audio Feature Extraction Pipeline section."""
    flow = []
    flow.extend(sec_header("11. Audio Extraction Pipeline (Tier 1 DSP)", styles))
    p1 = (
        "<b>Motivation & Design:</b> An optional capability was added enabling users to upload local MP3 audio files to estimate "
        "features directly on their machine using open-source Python libraries (librosa, soundfile, pyloudnorm). "
        "In accordance with project constraints, uploaded audio is processed in temporary memory and deleted immediately without cloud APIs."
    )
    flow.append(Paragraph(p1, styles["Body"]))

    headers = ["Feature", "Extraction Method", "Signal Processing Technique", "Status / Mode"]
    rows = [
        ["duration_sec", "Direct Sample Length", "Container sample count divided by sample rate (N / fs).", "Automatic (Physical)"],
        ["tempo", "Onset Autocorrelation", "Librosa onset envelope autocorrelation with [50, 180] BPM octave fold.", "Manual / Estimated"],
        ["loudness", "ITU-R BS.1770-4 LUFS", "Integrated physical loudness via PyLoudNorm meter. Distinct from Spotify.", "Manual / Estimated"],
        ["energy", "RMS Amplitude Proxy", "Normalized frame-wise RMS amplitude (clip(RMS / 0.20, 0, 1)).", "Manual / Estimated"],
        ["danceability", "Pulse Clarity Proxy", "Onset envelope autocorrelation peak prominence across [0.3s, 1.2s] lags.", "Manual / Estimated"],
        ["valence", "Omitted (Subjective)", "Emotional valence cannot be determined via simple DSP without heavy deep learning.", "Manual Only"],
        ["acousticness", "Omitted (Complex)", "Requires pretrained acoustic classification model; omitted to preserve lightweight stack.", "Manual Only"],
        ["instrumentalness", "Omitted (Complex)", "Requires vocal source separation (e.g. Spleeter/Demucs); omitted from Tier 1.", "Manual Only"],
        ["speechiness", "Omitted (Complex)", "Requires speech/voice activity detection model; omitted from Tier 1.", "Manual Only"],
        ["genre", "Omitted (Categorical)", "Genre is never inferred from audio; remains strictly a manual dropdown choice.", "Manual Only"],
    ]
    flow.append(make_table(headers, rows, [75, 110, 215, 96], styles))
    flow.append(Spacer(1, 10))
    return flow


def build_validation_audit_section(styles):
    """Construct Empirical Audio Validation & Audit section."""
    flow = []
    flow.extend(sec_header("12. Empirical Audio Validation & Acceptance Audit", styles))
    p1 = (
        "<b>Validation on 20 Held-Out Tracks:</b> To measure whether open-source DSP estimates match Spotify's proprietary API features, "
        "20 MP3 files corresponding to songs in the held-out test split were analyzed and evaluated against the frozen model pipeline. "
        "<b>The validation run failed all four equivalence acceptance gates</b>:"
    )
    flow.append(Paragraph(p1, styles["Body"]))

    headers = ["Validation Gate Criterion", "Target Requirement", "Observed Outcome (N=20)", "Audit Verdict"]
    rows = [
        ["Automatic Features (\u03c1 \u2265 0.60)", "\u2265 6 of 9 features", "1 / 9 features (duration_sec: \u03c1 = 0.7188)", "FAIL"],
        ["Model Label Agreement", "\u2265 80.0%", "55.0% (11 of 20 songs agree)", "FAIL"],
        ["Mean Absolute Probability Diff", "\u2264 0.1500 (15.0 pp)", "0.3451 (34.51 percentage points)", "FAIL"],
        ["Hit Share Discrepancy", "\u2264 15.0 pp", "45.0 pp (Spotify: 65.0%, Extracted: 20.0%)", "FAIL"],
        ["Overall Equivalence Gate", "All criteria pass", "0 of 4 criteria passed", "OVERALL FAIL"],
    ]
    flow.append(make_table(headers, rows, [140, 100, 176, 80], styles, align_right_cols=[3]))
    flow.append(Spacer(1, 8))

    p2 = (
        "<b>Root Causes & Remaining Limitations:</b><br/>"
        "1. <i>LUFS vs. Spotify Loudness:</i> ITU-R BS.1770-4 integrated loudness averages -14.22 LUFS on unmastered files compared to "
        "-7.31 dB on Spotify's streaming masters (offset \u2248 -6.91 dB). This offset penalizes positive loudness log-odds in the model.<br/>"
        "2. <i>Audio Cut & Edit Discrepancies:</i> Track versions varied substantially (e.g. Daddy Come Home was a 4m58s extended compilation "
        "track vs 2m22s on Spotify; My Way was the explicit video cut with a 63s outro).<br/>"
        "3. <i>Subjective Attributes:</i> Signal proxies cannot capture perceptual psychoacoustics without deep models.<br/>"
        "<b>Ethical AI Resolution:</b> The feature is integrated strictly as an <i>experimental tool</i> with an explicit UI disclaimer. "
        "Estimated features are never presented as true Spotify features, and all sliders remain manually adjustable."
    )
    flow.append(Paragraph(p2, styles["Body"]))
    return flow


def build_conclusion_section(styles):
    """Construct Conclusions, Reproducibility, and Defense Summary."""
    flow = []
    flow.extend(sec_header("13. Conclusion, System Limits & Defense Summary", styles))
    p1 = (
        "<b>Key Technical Takeaways:</b><br/>"
        "\u2022 <b>Generalization to Unseen Artists:</b> Achieved 0.929 test ROC-AUC with zero artist overlap.<br/>"
        "\u2022 <b>Commercial A&R Operating Point:</b> Threshold \u03c4 = 0.48 captures 85.2% of hits while rejecting 85.7% of non-hits.<br/>"
        "\u2022 <b>Acoustic Drivers of Chart Success:</b> Commercial hits strongly favor high mastering loudness, prominent vocal presence, "
        "and clear rhythmic danceability.<br/>"
        "\u2022 <b>Software Engineering Rigor:</b> 52 unit and integration tests passing; clean decoupled architecture.<br/>"
        "<br/>"
        "<b>System Boundaries (What the Model Cannot Do):</b><br/>"
        "\u2022 <i>Not a Virality Model:</i> Predicts acoustic feature viability, not marketing budgets, social media trends, or playlisting.<br/>"
        "\u2022 <i>Not Causal:</i> Positive coefficients reflect historical empirical associations, not a guarantee that boosting volume creates a hit.<br/>"
        "\u2022 <i>Temporal Horizon:</i> Trained on 2000–2023 commercial patterns; periodic retraining is required as music production evolves."
    )
    flow.append(Paragraph(p1, styles["Body"]))
    flow.append(Spacer(1, 10))
    flow.append(make_callout(
        "<b>End-to-End Reproduction Command:</b><br/>"
        "<code>pytest tests/ -v</code> (Runs all 52 automated tests)<br/>"
        "<code>python -m src.generate_pdf_report</code> (Regenerates this complete PDF report)<br/>"
        "<code>uvicorn app.main:app --host 127.0.0.1 --port 8000</code> (Launches the web application)",
        title="REPRODUCIBILITY & VERIFICATION", border_hex="#2B6CB0", bg_hex="#EDF2F7", styles=styles
    ))
    return flow


def generate_pdf(output_path="reports/HitSong_Comprehensive_Report.pdf"):
    """Compile and render the complete PDF report using ReportLab."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54
    )
    styles = get_report_styles()
    story = []

    story.extend(build_cover(styles))
    story.extend(build_exec_summary(styles))
    story.extend(build_dataset_section(styles))
    story.append(PageBreak())
    story.extend(build_eda_section(styles))
    story.append(PageBreak())
    story.extend(build_modeling_section(styles))
    story.extend(build_evaluation_section(styles))
    story.append(PageBreak())
    story.extend(build_selection_section(styles))
    story.extend(build_explainability_section(styles))
    story.append(PageBreak())
    story.extend(build_threshold_section(styles))
    story.append(PageBreak())
    story.extend(build_clustering_section(styles))
    story.append(PageBreak())
    story.extend(build_architecture_section(styles))
    story.extend(build_audio_extraction_section(styles))
    story.append(PageBreak())
    story.extend(build_validation_audit_section(styles))
    story.extend(build_conclusion_section(styles))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF Report successfully generated: {output_path}")


if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "reports/HitSong_Comprehensive_Report.pdf"
    generate_pdf(out_file)
