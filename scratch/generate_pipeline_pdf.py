"""Script to generate a publication-quality 3-page PDF explaining the Hit Song Predictor pipeline."""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas that writes dynamic running header and 'Page X of Y' footer."""

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
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Running Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(
                54, 11 * inch - 36, "Hit Song Predictor — System Architecture & Inference Pipeline"
            )
            self.drawRightString(8.5 * inch - 54, 11 * inch - 36, "Production Engineering Guide")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 11 * inch - 40, 8.5 * inch - 54, 11 * inch - 40)

        # Running Footer (all pages)
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 54, 30, footer_text)
        self.drawString(54, 30, "Academic ML Project &bull; Machine Learning Engineering Reference")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 40, 8.5 * inch - 54, 40)

        self.restoreState()


def build_pdf(filename="reports/Hit_Song_Predictor_Pipeline_Architecture.pdf"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=46,
        bottomMargin=46,
    )

    styles = getSampleStyleSheet()

    # Color Palette
    c_primary = colors.HexColor("#0F172A")    # slate 900
    c_accent = colors.HexColor("#2563EB")     # blue 600
    c_sub = colors.HexColor("#475569")        # slate 600
    c_bg_light = colors.HexColor("#F8FAFC")   # slate 50
    c_code_bg = colors.HexColor("#F1F5F9")    # slate 100
    c_border = colors.HexColor("#CBD5E1")     # slate 300
    c_blue_border = colors.HexColor("#93C5FD")

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=c_primary,
        spaceAfter=3,
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=14,
        textColor=c_accent,
        spaceAfter=8,
    )
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=c_sub,
    )
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=c_primary,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=c_accent,
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=c_primary,
        spaceAfter=4,
    )
    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=c_primary,
        leftIndent=10,
        spaceAfter=2,
    )
    code_inline_style = ParagraphStyle(
        'CodeSnippet',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#0F172A"),
    )
    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=c_sub,
    )
    table_text_style = ParagraphStyle(
        'TableText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=10.5,
        textColor=c_primary,
    )
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10.5,
        textColor=colors.white,
    )

    def code_box(code_text):
        p = Paragraph(code_text.replace("\n", "<br/>").replace(" ", "&nbsp;"), code_inline_style)
        t = Table([[p]], colWidths=[504])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), c_code_bg),
            ('BOX', (0, 0), (-1, -1), 0.5, c_border),
            ('TOPPADDING', (0, 0), (-1, -1), 3.5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        return t

    story = []

    # =========================================================================
    # PAGE 1: TITLE, EXECUTIVE OVERVIEW, ARCHITECTURE FLOW TABLE, STEPS 1 & 2
    # =========================================================================
    story.append(Paragraph("Hit Song Predictor — System Architecture & Inference Pipeline", title_style))
    story.append(Paragraph("End-to-End Technical Specification: Web Frontend, FastAPI Microservice, & ML Engine", subtitle_style))

    meta_text = (
        "<b>Artifact:</b> System Engineering Guide &bull; "
        "<b>Stack:</b> FastAPI, Scikit-Learn, Vanilla HTML5/CSS3/JS &bull; "
        "<b>Inference Host:</b> <font name='Courier'>http://127.0.0.1:8000</font> &bull; "
        "<b>Model:</b> Logistic Regression (<font name='Courier'>C=1.0, balanced</font>)"
    )
    story.append(Paragraph(meta_text, meta_style))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1, color=c_border, spaceBefore=2, spaceAfter=6))

    story.append(Paragraph("1. Executive Overview & Data Flow Architecture", h1_style))
    story.append(Paragraph(
        "The <b>Hit Song Predictor</b> predicts whether a song will chart on the <i>Billboard Hot 100</i> using "
        "Spotify audio features. The system couples a zero-build vanilla JavaScript interface with an asynchronous "
        "FastAPI service executing an in-memory scikit-learn pipeline with sub-5ms latency.",
        body_style
    ))

    seq_data = [
        [Paragraph("Pipeline Phase", table_header_style), Paragraph("Component Flow", table_header_style), Paragraph("Technical Mechanism & Operational Guarantee", table_header_style)],
        [Paragraph("<b>1. Server Startup</b>", table_text_style), Paragraph("FastAPI Lifespan &rarr; Disk", table_text_style), Paragraph("Loads <font name='Courier'>model.joblib</font> & <font name='Courier'>metadata.json</font> into memory once. Zero per-request disk I/O.", table_text_style)],
        [Paragraph("<b>2. Client Init</b>", table_text_style), Paragraph("Browser &rarr; <font name='Courier'>GET /api/schema</font>", table_text_style), Paragraph("Builds sliders from training min/max/step bounds. Zero hardcoded ranges in frontend.", table_text_style)],
        [Paragraph("<b>3. User Input</b>", table_text_style), Paragraph("Sliders &rarr; JS Debouncer", table_text_style), Paragraph("250ms debounce window prevents server request spamming during rapid slider adjustment.", table_text_style)],
        [Paragraph("<b>4. Validation</b>", table_text_style), Paragraph("Browser &rarr; <font name='Courier'>POST /api/predict</font>", table_text_style), Paragraph("Pydantic <font name='Courier'>SongFeatures</font> model bounds-checks all inputs; returns explicit HTTP 422 on violations.", table_text_style)],
        [Paragraph("<b>5. ML Inference</b>", table_text_style), Paragraph("FastAPI &rarr; ColumnTransformer", table_text_style), Paragraph("Standardizes numeric features (<font name='Courier'>StandardScaler</font>) and one-hot encodes genre, then evaluates sigmoid probability.", table_text_style)],
        [Paragraph("<b>6. Decision & XAI</b>", table_text_style), Paragraph("LogReg &rarr; Attribution Engine", table_text_style), Paragraph("Compares against frozen threshold <b>0.48</b> and computes exact linear feature contributions (<font name='Courier'>z_i &times; &beta;_i</font>).", table_text_style)],
        [Paragraph("<b>7. UI Render</b>", table_text_style), Paragraph("FastAPI &rarr; Browser DOM", table_text_style), Paragraph("Smooth cubic-bezier numeral count-up, threshold gauge fill, and horizontal diverging feature contribution bars.", table_text_style)],
    ]
    t_seq = Table(seq_data, colWidths=[70, 145, 289])
    t_seq.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_bg_light]),
    ]))
    story.append(t_seq)
    story.append(Spacer(1, 6))

    story.append(Paragraph("2. Step-by-Step Technical Breakdown (Steps 1 & 2)", h1_style))

    story.append(Paragraph("Step 1: Server Initialization & In-Memory Model Persistence", h2_style))
    story.append(Paragraph(
        "When FastAPI boots, it executes an <font name='Courier'>asynccontextmanager</font> lifespan function "
        "defined in <font name='Courier'>app/main.py</font>. The trained pipeline and metadata are cached directly in memory:",
        body_style
    ))
    story.append(code_box(
        "@asynccontextmanager\n"
        "async def lifespan(app_instance: FastAPI):\n"
        "    app_instance.state.model = joblib.load('models/final/model.joblib')\n"
        "    app_instance.state.metadata = load_metadata('models/final/metadata.json')\n"
        "    yield"
    ))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "&bull; <b>Pre-warmed Memory:</b> The pipeline is never deserialized during user requests. All transformation parameters (<font name='Courier'>StandardScaler</font> means/stds, <font name='Courier'>OneHotEncoder</font> categories) and Logistic Regression weights remain resident in RAM.<br/>"
        "&bull; <b>Artifact Integrity:</b> Contains the fitted pipeline and operational metadata produced by <font name='Courier'>src/select_final.py</font>.",
        bullet_style
    ))

    story.append(Paragraph("Step 2: Client Bootstrap & Dynamic Schema Construction", h2_style))
    story.append(Paragraph(
        "Upon loading <font name='Courier'>http://localhost:8000</font>, the client queries <font name='Courier'>GET /api/schema</font>. "
        "The backend returns feature names, empirical min/max/step ranges, genre categories, and the operational threshold (0.48).",
        body_style
    ))
    story.append(Paragraph(
        "&bull; <b>Single Source of Truth:</b> Sliders and dropdowns are dynamically generated from <font name='Courier'>metadata.json</font>. If training features or bounds change, the UI automatically updates without manual frontend edits.",
        bullet_style
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: STEPS 3 THROUGH 6
    # =========================================================================
    story.append(Paragraph("2. Step-by-Step Technical Breakdown (Steps 3 — 6)", h1_style))

    story.append(Paragraph("Step 3: User Input Handling & Debouncing", h2_style))
    story.append(Paragraph(
        "When users adjust sliders or click presets (<i>Mainstream Pop</i>, <i>Acoustic Ballad</i>, <i>Ambient</i>), "
        "the frontend updates visual labels instantly but buffers network requests with a <b>250ms debounce timer</b>. "
        "This prevents server request flooding while user scrubs sliders, transmitting a clean JSON payload:",
        body_style
    ))
    story.append(code_box(
        "{\n"
        '  "tempo": 122.0, "energy": 0.80, "danceability": 0.75, "valence": 0.65,\n'
        '  "loudness": -5.0, "acousticness": 0.12, "instrumentalness": 0.0,\n'
        '  "speechiness": 0.06, "duration_sec": 205.0, "genre": "pop"\n'
        "}"
    ))

    story.append(Paragraph("Step 4: Backend Ingestion & Pydantic Validation", h2_style))
    story.append(Paragraph(
        "Requests to <font name='Courier'>POST /api/predict</font> are validated by Pydantic's "
        "<font name='Courier'>SongFeatures</font> schema. Custom field validators enforce empirical training boundaries "
        "(e.g., loudness &isin; [-60.0, 5.0], danceability &isin; [0.0, 1.0]). If a boundary is violated:",
        body_style
    ))
    story.append(Paragraph(
        "&bull; <b>Error Handling:</b> Returns a structured <b>HTTP 422 Unprocessable Entity</b> specifying the exact invalid field and allowed range.<br/>"
        "&bull; <b>UI Integration:</b> The frontend highlights the offending slider and displays inline guidance without crashing.",
        bullet_style
    ))

    story.append(Paragraph("Step 5: Machine Learning Inference Pipeline", h2_style))
    story.append(Paragraph(
        "Validated input is structured into a 1-row Pandas DataFrame ordered strictly by feature name. "
        "The fitted scikit-learn pipeline executes three mathematical stages:",
        body_style
    ))
    story.append(Paragraph(
        "<b>1. Numeric Standardization:</b> Each of the 9 continuous audio features is transformed via training mean (&mu;) and standard deviation (&sigma;): "
        "<font name='Courier'>z_i = (x_i - &mu;_i) / &sigma;_i</font>.",
        bullet_style
    ))
    story.append(Paragraph(
        "<b>2. One-Hot Categorical Encoding:</b> The <font name='Courier'>genre</font> feature is mapped to binary indicator columns matching the training vocabulary.",
        bullet_style
    ))
    story.append(Paragraph(
        "<b>3. Probability Calculation:</b> Logistic Regression evaluates linear log-odds (&eta;) and computes charting probability (p):",
        bullet_style
    ))
    story.append(code_box(
        "eta = beta_0 + sum(beta_i * z_i)    # Linear combination in log-odds space\n"
        "p   = 1.0 / (1.0 + exp(-eta))       # Sigmoid activation function -> p in [0, 1]"
    ))

    story.append(Paragraph("Step 6: Decision Logic & The Frozen Operational Threshold", h2_style))
    story.append(Paragraph(
        "Rather than defaulting to an arbitrary 0.50 threshold, our system applies the <b>leakage-safe Youden's J "
        "optimal threshold of 0.48</b>, selected via out-of-fold cross-validation on training data:",
        body_style
    ))
    story.append(code_box(
        "threshold = 0.48\n"
        "label = 1 if probability >= threshold else 0    # 1: 'Likely to chart', 0: 'Unlikely to chart'"
    ))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "&bull; <b>Commercial A&R Screening:</b> At 0.48, the model captures <b>85.19% of true hits</b> (115 of 135 hits caught on held-out test data), prioritizing talent discovery over false-negative risk.<br/>"
        "&bull; <b>Leakage Safety:</b> The threshold was locked prior to final testing; single-pass held-out test evaluation confirmed Youden's J of 0.7091, verifying zero overfitting.",
        bullet_style
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: STEPS 7 & 8, LATENCY TABLE, POWERSHELL RUN GUIDE
    # =========================================================================
    story.append(Paragraph("2. Step-by-Step Technical Breakdown (Steps 7 & 8)", h1_style))

    story.append(Paragraph("Step 7: Local Feature Attribution (Explainability Engine)", h2_style))
    story.append(Paragraph(
        "Because Logistic Regression is linear in log-odds space, exact feature contributions are computed "
        "analytically in <b>< 0.5 milliseconds</b> without stochastic approximations like KernelSHAP:",
        body_style
    ))
    story.append(code_box(
        "Effect_i = z_i * beta_i    # Standardized feature value multiplied by model coefficient"
    ))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "&bull; <b>Positive Effect (&gt; 0):</b> Pushes the song's probability upward toward charting (e.g., loudness +0.42, pop genre +0.28).<br/>"
        "&bull; <b>Negative Effect (&lt; 0):</b> Pulls probability downward away from charting (e.g., instrumentalness -0.31, excessive duration -0.11).<br/>"
        "The API sorts features by absolute impact and returns the top 5 drivers in the response payload.",
        body_style
    ))

    story.append(Paragraph("Step 8: Frontend Rendering & Visual Feedback", h2_style))
    story.append(Paragraph(
        "Upon receiving the JSON payload, <font name='Courier'>web/app.js</font> updates the user interface in real time:",
        body_style
    ))
    story.append(Paragraph(
        "&bull; <b>Numeral Animation:</b> Probability percentage counts up smoothly using a cubic-bezier easing curve (<font name='Courier'>requestAnimationFrame</font>).<br/>"
        "&bull; <b>Horizontal Threshold Gauge:</b> Fill bar and knob glide to the calculated percentage relative to a permanent marker at 0.48.<br/>"
        "&bull; <b>Verdict Banner:</b> Switches dynamically between <i>'Likely to chart'</i> (green) and <i>'Unlikely to chart'</i> (slate).<br/>"
        "&bull; <b>Diverging Contribution Bars:</b> Renders horizontal bars showing positive drivers branching right and negative dampeners branching left.",
        bullet_style
    ))

    story.append(Spacer(1, 4))
    story.append(Paragraph("3. End-to-End Latency Profile", h1_style))
    story.append(Paragraph(
        "The system delivers end-to-end feedback in approximately <b>270 ms</b> (instantaneous to human perception):",
        body_style
    ))

    lat_data = [
        [Paragraph("Pipeline Stage", table_header_style), Paragraph("Underlying Mechanism", table_header_style), Paragraph("Duration", table_header_style)],
        [Paragraph("Client Event", table_text_style), Paragraph("Slider scrub & local DOM label update", table_text_style), Paragraph("< 1 ms", table_text_style)],
        [Paragraph("Input Debounce", table_text_style), Paragraph("Buffers rapid UI slider events to prevent request spamming", table_text_style), Paragraph("250 ms", table_text_style)],
        [Paragraph("HTTP Transport", table_text_style), Paragraph("Local loopback fetch() over HTTP/1.1", table_text_style), Paragraph("< 2 ms", table_text_style)],
        [Paragraph("Pydantic Validation", table_text_style), Paragraph("In-memory type checking & boundary validation", table_text_style), Paragraph("< 0.5 ms", table_text_style)],
        [Paragraph("Pipeline Inference", table_text_style), Paragraph("ColumnTransformer scaling + Logistic Sigmoid evaluation", table_text_style), Paragraph("< 1.5 ms", table_text_style)],
        [Paragraph("Feature Attribution", table_text_style), Paragraph("Dot-product linear effect calculation (<font name='Courier'>z_i &times; &beta;_i</font>)", table_text_style), Paragraph("< 0.5 ms", table_text_style)],
        [Paragraph("DOM Render", table_text_style), Paragraph("Eased count-up animation & SVG/CSS bar styling", table_text_style), Paragraph("16 ms (1 frame)", table_text_style)],
        [Paragraph("<b>Total Turnaround</b>", table_header_style), Paragraph("<b>Slider release to full UI visual verdict</b>", table_header_style), Paragraph("<b>~ 270 ms</b>", table_header_style)],
    ]
    t_lat = Table(lat_data, colWidths=[105, 295, 104])
    t_lat.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, c_bg_light]),
        ('BACKGROUND', (0, -1), (-1, -1), c_accent),
    ]))
    story.append(t_lat)

    story.append(Spacer(1, 4))
    story.append(Paragraph("4. Server Execution & Windows PowerShell Troubleshooting", h1_style))
    story.append(Paragraph(
        "To launch the production server from Windows PowerShell without PATH conflicts:",
        body_style
    ))
    story.append(code_box(
        "# Navigate to repository root (HitSong) and launch via Python module:\n"
        "python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"
    ))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "<b>Important Operational Notes:</b><br/>"
        "1. <b>Execution Directory:</b> Always run from the root directory (<font name='Courier'>HitSong</font>), not from inside <font name='Courier'>app/</font>. Relative artifact paths (<font name='Courier'>models/final/</font>, <font name='Courier'>web/</font>) resolve from repo root.<br/>"
        "2. <b>Command Syntax:</b> Invoking <font name='Courier'>python -m uvicorn</font> uses Python's environment directly, bypassing Windows PATH errors where <font name='Courier'>uvicorn.exe</font> is not recognized.",
        callout_style
    ))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated: {filename}")


if __name__ == "__main__":
    build_pdf()
