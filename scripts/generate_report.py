"""Generate updated one-page PDF report with verified results and honest evaluation status."""
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

doc = SimpleDocTemplate(
    "report/RAG_Evaluation_Report.pdf",
    pagesize=A4,
    leftMargin=1.3*cm, rightMargin=1.3*cm,
    topMargin=0.9*cm, bottomMargin=0.9*cm,
)

styles = getSampleStyleSheet()
title_style = ParagraphStyle("title", parent=styles["Heading1"],
    fontSize=13, spaceAfter=1, spaceBefore=0, alignment=TA_CENTER, textColor=colors.HexColor("#1a237e"))
h2_style = ParagraphStyle("h2", parent=styles["Heading2"],
    fontSize=9, spaceAfter=2, spaceBefore=3, textColor=colors.HexColor("#283593"))
body = ParagraphStyle("body", parent=styles["Normal"],
    fontSize=7.8, leading=10.5, spaceAfter=2)
small = ParagraphStyle("small", parent=styles["Normal"],
    fontSize=7.0, leading=9.0, textColor=colors.HexColor("#444444"))
cell_style = ParagraphStyle("cell", parent=styles["Normal"],
    fontSize=7.0, leading=8.5, textColor=colors.black)
hcell_style = ParagraphStyle("hcell", parent=styles["Normal"],
    fontSize=7.0, leading=8.5, textColor=colors.white, fontName="Helvetica-Bold")

def tbl(data, col_widths, header=True):
    wrapped_data = []
    for r_idx, row in enumerate(data):
        wrapped_row = []
        for cell in row:
            if isinstance(cell, str):
                st = hcell_style if (header and r_idx == 0) else cell_style
                wrapped_row.append(Paragraph(cell, st))
            else:
                wrapped_row.append(cell)
        wrapped_data.append(wrapped_row)
        
    t = Table(wrapped_data, colWidths=col_widths)
    style = [
        ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f8f9fa")]),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 3),
        ("RIGHTPADDING", (0,0), (-1,-1), 3),
        ("TOPPADDING", (0,0), (-1,-1), 1.2),
        ("BOTTOMPADDING", (0,0), (-1,-1), 1.2),
    ]
    if header:
        style += [
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#283593")),
        ]
    t.setStyle(TableStyle(style))
    return t

story = []

story.append(Paragraph("Open-Source RAG System — Evaluation Report", title_style))
story.append(Paragraph("RBI Document QA · qwen2.5:7b · BAAI/bge-base-en-v1.5 · BAAI/bge-reranker-base · Local Ollama", 
    ParagraphStyle("sub", parent=body, fontSize=7.2, alignment=TA_CENTER, textColor=colors.HexColor("#555555"))))
story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#283593"), spaceAfter=2))

# 1. System Overview
story.append(Paragraph("1. System Overview", h2_style))
story.append(Paragraph(
    "Retrieval-Augmented Generation pipeline over RBI Master Directions corpus. "
    "Documents chunked (512 tokens, 64 overlap), indexed with FAISS + BGE embeddings. "
    "Optional cross-encoder reranking (bge-reranker-base). Generation and LLM judge use local qwen2.5:7b via Ollama. "
    "Evaluations run on 40-question benchmark dataset (26 retrieval-labeled, 8 unanswerable, 6 ambiguous).",
    body))

# 2. Retrieval comparison
story.append(Paragraph("2. Retrieval: Baseline vs. Reranking (n=40, same dataset)", h2_style))
story.append(tbl([
    ["Configuration", "Recall@5", "Recall@10", "MRR@10", "Retrieval p50", "Retrieval p95"],
    ["Baseline (FAISS only)", "0.769", "0.885", "0.648", "—", "—"],
    ["+ bge-reranker-base (×4 candidates)", "0.962 (+25pp)", "0.962 (+8pp)", "0.862 (+33pp)", "14.1 s", "15.2 s"],
], [6.2*cm, 2.4*cm, 2.4*cm, 2.4*cm, 2.4*cm, 2.4*cm]))
story.append(Paragraph(
    "Source: results/reranking.json (retrieval_only=True, 40 Qs). Reranking retrieves 4× candidates then re-scores with cross-encoder; final top-10 returned.",
    small))

# 3. Answer quality
story.append(Paragraph("3. Answer Quality — Full-RAG with LLM Judge (n=40)", h2_style))
story.append(tbl([
    ["Metric", "Value", "Notes"],
    ["LLM judge mean correctness", "0.765", "34 judged (6 skipped: ambiguous, no reference answer)"],
    ["LLM judge mean faithfulness", "0.894", "Proportion of generated factual claims verifiable in retrieved context"],
    ["Exact-substring correctness", "0.115", "Naive baseline heuristic; prefer judge score for semantic correctness"],
    ["Judge API failures", "0", "0/34 local Ollama calls failed; all valid JSON returned"],
], [5.5*cm, 2.2*cm, 10.5*cm]))
story.append(Paragraph("Source: results/baseline_qwen.json (full_rag, qwen2.5:7b judge).", small))

# 4. Abstention
story.append(Paragraph("4. Abstention & Refusal Detection", h2_style))
story.append(tbl([
    ["Run", "Abstention Recall", "Abstention Precision", "Hallucination Rate", "n"],
    ["Targeted (unanswerable-only)", "1.000 (8/8)", "1.000 (0 FP)", "0.000", "8"],
    ["Full 40-q run (corrected detector)", "1.000 (8/8)", "0.889 (1 FP)", "0.000", "40"],
    ["Full 40-q run (original detector)", "0.000 (0/8)", "—", "1.000", "40"],
], [5.4*cm, 3.2*cm, 3.2*cm, 3.2*cm, 3.2*cm]))
story.append(Paragraph(
    "Original run: model produced valid refusals using lexical variants ('does not contain data', etc.) that escaped 9-pattern detector. "
    "Fix: 5 additional patterns added to is_refusal_response() in generation.py; 8/8 unanswerable detected. 1 false positive in full run.",
    small))

# 5. LLM Judge Validation & Cohen's Kappa
story.append(Paragraph("5. LLM Judge Validation & Human Annotation Status", h2_style))
story.append(tbl([
    ["Validation Dimension", "Status / Result", "Methodology & Verification Details"],
    ["LLM Judge Execution", "34 judged / 6 skipped", "0 failures. Q35–Q40 skipped by design (ambiguous type, empty gold answer)."],
    ["Judge Bias & Failure Modes", "Identified & Verified", "Self-evaluation bias (same model qwen2.5:7b). Judge failed on 6/8 unanswerable questions due to distractor context attention drift."],
    ["Human-Labelled Samples (20)", "Template Prepared", "Stratified 20-sample template generated: eval/human_validation_20_template.csv (7 factual, 3 multi-hop, 3 table, 4 unanswerable, 3 ambig)."],
    ["Cohen's Kappa (Human vs Judge)", "Pending Human Labels", "Dataset 'answerable' flag is not human rating. Genuine kappa requires human scoring. Automated script: scripts/calculate_kappa.py."],
], [4.8*cm, 3.4*cm, 10.0*cm]))

# 6. Latency
story.append(Paragraph("6. Latency Breakdown (Measured, n=40 per row)", h2_style))
story.append(tbl([
    ["Component", "mean", "p50", "p95", "min", "max", "Source"],
    ["Retrieval + reranking (baseline)", "14.0 s", "14.1 s", "15.2 s", "12.2 s", "15.6 s", "reranking.json"],
    ["Retrieval + reranking (rerank run)", "18.6 s", "16.7 s", "30.4 s", "14.5 s", "37.0 s", "reranking_qwen.json"],
    ["Full RAG baseline (no judge)", "86.8 s", "83.4 s", "121.7 s", "12.1 s", "136.6 s", "baseline.json"],
    ["Full RAG + LLM judge", "102.5 s", "91.3 s", "174.7 s", "79.3 s", "208.5 s", "baseline_qwen.json"],
], [4.4*cm, 1.5*cm, 1.5*cm, 1.5*cm, 1.5*cm, 1.6*cm, 6.2*cm]))

# 7. Best Validated Configuration
story.append(Paragraph("7. Best Validated Configuration", h2_style))
story.append(tbl([
    ["Dimension", "Best Result", "Configuration"],
    ["Retrieval", "Recall@5 = 0.962, MRR@10 = 0.862", "FAISS + BGE + bge-reranker-base (retrieves 4× top_k candidates)"],
    ["Answer correctness & faith", "Corr = 0.765, Faith = 0.894 (LLM judge)", "qwen2.5:7b, temp=0.0, evidence-grounded prompt"],
    ["Abstention (unanswerable)", "Recall = 1.000, Precision = 0.889", "Explicit refusal prompt constraint + expanded refusal pattern detector"],
], [3.2*cm, 5.0*cm, 10.0*cm]))

# 8. Limitations & Honest Caveats
story.append(Paragraph("8. Key Limitations & Methodological Caveats", h2_style))
limitations = [
    "Cohen's kappa: Cannot claim κ=1.0 without genuine human annotation. Dataset metadata was previously conflated with human labels; template and calculation script are now prepared for human input.",
    "Judge failure on negative queries: LLM judge failed on 6/8 valid refusals because retrieved distractor context distracted the judge prompt.",
    "Self-evaluation bias: Both generation and LLM judge use qwen2.5:7b, inflating apparent alignment; external oracle recommended.",
    "Hardware latency: Generation requires 83–102 s per question on CPU/local hardware; requires streaming or GPU acceleration for interactive demo.",
]
for lim in limitations:
    story.append(Paragraph(f"• {lim}", small))

story.append(Spacer(1, 1))
story.append(HRFlowable(width="100%", thickness=0.3, color=colors.HexColor("#999999"), spaceAfter=1))
story.append(Paragraph(
    "Tests: 63/63 pass (pytest -q) · Demo: streamlit run app.py · Human Validation: python scripts/calculate_kappa.py",
    ParagraphStyle("footer", parent=small, alignment=TA_CENTER, fontSize=6.5)))

doc.build(story)
print("PDF written to report/RAG_Evaluation_Report.pdf")