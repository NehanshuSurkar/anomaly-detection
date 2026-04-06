import json, os, sys
import numpy as np
from datetime import datetime
from fpdf import FPDF
sys.path.insert(0, "/content/anomaly-detection/src")
from config import Config
cfg = Config()

def get_severity(score, threshold):
    ratio = score / threshold
    if ratio < 1.2:   return "LOW"
    elif ratio < 1.6: return "MEDIUM"
    else:             return "HIGH"

def save_json(report):
    os.makedirs(cfg.REPORT_DIR, exist_ok=True)
    path = os.path.join(cfg.REPORT_DIR, report["report_id"] + ".json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print("JSON saved: " + path)
    return path

def save_pdf(report):
    os.makedirs(cfg.REPORT_DIR, exist_ok=True)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_fill_color(31, 78, 121)
    pdf.rect(0, 0, 210, 28, "F")
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(10, 7)
    pdf.cell(0, 10, "CCTV ANOMALY DETECTION REPORT", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_xy(10, 18)
    pdf.cell(0, 6, "Model: ConvLSTM Autoencoder  |  Generated: " +
             report["generated_at"][:19], ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_xy(10, 33)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "SUMMARY", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "Report ID       : " + report["report_id"], ln=True)
    pdf.cell(0, 6, "Video Source    : " + report.get("video_source", "N/A"), ln=True)
    pdf.cell(0, 6, "Total Events    : " + str(report["total_events"]), ln=True)
    pdf.cell(0, 6, "HIGH severity   : " + str(report["by_severity"]["HIGH"]), ln=True)
    pdf.cell(0, 6, "MEDIUM severity : " + str(report["by_severity"]["MEDIUM"]), ln=True)
    pdf.cell(0, 6, "LOW severity    : " + str(report["by_severity"]["LOW"]), ln=True)
    pdf.cell(0, 6, "Threshold       : " + str(report["threshold"]) +
             "  (baseline: " + str(report["normal_baseline"]) + ")", ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "DETECTED EVENTS", ln=True)
    pdf.ln(2)
    sev_colors = {
        "HIGH"  : (255, 200, 200),
        "MEDIUM": (255, 243, 200),
        "LOW"   : (200, 230, 200)
    }
    for ev in report["events"]:
        r, g, b = sev_colors.get(ev["severity"], (240, 240, 240))
        pdf.set_fill_color(r, g, b)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7,
            "  Event {:03d} | {:12s} | {:6s} | Confidence: {:.1f}%".format(
                ev["event_id"], ev.get("anomaly_type", "Anomaly"),
                ev["severity"], ev.get("confidence", 0.0)),
            ln=True, fill=True)
        pdf.set_font("Helvetica", "", 9)
        if "start_time" in ev:
            pdf.cell(0, 5, "    Time  : " + ev["start_time"] +
                     " -> " + ev["end_time"], ln=True)
        pdf.cell(0, 5, "    Score : {:.5f}  |  Threshold: {:.5f}".format(
            ev.get("max_score", ev.get("score", 0)), ev["threshold"]), ln=True)
        thumb = ev.get("thumbnail")
        if thumb and os.path.exists(str(thumb)):
            try:
                pdf.image(thumb, w=40)
            except Exception:
                pass
        pdf.ln(2)
    path = os.path.join(cfg.REPORT_DIR, report["report_id"] + ".pdf")
    pdf.output(path)
    print("PDF saved: " + path)
    return path
