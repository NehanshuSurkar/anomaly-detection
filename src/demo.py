import torch
import cv2
import os
import sys
import json
import numpy as np
from datetime import datetime, timedelta
from fpdf import FPDF
sys.path.insert(0, "/content/anomaly-detection/src")
from config import Config
from model import ConvLSTMAE
from data_loader import make_sequences
from inference import get_adaptive_threshold, compute_scores_on_folder

cfg = Config()


# ── Utility ───────────────────────────────────────────────────────

def frames_to_time(frame_idx, fps):
    seconds = frame_idx / max(fps, 1)
    return str(timedelta(seconds=int(seconds)))


def get_severity(score, threshold):
    ratio = score / threshold
    if ratio < 1.2:
        return "LOW"
    elif ratio < 1.6:
        return "MEDIUM"
    else:
        return "HIGH"


# ── Frame Extraction ─────────────────────────────────────────────────────────

def extract_frames_from_video(video_path):
    """Extract and preprocess all frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video: " + video_path)

    fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("  FPS: {:.1f} | Total frames: {}".format(fps, total))

    raw_frames  = []   # original BGR frames for annotation
    gray_frames = []   # preprocessed for model

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (cfg.FRAME_W, cfg.FRAME_H))
        gray_frames.append(gray.astype(np.float32) / 255.0)

    cap.release()
    print("  Extracted {} frames".format(len(raw_frames)))
    return np.array(gray_frames), raw_frames, fps


# ── Scoring ──────────────────────────────────────────────────────────────────

def score_all_windows(model, gray_frames, device):
    """Score every SEQ_LEN-frame sliding window."""
    seqs    = make_sequences(gray_frames, cfg.SEQ_LEN, cfg.STRIDE)
    scores  = []
    indices = []

    model.eval()
    with torch.no_grad():
        for i, seq in enumerate(seqs):
            t     = torch.from_numpy(seq).unsqueeze(1).unsqueeze(0).to(device)
            r     = model(t)
            score = model.anomaly_score(t, r).item()
            # Map score to centre frame of this window
            centre = i * cfg.STRIDE + cfg.SEQ_LEN // 2
            scores.append(score)
            indices.append(centre)

    return np.array(scores), np.array(indices)


def build_frame_score_array(scores, indices, total_frames):
    """Interpolate window scores to per-frame scores."""
    frame_scores = np.zeros(total_frames)
    for idx, score in zip(indices, scores):
        if idx < total_frames:
            frame_scores[idx] = score
    # Fill gaps with nearest scored frame
    last = scores[0] if len(scores) > 0 else 0.0
    for i in range(total_frames):
        if frame_scores[i] == 0:
            frame_scores[i] = last
        else:
            last = frame_scores[i]
    # Smooth with moving average to reduce flicker
    kernel = np.ones(5) / 5
    frame_scores = np.convolve(frame_scores, kernel, mode="same")
    return frame_scores


# ── Event Detection ───────────────────────────────────────────────────────────

def detect_events(scores, indices, threshold, fps):
    """Group consecutive anomaly windows into discrete events."""
    events       = []
    event_id     = 1
    in_event     = False
    event_scores = []
    event_start  = 0

    for frame_idx, score in zip(indices, scores):
        if score > threshold:
            if not in_event:
                in_event     = True
                event_start  = frame_idx
                event_scores = []
            event_scores.append(score)
        else:
            if in_event:
                severity = get_severity(max(event_scores), threshold)
                events.append({
                    "event_id"    : event_id,
                    "start_frame" : int(event_start),
                    "end_frame"   : int(frame_idx),
                    "start_time"  : frames_to_time(event_start, fps),
                    "end_time"    : frames_to_time(frame_idx, fps),
                    "anomaly_type": "Anomaly",
                    "severity"    : severity,
                    "max_score"   : round(float(max(event_scores)), 6),
                    "threshold"   : round(float(threshold), 6),
                    "confidence"  : round(
                        min((max(event_scores) / threshold - 1.0) * 100, 99.9), 1)
                })
                event_id += 1
                in_event  = False

    if in_event and event_scores:
        severity = get_severity(max(event_scores), threshold)
        events.append({
            "event_id"    : event_id,
            "start_frame" : int(event_start),
            "end_frame"   : int(indices[-1]),
            "start_time"  : frames_to_time(event_start, fps),
            "end_time"    : frames_to_time(indices[-1], fps),
            "anomaly_type": "Anomaly",
            "severity"    : severity,
            "max_score"   : round(float(max(event_scores)), 6),
            "threshold"   : round(float(threshold), 6),
            "confidence"  : round(
                min((max(event_scores) / threshold - 1.0) * 100, 99.9), 1)
        })
    return events


# ── Annotated Video ───────────────────────────────────────────────────────────

def create_annotated_video(raw_frames, frame_scores, threshold, fps, out_path):
    """Write video with real-time anomaly overlay."""
    if not raw_frames:
        print("No frames to annotate.")
        return

    h, w    = raw_frames[0].shape[:2]
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    writer  = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    max_err = frame_scores.max() + 1e-8

    for i, frame in enumerate(raw_frames):
        frame      = frame.copy()
        score      = frame_scores[i] if i < len(frame_scores) else 0.0
        is_anomaly = score > threshold

        color  = (0, 0, 255) if is_anomaly else (0, 200, 0)
        status = "ANOMALY DETECTED" if is_anomaly else "NORMAL"

        # Border
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 5)

        # Top bar
        cv2.rectangle(frame, (0, 0), (w, 48), color, -1)

        # Status text
        cv2.putText(frame, status, (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Error value top-right
        cv2.putText(frame, "Err:{:.4f}".format(score), (w - 180, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Timestamp bottom-left
        ts = frames_to_time(i, fps)
        cv2.putText(frame, ts, (12, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        # Error bar bottom
        bar_w = int((score / max_err) * (w - 20))
        cv2.rectangle(frame, (10, h - 30), (10 + bar_w, h - 15), color, -1)

        # Threshold marker on bar
        thresh_x = int((threshold / max_err) * (w - 20)) + 10
        cv2.line(frame, (thresh_x, h - 38), (thresh_x, h - 10),
                 (255, 255, 0), 2)
        cv2.putText(frame, "Threshold", (thresh_x + 4, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

        writer.write(frame)

    writer.release()
    print("Annotated video saved: " + out_path)


# ── PDF Report ────────────────────────────────────────────────────────────────

def save_pdf_report(report, out_path):
    """Generate professional PDF report."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header
    pdf.set_fill_color(31, 78, 121)
    pdf.rect(0, 0, 210, 30, "F")
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(10, 8)
    pdf.cell(0, 10, "CCTV ANOMALY DETECTION REPORT", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_xy(10, 20)
    pdf.cell(0, 6, "Model: Hybrid ConvLSTM Autoencoder  |  Generated: " +
             report["generated_at"][:19], ln=True)

    # Summary
    pdf.set_text_color(0, 0, 0)
    pdf.set_xy(10, 35)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "ANALYSIS SUMMARY", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_fill_color(240, 248, 255)
    pdf.cell(0, 6, "Report ID       : " + report["report_id"],
             ln=True, fill=True)
    pdf.cell(0, 6, "Video Source    : " + report.get("video_source", "N/A"),
             ln=True, fill=True)
    pdf.cell(0, 6, "Total Duration  : " + report.get("duration", "N/A"),
             ln=True, fill=True)
    pdf.cell(0, 6, "AUC Score       : 0.9371  (Model Performance)",
             ln=True, fill=True)
    pdf.cell(0, 6, "Total Events    : " + str(report["total_events"]),
             ln=True, fill=True)
    pdf.cell(0, 6, "HIGH severity   : " + str(report["by_severity"]["HIGH"]),
             ln=True, fill=True)
    pdf.cell(0, 6, "MEDIUM severity : " + str(report["by_severity"]["MEDIUM"]),
             ln=True, fill=True)
    pdf.cell(0, 6, "LOW severity    : " + str(report["by_severity"]["LOW"]),
             ln=True, fill=True)
    pdf.cell(0, 6, "Threshold       : " + str(report["threshold"]) +
             "  (Normal baseline: " + str(report["normal_baseline"]) + ")",
             ln=True, fill=True)
    pdf.ln(4)

    # Result box
    if report["total_events"] == 0:
        pdf.set_fill_color(200, 240, 200)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 12, "  RESULT: NO ANOMALY DETECTED", ln=True, fill=True)
    else:
        pdf.set_fill_color(255, 200, 200)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 12,
                 "  RESULT: {} ANOMALOUS EVENT(S) DETECTED".format(
                     report["total_events"]),
                 ln=True, fill=True)
    pdf.ln(4)

    # Events
    if report["events"]:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "DETECTED EVENTS", ln=True)
        pdf.ln(2)

        sev_colors = {
            "HIGH"  : (255, 200, 200),
            "MEDIUM": (255, 243, 200),
            "LOW"   : (200, 230, 200),
        }

        for ev in report["events"]:
            r, g, b = sev_colors.get(ev["severity"], (240, 240, 240))
            pdf.set_fill_color(r, g, b)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7,
                     "  Event {:03d} | {} | Severity: {} | Confidence: {:.1f}%".format(
                         ev["event_id"], ev.get("anomaly_type", "Anomaly"),
                         ev["severity"], ev.get("confidence", 0.0)),
                     ln=True, fill=True)
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(0, 5, "    Time  : {} --> {}".format(
                ev["start_time"], ev["end_time"]), ln=True)
            pdf.cell(0, 5, "    Score : {:.5f}  |  Threshold: {:.5f}".format(
                ev["max_score"], ev["threshold"]), ln=True)
            pdf.ln(3)

    # Footer
    pdf.set_y(-20)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 5,
             "Generated by Anomaly Detection System | "
             "Hybrid ConvLSTM Autoencoder | AUC: 0.9371",
             align="C")

    pdf.output(out_path)
    print("PDF report saved: " + out_path)


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_demo(video_path, output_dir=None):
    """
    Complete demo pipeline.
    Input : any video file (.mp4, .avi, .mov)
    Output: annotated video + PDF report + JSON report
    """
    assert os.path.exists(video_path), "Video not found: " + video_path

    if output_dir is None:
        output_dir = cfg.REPORT_DIR
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cfg.REPORT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 55)
    print("  ANOMALY DETECTION SYSTEM")
    print("=" * 55)
    print("  Video  : " + os.path.basename(video_path))
    print("  Device : " + str(device))
    print("=" * 55)

    # Load model
    print("\n[1/5] Loading model...")
    model = ConvLSTMAE().to(device)
    state = torch.load(cfg.CHECKPOINT, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    print("  Loaded from epoch {}, loss={:.5f}".format(
        state["epoch"] + 1, state["loss"]))

    # Compute threshold from normal data
    print("\n[2/5] Computing adaptive threshold...")
    normal_scores         = compute_scores_on_folder(model, cfg.NORMAL_DIR, device)
    threshold, mu, sigma  = get_adaptive_threshold(normal_scores)

    # Extract frames
    print("\n[3/5] Extracting frames from video...")
    gray_frames, raw_frames, fps = extract_frames_from_video(video_path)

    if len(gray_frames) < cfg.SEQ_LEN:
        print("ERROR: Video too short. Need at least {} frames.".format(cfg.SEQ_LEN))
        return None

    # Score windows
    print("\n[4/5] Running anomaly detection...")
    scores, indices = score_all_windows(model, gray_frames, device)
    frame_scores    = build_frame_score_array(scores, indices, len(raw_frames))

    n_anomaly = (scores > threshold).sum()
    print("  Windows scored    : {}".format(len(scores)))
    print("  Anomaly windows   : {} ({:.1f}%)".format(
        n_anomaly, 100 * n_anomaly / len(scores)))
    print("  Score range       : {:.5f} - {:.5f}".format(
        scores.min(), scores.max()))
    print("  Threshold         : {:.5f}".format(threshold))

    # Detect events
    events = detect_events(scores, indices, threshold, fps)
    print("  Events detected   : {}".format(len(events)))

    # Generate outputs
    print("\n[5/5] Generating outputs...")

    base       = os.path.splitext(os.path.basename(video_path))[0]
    report_id  = "RPT-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    ann_path   = os.path.join(output_dir, base + "_annotated.mp4")
    pdf_path   = os.path.join(output_dir, report_id + ".pdf")
    json_path  = os.path.join(output_dir, report_id + ".json")

    # Annotated video
    create_annotated_video(raw_frames, frame_scores, threshold, fps, ann_path)

    # Build report dict
    duration = frames_to_time(len(raw_frames), fps)
    report = {
        "report_id"      : report_id,
        "generated_at"   : datetime.now().isoformat(),
        "video_source"   : os.path.basename(video_path),
        "duration"       : duration,
        "total_frames"   : len(raw_frames),
        "fps"            : round(fps, 2),
        "threshold"      : round(float(threshold), 6),
        "normal_baseline": round(float(mu), 6),
        "total_events"   : len(events),
        "by_severity"    : {
            "HIGH"  : sum(1 for e in events if e["severity"] == "HIGH"),
            "MEDIUM": sum(1 for e in events if e["severity"] == "MEDIUM"),
            "LOW"   : sum(1 for e in events if e["severity"] == "LOW"),
        },
        "events": events
    }

    # Save JSON
    import json as _json
    with open(json_path, "w") as f:
        _json.dump(report, f, indent=2)
    print("JSON report saved : " + json_path)

    # Save PDF
    save_pdf_report(report, pdf_path)

    # Final summary
    print("\n" + "=" * 55)
    print("  DETECTION COMPLETE")
    print("=" * 55)
    if len(events) == 0:
        print("  RESULT : NO ANOMALY DETECTED")
        print("  The video appears to contain normal activity.")
    else:
        print("  RESULT : {} ANOMALOUS EVENT(S) DETECTED".format(len(events)))
        for ev in events:
            print("  [{:6s}] {} --> {}  score={:.5f}  conf={:.1f}%".format(
                ev["severity"], ev["start_time"], ev["end_time"],
                ev["max_score"], ev["confidence"]))
    print("=" * 55)
    print("  Annotated video : " + ann_path)
    print("  PDF report      : " + pdf_path)
    print("  JSON report     : " + json_path)
    print("=" * 55)

    return ann_path, pdf_path, json_path, report
