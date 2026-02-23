"""
Generate CEFR scores on a per-month basis by passing the full month's
consolidated transcripts to Gemini as a single scoring request.

Reads:  data/weekly_transcripts.csv
Writes: data/monthly_cefr_ai.csv   (one row per student per month)

Usage:
  python scripts/generate_monthly_cefr.py

Env:
  GEMINI_API_KEY - Required. Set in .env file.
"""

import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types

# ─── Config ───────────────────────────────────────────────────────────────────
INPUT_CSV = Path("data/weekly_transcripts.csv")
OUTPUT_CSV = Path("data/monthly_cefr_ai.csv")
PROMPT_FILE = Path("data/prompt.md")
CHECKPOINT_FILE = Path("data/.monthly_cefr_checkpoint.json")

MODEL = "gemini-2.0-flash"
REQUESTS_PER_MINUTE = 15
DELAY_BETWEEN_REQUESTS = 60 / REQUESTS_PER_MINUTE

# Only score these months (February excluded — partial data)
TARGET_MONTHS = ["December", "January"]

CEFR_LEVELS = {
    "Pre-A1": 0,
    "A1": 1,
    "Strong A1": 1.5,
    "A2": 2,
    "Strong A2": 2.5,
    "B1": 3,
    "B2": 4,
}


# ─── Helpers ──────────────────────────────────────────────────────────────────
def load_prompt() -> str:
    with open(PROMPT_FILE, "r") as f:
        return f.read()


def create_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in environment.")
        sys.exit(1)
    return genai.Client(api_key=api_key)


def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint: dict):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)


# ─── Score one student-month ──────────────────────────────────────────────────
def score_monthly_transcript(
    client: genai.Client,
    system_prompt: str,
    transcript: str,
    student: str,
    month: str,
) -> dict | None:
    if not transcript.strip():
        print(f"  SKIP {student} {month}: Empty transcript")
        return None

    # Truncate to avoid token limits (keep first 12000 chars — more than weekly)
    if len(transcript) > 12000:
        transcript = transcript[:12000] + "\n\n[... transcript truncated ...]"

    user_message = f"""Score the following student's English speaking transcript.
The student is: {student}
Time period: {month} (full month — multiple sessions combined)

IMPORTANT: Return ONLY valid JSON matching the output format specified. No markdown, no explanation, just the JSON object.

--- TRANSCRIPT START ---
{transcript}
--- TRANSCRIPT END ---"""

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=user_message)],
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.1,
            ),
        )

        text = response.text.strip()
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        return json.loads(text)

    except json.JSONDecodeError as e:
        print(f"  ERROR {student} {month}: JSON parse failed: {e}")
        return None
    except Exception as e:
        print(f"  ERROR {student} {month}: API call failed: {e}")
        return None


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("GENERATE MONTHLY CEFR: Full-month transcripts → CEFR scores")
    print("=" * 60)

    system_prompt = load_prompt()
    print(f"Loaded prompt ({len(system_prompt)} chars)")

    client = create_client()
    print(f"Gemini client ready (model: {MODEL})")

    # Load and group transcripts by (student, month)
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    student_month = defaultdict(
        lambda: {"weeks": [], "sessions": 0, "mins": 0.0, "active_days": 0}
    )
    for r in rows:
        month = r.get("month", "")
        if month not in TARGET_MONTHS:
            continue
        key = (r["student"], month)
        student_month[key]["weeks"].append(r)
        student_month[key]["sessions"] += int(r.get("session_count", 0))
        student_month[key]["mins"] += float(r.get("total_duration_mins", 0))
        student_month[key]["active_days"] += int(r.get("active_days", 0))

    # Sort keys: by month order then student name
    month_order = {m: i for i, m in enumerate(TARGET_MONTHS)}
    sorted_keys = sorted(
        student_month.keys(), key=lambda k: (month_order.get(k[1], 99), k[0])
    )

    print(f"Found {len(sorted_keys)} student-month pairs to score")

    checkpoint = load_checkpoint()
    print(f"Checkpoint has {len(checkpoint)} already-scored records\n")

    results = []
    scored = 0
    skipped = 0
    errors = 0

    for i, (student, month) in enumerate(sorted_keys):
        key = f"{student}|{month}"

        if key in checkpoint:
            results.append(checkpoint[key])
            skipped += 1
            continue

        data = student_month[(student, month)]
        weeks = sorted(data["weeks"], key=lambda w: w.get("week_start", ""))

        # Concatenate all weekly transcripts with week labels
        combined_transcript = ""
        for w in weeks:
            week_label = w.get("week_start", w.get("week", ""))
            combined_transcript += f"\n\n[Week: {week_label}]\n{w.get('consolidated_transcript', '').strip()}"
        combined_transcript = combined_transcript.strip()

        print(
            f"[{i + 1}/{len(sorted_keys)}] {student} — {month} ({len(weeks)} weeks, {len(combined_transcript)} chars)..."
        )

        result = score_monthly_transcript(
            client, system_prompt, combined_transcript, student, month
        )

        if result:
            row = {
                "student": student,
                "month": month,
                "weeks_scored": len(weeks),
                "total_sessions": data["sessions"],
                "total_duration_mins": round(data["mins"], 1),
                "total_active_days": data["active_days"],
                # CEFR scores
                "fluency": result.get("cefr_scores", {}).get("fluency", ""),
                "accuracy": result.get("cefr_scores", {}).get("accuracy", ""),
                "range": result.get("cefr_scores", {}).get("range", ""),
                "coherence": result.get("cefr_scores", {}).get("coherence", ""),
                "overall_level": result.get("overall_level", ""),
                # Evidence
                "fluency_evidence": result.get("key_evidence", {}).get(
                    "fluency_evidence", ""
                ),
                "accuracy_errors": json.dumps(
                    result.get("key_evidence", {}).get("accuracy_errors", [])
                ),
                "range_vocabulary": json.dumps(
                    result.get("key_evidence", {}).get("range_vocabulary", [])
                ),
                "coherence_structure": result.get("key_evidence", {}).get(
                    "coherence_structure", ""
                ),
            }

            results.append(row)
            checkpoint[key] = row
            scored += 1

            if scored % 10 == 0:
                save_checkpoint(checkpoint)
                print(f"  ✓ Checkpoint saved ({scored} scored so far)")
        else:
            errors += 1

        time.sleep(DELAY_BETWEEN_REQUESTS)

    save_checkpoint(checkpoint)

    # Write output CSV
    if results:
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(results[0].keys())
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nWrote {len(results)} rows to {OUTPUT_CSV}")
    else:
        print("\nNo results to write.")

    print(f"\n{'=' * 60}")
    print(f"DONE — Scored: {scored} | From checkpoint: {skipped} | Errors: {errors}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
