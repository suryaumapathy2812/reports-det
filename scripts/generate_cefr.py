"""
Generate CEFR scores for weekly student transcripts using Gemini API.

Reads:  data/weekly_transcripts.csv
Writes: data/weekly_cefr.csv     (per-student per-week CEFR scores)
        data/monthly_cefr.csv    (exponentially weighted monthly aggregation)

Usage:
  python scripts/generate_cefr.py

Env:
  GEMINI_API_KEY - Required. Set in .env file.
"""

import csv
import json
import os
import re
import time
import sys
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

# Load .env
load_dotenv()

from google import genai
from google.genai import types

# ─── Config ───────────────────────────────────────────────────────────────────
INPUT_CSV = Path("data/weekly_transcripts.csv")
WEEKLY_OUTPUT = Path("data/weekly_cefr.csv")
MONTHLY_OUTPUT = Path("data/monthly_cefr.csv")
PROMPT_FILE = Path("data/prompt.md")
CHECKPOINT_FILE = Path("data/.cefr_checkpoint.json")

MODEL = "gemini-2.0-flash"  # gemini-flash-latest resolves to this
THINKING = None  # No thinking mode

# Rate limiting
REQUESTS_PER_MINUTE = 15  # Conservative to avoid 429s
DELAY_BETWEEN_REQUESTS = 60 / REQUESTS_PER_MINUTE

# CEFR level ordering for numerical conversion
CEFR_LEVELS = {
    "Pre-A1": 0,
    "A1": 1,
    "Strong A1": 1.5,
    "A2": 2,
    "Strong A2": 2.5,
    "B1": 3,
    "B2": 4,
    "C1": 5,
    "C2": 6,
}

CEFR_REVERSE = {v: k for k, v in CEFR_LEVELS.items()}

# Month ordering for exponential weighting
MONTH_ORDER = {"December": 0, "January": 1, "February": 2}


# ─── Load CEFR Prompt ────────────────────────────────────────────────────────
def load_cefr_prompt() -> str:
    """Load the CEFR assessment prompt from prompt.md."""
    with open(PROMPT_FILE, "r") as f:
        return f.read()


# ─── Gemini Client ────────────────────────────────────────────────────────────
def create_client() -> genai.Client:
    """Create Gemini API client."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in environment.")
        print("Add it to .env file: GEMINI_API_KEY=your-key-here")
        sys.exit(1)

    return genai.Client(api_key=api_key)


def score_transcript(
    client: genai.Client, system_prompt: str, transcript: str, student: str, week: str
) -> dict | None:
    """Score a single transcript using Gemini API. Returns parsed CEFR dict or None."""
    if not transcript.strip():
        print(f"  SKIP {student} {week}: Empty transcript")
        return None

    # Truncate very long transcripts to avoid token limits (keep first ~8000 chars)
    if len(transcript) > 8000:
        transcript = (
            transcript[:8000] + "\n\n[... transcript truncated for scoring ...]"
        )

    user_message = f"""Score the following student's English speaking transcript.
The student is: {student}
Time period: {week}

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
                temperature=0.1,  # Low temperature for consistent scoring
            ),
        )

        # Parse JSON from response
        text = response.text.strip()

        # Remove markdown code fences if present
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        result = json.loads(text)
        return result

    except json.JSONDecodeError as e:
        print(f"  ERROR {student} {week}: JSON parse failed: {e}")
        print(f"  Raw response: {text[:200]}...")
        return None
    except Exception as e:
        print(f"  ERROR {student} {week}: API call failed: {e}")
        return None


# ─── Checkpoint Management ────────────────────────────────────────────────────
def load_checkpoint() -> dict:
    """Load checkpoint of already-scored records."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint: dict):
    """Save checkpoint."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)


# ─── Score All Weekly Transcripts ─────────────────────────────────────────────
def score_all_weekly(client: genai.Client, system_prompt: str) -> list[dict]:
    """Score all weekly transcripts, with checkpointing."""
    # Load input
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)

    print(f"Loaded {len(records)} weekly transcript records")

    # Load checkpoint
    checkpoint = load_checkpoint()
    print(f"Checkpoint has {len(checkpoint)} already-scored records")

    results = []
    scored_count = 0
    skipped_count = 0
    error_count = 0

    for i, record in enumerate(records):
        student = record["student"]
        week = record["week"]
        key = f"{student}|{week}"

        # Check checkpoint
        if key in checkpoint:
            results.append(checkpoint[key])
            skipped_count += 1
            continue

        # Score
        print(f"[{i + 1}/{len(records)}] Scoring {student} - {week}...")
        transcript = record.get("consolidated_transcript", "")

        cefr_result = score_transcript(client, system_prompt, transcript, student, week)

        if cefr_result:
            row = {
                "student": student,
                "week": week,
                "week_start": record.get("week_start", ""),
                "month": record.get("month", ""),
                "session_count": record.get("session_count", 0),
                "total_duration_mins": record.get("total_duration_mins", 0),
                "active_days": record.get("active_days", 0),
                "has_unintelligible": record.get("has_unintelligible", "False"),
                # CEFR scores
                "fluency": cefr_result.get("cefr_scores", {}).get("fluency", ""),
                "accuracy": cefr_result.get("cefr_scores", {}).get("accuracy", ""),
                "range": cefr_result.get("cefr_scores", {}).get("range", ""),
                "coherence": cefr_result.get("cefr_scores", {}).get("coherence", ""),
                "overall_level": cefr_result.get("overall_level", ""),
                # Evidence
                "fluency_evidence": cefr_result.get("key_evidence", {}).get(
                    "fluency_evidence", ""
                ),
                "accuracy_errors": json.dumps(
                    cefr_result.get("key_evidence", {}).get("accuracy_errors", [])
                ),
                "range_vocabulary": json.dumps(
                    cefr_result.get("key_evidence", {}).get("range_vocabulary", [])
                ),
                "coherence_structure": cefr_result.get("key_evidence", {}).get(
                    "coherence_structure", ""
                ),
            }

            results.append(row)
            checkpoint[key] = row
            scored_count += 1

            # Save checkpoint every 10 records
            if scored_count % 10 == 0:
                save_checkpoint(checkpoint)
                print(f"  Checkpoint saved ({scored_count} scored so far)")
        else:
            error_count += 1

        # Rate limiting
        time.sleep(DELAY_BETWEEN_REQUESTS)

    # Final checkpoint save
    save_checkpoint(checkpoint)

    print(f"\nScoring complete:")
    print(f"  Scored: {scored_count}")
    print(f"  From checkpoint: {skipped_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total results: {len(results)}")

    return results


# ─── Aggregate Monthly CEFR ──────────────────────────────────────────────────
def cefr_to_numeric(level: str) -> float | None:
    """Convert CEFR level string to numeric value."""
    level = level.strip()
    if level in CEFR_LEVELS:
        return CEFR_LEVELS[level]
    return None


def numeric_to_cefr(value: float) -> str:
    """Convert numeric value back to nearest CEFR level."""
    if value is None:
        return ""
    # Find closest level
    closest = min(CEFR_REVERSE.keys(), key=lambda x: abs(x - value))
    return CEFR_REVERSE[closest]


def aggregate_monthly_cefr(weekly_results: list[dict]) -> list[dict]:
    """
    Aggregate weekly CEFR scores into monthly scores using exponential weighting.
    More recent weeks within a month get higher weight.
    """
    # Group by (student, month)
    student_month = defaultdict(list)
    for r in weekly_results:
        if r.get("month") and r.get("overall_level"):
            student_month[(r["student"], r["month"])].append(r)

    results = []
    dimensions = ["fluency", "accuracy", "range", "coherence", "overall_level"]

    for (student, month), weeks in sorted(student_month.items()):
        # Sort weeks chronologically
        weeks.sort(key=lambda x: x.get("week_start", ""))

        # Exponential weights: more recent = higher weight
        # decay factor 0.7 means each older week is 70% of the next
        n = len(weeks)
        decay = 0.7
        raw_weights = [decay ** (n - 1 - i) for i in range(n)]
        total_weight = sum(raw_weights)
        weights = [w / total_weight for w in raw_weights]

        row = {
            "student": student,
            "month": month,
            "weeks_scored": n,
            "total_sessions": sum(int(w.get("session_count", 0)) for w in weeks),
            "total_duration_mins": sum(
                float(w.get("total_duration_mins", 0)) for w in weeks
            ),
            "total_active_days": sum(int(w.get("active_days", 0)) for w in weeks),
        }

        # Weighted average for each dimension
        for dim in dimensions:
            dim_key = dim if dim != "overall_level" else "overall_level"
            values = []
            dim_weights = []

            for i, w in enumerate(weeks):
                val = cefr_to_numeric(w.get(dim_key, ""))
                if val is not None:
                    values.append(val)
                    dim_weights.append(weights[i])

            if values:
                # Normalize weights for available values
                w_sum = sum(dim_weights)
                if w_sum > 0:
                    weighted_avg = sum(
                        v * w / w_sum for v, w in zip(values, dim_weights)
                    )
                else:
                    weighted_avg = sum(values) / len(values)

                col_name = dim if dim != "overall_level" else "overall"
                row[f"{col_name}_numeric"] = round(weighted_avg, 2)
                row[f"{col_name}_level"] = numeric_to_cefr(weighted_avg)
            else:
                col_name = dim if dim != "overall_level" else "overall"
                row[f"{col_name}_numeric"] = None
                row[f"{col_name}_level"] = ""

        results.append(row)

    print(f"Aggregated {len(results)} monthly CEFR records")
    return results


# ─── Write Outputs ────────────────────────────────────────────────────────────
def write_csv(data: list[dict], output_path: Path):
    """Write list of dicts to CSV."""
    if not data:
        print(f"No data to write to {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(data[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"Wrote {len(data)} rows to {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("GENERATE CEFR: Weekly transcripts → CEFR scores")
    print("=" * 60)

    # Load prompt
    system_prompt = load_cefr_prompt()
    print(f"Loaded CEFR prompt ({len(system_prompt)} chars)")

    # Create client
    client = create_client()
    print(f"Gemini client created (model: {MODEL})")

    # Score weekly transcripts
    weekly_results = score_all_weekly(client, system_prompt)

    # Write weekly CEFR
    write_csv(weekly_results, WEEKLY_OUTPUT)

    # Aggregate monthly
    monthly_results = aggregate_monthly_cefr(weekly_results)
    write_csv(monthly_results, MONTHLY_OUTPUT)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    from collections import Counter

    if weekly_results:
        overall_dist = Counter(r.get("overall_level", "") for r in weekly_results)
        print("Weekly overall level distribution:")
        for level in ["A1", "Strong A1", "A2", "Strong A2", "B1", "B2"]:
            print(f"  {level}: {overall_dist.get(level, 0)}")

    if monthly_results:
        for month in ["December", "January", "February"]:
            month_data = [r for r in monthly_results if r["month"] == month]
            if month_data:
                dist = Counter(r.get("overall_level", "") for r in month_data)
                print(f"\n{month} overall distribution:")
                for level in ["A1", "Strong A1", "A2", "Strong A2", "B1", "B2"]:
                    print(f"  {level}: {dist.get(level, 0)}")


if __name__ == "__main__":
    main()
