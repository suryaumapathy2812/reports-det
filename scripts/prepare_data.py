"""
Prepare raw session CSV into structured datasets for CEFR analysis.

Outputs:
  - data/weekly_transcripts.csv  (student × week: concatenated transcripts + effort metrics)
  - data/student_effort.csv      (student × month: sessions, talk time, consistency)

Usage:
  python scripts/prepare_data.py
"""

import csv
import re
import os
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────────────
INPUT_CSV = Path("data/det.csv")
OUTPUT_DIR = Path("data")
WEEKLY_OUTPUT = OUTPUT_DIR / "weekly_transcripts.csv"
EFFORT_OUTPUT = OUTPUT_DIR / "student_effort.csv"

# Program date range (from data analysis)
PROGRAM_START = datetime(2025, 12, 12)
PROGRAM_END = datetime(2026, 2, 19)

# Month boundaries for reporting (end dates include full day via 23:59:59)
MONTHS = {
    "December": (datetime(2025, 12, 1), datetime(2025, 12, 31, 23, 59, 59)),
    "January": (datetime(2026, 1, 1), datetime(2026, 1, 31, 23, 59, 59)),
    "February": (datetime(2026, 2, 1), datetime(2026, 2, 28, 23, 59, 59)),
}


# ─── Helpers ──────────────────────────────────────────────────────────────────
def parse_duration(duration_str: str) -> int:
    """Parse '2m 35s' or '0m 10s' into total seconds."""
    match = re.match(r"(\d+)m\s*(\d+)s", duration_str.strip())
    if match:
        return int(match.group(1)) * 60 + int(match.group(2))
    # Try seconds only
    match = re.match(r"(\d+)s", duration_str.strip())
    if match:
        return int(match.group(1))
    return 0


def get_week_label(dt: datetime) -> str:
    """Get ISO week label like '2025-W50'."""
    iso = dt.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def get_week_start(dt: datetime) -> datetime:
    """Get Monday of the week for a given date."""
    return dt - timedelta(days=dt.weekday())


def get_month_label(dt: datetime) -> str:
    """Get month name for reporting periods."""
    for name, (start, end) in MONTHS.items():
        if start <= dt <= end:
            return name
    return None


def extract_student_text(transcript: str) -> str:
    """Extract only student speech from transcript, removing interviewer lines."""
    lines = transcript.split("\n")
    student_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith("[Student]:") or line.startswith("[SPEAKER_00]:"):
            # Remove the speaker tag
            text = re.sub(r"^\[(Student|SPEAKER_00)\]:\s*", "", line)
            if text and text != "[unintelligible]":
                student_lines.append(text)
    return " ".join(student_lines)


def has_unintelligible(transcript: str) -> bool:
    """Check if transcript contains unintelligible markers."""
    return "[unintelligible]" in transcript.lower()


# ─── Parse Raw CSV ────────────────────────────────────────────────────────────
def parse_sessions(csv_path: Path) -> list[dict]:
    """Parse the raw CSV into a list of session records."""
    sessions = []

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            student = row.get("Student", "").strip()
            created_at = row.get("Created At", "").strip()
            duration = row.get("Duration", "").strip()
            transcript = row.get("Transcript", "").strip()
            activity = row.get("Activity", "").strip()
            topic = row.get("Topic", "").strip()

            if not student or not created_at:
                continue

            try:
                dt = datetime.fromisoformat(created_at.replace("Z", ""))
            except ValueError:
                continue

            duration_secs = parse_duration(duration)
            student_text = extract_student_text(transcript)

            sessions.append(
                {
                    "student": student,
                    "datetime": dt,
                    "date": dt.date(),
                    "week": get_week_label(dt),
                    "week_start": get_week_start(dt).date(),
                    "month": get_month_label(dt),
                    "activity": activity,
                    "topic": topic,
                    "duration_secs": duration_secs,
                    "student_text": student_text,
                    "full_transcript": transcript,
                    "has_unintelligible": has_unintelligible(transcript),
                }
            )

    print(f"Parsed {len(sessions)} sessions from {csv_path}")
    return sessions


# ─── Build Weekly Transcripts ─────────────────────────────────────────────────
def build_weekly_transcripts(sessions: list[dict]) -> list[dict]:
    """Consolidate sessions into weekly transcripts per student."""
    # Group by (student, week)
    weekly = defaultdict(
        lambda: {
            "transcripts": [],
            "session_count": 0,
            "total_duration_secs": 0,
            "active_days": set(),
            "activities": [],
            "has_unintelligible": False,
            "month": None,
            "week_start": None,
        }
    )

    for s in sessions:
        if not s["month"]:  # Outside reporting period
            continue

        key = (s["student"], s["week"])
        w = weekly[key]
        w["transcripts"].append(s["student_text"])
        w["session_count"] += 1
        w["total_duration_secs"] += s["duration_secs"]
        w["active_days"].add(s["date"])
        w["activities"].append(f"{s['activity']} ({s['topic']})")
        w["has_unintelligible"] = w["has_unintelligible"] or s["has_unintelligible"]
        w["month"] = s["month"]
        w["week_start"] = s["week_start"]

    # Convert to list
    results = []
    for (student, week), data in sorted(weekly.items()):
        consolidated_transcript = "\n\n---\n\n".join(
            [t for t in data["transcripts"] if t.strip()]
        )

        results.append(
            {
                "student": student,
                "week": week,
                "week_start": str(data["week_start"]),
                "month": data["month"],
                "session_count": data["session_count"],
                "total_duration_secs": data["total_duration_secs"],
                "total_duration_mins": round(data["total_duration_secs"] / 60, 1),
                "active_days": len(data["active_days"]),
                "activities": "; ".join(data["activities"][:5])
                + (
                    f" (+{len(data['activities']) - 5} more)"
                    if len(data["activities"]) > 5
                    else ""
                ),
                "has_unintelligible": data["has_unintelligible"],
                "consolidated_transcript": consolidated_transcript,
            }
        )

    print(f"Built {len(results)} weekly transcript records")
    return results


# ─── Build Student Effort (Monthly) ──────────────────────────────────────────
def build_student_effort(sessions: list[dict]) -> list[dict]:
    """Aggregate effort metrics per student per month."""
    # Group by (student, month)
    monthly = defaultdict(
        lambda: {
            "session_count": 0,
            "total_duration_secs": 0,
            "active_days": set(),
            "activities": set(),
            "topics": set(),
        }
    )

    for s in sessions:
        if not s["month"]:
            continue

        key = (s["student"], s["month"])
        m = monthly[key]
        m["session_count"] += 1
        m["total_duration_secs"] += s["duration_secs"]
        m["active_days"].add(s["date"])
        m["activities"].add(s["activity"])
        m["topics"].add(s["topic"])

    # Calculate consistency: active days / total weekdays in that month's program period
    def weekdays_in_range(start: datetime, end: datetime) -> int:
        """Count weekdays (Mon-Sat) in range, capped by program dates."""
        start = max(start, PROGRAM_START)
        end = min(end, PROGRAM_END)
        count = 0
        current = start
        while current <= end:
            if current.weekday() < 6:  # Mon-Sat
                count += 1
            current += timedelta(days=1)
        return count

    month_weekdays = {}
    for name, (start, end) in MONTHS.items():
        month_weekdays[name] = weekdays_in_range(start, end)

    results = []
    for (student, month), data in sorted(monthly.items()):
        total_weekdays = month_weekdays.get(month, 1)
        active_days = len(data["active_days"])
        consistency = round(active_days / total_weekdays * 100, 1)
        total_mins = round(data["total_duration_secs"] / 60, 1)
        avg_daily_mins = round(total_mins / active_days, 1) if active_days > 0 else 0

        results.append(
            {
                "student": student,
                "month": month,
                "session_count": data["session_count"],
                "total_duration_secs": data["total_duration_secs"],
                "total_duration_mins": total_mins,
                "active_days": active_days,
                "total_program_days": total_weekdays,
                "consistency_pct": consistency,
                "avg_daily_mins": avg_daily_mins,
                "unique_activities": len(data["activities"]),
                "unique_topics": len(data["topics"]),
            }
        )

    print(f"Built {len(results)} monthly effort records")
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
    print("PREPARE DATA: Raw sessions → Weekly transcripts + Effort")
    print("=" * 60)

    # Parse
    sessions = parse_sessions(INPUT_CSV)

    # Build weekly transcripts
    weekly = build_weekly_transcripts(sessions)
    write_csv(weekly, WEEKLY_OUTPUT)

    # Build monthly effort
    effort = build_student_effort(sessions)
    write_csv(effort, EFFORT_OUTPUT)

    # Summary stats
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    students = set(r["student"] for r in weekly)
    print(f"Students: {len(students)}")
    print(f"Weekly records: {len(weekly)}")
    print(f"Monthly effort records: {len(effort)}")

    # Weekly records per month
    from collections import Counter

    month_counts = Counter(r["month"] for r in weekly)
    for m in ["December", "January", "February"]:
        print(f"  {m}: {month_counts.get(m, 0)} weekly records")

    # Students with data in all 3 months
    student_months = defaultdict(set)
    for r in effort:
        student_months[r["student"]].add(r["month"])
    all_three = [s for s, months in student_months.items() if len(months) == 3]
    print(f"\nStudents with data in all 3 months: {len(all_three)}")
    two_months = [s for s, months in student_months.items() if len(months) == 2]
    print(f"Students with data in 2 months: {len(two_months)}")
    one_month = [s for s, months in student_months.items() if len(months) == 1]
    print(f"Students with data in 1 month: {len(one_month)}")


if __name__ == "__main__":
    main()
