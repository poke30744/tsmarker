"""
Fix Whisper SRT timing gaps caused by faster-whisper VAD bugs.

Background:
  faster-whisper's VAD integration has a known bug (SYSTRAN/faster-whisper#1119)
  where segment.end is set to the NEXT VAD segment's start time instead of the
  actual speech end. This causes subtitles to disappear before speech finishes.

  Another bug (#125) causes the first word after a long silence to have an
  incorrect start timestamp, but start times are harder to fix deterministically
  and are left untouched here.

Algorithm (parameters derived from statistical analysis of 584-entry anime SRT):
  k = 0.8s  (merge threshold: gaps below this are VAD artifacts, not natural pauses)
  m = 0.5s  (extension: for larger natural pauses, add 0.5s to cover trailing syllables)

  For each adjacent pair of entries:
    gap = next.start - current.end
    if gap < k:  current.end = next.start  (merge — VAD cut off speech tail)
    else:        current.end += 0.5s        (extend — add room for trailing syllables)

Only end times are modified. Start times are never changed.
The file is modified in-place.
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger("tsmarker.fix_srt_gaps")

K = 0.8  # merge threshold (seconds)
M = 0.5  # extension for large gaps (seconds)


def _to_seconds(ts_str: str) -> float:
    h, mi, s, ms = map(int, re.match(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", ts_str).groups())
    return h * 3600 + mi * 60 + s + ms / 1000


def _format_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    seconds %= 3600
    mi = int(seconds // 60)
    seconds %= 60
    s = int(seconds)
    ms = int(round((seconds - s) * 1000))
    return f"{h:02d}:{mi:02d}:{s:02d},{ms:03d}"


def fix_srt_gaps(srt_path: Path) -> None:
    """Fix subtitle end-time gaps in a .corrected.srt file (modified in-place).

    Only modifies end timestamps. Start timestamps are left unchanged because
    there is no deterministic rule for fixing early-start artifacts from
    faster-whisper VAD bug #125.
    """
    content = srt_path.read_text(encoding="utf-8")
    blocks = content.strip().split("\n\n")

    entries = []
    for blk in blocks:
        lines = blk.strip().split("\n")
        m = re.match(r"(\S+)\s*-->\s*(\S+)", lines[1])
        if not m:
            entries.append({"lines": lines, "start": None, "end": None})
            continue
        entries.append({
            "lines": lines,
            "start": _to_seconds(m.group(1)),
            "end": _to_seconds(m.group(2)),
        })

    changed = 0
    for i in range(len(entries) - 1):
        if entries[i]["start"] is None or entries[i+1]["start"] is None:
            continue
        gap = entries[i+1]["start"] - entries[i]["end"]
        old_end = entries[i]["end"]

        if gap < K:
            entries[i]["end"] = entries[i+1]["start"]
        else:
            entries[i]["end"] += M

        if abs(entries[i]["end"] - old_end) > 0.001:
            changed += 1

    new_blocks = []
    for entry in entries:
        if entry["start"] is not None:
            entry["lines"][1] = f"{_format_ts(entry['start'])} --> {_format_ts(entry['end'])}"
        new_blocks.append("\n".join(entry["lines"]))

    srt_path.write_text("\n\n".join(new_blocks) + "\n", encoding="utf-8")
    logger.info(f"Fixed {changed}/{len(entries)} entries in {srt_path.name}")
