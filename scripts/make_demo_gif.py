"""Render examples/demo.gif: the demo audit playing out in a styled terminal.

Runs the real CLI on examples/demo_trace.json and renders the output
frame-by-frame with Pillow. Regenerate after report format changes:

    python scripts/make_demo_gif.py
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent.parent
OUT = ROOT / "examples" / "demo.gif"
FONT_PATH = "/System/Library/Fonts/Menlo.ttc"
FONT_SIZE = 14
LINE_H = 21
PAD = 18
WIDTH = 920

BG = "#0d1117"
FG = "#e6edf3"
DIM = "#8b949e"
GREEN = "#3fb950"
RED = "#f85149"
YELLOW = "#d29922"
CYAN = "#79c0ff"
MAGENTA = "#d2a8ff"

CMD = (
    'agentaudit examples/demo_trace.json \\\n'
    '    --failure "equipment revenue wrong, total transposed to 12.2 instead of 8.6"'
)

EMOJI = {"✅": ("✔", GREEN), "❌": ("✘", RED), "🔁": ("↻", RED), "💸": ("$", YELLOW), "⚠️": ("!", YELLOW)}


def color_for(line: str) -> str:
    if line.startswith("#"):
        return MAGENTA
    if "defect first appears" in line:
        return RED
    if line.startswith("|"):
        return DIM
    return FG


def styled_lines(report: str) -> list[tuple[str, str]]:
    lines = []
    for raw in report.rstrip().splitlines():
        color = color_for(raw)
        for emoji, (repl, ecolor) in EMOJI.items():
            if emoji in raw:
                raw = raw.replace(emoji, repl)
                color = ecolor if color is FG else color
        text = raw.replace("**", "")
        wrapped = textwrap.wrap(text, width=104, subsequent_indent="    ") or [""]
        lines.extend((w, color) for w in wrapped)
    return lines


def main() -> None:
    report = subprocess.run(
        [sys.executable, "-m", "agentaudit.cli", str(ROOT / "examples" / "demo_trace.json"),
         "--failure", "equipment revenue wrong, total transposed to 12.2 instead of 8.6"],
        capture_output=True, text=True, check=True, cwd=ROOT,
    ).stdout

    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    cmd_lines = CMD.split("\n")
    out_lines = styled_lines(report)
    total_lines = len(cmd_lines) + len(out_lines) + 2
    height = total_lines * LINE_H + 2 * PAD

    frames: list[Image.Image] = []
    durations: list[int] = []

    def render(cmd_chars: int, out_count: int, hold_ms: int) -> None:
        img = Image.new("RGB", (WIDTH, height), BG)
        d = ImageDraw.Draw(img)
        y = PAD
        shown, budget = [], cmd_chars
        for cl in cmd_lines:
            take = cl[: max(0, budget)]
            shown.append(take)
            budget -= len(cl)
        d.text((PAD, y), "$ ", font=font, fill=GREEN)
        for i, cl in enumerate(shown):
            x = PAD + (d.textlength("$ ", font=font) if i == 0 else d.textlength("    ", font=font) * 0)
            d.text((x if i == 0 else PAD, y), cl, font=font, fill=CYAN)
            if cl:
                y += LINE_H
        y = PAD + len([c for c in shown if c]) * LINE_H or PAD + LINE_H
        y += LINE_H // 2
        for text, color in out_lines[:out_count]:
            d.text((PAD, y), text, font=font, fill=color)
            y += LINE_H
        frames.append(img)
        durations.append(hold_ms)

    total_cmd = sum(len(c) for c in cmd_lines)
    for n in range(0, total_cmd + 1, 7):
        render(n, 0, 55)
    render(total_cmd, 0, 700)

    section_breaks = {i for i, (t, _) in enumerate(out_lines) if t.startswith("#")}
    for n in range(1, len(out_lines) + 1):
        pause = 650 if n in section_breaks else 90
        render(total_cmd, n, pause)
    render(total_cmd, len(out_lines), 6000)

    frames[0].save(
        OUT, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True,
    )
    total_s = sum(durations) / 1000
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB, {len(frames)} frames, {total_s:.1f}s)")


if __name__ == "__main__":
    main()
