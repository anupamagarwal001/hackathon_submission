"""Generate committed README assets from the verified Colab smoke run."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "assets"

REWARD_SERIES = [0.019333332777023315, 0.01599554717540741, 0.000563124951440841, 0.027521273121237755]
LOSS_SERIES = [0.36607930064201355, 0.4380803406238556, 0.0, -0.07474987953901291]
BASELINE_SCORE = {
    "heuristic": 0.4084,
    "random": 0.2325,
}

WIDTH = 1200
HEIGHT = 720
PADDING_LEFT = 140
PADDING_RIGHT = 120
PADDING_TOP = 90
PADDING_BOTTOM = 140
BG = "white"
TEXT = "#0f172a"
AXIS = "#475569"
GRID = "#cbd5e1"
BLUE = "#2563eb"
GREEN = "#16a34a"
RED = "#dc2626"


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/Library/Fonts/Arial.ttf",
    )
    if not bold:
        candidates = tuple(path for path in candidates if "Bold" not in path) + candidates
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


TITLE_FONT = load_font(34, bold=True)
LABEL_FONT = load_font(24)
TICK_FONT = load_font(20)
BODY_FONT = load_font(22)
COMPACT_BODY_FONT = load_font(20)
SMALL_FONT = load_font(18)
BIG_FONT = load_font(42, bold=True)
MID_FONT = load_font(28, bold=True)


def draw_axes(draw: ImageDraw.ImageDraw) -> tuple[int, int, int, int]:
    x0 = PADDING_LEFT
    y0 = HEIGHT - PADDING_BOTTOM
    x1 = WIDTH - PADDING_RIGHT
    y1 = PADDING_TOP
    draw.line((x0, y0, x1, y0), fill=AXIS, width=3)
    draw.line((x0, y0, x0, y1), fill=AXIS, width=3)
    return x0, y0, x1, y1


def draw_centered(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font, fill=TEXT) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = box[0] + (box[2] - box[0] - tw) / 2
    y = box[1] + (box[3] - box[1] - th) / 2
    draw.text((x, y), text, font=font, fill=fill)


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font,
    fill=TEXT,
    max_width: int = 340,
    line_gap: int = 8,
) -> int:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if current and bbox[2] - bbox[0] > max_width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))

    x, y = xy
    line_height = draw.textbbox((0, 0), "Ag", font=font)[3]
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height + line_gap
    return y


def generate_reward_curve() -> Path:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = draw_axes(draw)

    title = "Smoke Training Reward Curve"
    subtitle = "HF Jobs/Colab T4, Qwen3-0.6B, LoRA, 4 trainer steps"
    draw.text((PADDING_LEFT, 24), title, font=TITLE_FONT, fill=TEXT)
    draw.text((PADDING_LEFT, 62), subtitle, font=BODY_FONT, fill=AXIS)

    min_value = min(REWARD_SERIES)
    max_value = max(REWARD_SERIES)
    span = max_value - min_value or 1.0
    y_min = min_value - 0.003
    y_max = max_value + 0.003

    for idx in range(5):
        frac = idx / 4
        y = y0 - frac * (y0 - y1)
        value = y_min + frac * (y_max - y_min)
        draw.line((x0, y, x1, y), fill=GRID, width=1)
        label = f"{value:.3f}"
        bbox = draw.textbbox((0, 0), label, font=TICK_FONT)
        draw.text((x0 - 20 - (bbox[2] - bbox[0]), y - 10), label, font=TICK_FONT, fill=TEXT)

    step_gap = (x1 - x0) / (len(REWARD_SERIES) - 1)
    points: list[tuple[float, float]] = []
    for idx, value in enumerate(REWARD_SERIES):
        x = x0 + idx * step_gap
        y = y0 - ((value - y_min) / (y_max - y_min)) * (y0 - y1)
        points.append((x, y))
        draw.line((x, y0, x, y0 + 8), fill=AXIS, width=2)
        label = str(idx + 1)
        bbox = draw.textbbox((0, 0), label, font=TICK_FONT)
        draw.text((x - (bbox[2] - bbox[0]) / 2, y0 + 16), label, font=TICK_FONT, fill=TEXT)

    draw.line(points, fill=BLUE, width=6)
    for idx, (x, y) in enumerate(points):
        color = GREEN if idx == len(points) - 1 else RED if idx == 2 else BLUE
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline="white", width=2)
        value = f"{REWARD_SERIES[idx]:.4f}"
        label_x = x + 10 if idx < len(points) - 1 else x - 74
        draw.text((label_x, y - 26), value, font=SMALL_FONT, fill=TEXT)

    draw_centered(draw, (x0, HEIGHT - 90, x1, HEIGHT - 56), "training step", LABEL_FONT)
    draw.text((PADDING_LEFT, HEIGHT - 34), "Verified smoke run from the Colab T4 execution used in the on-site demo.", font=SMALL_FONT, fill=AXIS)

    path = OUTPUT_DIR / "reward_curve.png"
    image.save(path)
    return path


def generate_loss_curve() -> Path:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = draw_axes(draw)

    title = "GRPO Training Loss / Objective"
    subtitle = "HF Jobs T4, Qwen3-0.6B, LoRA, 4 trainer steps. Objective-style loss can cross below zero."
    draw.text((PADDING_LEFT, 24), title, font=TITLE_FONT, fill=TEXT)
    draw.text((PADDING_LEFT, 62), subtitle, font=BODY_FONT, fill=AXIS)

    y_min = min(LOSS_SERIES) - 0.08
    y_max = max(LOSS_SERIES) + 0.08

    for idx in range(5):
        frac = idx / 4
        y = y0 - frac * (y0 - y1)
        value = y_min + frac * (y_max - y_min)
        draw.line((x0, y, x1, y), fill=GRID, width=1)
        label = f"{value:.2f}"
        bbox = draw.textbbox((0, 0), label, font=TICK_FONT)
        draw.text((x0 - 20 - (bbox[2] - bbox[0]), y - 10), label, font=TICK_FONT, fill=TEXT)

    zero_y = y0 - ((0 - y_min) / (y_max - y_min)) * (y0 - y1)
    if y1 <= zero_y <= y0:
        draw.line((x0, zero_y, x1, zero_y), fill="#94a3b8", width=2)
        draw.text((x1 - 104, zero_y - 28), "zero", font=SMALL_FONT, fill=AXIS)

    step_gap = (x1 - x0) / (len(LOSS_SERIES) - 1)
    points: list[tuple[float, float]] = []
    for idx, value in enumerate(LOSS_SERIES):
        x = x0 + idx * step_gap
        y = y0 - ((value - y_min) / (y_max - y_min)) * (y0 - y1)
        points.append((x, y))
        draw.line((x, y0, x, y0 + 8), fill=AXIS, width=2)
        label = str(idx + 1)
        bbox = draw.textbbox((0, 0), label, font=TICK_FONT)
        draw.text((x - (bbox[2] - bbox[0]) / 2, y0 + 16), label, font=TICK_FONT, fill=TEXT)

    draw.line(points, fill=RED, width=6)
    for idx, (x, y) in enumerate(points):
        color = GREEN if idx == len(points) - 1 else RED
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline="white", width=2)
        value = f"{LOSS_SERIES[idx]:.3f}"
        if idx < len(points) - 1:
            label_x, label_y = x + 10, y - 26
        else:
            label_x, label_y = x - 96, y - 34
        draw.text((label_x, label_y), value, font=SMALL_FONT, fill=TEXT)

    draw_centered(draw, (x0, HEIGHT - 90, x1, HEIGHT - 56), "training step", LABEL_FONT)
    draw.text((PADDING_LEFT, HEIGHT - 34), "Metric copied from the real HF Jobs trainer log for the same smoke run.", font=SMALL_FONT, fill=AXIS)

    path = OUTPUT_DIR / "loss_curve.png"
    image.save(path)
    return path


def generate_baseline_chart() -> Path:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = draw_axes(draw)

    title = "Deterministic Baseline Comparison (overall score)"
    subtitle = "Higher is better; heuristic Portfolio Manager beats random across the 4 task suite."
    draw.text((PADDING_LEFT, 24), title, font=TITLE_FONT, fill=TEXT)
    draw.text((PADDING_LEFT, 60), subtitle, font=BODY_FONT, fill=AXIS)

    for idx in range(6):
        frac = idx / 5
        y = y0 - frac * (y0 - y1)
        value = frac
        draw.line((x0, y, x1, y), fill=GRID, width=1)
        label = f"{value:.1f}"
        bbox = draw.textbbox((0, 0), label, font=TICK_FONT)
        draw.text((x0 - 20 - (bbox[2] - bbox[0]), y - 10), label, font=TICK_FONT, fill=TEXT)

    labels = list(BASELINE_SCORE.keys())
    bar_gap = 180
    bar_width = 240
    start_x = x0 + 140
    colors = [BLUE, GREEN]
    for idx, label in enumerate(labels):
        value = BASELINE_SCORE[label]
        left = start_x + idx * (bar_width + bar_gap)
        right = left + bar_width
        top = y0 - (value / 1.0) * (y0 - y1)
        draw.rounded_rectangle((left, top, right, y0), radius=16, fill=colors[idx])
        value_label = f"{value:.4f}"
        bbox = draw.textbbox((0, 0), value_label, font=BODY_FONT)
        draw.text((left + (bar_width - (bbox[2] - bbox[0])) / 2, top - 36), value_label, font=BODY_FONT, fill=TEXT)
        label_bbox = draw.textbbox((0, 0), label, font=LABEL_FONT)
        draw.text((left + (bar_width - (label_bbox[2] - label_bbox[0])) / 2, y0 + 18), label, font=LABEL_FONT, fill=TEXT)

    draw.text((PADDING_LEFT, HEIGHT - 36), "Numbers come from the verified multi-seed baseline snapshot embedded in the Colab notebook.", font=SMALL_FONT, fill=AXIS)

    path = OUTPUT_DIR / "baseline_score_comparison.png"
    image.save(path)
    return path


def generate_conflict_snapshot() -> Path:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#f8fafc")
    draw = ImageDraw.Draw(image)

    draw.text((64, 38), "Conflict Resolution Snapshot", font=BIG_FONT, fill=TEXT)
    draw.text(
        (64, 92),
        "The benchmark exposes when a PM blindly follows Research while Risk is warning about mandate pressure.",
        font=BODY_FONT,
        fill=AXIS,
    )

    panel_y = 150
    panel_h = 270
    gap = 28
    panel_w = (WIDTH - 128 - gap * 2) // 3
    panels = [
        (
            "Research",
            "#dbeafe",
            BLUE,
            "BUY high-momentum tech. Signals are positive and sector trend is improving.",
        ),
        (
            "Risk",
            "#fee2e2",
            RED,
            "Volatility and concentration are above mandate. Aggressive allocation triggers penalty.",
        ),
        (
            "PM Decision",
            "#dcfce7",
            GREEN,
            "Good behavior: query both agents, reduce concentration, and keep risk-adjusted exposure.",
        ),
    ]
    for idx, (title, bg, accent, body) in enumerate(panels):
        x0 = 64 + idx * (panel_w + gap)
        y0 = panel_y
        x1 = x0 + panel_w
        y1 = y0 + panel_h
        draw.rounded_rectangle((x0, y0, x1, y1), radius=18, fill=bg, outline=accent, width=3)
        draw.text((x0 + 24, y0 + 24), title, font=MID_FONT, fill=accent)
        draw_wrapped(draw, (x0 + 24, y0 + 78), body, BODY_FONT, fill=TEXT, max_width=panel_w - 48)

    lower_y = 456
    lower_bottom = 652
    draw.rounded_rectangle((64, lower_y, 556, lower_bottom), radius=18, fill="white", outline=RED, width=3)
    draw.text((92, lower_y + 26), "Failure Mode: Ignoring Risk", font=MID_FONT, fill=RED)
    draw_wrapped(
        draw,
        (92, lower_y + 78),
        "Baseline PM chases the buy signal, violates constraints, and loses reward to compliance and drawdown penalties.",
        COMPACT_BODY_FONT,
        fill=TEXT,
        max_width=430,
        line_gap=6,
    )

    draw.rounded_rectangle((644, lower_y, 1136, lower_bottom), radius=18, fill="white", outline=GREEN, width=3)
    draw.text((672, lower_y + 26), "Target Behavior: Resolve Conflict", font=MID_FONT, fill=GREEN)
    draw_wrapped(
        draw,
        (672, lower_y + 78),
        "The environment rewards balanced decisions: use Research, respect Risk, reduce concentration, preserve return signal.",
        COMPACT_BODY_FONT,
        fill=TEXT,
        max_width=430,
        line_gap=6,
    )

    draw.text(
        (64, HEIGHT - 42),
        "Falsifiable claim: GRPO smoke reward improved from 0.0193 to 0.0275 under this verifier.",
        font=SMALL_FONT,
        fill=AXIS,
    )

    path = OUTPUT_DIR / "conflict_resolution_snapshot.png"
    image.save(path)
    return path


def generate_trace_comparison() -> Path:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#f8fafc")
    draw = ImageDraw.Draw(image)

    draw.text((64, 36), "Demo Trace: Random PM vs Heuristic PM", font=BIG_FONT, fill=TEXT)
    draw.text(
        (64, 90),
        "research_risk_conflict task, seed=7. Same environment; different decision discipline.",
        font=BODY_FONT,
        fill=AXIS,
    )

    card_y = 140
    card_h = 480
    card_w = 508
    left_x = 64
    right_x = 628

    def card(
        x: int,
        title: str,
        subtitle: str,
        border: str,
        bg: str,
        lines: list[str],
        score: str,
        violations: str,
        takeaway: str,
    ) -> None:
        draw.rounded_rectangle((x, card_y, x + card_w, card_y + card_h), radius=22, fill=bg, outline=border, width=3)
        draw.text((x + 28, card_y + 28), title, font=MID_FONT, fill=border)
        draw.text((x + 28, card_y + 66), subtitle, font=SMALL_FONT, fill=AXIS)

        y = card_y + 112
        for idx, line in enumerate(lines, start=1):
            draw.text((x + 28, y), f"Step {idx}", font=SMALL_FONT, fill=border)
            draw_wrapped(draw, (x + 116, y - 2), line, COMPACT_BODY_FONT, fill=TEXT, max_width=350, line_gap=4)
            y += 58

        metric_y = card_y + 350
        draw.rounded_rectangle((x + 28, metric_y, x + 222, metric_y + 56), radius=14, fill="white", outline="#cbd5e1", width=2)
        draw.text((x + 48, metric_y + 12), f"Score: {score}", font=COMPACT_BODY_FONT, fill=TEXT)
        draw.rounded_rectangle((x + 242, metric_y, x + 480, metric_y + 56), radius=14, fill="white", outline="#cbd5e1", width=2)
        draw.text((x + 262, metric_y + 12), f"Violations: {violations}", font=COMPACT_BODY_FONT, fill=TEXT)

        draw_wrapped(draw, (x + 28, card_y + 424), takeaway, SMALL_FONT, fill=AXIS, max_width=450, line_gap=4)

    card(
        left_x,
        "Random PM",
        "No committee discipline",
        RED,
        "#fff1f2",
        [
            "allocate top2_conviction before reading the committee",
            "query_research randomly",
            "hold without a clear risk response",
            "query_risk randomly after exposure is already live",
        ],
        "0.2000",
        "9",
        "Takeaway: action order is noisy; conflict is mostly accidental.",
    )
    card(
        right_x,
        "Heuristic PM",
        "Committee-aware baseline",
        GREEN,
        "#ecfdf5",
        [
            "query_research on the sector first",
            "hold until the signal/risk mix is clearer",
            "allocate balanced_top3 instead of all-in conviction",
            "move_to_cash when risk pressure rises",
        ],
        "0.4071",
        "3",
        "Takeaway: still imperfect, but visibly more disciplined.",
    )

    draw.text(
        (64, HEIGHT - 48),
        "Training evidence is separate: the GRPO smoke run improves verifier reward from 0.0193 to 0.0275.",
        font=SMALL_FONT,
        fill=AXIS,
    )

    path = OUTPUT_DIR / "demo_trace_comparison.png"
    image.save(path)
    return path


def generate_grpo_behavior_sample() -> Path:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#f8fafc")
    draw = ImageDraw.Draw(image)

    draw.text((64, 36), "GRPO Behavior Sample: Reward Prefers Committee Use", font=BIG_FONT, fill=TEXT)
    draw.text(
        (64, 90),
        "Actual completion trace from HF job hf-job-20260426-trained-trace, trainer step 4.",
        font=BODY_FONT,
        fill=AXIS,
    )

    card_y = 144
    card_h = 438
    card_w = 508
    left_x = 64
    right_x = 628

    def card(
        x: int,
        title: str,
        border: str,
        bg: str,
        steps: list[str],
        metrics: list[tuple[str, str]],
        takeaway: str,
    ) -> None:
        draw.rounded_rectangle((x, card_y, x + card_w, card_y + card_h), radius=22, fill=bg, outline=border, width=3)
        draw.text((x + 28, card_y + 28), title, font=MID_FONT, fill=border)
        draw.text((x + 28, card_y + 66), "Task: mandate_drift", font=SMALL_FONT, fill=AXIS)

        y = card_y + 112
        for idx, step in enumerate(steps, start=1):
            draw.text((x + 28, y), f"{idx}.", font=COMPACT_BODY_FONT, fill=border)
            draw_wrapped(draw, (x + 62, y - 2), step, COMPACT_BODY_FONT, fill=TEXT, max_width=410, line_gap=4)
            y += 54

        metric_y = card_y + 282
        metric_w = 140
        for idx, (label, value) in enumerate(metrics):
            mx = x + 28 + idx * (metric_w + 14)
            draw.rounded_rectangle((mx, metric_y, mx + metric_w, metric_y + 70), radius=14, fill="white", outline="#cbd5e1", width=2)
            draw_centered(draw, (mx, metric_y + 10, mx + metric_w, metric_y + 34), label, SMALL_FONT, fill=AXIS)
            draw_centered(draw, (mx, metric_y + 34, mx + metric_w, metric_y + 64), value, COMPACT_BODY_FONT, fill=TEXT)

        draw_wrapped(draw, (x + 28, card_y + 374), takeaway, SMALL_FONT, fill=AXIS, max_width=450, line_gap=4)

    card(
        left_x,
        "Lower-Reward Candidate",
        RED,
        "#fff1f2",
        [
            "query_risk",
            "allocate balanced_top3",
            "takes exposure before refreshing Research",
        ],
        [("task", "0.0119"), ("compliance", "0.0044"), ("advantage", "-0.7026")],
        "The model asks Risk, but still allocates before resolving the full committee conflict.",
    )
    card(
        right_x,
        "Higher-Reward Candidate",
        GREEN,
        "#ecfdf5",
        [
            "query_risk",
            "query_research",
            "collects both committee views before exposure",
        ],
        [("task", "0.0298"), ("compliance", "0.0089"), ("advantage", "+0.7026")],
        "The verifier assigns higher reward to using both advisors before committing capital.",
    )

    draw.text(
        (64, HEIGHT - 78),
        "Scope note: this is a GRPO training-rollout sample, not a claim that the saved adapter beats the heuristic in deployment.",
        font=SMALL_FONT,
        fill=AXIS,
    )
    draw.text(
        (64, HEIGHT - 48),
        "It shows what the reward model is reinforcing: conflict resolution through Research + Risk evidence gathering.",
        font=SMALL_FONT,
        fill=AXIS,
    )

    path = OUTPUT_DIR / "grpo_behavior_sample.png"
    image.save(path)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reward_path = generate_reward_curve()
    loss_path = generate_loss_curve()
    baseline_path = generate_baseline_chart()
    conflict_path = generate_conflict_snapshot()
    trace_path = generate_trace_comparison()
    grpo_path = generate_grpo_behavior_sample()
    print(f"wrote {reward_path}")
    print(f"wrote {loss_path}")
    print(f"wrote {baseline_path}")
    print(f"wrote {conflict_path}")
    print(f"wrote {trace_path}")
    print(f"wrote {grpo_path}")


if __name__ == "__main__":
    main()
