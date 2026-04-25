"""Generate committed README assets from the verified Colab smoke run."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "assets"

REWARD_SERIES = [0.019333332777023315, 0.0160, 0.0006, 0.027521273121237755]
BASELINE_SCORE = {
    "heuristic": 0.4084,
    "random": 0.2504,
}

WIDTH = 1200
HEIGHT = 720
PADDING_LEFT = 140
PADDING_RIGHT = 60
PADDING_TOP = 90
PADDING_BOTTOM = 110
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
SMALL_FONT = load_font(18)


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


def generate_reward_curve() -> Path:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = draw_axes(draw)

    title = "Smoke Training Reward Curve (Colab T4, Qwen3-0.6B, LoRA, 4 steps)"
    draw.text((PADDING_LEFT, 24), title, font=TITLE_FONT, fill=TEXT)

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
        draw.text((x + 10, y - 26), value, font=SMALL_FONT, fill=TEXT)

    draw_centered(draw, (x0, HEIGHT - 60, x1, HEIGHT - 20), "training step", LABEL_FONT)
    draw.text((28, (y1 + y0) / 2), "reward", font=LABEL_FONT, fill=TEXT)
    draw.text((PADDING_LEFT, HEIGHT - 36), "Verified smoke run from the Colab T4 execution used in the on-site demo.", font=SMALL_FONT, fill=AXIS)

    path = OUTPUT_DIR / "reward_curve.png"
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

    draw_centered(draw, (x0, HEIGHT - 60, x1, HEIGHT - 20), "policy", LABEL_FONT)
    draw.text((28, (y1 + y0) / 2), "overall score (0-1)", font=LABEL_FONT, fill=TEXT)
    draw.text((PADDING_LEFT, HEIGHT - 36), "Numbers come from the verified multi-seed baseline snapshot embedded in the Colab notebook.", font=SMALL_FONT, fill=AXIS)

    path = OUTPUT_DIR / "baseline_score_comparison.png"
    image.save(path)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reward_path = generate_reward_curve()
    baseline_path = generate_baseline_chart()
    print(f"wrote {reward_path}")
    print(f"wrote {baseline_path}")


if __name__ == "__main__":
    main()
