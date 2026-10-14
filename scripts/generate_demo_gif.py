"""Generate authentic animated demo GIF for SiliconMark README and documentation."""

from __future__ import annotations

import io
import os
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_IMG = ROOT_DIR / "docs" / "images"
DOCS_IMG.mkdir(parents=True, exist_ok=True)

WIDTH, HEIGHT = 1200, 750
BG_COLOR = (13, 17, 23)
PANEL_COLOR = (22, 27, 34)
BORDER_COLOR = (48, 54, 61)
TITLE_COLOR = (139, 148, 158)
TEXT_COLOR = (230, 237, 243)
CYAN = (56, 189, 248)
GREEN = (74, 222, 128)
YELLOW = (250, 204, 21)
PURPLE = (192, 132, 252)

FONT_PATH = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "consola.ttf")
FONT_BOLD_PATH = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "consolab.ttf")

font = ImageFont.truetype(FONT_PATH, 16)
font_bold = ImageFont.truetype(FONT_BOLD_PATH, 16)
font_title = ImageFont.truetype(FONT_BOLD_PATH, 14)


def create_terminal_window(command: str, lines: list[tuple[str, tuple[int, int, int]]]) -> Image.Image:
    """Create a dark mode terminal screenshot with window chrome and colorized output."""
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Outer border & header
    draw.rectangle([(20, 20), (WIDTH - 20, HEIGHT - 20)], fill=PANEL_COLOR, outline=BORDER_COLOR, width=1)
    draw.rectangle([(20, 20), (WIDTH - 20, 60)], fill=(28, 33, 40))
    draw.line([(20, 60), (WIDTH - 20, 60)], fill=BORDER_COLOR, width=1)

    # Window controls (macOS style dots)
    draw.ellipse([(35, 36), (47, 48)], fill=(255, 95, 86))
    draw.ellipse([(55, 36), (67, 48)], fill=(255, 189, 46))
    draw.ellipse([(75, 36), (87, 48)], fill=(39, 201, 63))

    # Title
    title = f"siliconmark ~ {command}"
    draw.text((WIDTH // 2 - len(title) * 4, 34), title, font=font_title, fill=TITLE_COLOR)

    # Prompt
    prompt_y = 75
    draw.text((40, prompt_y), "$ ", font=font_bold, fill=GREEN)
    draw.text((60, prompt_y), f"siliconmark {command}", font=font_bold, fill=TEXT_COLOR)

    y = prompt_y + 30
    for text, color in lines:
        if y > HEIGHT - 40:
            break
        draw.text((40, y), text, font=font, fill=color)
        y += 20

    return img


def run_command_capture(args: list[str]) -> list[str]:
    """Run CLI command and return output lines."""
    py_exe = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
    res = subprocess.run([str(py_exe), "-m", "siliconmark.cli.main"] + args, capture_output=True, text=True, cwd=str(ROOT_DIR))
    return res.stdout.splitlines()


def main() -> None:
    frames: list[Image.Image] = []

    # Frame 1: SiliconMark Info
    info_raw = run_command_capture(["info"])
    info_lines = []
    for line in info_raw[:32]:
        color = TEXT_COLOR
        if "Property" in line or "Value" in line:
            color = CYAN
        elif "Device Name" in line or "Theoretical" in line or "Backend" in line:
            color = YELLOW
        info_lines.append((line, color))
    frames.append(create_terminal_window("info", info_lines))

    # Frame 2: Roofline Model
    roof_raw = run_command_capture(["roofline"])
    roof_lines = []
    for line in roof_raw[:32]:
        color = TEXT_COLOR
        if "ROOFLINE" in line or "Knee Point" in line:
            color = CYAN
        elif "Compute Ceiling" in line:
            color = GREEN
        elif "Slope" in line or "Bandwidth" in line:
            color = YELLOW
        roof_lines.append((line, color))
    frames.append(create_terminal_window("roofline", roof_lines))

    # Frame 3: Model Estimator
    est_raw = run_command_capture(["estimate", "--model", "llama-3-8b", "--quant", "q4_k_m"])
    est_lines = []
    for line in est_raw[:32]:
        color = TEXT_COLOR
        if "Estimator" in line or "Llama-3-8B" in line:
            color = PURPLE
        elif "Expected Decode Speed" in line or "Tokens/sec" in line:
            color = GREEN
        elif "VRAM" in line or "Memory Fit" in line:
            color = CYAN
        est_lines.append((line, color))
    frames.append(create_terminal_window("estimate --model llama-3-8b --quant q4_k_m", est_lines))

    # Frame 4: Run Quick Benchmark Scorecard
    run_raw = run_command_capture(["run", "--quick"])
    run_lines = []
    for line in run_raw[:32]:
        color = TEXT_COLOR
        if "SCORE" in line or "Scorecard" in line:
            color = YELLOW
        elif "Peak" in line or "Sustained" in line:
            color = GREEN
        elif "Classification" in line or "Grade" in line:
            color = CYAN
        run_lines.append((line, color))
    frames.append(create_terminal_window("run --quick", run_lines))

    # Frame 5: Hardware Comparison
    comp_raw = run_command_capture(["compare", "latest", "rtx4090"])
    comp_lines = []
    for line in comp_raw[:32]:
        color = TEXT_COLOR
        if "Comparison" in line:
            color = CYAN
        elif "+" in line:
            color = GREEN
        elif "Overall" in line or "Bandwidth" in line:
            color = YELLOW
        comp_lines.append((line, color))
    frames.append(create_terminal_window("compare latest rtx4090", comp_lines))

    # Frame 6: Dashboard View
    dash_path = DOCS_IMG / "siliconmark_dashboard.png"
    if dash_path.exists():
        dash_im = Image.open(dash_path).convert("RGB")
        # Resize preserving aspect ratio into 1200x750 canvas
        dash_im.thumbnail((WIDTH - 40, HEIGHT - 80), Image.Resampling.LANCZOS)
        dash_frame = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
        dash_draw = ImageDraw.Draw(dash_frame)
        dash_draw.rectangle([(20, 20), (WIDTH - 20, HEIGHT - 20)], fill=PANEL_COLOR, outline=BORDER_COLOR, width=1)
        dash_draw.rectangle([(20, 20), (WIDTH - 20, 60)], fill=(28, 33, 40))
        dash_draw.line([(20, 60), (WIDTH - 20, 60)], fill=BORDER_COLOR, width=1)
        dash_draw.ellipse([(35, 36), (47, 48)], fill=(255, 95, 86))
        dash_draw.ellipse([(55, 36), (67, 48)], fill=(255, 189, 46))
        dash_draw.ellipse([(75, 36), (87, 48)], fill=(39, 201, 63))
        dash_draw.text((WIDTH // 2 - 120, 34), "http://localhost:8000 - SiliconMark Web Dashboard", font=font_title, fill=TITLE_COLOR)
        # Paste centered
        ox = (WIDTH - dash_im.width) // 2
        oy = 70 + (HEIGHT - 80 - dash_im.height) // 2
        dash_frame.paste(dash_im, (ox, oy))
        frames.append(dash_frame)

    # Save to temp folder and invoke ffmpeg for ultra-high quality palette-optimized GIF
    temp_dir = ROOT_DIR / "results" / "gif_frames"
    temp_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        # Repeat frame 2 times to give 2 seconds dwell time
        frame.save(temp_dir / f"frame_{i:02d}.png")

    # ffmpeg loop: 2.2 seconds per frame
    output_gif = DOCS_IMG / "siliconmark_demo.gif"
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        "0.45",
        "-i",
        str(temp_dir / "frame_%02d.png"),
        "-vf",
        "scale=1000:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
        str(output_gif),
    ]
    subprocess.run(cmd, check=True)
    print(f"Generated {output_gif} successfully ({output_gif.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
