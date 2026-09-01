from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_DIR / "results" / "full_image_predictions"
OUTPUT_DIR = PROJECT_DIR / "results" / "report_figures"
OUTPUT_PATH = OUTPUT_DIR / "stitching_comparison_p128_v25.png"

CROP_BOX = (512, 512, 1280, 1280)
PANEL_SIZE = 310
LABEL_W = 320
PAD = 24
GAP = 18
TITLE_H = 86
HEADER_H = 46
ROW_H = 380
FOOTER_H = 46

METHODS = [
    {
        "name": "No overlap",
        "setting": "patch 128, stride 128",
        "metrics": "Mean MAE 0.08521 | RMSE 0.11250",
        "folder": "V25_O_A$3D_small_unet_p128_s128_none",
        "accent": (155, 80, 76),
    },
    {
        "name": "Uniform overlap",
        "setting": "patch 128, stride 64",
        "metrics": "Mean MAE 0.08486 | RMSE 0.11205",
        "folder": "V25_O_A$3D_small_unet_p128_s64_uniform",
        "accent": (70, 115, 165),
    },
    {
        "name": "Gaussian overlap",
        "setting": "patch 128, stride 64",
        "metrics": "Mean MAE 0.08465 | RMSE 0.11176",
        "folder": "V25_O_A$3D_small_unet_p128_s64_gaussian",
        "accent": (55, 135, 98),
    },
]


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def image_crop(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    image = Image.open(path).convert("RGB")
    crop_box = CROP_BOX
    if crop_box[2] > image.width or crop_box[3] > image.height:
        side = min(image.width, image.height, CROP_BOX[2] - CROP_BOX[0])
        left = (image.width - side) // 2
        top = (image.height - side) // 2
        crop_box = (left, top, left + side, top + side)
    return image.crop(crop_box).resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.LANCZOS)


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill: tuple[int, int, int]) -> None:
    draw.text(xy, text, font=font, fill=fill)


def draw_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    x: int,
    y: int,
    label: str,
    font,
) -> None:
    draw.rounded_rectangle(
        (x - 1, y - 1, x + PANEL_SIZE + 1, y + PANEL_SIZE + 1),
        radius=4,
        outline=(185, 195, 205),
        width=2,
    )
    canvas.paste(image, (x, y))
    draw_text(draw, (x, y + PANEL_SIZE + 8), label, font, (65, 72, 82))


def build_figure() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    width = PAD * 2 + LABEL_W + GAP * 2 + PANEL_SIZE * 2
    height = TITLE_H + HEADER_H + ROW_H * len(METHODS) + FOOTER_H
    canvas = Image.new("RGB", (width, height), (247, 248, 250))
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(30, bold=True)
    subtitle_font = load_font(17)
    header_font = load_font(17, bold=True)
    method_font = load_font(23, bold=True)
    body_font = load_font(16)
    small_font = load_font(14)

    draw_text(draw, (PAD, 22), "Stitching artifact comparison, sample V25_O_A$3D", title_font, (28, 48, 70))
    draw_text(
        draw,
        (PAD, 58),
        "Same crop from full-image prediction. Lower error and softer boundaries indicate better stitching.",
        subtitle_font,
        (85, 94, 105),
    )

    label_x = PAD
    pred_x = PAD + LABEL_W
    error_x = pred_x + PANEL_SIZE + GAP
    header_y = TITLE_H + 8
    draw_text(draw, (pred_x, header_y), "Predicted height crop", header_font, (40, 56, 72))
    draw_text(draw, (error_x, header_y), "Absolute error crop", header_font, (40, 56, 72))

    for row_index, method in enumerate(METHODS):
        y = TITLE_H + HEADER_H + row_index * ROW_H
        row_top = y + 6
        row_bottom = y + ROW_H - 12
        draw.rounded_rectangle(
            (PAD, row_top, width - PAD, row_bottom),
            radius=7,
            fill=(255, 255, 255),
            outline=(221, 226, 232),
            width=1,
        )
        accent = method["accent"]
        draw.rounded_rectangle(
            (PAD, row_top, PAD + 9, row_bottom),
            radius=4,
            fill=accent,
        )

        text_y = row_top + 34
        draw_text(draw, (label_x + 26, text_y), method["name"], method_font, (30, 37, 45))
        draw_text(draw, (label_x + 26, text_y + 38), method["setting"], body_font, (80, 90, 102))
        draw_text(draw, (label_x + 26, text_y + 70), method["metrics"], small_font, (80, 90, 102))

        folder = BASE_DIR / method["folder"]
        pred = image_crop(folder / "predicted_height_norm.png")
        error = image_crop(folder / "absolute_error_norm.png")
        image_y = row_top + 28
        draw_panel(canvas, draw, pred, pred_x, image_y, "prediction", small_font)
        draw_panel(canvas, draw, error, error_x, image_y, "absolute error", small_font)

    footer_y = height - FOOTER_H + 10
    draw_text(
        draw,
        (PAD, footer_y),
        "Visual takeaway: overlap averaging reduces hard patch transitions; Gaussian weighting is the current best setting.",
        body_font,
        (70, 78, 88),
    )

    canvas.save(OUTPUT_PATH)
    print(OUTPUT_PATH)
    return OUTPUT_PATH


if __name__ == "__main__":
    build_figure()
