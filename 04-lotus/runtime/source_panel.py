"""Draw source excerpts with their original line numbers."""
import textwrap

from PIL import Image, ImageDraw, ImageFont


def source_panel(section, width, height):
    path, marker = section
    lines = path.read_text(encoding='utf-8').splitlines()
    start = next(i for i, line in enumerate(lines) if line.lstrip().startswith(marker))
    # Match the reference's subdued gutter, divider and green section opening.
    # Use a slightly larger type size for the longer names in the duet source.
    scale = width / 1080
    font = ImageFont.truetype('/System/Library/Fonts/Menlo.ttc', round(25 * scale))
    line_height = round(34 * scale)
    margin = round(13 * scale)
    code_x = round(86 * scale)
    count = (height - margin) // line_height
    excerpt = textwrap.dedent('\n'.join(lines[start : start + count])).splitlines()
    pane = Image.new('RGB', (width, height), (8, 10, 14))
    draw = ImageDraw.Draw(pane)
    draw.line((0, 0, width, 0), fill=(40, 48, 60))
    for row, line in enumerate(excerpt):
        y = margin + row * line_height
        draw.text((margin, y), f'{start + row + 1:4d}', font=font, fill=(70, 80, 96))
        color = (110, 150, 120) if line.lstrip().startswith('#') else (176, 186, 204)
        draw.text((code_x, y), line, font=font, fill=(150, 220, 170) if row == 0 else color)
    return pane
