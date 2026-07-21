#!/usr/bin/env python3
"""Patch panel (c) of framework.png: the original artwork shows
confidence-weighted pooling, but the final method uses mean pooling +
calibrated shrinkage. Repaint the banner title and the formula box.
Run from repo root with an env that has PIL + matplotlib.
"""
from PIL import Image, ImageDraw
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "paper_v2/latex/framework.png"
DST = "paper_v3/figures/framework.png"
S = 2.59  # displayed(2000px) -> original scale

img = Image.open(SRC).convert("RGB")
d = ImageDraw.Draw(img)


def orig(x, y):
    return int(x * S), int(y * S)


def render_text(text, color, weight="bold", fontsize=60):
    """Render text/mathtext on transparent bg, tightly cropped."""
    fig = plt.figure(figsize=(30, 6), dpi=100)
    fig.patch.set_alpha(0.0)
    t = fig.text(0.5, 0.5, text, ha="center", va="center",
                 fontsize=fontsize, color=color, weight=weight,
                 family="DejaVu Sans")
    fig.canvas.draw()
    bbox = t.get_window_extent()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    hgt = buf.shape[0]
    x0, y0 = max(int(bbox.x0) - 4, 0), max(int(hgt - bbox.y1) - 4, 0)
    x1, y1 = int(bbox.x1) + 4, int(hgt - bbox.y0) + 4
    return Image.fromarray(buf[y0:y1, x0:x1])


def paste_fit(base, im, cx, cy, max_w, max_h):
    """Scale im to fit in (max_w, max_h), paste centered at (cx, cy)."""
    s = min(max_w / im.width, max_h / im.height)
    im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))),
                   Image.LANCZOS)
    base.paste(im, (cx - im.width // 2, cy - im.height // 2), im)


# ---- 1. banner ----
bx0, by0 = orig(1408, 8)
bx1, by1 = orig(1935, 46)
banner_color = img.getpixel(orig(1520, 27))
r = (by1 - by0) // 2
d.rounded_rectangle([bx0, by0, bx1, by1], radius=r, fill=banner_color)
title = render_text("(c) Calibrated Aggregation", "white")
paste_fit(img, title, (bx0 + bx1) // 2, (by0 + by1) // 2,
          0.92 * (bx1 - bx0), 0.62 * (by1 - by0))

# ---- 2. formula box interior ----
fx0, fy0 = orig(1412, 296)
fx1, fy1 = orig(1933, 497)
box_bg = img.getpixel(orig(1670, 420))
d.rounded_rectangle([fx0 + 6, fy0 + 6, fx1 - 6, fy1 - 6], radius=20, fill=box_bg)

bw, bh = fx1 - fx0, fy1 - fy0
cx = (fx0 + fx1) // 2
heading = render_text("Mean Pooling + Calibrated Shrinkage", "#282828")
paste_fit(img, heading, cx, fy0 + int(0.20 * bh), 0.85 * bw, 0.16 * bh)

f1 = render_text(r"$\bar{p} \,=\, \frac{1}{J}\sum_{j=1}^{J} p_j^{(R)}$",
                 "#141414", weight="normal")
paste_fit(img, f1, cx, fy0 + int(0.48 * bh), 0.55 * bw, 0.30 * bh)

f2 = render_text(r"$\hat{p} \,=\, p_0 + w\,(\bar{p}-p_0),\ \ "
                 r"w=w_{\mathrm{hi}}\ \mathrm{if}\ \bar{p}>p_0\ "
                 r"\mathrm{else}\ w_{\mathrm{lo}}$",
                 "#141414", weight="normal")
paste_fit(img, f2, cx, fy0 + int(0.77 * bh), 0.88 * bw, 0.18 * bh)

img.save(DST)
print("Wrote", DST)
