import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from colormath.color_conversions import convert_color
from colormath.color_objects import sRGBColor, HSVColor


def rgb_col(r: int, g: int, b: int, *a):
    values = np.array([r, g, b, *a], dtype=float)
    values /= 255
    return tuple(values.tolist())


def hex_col(string: str):
    g = string.lstrip("#")
    col = tuple(int(g[i:i + 2], 16) for i in (0, 2, 4))
    return rgb_col(*col, 255)


rgb1 = sRGBColor(*hex_col("#2d82b7")[:3])
rgb2 = sRGBColor(*hex_col("#07004d")[:3])

hsv1 = convert_color(rgb1, HSVColor)
hsv2 = convert_color(rgb2, HSVColor)

colors = []
width, height = 250, 100

for t in range(0, width):
    t /= width
    hsv_h = hsv1.hsv_h * t + hsv2.hsv_h * (1 - t)
    hsv_s = hsv1.hsv_s * t + hsv2.hsv_s * (1 - t)
    hsv_v = hsv1.hsv_v * t + hsv2.hsv_v * (1 - t)

    lab3 = HSVColor(hsv_h, hsv_s, hsv_v)

    rgb3 = convert_color(lab3, sRGBColor)

    colors.append([rgb3.rgb_r * 255, rgb3.rgb_g * 255, rgb3.rgb_b * 255])

img = np.zeros((height, width, 3), dtype=np.uint8)

for w in range(width):
    for h in range(height):
        img[h][w] = colors[w]

plt.imshow(img)
plt.show()

im = Image.fromarray(img)
im.save("image.png")
