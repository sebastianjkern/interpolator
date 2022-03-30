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


c1 = "#b7efc5"
c2 = "#10451d"

from scipy.special import comb


def smoothstep(x, x_min=0, x_max=1, N=1):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
        result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)

    return result


def hsv_interpolate(color1: str, color2: str, factor, use_smoothstep=False):
    rgb1 = sRGBColor.new_from_rgb_hex(color1)
    rgb2 = sRGBColor.new_from_rgb_hex(color2)

    hsv1 = np.array(convert_color(rgb1, HSVColor).get_value_tuple())
    hsv2 = np.array(convert_color(rgb2, HSVColor).get_value_tuple())

    if use_smoothstep:
        factor = smoothstep(factor)

    return convert_color(HSVColor(*(hsv1 * factor + hsv2 * (1 - factor)).tolist()), sRGBColor).get_rgb_hex()


def rgb_interpolate(color1: str, color2: str, factor, use_smoothstep=False):
    rgb1 = np.array(sRGBColor.new_from_rgb_hex(color1).get_value_tuple())
    rgb2 = np.array(sRGBColor.new_from_rgb_hex(color2).get_value_tuple())

    if use_smoothstep:
        factor = smoothstep(factor)

    return sRGBColor(*(rgb1 * factor + rgb2 * (1 - factor))).get_rgb_hex()


colors = []
width, height = 250, 80

for t in range(0, width):
    t /= width

    rgb3 = sRGBColor.new_from_rgb_hex(hsv_interpolate(c1, c2, t))

    print(rgb3.get_rgb_hex())

    colors.append([rgb3.rgb_r * 255, rgb3.rgb_g * 255, rgb3.rgb_b * 255])

img = np.zeros((height, width, 3), dtype=np.uint8)

for w in range(width):
    for h in range(height):
        img[h][w] = colors[w]

plt.imshow(img)
plt.show()

im = Image.fromarray(img)
im.save("image.png")
