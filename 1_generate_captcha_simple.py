import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os.path
count = 1


def make(strs, width=72, height=24):
    print(strs)
    im = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype('verdana.ttf', width // 4)
    font_width, font_height = font.getsize(strs)
    strs_len = len(strs)
    x = (width - font_width) // 2
    y = (height - font_height) // 2
    for i in strs:
        draw.text((x, y), i, (0, 0, 0), font) # position, string, color, font
        draw = ImageDraw.Draw(im)
        x += font_width / strs_len

    if train_or_test < 700:
        name = 'train_data/'+str(strs) + '.png'
    else:
        name = 'test_data/'+str(strs) + '.png'
    im = im.convert('L')
    im.save(name)

if __name__ == '__main__':
    for i in range(1000):
        num = ''.join(random.sample('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', 4))
        train_or_test = i
        make(num)
        count = count + 1
