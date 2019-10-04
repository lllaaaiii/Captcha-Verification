import numpy as np
import cv2
import os
from imutils import paths
counts = {}


def split(filename):
    image = cv2.imread(filename)
    filename = filename.split('\\')[1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    _,contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    letter_image_regions = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        # print((x, y, w, h))
        if w == 225:
            continue
        if w > 25 and h > 40:
            if 60 < w <= 120:
                half_width = int(w / 2)
                letter_image_regions.append((x, y, half_width, h))
                letter_image_regions.append((x + half_width, y, half_width, h))
            elif 120 < w <= 180:
                width = int(w / 3)
                letter_image_regions.append((x, y, width, h))
                letter_image_regions.append((x + width, y, width, h))
                letter_image_regions.append((x + width*2, y, width, h))
            elif 180 < w < 205:
                width = int(w / 4)
                letter_image_regions.append((x, y, width, h))
                letter_image_regions.append((x + width, y, width, h))
                letter_image_regions.append((x + width * 2, y, width, h))
                letter_image_regions.append((x + width * 3, y, width, h))
            else:
                letter_image_regions.append((x, y, w, h))
    if len(letter_image_regions) != 4:
        letter_image_regions.clear()
        w = gray.shape[1] - 30  # 寬
        h = gray.shape[0]
        x = 25
        y = 2
        w = int(w / 4)
        for i in range(4):
            letter_image_regions.append((x, y, w, h))
            x = x + w
        
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
    for letter_bounding_box, letter_text in zip(letter_image_regions, filename):
        x, y, w, h = letter_bounding_box
        # print(x, y, w, h)
        letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]
        save_path = os.path.join(OUTPUT_FOLDER, letter_text)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(p, letter_image)
        counts[letter_text] = count + 1


OUTPUT_FOLDER = "extracted_letter_rail"
captcha_image_files = list(paths.list_images('train_data_rail_preprocess'))
for image_file in captcha_image_files:
    print(image_file)
    split(image_file)