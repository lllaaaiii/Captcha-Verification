import os.path
import cv2
import glob
import imutils

CAPTCHA_IMAGE_FOLDER = "train_data"
OUTPUT_FOLDER = "extracted_letter"

captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
counts = {}
for (i, captcha_image_file) in enumerate(captcha_image_files):
    print(i + 1)
    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = os.path.splitext(filename)[0]
    image = cv2.imread(captcha_image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]    # 轉為非黑即白的圖片(要先轉為灰階才可用)

    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # cv2.RETR_EXTERNAL=>表示只檢測外輪廓
    contours = contours[0] if imutils.is_cv2() else contours[1]
    letter_image_regions = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)    # x，y是矩陣左上點的坐標，w，h是矩陣的寬和高
        if w / h > 1.25:
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            letter_image_regions.append((x, y, w, h))
    if len(letter_image_regions) != 4:
        continue

    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
    for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
        x, y, w, h = letter_bounding_box
        letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]
        save_path = os.path.join(OUTPUT_FOLDER, letter_text)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(p, letter_image)
        counts[letter_text] = count + 1
