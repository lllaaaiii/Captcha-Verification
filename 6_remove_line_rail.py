import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import cv2
from imutils import paths


def rm_regression(img, border):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w = gray.shape[1]
    scale = 225 / w
    new_im = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    denoise = cv2.fastNlMeansDenoising(new_im, h=50)
    ret, thres = cv2.threshold(denoise, 127, 255,
                               cv2.THRESH_BINARY_INV)
    ori = thres.copy()
    width = thres.shape[1]
    height = thres.shape[0]
    thres[:, border:width - border] = 0

    border_data = np.where(thres == 255)
    Y_label = border_data[0]
    samples = Y_label.shape[0]
    X = border_data[1].reshape(samples, 1)
    regs = LinearRegression()
    feature = PolynomialFeatures(degree=2)
    X_input = feature.fit_transform(X)
    regs.fit(X_input, Y_label)

    newX = np.array([i for i in range(width)])
    newX = newX.reshape(newX.shape[0], 1)
    newX_input = feature.fit_transform(newX)
    newY = regs.predict(newX_input)

    plt.ylim(bottom=0, top=height)
    plt.scatter(X, height - Y_label, color='blue', s=1)
    plt.scatter(newX, height - newY, color='red', s=1)

    img_cuv = np.zeros_like(ori)
    newY = newY.round(0)
    for point in np.column_stack([newY, newX]):
        py = int(point[0])
        px = int(point[1])
        w = 4
        img_cuv[py - w:py + w, px] = 255

    diff = cv2.absdiff(ori, img_cuv)

    denoise = cv2.fastNlMeansDenoising(diff, h=80)
    return denoise


captcha_image_files = list(paths.list_images('train_data_rail'))
for image_file in captcha_image_files:
    img = cv2.imread(image_file)
    file_name = image_file.split('\\')[1]
    result_img = rm_regression(img, border=9)
    cv2.imwrite('train_data_rail_preprocess/' + file_name, result_img)
