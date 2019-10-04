from keras.models import load_model
from helpers import resize_to_fit
from imutils import paths
import numpy as np
import imutils
import cv2
import pickle
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
CAPTCHA_IMAGE_FOLDER = "test_data_rail"

with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)
model = load_model(MODEL_FILENAME)


def rm_regression(img, border):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    w = gray.shape[1]
    scale = 225 / w
    new_im = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    denoise=cv2.fastNlMeansDenoising(new_im, h=50)    
    ret, thres = cv2.threshold(denoise, 127, 255,  
                                cv2.THRESH_BINARY_INV)
    ori = thres.copy()          
    width = thres.shape[1]     
    height = thres.shape[0]    
    thres[:, border:width-border] = 0      
    
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
        img_cuv[py-w:py+w, px] = 255   
    
    diff = cv2.absdiff(ori, img_cuv)  
    
    denoise = cv2.fastNlMeansDenoising(diff, h = 80)  
    return denoise


captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
last_prob = 0
predict = 0
count = 0
for image_file in captcha_image_files:
    count += 1
    image = cv2.imread(image_file)
    filename = image_file.split('\\')[1]
    image = rm_regression(image, border=9)

    _, contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    letter_image_regions = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
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
        w = image.shape[1] - 30  # 寬
        h = image.shape[0]
        x = 25
        y = 0
        w = int(w / 4)
        for i in range(4):
            letter_image_regions.append((x, y, w, h))
            x = x + w

    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
    predictions = []
    for letter_bounding_box in letter_image_regions:
        x, y, w, h = letter_bounding_box
        letter_image = image[y:y + h + 2, x:x + w + 2]

        letter_image = resize_to_fit(letter_image, 20, 20)
        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)
    
        prediction = model.predict(letter_image)

        letter = lb.inverse_transform(prediction)[0]
        predictions.append(letter)

    print('right:', filename)
    captcha_text = "".join(predictions)
    print("CAPTCHA text is: {}".format(captcha_text))

    predict = 0
    for p in range(4):
        if filename[p] == captcha_text[p]:
            predict += 1
    predict /= 4
    print(predict)
    last_prob += predict
print(last_prob/count)


