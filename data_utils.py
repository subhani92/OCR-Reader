import cv2
import numpy as np
from spellchecker import SpellChecker

def smoothening(img):
    ret1, th1 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3

def preprocess(image, filename):
    # resizing image to 1024x1024 dpi as ocr gives better result more than 300dpi
    img = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_CUBIC)
    # converting image to graysacle 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #in order to remove noise we can apply, and smoothening image
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)  # dilation 
    img = cv2.erode(img, kernel, iterations=1)    #erosion

    #img = smoothening(img)
    #let's save this image
    image_id = filename.split("/")[-1].split(".")[0]
    cv2.imwrite(f"logdir/{image_id}.png", img)
    return img




def postprocess(text):

    # removing white-space 
    text = " ".join(text.split())
    spell = SpellChecker()
    s = text
    return ' '.join([spell.correction(word) for word in s.split(' ')])