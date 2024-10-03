import cv2
import numpy as np

img = cv2.imread('samples/test2.jpg')
temp = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 115, 255, cv2.THRESH_BINARY_INV)
contours, heirarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

x = y = 50000
w = h = 0

for contour in contours:
    x_t,y_t,w_t,h_t = cv2.boundingRect(contour)
    x = x_t if x_t < x else x
    w = w_t if w_t > w else w
    y = y_t if y_t < y else y
    h = h_t if h_t > h else h

cv2.rectangle(temp, (x,y), (x+w, y+h), (0,255,0), 5)

cv2.imshow('thresh', thresh)
cv2.imshow('temp', temp)
cv2.waitKey(0)

cv2.destroyAllWindows()


