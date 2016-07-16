import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2 as cv
size = 3

image = cv.imread('/Volumes/DT01ACA200/NNCT/GS_H28/Carbon/Data/Data1_Default_int/9/05/15.bmp',1)
hight = image.shape[0]
width = image.shape[1]
image = cv.resize(image,(hight/size,width/size))
image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

thresh = 50
max_pixel = 255
#ret, image_dst = cv.threshold(image_gray,thresh,max_pixel,cv.THRESH_BINARY)
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
image = cv.adaptiveThreshold(image,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
cv.imshow('image', image)
cv.waitKey(0)
cv.destroyAllWindows()
