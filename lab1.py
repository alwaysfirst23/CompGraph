import numpy as np
from PIL import Image, ImageOps
from math import *

def draw_line_easiest(img_mat, x0, y0, x1, y1, color, count):
    step = 1.0/count

    for t in np.arange (0, 1, step):
        x = round ((1.0 - t)*x0 + t*x1)
        y = round ((1.0 - t)*y0 + t*y1)
        img_mat[y, x] = color

def x_loop_line(image, x0, y0, x1, y1, color):

    for x in range (round(x0), round(x1)):
        t = (x-x0)/(x1 - x0)
        y = round ((1.0 - t)*y0 + t*y1)
        image[y, x] = color

def x_loop_line_fix1(image, x0, y0, x1, y1, color):

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    for x in range (round(x0), round(x1)):
        t = (x-x0)/(x1 - x0)
        y = round ((1.0 - t)*y0 + t*y1)
        image[y, x] = color

def x_loop_line_fix2(image, x0, y0, x1, y1, color):

    xchange = False

    if (abs(x0 -x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    for x in range (round(x0), round(x1)):
        t = (x-x0)/(x1 - x0)
        y = round ((1.0 - t)*y0 + t*y1)
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color

def draw_without_y(image, x0, y0, x1, y1, color):

    xchange = False

    if (abs(x0 -x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = abs(y1 - y0)/(x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1


    for x in range (round(x0), round(x1)):
        if (xchange):
            image[round(x), round(y)] = color
        else:
            image[round(y), round(x)] = color
        derror += dy
        if (derror > 2.0*(x1-x0)*0.5):
            derror -= 2.0*(x1-x0)*1.0
            y += y_update

def draw_perfect_line(image, x0, y0, x1, y1, color):

    xchange = False

    if (abs(x0 -x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = 2*abs(y1 - y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1



    for x in range (round(x0), round(x1)):
        if (xchange):
            image[round(x), round(y)] = color
        else:
            image[round(y), round(x)] = color
        derror += dy
        if (derror > (x1 - x0)):
            derror -= 2*(x1 - x0)
            y += y_update


img_mat = np.zeros((4000, 4000, 3), dtype=np.uint8)
def print_star(img_mat):
    for i in range(13): # Рисуем звёздочку
        x0 = 100
        y0 = 100
        x1 = 100+95*cos(i*2*pi/13)
        y1 = 100+95*sin(i*2*pi/13)
        draw_perfect_line(img_mat, x0, y0, x1, y1, 255)

coordinates = []

f = open("model_1.obj")
for line in f:
    s = line.split()
    if (s[0]=='v'):
        x = round(20000*float(s[2]))+1600
        y = round(20000*float(s[1]))-2000
        img_mat[x, y] = 255
        coordinates.append((y,x))

f = open("model_1.obj")
for line in f:
    s = line.split()
    if (s[0]=='f'):
        point1 = int(s[1].split('/')[0])
        point2 = int(s[2].split('/')[0])
        point3 = int(s[3].split('/')[0])
        
        draw_perfect_line(img_mat, coordinates[point1-1][0], \
                        coordinates[point1-1][1], coordinates[point2-1][0], coordinates[point2-1][1], 255)
        draw_perfect_line(img_mat, coordinates[point2-1][0], \
                        coordinates[point2-1][1], coordinates[point3-1][0], coordinates[point3-1][1], 255)
        draw_perfect_line(img_mat, coordinates[point1-1][0], \
                        coordinates[point1-1][1], coordinates[point3-1][0], coordinates[point3-1][1], 255)
    
img = Image.fromarray(img_mat, mode = 'RGB')
img = ImageOps.flip(img)

img.save('image1.png')
