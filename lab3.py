import numpy as np
from PIL import Image, ImageOps
from math import *
import random as random

coordinates = []

alpha = np.radians(270)
beta = np.radians(0)
gamma = np.radians(0)

R1 = np.array([
    [1, 0, 0],
    [0, np.cos(alpha), np.sin(alpha)],
    [0, -np.sin(alpha), np.cos(alpha)]
])

R2 = np.array([
    [np.cos(beta), 0, np.sin(beta)],
    [0, 1, 0],
    [-np.sin(beta), 0, np.cos(beta)]
])

R3 = np.array([
    [np.cos(gamma), np.sin(gamma), 0],
    [-np.sin(gamma), np.cos(gamma), 0],
    [0, 0, 1]
])

# Перемножение матриц
R = np.dot(R1, np.dot(R2, R3))
t = np.array([-0.03,0,0.15])

def draw_line1(img_mat, x0, y0, x1, y1, color):
    count = sqrt((x0-x1)**2 + (y0-y1)**2)
    step = 1.0 / count
    for t in np.arange(0, 1, step):
        x = round((1.0-t)*x0 + t*x1)
        y = round((1.0-t)*y0 + t*y1)
        img_mat[y, x] = color
def draw_line2(img_mat, x0, y0, x1, y1, color):

    for x in range (round(x0), round(x1)):
        t = (x-x0)/(x1 - x0)
        y = round ((1.0 - t)*y0 + t*y1)
        img_mat[y, x] = color
def draw_line3(img_mat, x0, y0, x1, y1, color):
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    for x in range (round(x0), round(x1)):
        t = (x-x0)/(x1 - x0)
        y = round ((1.0 - t)*y0 + t*y1)
        img_mat[y, x] = color

def draw_line4(img_mat, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
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
            img_mat[x, y] = color
        else:
            img_mat[y, x] = color

def draw_line5(img_mat, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = abs(y1 - y0) / (x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1


    for x in range (round(x0), round(x1)):
        if (xchange):
            img_mat[round(x), round(y)] = color
        else:
            img_mat[round(y), round(x)] = color
        derror += dy
        if (derror > 0.5):
            derror -= 1.0
            y += y_update

def draw_line6(img_mat, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = 2.0*(x1-x0)*abs(y1 - y0) / (x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1

    for x in range(round(x0), round(x1)):
        if (xchange):
            img_mat[round(x), round(y)] = color
        else:
            img_mat[round(y), round(x)] = color
        derror += dy
        if (derror > 2.0*(x1-x0)*0.5):
            derror -= 2.0*(x1-x0)*1.0
            y += y_update

def draw_line7(img_mat, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = 2*abs(y1-y0)
    derror = 0
    y_update = 1 if y1 > y0 else -1

    for x in range(round(x0), round(x1)):
        if (xchange):
            img_mat[round(x), round(y)] = color
        else:
            img_mat[round(y), round(x)] = color
        derror += dy
        if (derror > (x1 - x0)):
            derror -= 2*(x1 - x0)
            y += y_update

def bary(x0, y0, x1, y1, x2, y2, x, y):
    labmda0 = ((x-x2) * (y1-y2) - (x1-x2) * (y-y2)) / ((x0 - x2) * (y1 - y2) - (x1-x2) * (y0 - y2))
    lambda1 =  ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - labmda0 - lambda1
    return labmda0, lambda1, lambda2



def draw(x0, y0, z0, x1, y1, z1, x2, y2, z2, img_mat, color, z_buf):
    x0 = 5000 * x0/z0 + 2500
    y0 = 5000 * y0/z0 + 2500
    x1 = 5000 * x1/z1 + 2500
    y1 = 5000 * y1/z1 + 2500
    x2 = 5000 * x2/z2 + 2500
    y2 = 5000 * y2/z2 + 2500

    xmin = max(int(np.floor(min(x0, x1, x2))), 0)
    xmax = min(5000, int(np.ceil(max(x0, x1, x2))))
    ymin = max(0, int(np.floor(min(y0, y1, y2))))
    ymax = min(5000, int(np.ceil(max(y0, y1, y2))))
    firstcol = color[0]
    secondcol = color[1]
    thirdcol = color[2]
    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            lambda0, lambda1, lambda2 = bary(x0, y0, x1, y1, x2, y2, x, y)
            if (lambda0 > 0) and (lambda1 > 0) and (lambda2 > 0):
                z_coord = lambda0 * z0 + lambda1 * z1 + lambda2 * z2
                if (z_coord < z_buf[x, y]):
                    z_buf[x, y] = z_coord
                    img_mat[y, x] = [firstcol, secondcol, thirdcol]


def normal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    first = [x1 - x2, y1 - y2, z1 - z2]
    second = [x1 - x0, y1 - y0, z1 - z0]
    return np.cross(first, second)


def cosfall(norma, vector):
    return np.dot(norma, vector) / (np.linalg.norm(norma) * np.linalg.norm(vector))


img_mat = np.zeros((5000, 5000, 3), dtype=np.uint8)
z_buf = np.full((5000, 5000), np.inf, dtype=np.float32)

f = open("model_1.obj")
for line in f:
    s = line.split()
    if (s[0] == 'v'):
        x = float(s[2])
        y = float(s[1])
        z = float(s[3])

        [new_x, new_y, new_z] = np.dot(R, (x, y, z)) + t
        coordinates.append((new_y, new_x, new_z))
f.close()
f = open("model_1.obj")
for line in f:
    s = line.split()
    if (s[0] == 'f'):
        point1 = int(s[1].split('/')[0])
        point2 = int(s[2].split('/')[0])
        point3 = int(s[3].split('/')[0])
        norma = normal(coordinates[point1 - 1][0], coordinates[point1 - 1][1], coordinates[point1 - 1][2],
                       coordinates[point2 - 1][0], coordinates[point2 - 1][1], coordinates[point2 - 1][2],
                       coordinates[point3 - 1][0], coordinates[point3 - 1][1], coordinates[point3 - 1][2])
        vector = [0, 0, 1]
        cos_angle = cosfall(norma, vector)
        color = [-255 * cos_angle, 0, 0]
        if (cos_angle < 0):
            draw(coordinates[point1 - 1][0], coordinates[point1 - 1][1], coordinates[point1 - 1][2],
                 coordinates[point2 - 1][0], coordinates[point2 - 1][1], coordinates[point2 - 1][2],
                 coordinates[point3 - 1][0], coordinates[point3 - 1][1], coordinates[point3 - 1][2], img_mat, color,
                 z_buf)

f.close()

img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)

img.save('image1.png')
