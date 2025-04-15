import numpy as np
from PIL import Image, ImageOps
from math import *
import random as random

coordinates = []
alpha = np.radians(180)
beta = np.radians(0)
gamma = np.radians(0)
texture_image = Image.open("bunny-atlas.jpg")

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

R = np.dot(R1, np.dot(R2, R3))
t = np.array([-0.03,0,0.15])

texture_coords = []
texture_indices = []

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
    labmda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - labmda0 - lambda1
    return labmda0, lambda1, lambda2


def draw(x0, y0, z0, x1, y1, z1, x2, y2, z2, img_mat, z_buf, cos1, cos2, cos3, vt0, vt1, vt2):
    x0 = 5000 * x0 / z0 + 2500
    y0 = 5000 * y0 / z0 + 2500
    x1 = 5000 * x1 / z1 + 2500
    y1 = 5000 * y1 / z1 + 2500
    x2 = 5000 * x2 / z2 + 2500
    y2 = 5000 * y2 / z2 + 2500

    xmin = max(int(np.floor(min(x0, x1, x2))), 0)
    xmax = min(5000, int(np.ceil(max(x0, x1, x2))))
    ymin = max(0, int(np.floor(min(y0, y1, y2))))
    ymax = min(5000, int(np.ceil(max(y0, y1, y2))))

    u0, v0 = texture_coords[vt0 - 1]
    u1, v1 = texture_coords[vt1 - 1]
    u2, v2 = texture_coords[vt2 - 1]
    w_t, h_t = texture_image.size

    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            lambda0, lambda1, lambda2 = bary(x0, y0, x1, y1, x2, y2, x, y)
            if (lambda0 > 0) and (lambda1 > 0) and (lambda2 > 0):
                z_coord = lambda0 * z0 + lambda1 * z1 + lambda2 * z2
                if (z_coord < z_buf[x, y]):
                    z_buf[x, y] = z_coord
                    I = -(lambda0 * cos1 + lambda1 * cos2 + lambda2 * cos3)
                    if (I < 0): continue

                    u = lambda0 * u0 + lambda1 * u1 + lambda2 * u2
                    v = lambda0 * v0 + lambda1 * v1 + lambda2 * v2

                    tex_x = int(u * w_t)
                    tex_y = int((1 - v) * h_t)

                    texture_color = texture_image.getpixel((tex_x, tex_y))

                    img_mat[y, x] = [
                        int(texture_color[0] * I),
                        int(texture_color[1] * I),
                        int(texture_color[2] * I)
                    ]


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
    elif (s[0] == 'vt'):
        u = float(s[1])
        v = float(s[2])
        texture_coords.append((u, v))
    elif (s[0] == 'f'):
        parts1 = s[1].split('/')
        parts2 = s[2].split('/')
        parts3 = s[3].split('/')

        if len(parts1) > 1 and parts1[1]:
            vt1 = int(parts1[1])
            vt2 = int(parts2[1])
            vt3 = int(parts3[1])
            texture_indices.append((vt1, vt2, vt3))

vr_calc = np.zeros((len(coordinates), 3), dtype=float)
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
        vr_calc[point1 - 1] += norma
        vr_calc[point2 - 1] += norma
        vr_calc[point3 - 1] += norma

f.close()

magnitudes = np.linalg.norm(vr_calc, axis=1)
non_zero_mask = magnitudes != 0
vr_calc[non_zero_mask] = vr_calc[non_zero_mask] / magnitudes[non_zero_mask, np.newaxis]


vector = np.array([0, 0, 1])


f = open("model_1.obj")
for line in f:
    s = line.split()
    if (s[0] == 'f'):
        point1 = int(s[1].split('/')[0])
        point2 = int(s[2].split('/')[0])
        point3 = int(s[3].split('/')[0])
        point1_vt = int(s[1].split('/')[1])
        point2_vt = int(s[2].split('/')[1])
        point3_vt = int(s[3].split('/')[1])
        cos_angle1 = cosfall(vr_calc[point1-1], vector)
        cos_angle2 = cosfall(vr_calc[point2-1], vector)
        cos_angle3 = cosfall(vr_calc[point3-1], vector)

        draw(coordinates[point1 - 1][0], coordinates[point1 - 1][1], coordinates[point1 - 1][2],
            coordinates[point2 - 1][0], coordinates[point2 - 1][1], coordinates[point2 - 1][2],
            coordinates[point3 - 1][0], coordinates[point3 - 1][1], coordinates[point3 - 1][2], img_mat,
            z_buf, cos_angle1, cos_angle2, cos_angle3, point1_vt, point2_vt, point3_vt)
f.close()
img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)

img.save('image1.png')