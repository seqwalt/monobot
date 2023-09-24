#!/usr/bin/env python3

import numpy as np

def X_func(x, y, yaw, v, yaw_rate, cr1, cr2, cr3, cl1, cl2, cl3, x_tag_1, y_tag_1, x_tag_2, y_tag_2, x_tag_3, y_tag_3, x_tag_4, y_tag_4, x_tag_5, y_tag_5):
    return np.array([x, y, yaw, v, yaw_rate, cr1, cr2, cr3, cl1, cl2, cl3, x_tag_1, y_tag_1, x_tag_2, y_tag_2, x_tag_3, y_tag_3, x_tag_4, y_tag_4, x_tag_5, y_tag_5]).reshape(-1,1)

def A_func(dt, sin_yaw, cos_yaw, v, r, w_r, w_l, bl):
    return np.array([[1, 0, -dt*v*sin_yaw, dt*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, dt*v*cos_yaw, dt*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, r*w_r/2, r*w_r**2/2, r*w_r**3/2, r*w_l/2, r*w_l**2/2, r*w_l**3/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, r*w_r/bl, r*w_r**2/bl, r*w_r**3/bl, -r*w_l/bl, -r*w_l**2/bl, -r*w_l**3/bl, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

W = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# H_i matrix for tag i
def H_i_func(x, y, sin_yaw, cos_yaw, yaw_rate, i, x_tag_i, y_tag_i):
    # tag 0 (origin)
    if (i == 0):
        H_i = np.array([[-cos_yaw, -sin_yaw, x*sin_yaw - y*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [sin_yaw, -cos_yaw, x*cos_yaw + y*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [yaw_rate*sin_yaw, -yaw_rate*cos_yaw, yaw_rate*(x*cos_yaw + y*sin_yaw), -1, x*sin_yaw - y*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [yaw_rate*cos_yaw, yaw_rate*sin_yaw, -yaw_rate*(x*sin_yaw - y*cos_yaw), 0, x*cos_yaw + y*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    # tag 1
    elif (i == 1):
        H_i = np.array([[-cos_yaw, -sin_yaw, -(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0, cos_yaw, sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0],
                        [sin_yaw, -cos_yaw, (x - x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0, -sin_yaw, cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [yaw_rate*sin_yaw, -yaw_rate*cos_yaw, yaw_rate*((x - x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw), -1, -(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw, 0, 0, 0, 0, 0, 0, -yaw_rate*sin_yaw, yaw_rate*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0],
                        [yaw_rate*cos_yaw, yaw_rate*sin_yaw, -yaw_rate*(-(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw), 0, -(-x + x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw, 0, 0, 0, 0, 0, 0, -yaw_rate*cos_yaw, -yaw_rate*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    # tag 2
    elif (i == 2):
        H_i = np.array([[-cos_yaw, -sin_yaw, -(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, cos_yaw, sin_yaw, 0, 0, 0, 0, 0, 0],
                        [sin_yaw, -cos_yaw, (x - x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sin_yaw, cos_yaw, 0, 0, 0, 0, 0, 0],
                        [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [yaw_rate*sin_yaw, -yaw_rate*cos_yaw, yaw_rate*((x - x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw), -1, -(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0, -yaw_rate*sin_yaw, yaw_rate*cos_yaw, 0, 0, 0, 0, 0, 0],
                        [yaw_rate*cos_yaw, yaw_rate*sin_yaw, -yaw_rate*(-(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw), 0, -(-x + x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0, -yaw_rate*cos_yaw, -yaw_rate*sin_yaw, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    # tag 3
    elif (i == 3):
        H_i = np.array([[-cos_yaw, -sin_yaw, -(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, cos_yaw, sin_yaw, 0, 0, 0, 0],
                        [sin_yaw, -cos_yaw, (x - x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sin_yaw, cos_yaw, 0, 0, 0, 0],
                        [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [yaw_rate*sin_yaw, -yaw_rate*cos_yaw, yaw_rate*((x - x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw), -1, -(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -yaw_rate*sin_yaw, yaw_rate*cos_yaw, 0, 0, 0, 0],
                        [yaw_rate*cos_yaw, yaw_rate*sin_yaw, -yaw_rate*(-(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw), 0, -(-x + x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -yaw_rate*cos_yaw, -yaw_rate*sin_yaw, 0, 0, 0, 0],
                        [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    # tag 4
    elif (i == 4):
        H_i = np.array([[-cos_yaw, -sin_yaw, -(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, cos_yaw, sin_yaw, 0, 0],
                        [sin_yaw, -cos_yaw, (x - x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sin_yaw, cos_yaw, 0, 0],
                        [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [yaw_rate*sin_yaw, -yaw_rate*cos_yaw, yaw_rate*((x - x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw), -1, -(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -yaw_rate*sin_yaw, yaw_rate*cos_yaw, 0, 0],
                        [yaw_rate*cos_yaw, yaw_rate*sin_yaw, -yaw_rate*(-(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw), 0, -(-x + x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -yaw_rate*cos_yaw, -yaw_rate*sin_yaw, 0, 0],
                        [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    # tag 5
    elif (i == 5):
        H_i = np.array([[-cos_yaw, -sin_yaw, -(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, cos_yaw, sin_yaw],
                        [sin_yaw, -cos_yaw, (x - x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sin_yaw, cos_yaw],
                        [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [yaw_rate*sin_yaw, -yaw_rate*cos_yaw, yaw_rate*((x - x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw), -1, -(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -yaw_rate*sin_yaw, yaw_rate*cos_yaw],
                        [yaw_rate*cos_yaw, yaw_rate*sin_yaw, -yaw_rate*(-(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw), 0, -(-x + x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -yaw_rate*cos_yaw, -yaw_rate*sin_yaw],
                        [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    else:
        print("Invalid tag number.")
        exit()

    return H_i

# H is a vertical stack of H_i matrices, for each visible tag.
#   For example, is tags 0 and 2 are visible: H = np.vstack((H_0, H_2))
# V is identity matrix of size (6 * num visible tags)
#   For example, is n tags are visible: V = np.eye(6*n)

H_0 = H_i_func(1, 1, np.sin(np.pi/4), np.cos(np.pi/4), 1.0, 0, 0, 0)
H_1 = H_i_func(1, 1, np.sin(np.pi/4), np.cos(np.pi/4), 1.0, 1, 1, 2)
H = np.vstack((H_0, H_1))

print(H_0.shape)
print(H_1.shape)
print(H.shape)
