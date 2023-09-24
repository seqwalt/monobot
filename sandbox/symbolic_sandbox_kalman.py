#!/usr/bin/env python3

import sympy as sp

# Constants
r, bl, dt = sp.symbols('r bl dt') # wheel radius, base length btw wheels, delta time
yaw_i_world = sp.symbols('yaw_i_world') # yaw from tag i to world frame

# State vector (length 21)
x, y, yaw, v, w= sp.symbols('x y yaw v w')    # robot position, yaw, speed, yaw rate
cr1, cr2, cr3 = sp.symbols('cr1 cr2 cr3') # coefficients for right wheel angular velocity: w_r_true = cr1*w_r + cr2*w_r*w_r + cr3*w_r*w_r*w_r
cl1, cl2, cl3 = sp.symbols('cl1 cl2 cl3') # coefficients for left wheel angular velocity
# position of tag 0 (origin) does not need estimation
x_1, y_1 = sp.symbols('x_1 y_1')    # position estimate of tag 1
x_2, y_2 = sp.symbols('x_2 y_2')    # position estimate of tag 2
x_3, y_3 = sp.symbols('x_3 y_3')    # position estimate of tag 3
x_4, y_4 = sp.symbols('x_4 y_4')    # position estimate of tag 4
x_5, y_5 = sp.symbols('x_5 y_5')    # position estimate of tag 5
X = sp.Matrix([x, y, yaw, v, w, cr1, cr2, cr3, cl1, cl2, cl3, x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4, x_5, y_5])
#sp.pprint(X)

# Process noise
nx, ny, nyaw, nv, nw = sp.symbols('nx ny nyaw nv nw')
ncr1, ncr2, ncr3 = sp.symbols('ncr1 ncr2 ncr3')
ncl1, ncl2, ncl3 = sp.symbols('ncl1 ncl2 ncl3')
ProcNoise = sp.Matrix([nx, ny, nyaw, nv, nw, ncr1, ncr2, ncr3, ncl1, ncl2, ncl3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # Note tag positions are not used in the Kalman propogation step
N = sp.Matrix([nx, ny, nyaw, nv, nw, ncr1, ncr2, ncr3, ncl1, ncl2, ncl3])
#sp.pprint(N)

# Measurement noise
mi_1, mi_2, mi_3, mi_4, mi_5, mi_6 = sp.symbols('mi_1 mi_2 mi_3 mi_4 mi_5 mi_6')
M = sp.Matrix([mi_1, mi_2, mi_3, mi_4, mi_5, mi_6])
MeasNoise = M
#sp.pprint(M)

# Input vector (length 2)
w_r, w_l = sp.symbols('w_r w_l') # right and left wheel angular velocities
U = sp.Matrix([w_r, w_l])
#sp.pprint(U)

# Discretized dynamics
w_r_true = cr1*w_r + cr2*w_r*w_r + cr3*w_r*w_r*w_r
w_l_true = cl1*w_l + cl2*w_l*w_l + cl3*w_l*w_l*w_l
F = ProcNoise + sp.Matrix([ x + dt*v*sp.cos(yaw),
                            y + dt*v*sp.sin(yaw),
                            yaw + dt*w,
                            (r/2)*(w_r_true + w_l_true),
                            (r/bl)*(w_r_true - w_l_true),
                            cr1, cr2, cr3,
                            cl1, cl2, cl3,
                            x_1, y_1,
                            x_2, y_2,
                            x_3, y_3,
                            x_4, y_4,
                            x_5, y_5, ])
#sp.pprint(F)

# Measurement dynamics
def h_func(x_i, y_i):
    h_i = MeasNoise + sp.Matrix([ sp.cos(yaw)*(x_i - x) + sp.sin(yaw)*(y_i - y),
                                  sp.cos(yaw)*(y_i - y) - sp.sin(yaw)*(x_i - x),
                                  yaw_i_world - yaw,
                                  w*(sp.cos(yaw)*(y_i - y) - sp.sin(yaw)*(x_i - x)) - v,
                                  -w*(sp.cos(yaw)*(x_i - x) + sp.sin(yaw)*(y_i - y)),
                                  -w ])
    return h_i
h_0 = h_func(0, 0) # h function for tag 0 (origin)
h_1 = h_func(x_1, y_1) # h function for tag 1
h_2 = h_func(x_2, y_2) # h function for tag 2
h_3 = h_func(x_3, y_3) # h function for tag 3
h_4 = h_func(x_4, y_4) # h function for tag 4
h_5 = h_func(x_5, y_5) # h function for tag 5

#sp.pprint(h)

# ------- Linearization ------- #
A = F.jacobian(X)
W = F.jacobian(N)
H_0 = h_0.jacobian(X)
H_1 = h_1.jacobian(X)
H_2 = h_2.jacobian(X)
H_3 = h_3.jacobian(X)
H_4 = h_4.jacobian(X)
H_5 = h_5.jacobian(X)

#V_0 = h_0.jacobian(M) # always identity
# V is identity matrix of size (6 * num visible tags)

#sp.pprint(A)
#sp.pprint(W)
# sp.pprint(H_0)
# sp.pprint(H_1)
# sp.pprint(H_2)
# sp.pprint(H_3)
# sp.pprint(H_4)
# sp.pprint(H_5)

print("H_1")
print(sp.python(H_1))
print("H_2")
print(sp.python(H_2))
print("H_3")
print(sp.python(H_3))
print("H_4")
print(sp.python(H_4))
print("H_5")
print(sp.python(H_5))
