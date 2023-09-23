#!/usr/bin/env python3

import sympy as sp

# Constants
r, bl, dt = sp.symbols('r bl dt') # wheel radius, base length btw wheels, delta time
x_0, y_0 = sp.symbols('x_0 y_0')    # position of tag 0 (origin)

# State vector (length 15)
x, y, yaw = sp.symbols('x y yaw')    # robot position and yaw
bw_r, bw_l = sp.symbols('bw_r bw_l') # bias of wheel angular velocities
x_1, y_1 = sp.symbols('x_1 y_1')    # position estimate of tag 1
x_2, y_2 = sp.symbols('x_2 y_2')    # position estimate of tag 2
x_3, y_3 = sp.symbols('x_3 y_3')    # position estimate of tag 3
x_4, y_4 = sp.symbols('x_4 y_4')    # position estimate of tag 4
x_5, y_5 = sp.symbols('x_5 y_5')    # position estimate of tag 5
X = sp.Matrix([x, y, yaw, bw_l, bw_r, x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4, x_5, y_5])
#sp.pprint(X)

# Process noise
nx, ny, nyaw = sp.symbols('nx ny nyaw')
nbw_l, nbw_r = sp.symbols('nbw_l nbw_r')
ProcNoise = sp.Matrix([nx, ny, nyaw, nbw_l, nbw_r, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # Note tag positions are not used in the Kalman propogation step
N = sp.Matrix([nx, ny, nyaw, nbw_l, nbw_r])
#sp.pprint(N)

# Measurement noise
mx_0, my_0 = sp.symbols('mx_0 my_0')
mx_1, my_1 = sp.symbols('mx_1 my_1')
mx_2, my_2 = sp.symbols('mx_2 my_2')
mx_3, my_3 = sp.symbols('mx_3 my_3')
mx_4, my_4 = sp.symbols('mx_4 my_4')
mx_5, my_5 = sp.symbols('mx_5 my_5')
M = sp.Matrix([mx_0, my_0, mx_1, my_1, mx_2, my_2, mx_3, my_3, mx_4, my_4, mx_5, my_5])
MeasNoise = M
#sp.pprint(M)

# Input vector (length 2)
w_r, w_l = sp.symbols('w_r w_l') # right and left wheel angular velocities
U = sp.Matrix([w_r, w_l])
#sp.pprint(U)

# Discretized dynamics
F = ProcNoise + X + dt * sp.Matrix([ (r/2)*(w_r - bw_r + w_l - bw_l)*sp.cos(yaw),
                                     (r/2)*(w_r - bw_r + w_l - bw_l)*sp.sin(yaw),
                                     (r/bl)*(w_r - bw_r - (w_l - bw_l)),
                                     0, 0,
                                     0, 0,
                                     0, 0,
                                     0, 0,
                                     0, 0,
                                     0, 0 ])
#sp.pprint(F)

# Measurement dynamics
h = MeasNoise + sp.Matrix([ sp.cos(yaw)*(x_0 - x) + sp.sin(yaw)*(y_0 - y),
                            sp.cos(yaw)*(y_0 - y) - sp.sin(yaw)*(x_0 - x),
                            sp.cos(yaw)*(x_1 - x) + sp.sin(yaw)*(y_1 - y),
                            sp.cos(yaw)*(y_1 - y) - sp.sin(yaw)*(x_1 - x),
                            sp.cos(yaw)*(x_2 - x) + sp.sin(yaw)*(y_2 - y),
                            sp.cos(yaw)*(y_2 - y) - sp.sin(yaw)*(x_2 - x),
                            sp.cos(yaw)*(x_3 - x) + sp.sin(yaw)*(y_3 - y),
                            sp.cos(yaw)*(y_3 - y) - sp.sin(yaw)*(x_3 - x),
                            sp.cos(yaw)*(x_4 - x) + sp.sin(yaw)*(y_4 - y),
                            sp.cos(yaw)*(y_4 - y) - sp.sin(yaw)*(x_4 - x),
                            sp.cos(yaw)*(x_5 - x) + sp.sin(yaw)*(y_5 - y),
                            sp.cos(yaw)*(y_5 - y) - sp.sin(yaw)*(x_5 - x),])
#sp.pprint(h)

# ------- Linearization ------- #
A = F.jacobian(X)
W = F.jacobian(N)
H = h.jacobian(X)
V = h.jacobian(M)

#sp.pprint(A)
#sp.pprint(W)
#sp.pprint(H)
#sp.pprint(V)
