#!/usr/bin/env python3

import sympy as sp

# Constants
r, bl, dt = sp.symbols('r bl dt') # wheel radius, base length btw wheels, delta time
x_0, y_0 = sp.symbols('x_0 y_0')    # position of tag 0 (origin)
yaw_i_world = sp.symbols('yaw_i_world') # yaw from tag i to world frame

# State vector (length 17)
x, y, yaw, v, w= sp.symbols('x y yaw v w')    # robot position, yaw, speed, yaw rate
bw_r, bw_l = sp.symbols('bw_r bw_l') # bias of wheel angular velocities
x_1, y_1 = sp.symbols('x_1 y_1')    # position estimate of tag 1
x_2, y_2 = sp.symbols('x_2 y_2')    # position estimate of tag 2
x_3, y_3 = sp.symbols('x_3 y_3')    # position estimate of tag 3
x_4, y_4 = sp.symbols('x_4 y_4')    # position estimate of tag 4
x_5, y_5 = sp.symbols('x_5 y_5')    # position estimate of tag 5
X = sp.Matrix([x, y, yaw, v, w, bw_l, bw_r, x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4, x_5, y_5])
#sp.pprint(X)

# Process noise
nx, ny, nyaw, nv, nw = sp.symbols('nx ny nyaw nv nw')
nbw_l, nbw_r = sp.symbols('nbw_l nbw_r')
ProcNoise = sp.Matrix([nx, ny, nyaw, nv, nw, nbw_l, nbw_r, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # Note tag positions are not used in the Kalman propogation step
N = sp.Matrix([nx, ny, nyaw, nv, nw, nbw_l, nbw_r])
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
F = ProcNoise + sp.Matrix([ x + dt*v*sp.cos(yaw),
                            y + dt*v*sp.sin(yaw),
                            yaw + dt*w,
                            (r/2)*(w_r - bw_r + w_l - bw_l),
                            (r/bl)*(w_r - bw_r - (w_l - bw_l)),
                            bw_l, bw_r,
                            x_1, y_1,
                            x_2, y_2,
                            x_3, y_3,
                            x_4, y_4,
                            x_5, y_5, ])
#sp.pprint(F)

# Measurement dynamics for tag i
x_i, y_i = sp.symbols('x_i y_i')    # position estimate of tag i
h_i = MeasNoise + sp.Matrix([ sp.cos(yaw)*(x_i - x) + sp.sin(yaw)*(y_i - y),
                              sp.cos(yaw)*(y_i - y) - sp.sin(yaw)*(x_i - x),
                              yaw_i_world - yaw,
                              w*(sp.cos(yaw)*(y_i - y) - sp.sin(yaw)*(x_i - x)) - v,
                              -w*(sp.cos(yaw)*(x_i - x) + sp.sin(yaw)*(y_i - y)),
                              -w ])
#sp.pprint(h)

# ------- Linearization ------- #
A = F.jacobian(X)
W = F.jacobian(N)
H_i = h_i.jacobian(X)
V_i = h_i.jacobian(M)

#sp.pprint(A)
#sp.pprint(W)
#sp.pprint(H_i)
sp.pprint(V_i)
