#!/usr/bin/env python3

import sympy as sp
from sympy import sin, cos, sqrt, pi
from matplotlib import pyplot as plt
import numpy as np

# a, b, c, d, s, T, t = sp.symbols('a b c d s T t') # use for symbolic differentiation
t = sp.symbols('t')
a = 4.1148
b = 1.1557
c = 3.5814
d = 1.8034
s = 0.97    # trajectory corner sharpness, orig: 0.97
T = 60      # total trajectory time

# ----- Parametric equation for desired trajectory ----- #
# See "Squircular Calculations" https://arxiv.org/vc/arxiv/papers/1604/1604.02174v1.pdf
x_d = a + c*( (1/(2*s))*sqrt(2+2*s*sqrt(2)*cos((pi + 2*pi*t/T))+s*s*cos(2*(pi + 2*pi*t/T))) - (1/(2*s))*sqrt(2-2*s*sqrt(2)*cos((pi + 2*pi*t/T))+s*s*cos(2*(pi + 2*pi*t/T))) )
y_d = b + d*( (1/(2*s))*sqrt(2+2*s*sqrt(2)*sin((pi + 2*pi*t/T))-s*s*cos(2*(pi + 2*pi*t/T))) - (1/(2*s))*sqrt(2-2*s*sqrt(2)*sin((pi + 2*pi*t/T))-s*s*cos(2*(pi + 2*pi*t/T))) )

# ----- Symbolic differentiation ----- #
# dx_d = sp.diff(x_d, t)
# ddx_d = sp.diff(dx_d, t)
# print(dx_d)
# print()
# print(ddx_d)
# print()
#
# dy_d = sp.diff(y_d, t)
# ddy_d = sp.diff(dy_d, t)
# print(dy_d)
# print()
# print(ddy_d)
# print()

# ----- Derivative expressions ----- #
dx_d = c*(-(-2*pi*s**2*sin(4*pi*t/T)/T - 2*sqrt(2)*pi*s*sin(2*pi*t/T)/T)/(2*s*sqrt(s**2*cos(4*pi*t/T) + 2*sqrt(2)*s*cos(2*pi*t/T) + 2)) + (-2*pi*s**2*sin(4*pi*t/T)/T + 2*sqrt(2)*pi*s*sin(2*pi*t/T)/T)/(2*s*sqrt(s**2*cos(4*pi*t/T) - 2*sqrt(2)*s*cos(2*pi*t/T) + 2)))
ddx_d = c*(-(-2*pi*s**2*sin(4*pi*t/T)/T - 2*sqrt(2)*pi*s*sin(2*pi*t/T)/T)*(2*pi*s**2*sin(4*pi*t/T)/T + 2*sqrt(2)*pi*s*sin(2*pi*t/T)/T)/(2*s*(s**2*cos(4*pi*t/T) + 2*sqrt(2)*s*cos(2*pi*t/T) + 2)**(3/2)) + (-2*pi*s**2*sin(4*pi*t/T)/T + 2*sqrt(2)*pi*s*sin(2*pi*t/T)/T)*(2*pi*s**2*sin(4*pi*t/T)/T - 2*sqrt(2)*pi*s*sin(2*pi*t/T)/T)/(2*s*(s**2*cos(4*pi*t/T) - 2*sqrt(2)*s*cos(2*pi*t/T) + 2)**(3/2)) - (-8*pi**2*s**2*cos(4*pi*t/T)/T**2 - 4*sqrt(2)*pi**2*s*cos(2*pi*t/T)/T**2)/(2*s*sqrt(s**2*cos(4*pi*t/T) + 2*sqrt(2)*s*cos(2*pi*t/T) + 2)) + (-8*pi**2*s**2*cos(4*pi*t/T)/T**2 + 4*sqrt(2)*pi**2*s*cos(2*pi*t/T)/T**2)/(2*s*sqrt(s**2*cos(4*pi*t/T) - 2*sqrt(2)*s*cos(2*pi*t/T) + 2)))

dy_d = d*((2*pi*s**2*sin(4*pi*t/T)/T - 2*sqrt(2)*pi*s*cos(2*pi*t/T)/T)/(2*s*sqrt(-s**2*cos(4*pi*t/T) - 2*sqrt(2)*s*sin(2*pi*t/T) + 2)) - (2*pi*s**2*sin(4*pi*t/T)/T + 2*sqrt(2)*pi*s*cos(2*pi*t/T)/T)/(2*s*sqrt(-s**2*cos(4*pi*t/T) + 2*sqrt(2)*s*sin(2*pi*t/T) + 2)))
ddy_d = d*(-(-2*pi*s**2*sin(4*pi*t/T)/T - 2*sqrt(2)*pi*s*cos(2*pi*t/T)/T)*(2*pi*s**2*sin(4*pi*t/T)/T + 2*sqrt(2)*pi*s*cos(2*pi*t/T)/T)/(2*s*(-s**2*cos(4*pi*t/T) + 2*sqrt(2)*s*sin(2*pi*t/T) + 2)**(3/2)) + (-2*pi*s**2*sin(4*pi*t/T)/T + 2*sqrt(2)*pi*s*cos(2*pi*t/T)/T)*(2*pi*s**2*sin(4*pi*t/T)/T - 2*sqrt(2)*pi*s*cos(2*pi*t/T)/T)/(2*s*(-s**2*cos(4*pi*t/T) - 2*sqrt(2)*s*sin(2*pi*t/T) + 2)**(3/2)) - (8*pi**2*s**2*cos(4*pi*t/T)/T**2 - 4*sqrt(2)*pi**2*s*sin(2*pi*t/T)/T**2)/(2*s*sqrt(-s**2*cos(4*pi*t/T) + 2*sqrt(2)*s*sin(2*pi*t/T) + 2)) + (8*pi**2*s**2*cos(4*pi*t/T)/T**2 + 4*sqrt(2)*pi**2*s*sin(2*pi*t/T)/T**2)/(2*s*sqrt(-s**2*cos(4*pi*t/T) - 2*sqrt(2)*s*sin(2*pi*t/T) + 2)))

# ----- Desired control inputs ----- #
# See "Control of Wheeled Mobile Robots: An Experimental Overview" Sec. 5. https://web2.qatar.cmu.edu/~gdicaro/16311-Fall17/slides/control-theory-for-robotics.pdf
v_d = sqrt(dx_d**2 + dy_d**2)
w_d = (ddy_d*dx_d - ddx_d*dy_d)/(dx_d**2 + dy_d**2)

# ----- Yaw during trajectory ----- #
yaw = sp.atan2(dy_d, dx_d)

# ----- Plotting ----- #
x_func = sp.lambdify(t, x_d, 'numpy')
y_func = sp.lambdify(t, y_d, 'numpy')
dx_func = sp.lambdify(t, dx_d, 'numpy')
dy_func = sp.lambdify(t, dy_d, 'numpy')
ddx_func = sp.lambdify(t, ddx_d, 'numpy')
ddy_func = sp.lambdify(t, ddy_d, 'numpy')
v_func = sp.lambdify(t, v_d, 'numpy')
w_func = sp.lambdify(t, w_d, 'numpy')
yaw_func = sp.lambdify(t, yaw, 'numpy')
times = np.linspace(0,0.99*T,1000)

fig, ax1 = plt.subplots()
ax1.plot(x_func(times), y_func(times))
ax1.set_title('Trajectory')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.set_aspect('equal')

fig, (ax2, ax3) = plt.subplots(2,2,figsize=(14,8))
ax2[0].plot(times, x_func(times), color = 'black', label='x')
ax2[0].plot(times, dx_func(times), color = 'green', label='$\dot x$')
ax2[0].plot(times, ddx_func(times), color = 'red', label='$\ddot x$')
ax2[0].set_title('x derivatives')
ax2[0].legend()
ax2[1].plot(times, y_func(times), color = 'black', label='y')
ax2[1].plot(times, dy_func(times), color = 'green', label='$\dot y$')
ax2[1].plot(times, ddy_func(times), color = 'red', label='$\ddot y$')
ax2[1].set_title('y derivatives')
ax2[1].legend()

ax3[0].plot(times, v_func(times), label = 'speed (m/s)')
ax3[0].plot(times, w_func(times), label = 'yaw rate (rad/s)')
ax3[0].set_xlabel('time (s)')
ax3[0].set_title('Desired Control Inputs')
ax3[0].legend()

ax3[1].plot(times, yaw_func(times))
ax3[1].set_xlabel('time (s)')
ax3[1].set_ylabel('yaw (rad)')
ax3[1].set_title('Yaw Angle')

plt.show()
