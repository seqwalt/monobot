import time
import sys
import numpy as np
from numpy import sin, cos, sqrt, pi
from adafruit_servokit import ServoKit

# ----- Desired trajectory ----- #
# See "Squircular Calculations" https://arxiv.org/vc/arxiv/papers/1604/1604.02174v1.pdf
a = 4.1148
b = 1.1557
c = 3.5814
d = 1.8034
s = 0.97    # trajectory corner sharpness, orig: 0.97
T = 60      # total trajectory time
def x_d(t):
    return a + c*( (1/(2*s))*sqrt(2+2*s*sqrt(2)*cos((pi + 2*pi*t/T))+s*s*cos(2*(pi + 2*pi*t/T))) - (1/(2*s))*sqrt(2-2*s*sqrt(2)*cos((pi + 2*pi*t/T))+s*s*cos(2*(pi + 2*pi*t/T))) )
def y_d(t):
    return b + d*( (1/(2*s))*sqrt(2+2*s*sqrt(2)*sin((pi + 2*pi*t/T))-s*s*cos(2*(pi + 2*pi*t/T))) - (1/(2*s))*sqrt(2-2*s*sqrt(2)*sin((pi + 2*pi*t/T))-s*s*cos(2*(pi + 2*pi*t/T))) )
def dx_d(t):
    return c*(-(-2*pi*s**2*sin(4*pi*t/T)/T - 2*sqrt(2)*pi*s*sin(2*pi*t/T)/T)/(2*s*sqrt(s**2*cos(4*pi*t/T) + 2*sqrt(2)*s*cos(2*pi*t/T) + 2)) + (-2*pi*s**2*sin(4*pi*t/T)/T + 2*sqrt(2)*pi*s*sin(2*pi*t/T)/T)/(2*s*sqrt(s**2*cos(4*pi*t/T) - 2*sqrt(2)*s*cos(2*pi*t/T) + 2)))
def dy_d(t):
    return d*((2*pi*s**2*sin(4*pi*t/T)/T - 2*sqrt(2)*pi*s*cos(2*pi*t/T)/T)/(2*s*sqrt(-s**2*cos(4*pi*t/T) - 2*sqrt(2)*s*sin(2*pi*t/T) + 2)) - (2*pi*s**2*sin(4*pi*t/T)/T + 2*sqrt(2)*pi*s*cos(2*pi*t/T)/T)/(2*s*sqrt(-s**2*cos(4*pi*t/T) + 2*sqrt(2)*s*sin(2*pi*t/T) + 2)))
def ddx_d(t):
    return c*(-(-2*pi*s**2*sin(4*pi*t/T)/T - 2*sqrt(2)*pi*s*sin(2*pi*t/T)/T)*(2*pi*s**2*sin(4*pi*t/T)/T + 2*sqrt(2)*pi*s*sin(2*pi*t/T)/T)/(2*s*(s**2*cos(4*pi*t/T) + 2*sqrt(2)*s*cos(2*pi*t/T) + 2)**(3/2)) + (-2*pi*s**2*sin(4*pi*t/T)/T + 2*sqrt(2)*pi*s*sin(2*pi*t/T)/T)*(2*pi*s**2*sin(4*pi*t/T)/T - 2*sqrt(2)*pi*s*sin(2*pi*t/T)/T)/(2*s*(s**2*cos(4*pi*t/T) - 2*sqrt(2)*s*cos(2*pi*t/T) + 2)**(3/2)) - (-8*pi**2*s**2*cos(4*pi*t/T)/T**2 - 4*sqrt(2)*pi**2*s*cos(2*pi*t/T)/T**2)/(2*s*sqrt(s**2*cos(4*pi*t/T) + 2*sqrt(2)*s*cos(2*pi*t/T) + 2)) + (-8*pi**2*s**2*cos(4*pi*t/T)/T**2 + 4*sqrt(2)*pi**2*s*cos(2*pi*t/T)/T**2)/(2*s*sqrt(s**2*cos(4*pi*t/T) - 2*sqrt(2)*s*cos(2*pi*t/T) + 2)))
def ddy_d(t):
    return d*(-(-2*pi*s**2*sin(4*pi*t/T)/T - 2*sqrt(2)*pi*s*cos(2*pi*t/T)/T)*(2*pi*s**2*sin(4*pi*t/T)/T + 2*sqrt(2)*pi*s*cos(2*pi*t/T)/T)/(2*s*(-s**2*cos(4*pi*t/T) + 2*sqrt(2)*s*sin(2*pi*t/T) + 2)**(3/2)) + (-2*pi*s**2*sin(4*pi*t/T)/T + 2*sqrt(2)*pi*s*cos(2*pi*t/T)/T)*(2*pi*s**2*sin(4*pi*t/T)/T - 2*sqrt(2)*pi*s*cos(2*pi*t/T)/T)/(2*s*(-s**2*cos(4*pi*t/T) - 2*sqrt(2)*s*sin(2*pi*t/T) + 2)**(3/2)) - (8*pi**2*s**2*cos(4*pi*t/T)/T**2 - 4*sqrt(2)*pi**2*s*sin(2*pi*t/T)/T**2)/(2*s*sqrt(-s**2*cos(4*pi*t/T) + 2*sqrt(2)*s*sin(2*pi*t/T) + 2)) + (8*pi**2*s**2*cos(4*pi*t/T)/T**2 + 4*sqrt(2)*pi**2*s*sin(2*pi*t/T)/T**2)/(2*s*sqrt(-s**2*cos(4*pi*t/T) - 2*sqrt(2)*s*sin(2*pi*t/T) + 2)))

# ----- Desired Control Inputs ----- #
def speed_d(t):
    return sqrt(dx_d(t)**2 + dy_d(t)**2)
def yaw_rate_d(t):
    return (ddy_d(t)*dx_d(t) - ddx_d(t)*dy_d(t))/(dx_d(t)**2 + dy_d(t)**2)

# ----- Dynamic Feedback Linearization ----- #
# See "Control of Wheeled Mobile Robots: An Experimental Overview" Sec. 5. https://web2.qatar.cmu.edu/~gdicaro/16311-Fall17/slides/control-theory-for-robotics.pdf
kpx = 1.0 # control gains
kdx = 1.0
kpy = 1.0
kdy = 1.0
def u1(t, x, dx):
    return ddx_d(t) + kpx*(x_d(t) - x) + kdx*(dx_d(t) - dx)
def u2(t, y, dy):
    return ddy_d(t) + kpy*(y_d(t) - y) + kdy*(dy_d(t) - dy)

# ----- Vehicle Parameters ----- #
whl_rad = 0.066 # meters
base_line = 0.14089 # meters (dist btw wheels)

# ----- Initialization ----- #
kit = ServoKit(channels=16)
start_time = time.process_time()
speed = speed_d(0)
yaw_rate = yaw_rate_d(0)

def main():
    # ----- Control Loop ----- #
    while True:
        # Apply control to system
        left_whl_rate = (2*speed - yaw_rate*base_line)/(2*whl_rad)
        right_whl_rate = (2*speed + yaw_rate*base_line)/(2*whl_rad)
        c_servo = 0.04125 # measured const to convert from rad/s to servo throttle val
        kit.continuous_servo[7].throttle = c_servo*left_whl_rate    # left wheel
        kit.continuous_servo[8].throttle = -c_servo*right_whl_rate  # right wheel (motor flipped so need minus sign)

        # Get state estimate

        # Update control input
        curr_time = time.process_time() - start_time
        speed = speed_d(curr_time)
        yaw_rate = yaw_rate_d(curr_time)

if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt:
        # shut off servos
        kit.continuous_servo[7].throttle = 0
        kit.continuous_servo[8].throttle = 0
        exit()
