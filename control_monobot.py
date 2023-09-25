import time
import sys
import numpy as np
from numpy import sin, cos, sqrt, pi
from adafruit_servokit import ServoKit

kit = ServoKit(channels=16)

def main():
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
    start_t = time.time()
    speed = speed_d(0)
    yaw_rate = yaw_rate_d(0)
    rate2throttle = np.load('rate2throttle.npy', allow_pickle=True) # load wheel rate calibration
    r2t = rate2throttle.item() # scipy Akima1DInterpolator (see sanbox/calib_wheel_spd.py)
    curr_t = 0; prev_t = 0; temp_t = 0
    accel = 0
    print_hz = 10 # print frequency

    # ----- Control Loop ----- #
    while True:
        # Apply control to system
        left_rate = (2*speed - yaw_rate*base_line)/(2*whl_rad)  # left wheel rate
        right_rate = (2*speed + yaw_rate*base_line)/(2*whl_rad) # right wheel rate
        left_throttle = np.clip(r2t(left_rate), 0, 1)
        right_throttle = -np.clip(r2t(right_rate), 0, 1) # (-) due to flipped motor
        kit.continuous_servo[7].throttle = left_throttle  # left wheel
        kit.continuous_servo[8].throttle = right_throttle # right wheel

        # Update times
        curr_t = time.time() - start_t
        dt = curr_t - prev_t
        prev_t = curr_t

        # Get state estimate
        yaw = np.arctan2(dy_d(curr_t), dx_d(curr_t))  # placeholder

        # Update control input (dynamic feedback linearization)
        u1_ = u1(curr_t, x_d(curr_t), dx_d(curr_t)) # placeholder
        u2_ = u2(curr_t, y_d(curr_t), dy_d(curr_t)) # placeholder
        speed = speed + dt*accel
        accel = u1_*cos(yaw) + u2_*sin(yaw)
        yaw_rate = (u2_*cos(yaw) - u1_*sin(yaw))/speed

        speed = speed_d(curr_t)
        yaw_rate = yaw_rate_d(curr_t)

        # Printing
        if (curr_t - temp_t > 1.0/print_hz):
            temp_t = curr_t
            print('Left :  ' + str(left_throttle))
            print('Right: ' + str(right_throttle))
            print()

if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt:
        # shut off servos
        kit.continuous_servo[7].throttle = 0
        kit.continuous_servo[8].throttle = 0
        exit()
