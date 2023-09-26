import time
import sys
import numpy as np
import cv2
from numpy import sin, cos, sqrt, pi
from adafruit_servokit import ServoKit
from kalman_filter import ExtendedKalmanFilter
from fiducial_detect import TagDetect
from flask import Flask, render_template, Response
import threading

kit = ServoKit(channels=16)
camera = cv2.VideoCapture('/dev/video0')
if not camera.isOpened():
    raise RuntimeError('Could not start camera.')
Traj = np.nan*np.ones((1,21))

class Camera:
    def __init__(self):
        self.img = np.zeros((480, 640), dtype=np.uint8)
    def set_img(self, img):
        self.img = img
    def gen(self):
        yield b'--frame\r\n'
        while True:
            frame = cv2.imencode('.jpg', self.img)[1].tobytes()
            yield b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n--frame\r\n'
stream = Camera()

app = Flask(__name__)
@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(stream.gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
stream_thrd = threading.Thread(target=app.run, name="Flask video stream", kwargs={'host': '0.0.0.0', 'threaded': True})
stream_thrd.daemon = True
stream_thrd.start()

try:
    # ----- Desired trajectory ----- #
    # See "Squircular Calculations" https://arxiv.org/vc/arxiv/papers/1604/1604.02174v1.pdf
    a = 4.1148
    b = 1.1557
    c = 3.5814
    d = 1.8034
    s = 0.97    # trajectory corner sharpness, orig: 0.97
    T = 60      # total trajectory time
    def x_d(t):
        x_d = a + c*( (1/(2*s))*sqrt(2+2*s*sqrt(2)*cos((pi + 2*pi*t/T))+s*s*cos(2*(pi + 2*pi*t/T))) - (1/(2*s))*sqrt(2-2*s*sqrt(2)*cos((pi + 2*pi*t/T))+s*s*cos(2*(pi + 2*pi*t/T))) )
        return float(x_d)
    def y_d(t):
        y_d = b + d*( (1/(2*s))*sqrt(2+2*s*sqrt(2)*sin((pi + 2*pi*t/T))-s*s*cos(2*(pi + 2*pi*t/T))) - (1/(2*s))*sqrt(2-2*s*sqrt(2)*sin((pi + 2*pi*t/T))-s*s*cos(2*(pi + 2*pi*t/T))) )
        return float(y_d)
    def dx_d(t):
        dx_d = c*(-(-2*pi*s**2*sin(4*pi*t/T)/T - 2*sqrt(2)*pi*s*sin(2*pi*t/T)/T)/(2*s*sqrt(s**2*cos(4*pi*t/T) + 2*sqrt(2)*s*cos(2*pi*t/T) + 2)) + (-2*pi*s**2*sin(4*pi*t/T)/T + 2*sqrt(2)*pi*s*sin(2*pi*t/T)/T)/(2*s*sqrt(s**2*cos(4*pi*t/T) - 2*sqrt(2)*s*cos(2*pi*t/T) + 2)))
        return float(dx_d)
    def dy_d(t):
        dy_d = d*((2*pi*s**2*sin(4*pi*t/T)/T - 2*sqrt(2)*pi*s*cos(2*pi*t/T)/T)/(2*s*sqrt(-s**2*cos(4*pi*t/T) - 2*sqrt(2)*s*sin(2*pi*t/T) + 2)) - (2*pi*s**2*sin(4*pi*t/T)/T + 2*sqrt(2)*pi*s*cos(2*pi*t/T)/T)/(2*s*sqrt(-s**2*cos(4*pi*t/T) + 2*sqrt(2)*s*sin(2*pi*t/T) + 2)))
        return float(dy_d)
    def ddx_d(t):
        ddx_d = c*(-(-2*pi*s**2*sin(4*pi*t/T)/T - 2*sqrt(2)*pi*s*sin(2*pi*t/T)/T)*(2*pi*s**2*sin(4*pi*t/T)/T + 2*sqrt(2)*pi*s*sin(2*pi*t/T)/T)/(2*s*(s**2*cos(4*pi*t/T) + 2*sqrt(2)*s*cos(2*pi*t/T) + 2)**(3/2)) + (-2*pi*s**2*sin(4*pi*t/T)/T + 2*sqrt(2)*pi*s*sin(2*pi*t/T)/T)*(2*pi*s**2*sin(4*pi*t/T)/T - 2*sqrt(2)*pi*s*sin(2*pi*t/T)/T)/(2*s*(s**2*cos(4*pi*t/T) - 2*sqrt(2)*s*cos(2*pi*t/T) + 2)**(3/2)) - (-8*pi**2*s**2*cos(4*pi*t/T)/T**2 - 4*sqrt(2)*pi**2*s*cos(2*pi*t/T)/T**2)/(2*s*sqrt(s**2*cos(4*pi*t/T) + 2*sqrt(2)*s*cos(2*pi*t/T) + 2)) + (-8*pi**2*s**2*cos(4*pi*t/T)/T**2 + 4*sqrt(2)*pi**2*s*cos(2*pi*t/T)/T**2)/(2*s*sqrt(s**2*cos(4*pi*t/T) - 2*sqrt(2)*s*cos(2*pi*t/T) + 2)))
        return float(ddx_d)
    def ddy_d(t):
        ddy_d = d*(-(-2*pi*s**2*sin(4*pi*t/T)/T - 2*sqrt(2)*pi*s*cos(2*pi*t/T)/T)*(2*pi*s**2*sin(4*pi*t/T)/T + 2*sqrt(2)*pi*s*cos(2*pi*t/T)/T)/(2*s*(-s**2*cos(4*pi*t/T) + 2*sqrt(2)*s*sin(2*pi*t/T) + 2)**(3/2)) + (-2*pi*s**2*sin(4*pi*t/T)/T + 2*sqrt(2)*pi*s*cos(2*pi*t/T)/T)*(2*pi*s**2*sin(4*pi*t/T)/T - 2*sqrt(2)*pi*s*cos(2*pi*t/T)/T)/(2*s*(-s**2*cos(4*pi*t/T) - 2*sqrt(2)*s*sin(2*pi*t/T) + 2)**(3/2)) - (8*pi**2*s**2*cos(4*pi*t/T)/T**2 - 4*sqrt(2)*pi**2*s*sin(2*pi*t/T)/T**2)/(2*s*sqrt(-s**2*cos(4*pi*t/T) + 2*sqrt(2)*s*sin(2*pi*t/T) + 2)) + (8*pi**2*s**2*cos(4*pi*t/T)/T**2 + 4*sqrt(2)*pi**2*s*sin(2*pi*t/T)/T**2)/(2*s*sqrt(-s**2*cos(4*pi*t/T) - 2*sqrt(2)*s*sin(2*pi*t/T) + 2)))
        return float(ddy_d)

    # ----- Desired Control Inputs ----- #
    def speed_d(t):
        return sqrt(dx_d(t)**2 + dy_d(t)**2)
    def yaw_rate_d(t):
        return (ddy_d(t)*dx_d(t) - ddx_d(t)*dy_d(t))/(dx_d(t)**2 + dy_d(t)**2)

    # ----- Dynamic Feedback Linearization ----- #
    # See "Control of Wheeled Mobile Robots: An Experimental Overview" Sec. 5. https://web2.qatar.cmu.edu/~gdicaro/16311-Fall17/slides/control-theory-for-robotics.pdf
    kpx = 10.0 # control gains
    kdx = 1.0
    kpy = 10.0
    kdy = 1.0
    def u1(t, x, dx):
        return ddx_d(t) + kpx*(x_d(t) - x) + kdx*(dx_d(t) - dx)
    def u2(t, y, dy):
        return ddy_d(t) + kpy*(y_d(t) - y) + kdy*(dy_d(t) - dy)

    # ----- Vehicle Parameters ----- #
    whl_rad = 0.066 # meters
    base_line = 0.14089 # meters (dist btw wheels)

    # ----- Initialization ----- #

    # Init control and throttle mapping
    speed = speed_d(0)
    yaw_rate = yaw_rate_d(0)
    rate2throttle = np.load('rate2throttle.npy', allow_pickle=True) # load wheel rate calibration
    r2t = rate2throttle.item() # scipy Akima1DInterpolator (see sanbox/calib_wheel_spd.py)
    right_rate = left_rate = 0

    # Initial pose estimate. NOTE: Face camera to tag0
    td = TagDetect()
    detect_tag0 = False
    print('\nLooking for tag0...')
    while (not detect_tag0):
        _, img = camera.read()    # Read current camera frame
        tags, _ = td.DetectTags(img) # Detect AprilTag
        stream.set_img(td.GetTagImage(tags))
        detect_tag0, x_init, y_init, yaw_init = td.InitialPoseEst(tags)
    print('Found tag0!')
    EKF = ExtendedKalmanFilter(x_init, y_init, yaw_init)
    Traj = EKF.GetEKFState().T

    # Init timing
    prev_t = temp_t = 0
    accel = 0
    print_hz = 20
    start_t = time.time()

    # ----- Control Loop ----- #
    while True:
        # Update times
        curr_t = time.time() - start_t
        dt = curr_t - prev_t
        prev_t = curr_t

        # EKF Steps
        _, img = camera.read()    # Read current camera frame
        tags, detect_time = td.DetectTags(img) # Detect AprilTag
        EKF.ProcessTagData(tags, detect_time)  # Load tag pose data into EKF
        EKF.Propagate(right_rate, left_rate, dt) # Tell state estimator control inputs

        # Apply control to system
        left_rate = (2*speed - yaw_rate*base_line)/(2*whl_rad)  # left wheel rate
        #print("R:"+str(left_rate))
        right_rate = (2*speed + yaw_rate*base_line)/(2*whl_rad) # right wheel rate
        left_throttle = r2t(np.clip(left_rate, 0, 14))
        #print("T:"+str(left_throttle))
        right_throttle = -r2t(np.clip(right_rate, 0, 14)) # (-) due to flipped motor
        kit.continuous_servo[7].throttle = left_throttle  # left wheel
        kit.continuous_servo[8].throttle = right_throttle # right wheel

        # Get state estimate
        X_est = EKF.GetEKFState()
        x_est = X_est[0]
        y_est = X_est[1]
        yaw_est = X_est[2]
        true_spd_est = X_est[3]
        dx_est = true_spd_est*np.cos(yaw_est) # estimated for control law
        dy_est = true_spd_est*np.sin(yaw_est) # estimated for control law

        # Update control input (dynamic feedback linearization)
        u1_ = u1(curr_t, x_est, dx_est)
        #print(u1_)
        u2_ = u2(curr_t, y_est, dy_est)
        speed = speed + dt*accel
        accel = u1_[0]*cos(yaw_est) + u2_[0]*sin(yaw_est)
        yaw_rate = (u2_[0]*cos(yaw_est) - u1_[0]*sin(yaw_est))/speed

        # Printing
        if (curr_t - temp_t > 1.0/print_hz):
            temp_t = curr_t
            #print(dt)
            stream.set_img(td.GetTagImage(tags))
            # Save to trajectory for analysis
            Traj = np.vstack((Traj, X_est.T))

except KeyboardInterrupt:
    # shut off servos
    kit.continuous_servo[7].throttle = 0
    kit.continuous_servo[8].throttle = 0
    # Save last EKF state
    np.savetxt("traj.txt", Traj, fmt='%.5f', delimiter=",")
    # Stop video capture
    camera.release()
    exit()

#if __name__=="__main__":
#    try:
#        main()
#    except KeyboardInterrupt:
#        # shut off servos
#        kit.continuous_servo[7].throttle = 0
#        kit.continuous_servo[8].throttle = 0
#        # Save last EKF state
#        np.savetxt("traj.txt", Traj, delimiter=",")
#        # Stop video capture
#        camera.release()
#        exit()
