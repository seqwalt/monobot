import time
import sys
import os
import numpy as np
import cv2
from numpy import sin, cos, sqrt, pi
from adafruit_servokit import ServoKit
from flask import Flask, render_template, Response
import multiprocessing # for flask streaming
import threading       # for key press listener

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/src')
from kalman_filter import ExtendedKalmanFilter
from fiducial_detect import TagDetect
from flask_generators import PlotlyTrajectory, TagImage

# -------------------- Flask setup -------------------- #
max_queue_sz = 1
plt_queue = multiprocessing.Queue()
cam_queue = multiprocessing.Queue(maxsize=max_queue_sz)
tags_queue = multiprocessing.Queue(maxsize=max_queue_sz)
plt_stream = PlotlyTrajectory(plt_queue)
tag_stream = TagImage(tags_queue, cam_queue)
app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(tag_stream.gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/plot_feed')
def plot_feed():
    """Plot streaming route"""
    return Response(plt_stream.gen(),
                    content_type='text/event-stream')
# run Flask app in a process
stream_proc = multiprocessing.Process(target=app.run, name="Flask app", kwargs={'host': '0.0.0.0', 'threaded': True})
stream_proc.daemon = True
stream_proc.start()
# -------------------- Done Flask setup -------------------- #

# ----- Desired trajectory ----- #
# See "Squircular Calculations" https://arxiv.org/vc/arxiv/papers/1604/1604.02174v1.pdf
a = 4.1148
b = 1.1557
c = 3.5814
d = 1.8034
s = 0.97    # trajectory corner sharpness, orig: 0.97
T = 80      # total trajectory time (s)
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
kdx = 0.5
kpy = 1.0
kdy = 0.5
def u1(t, x, dx):
    return ddx_d(t) + kpx*(x_d(t) - x) + kdx*(dx_d(t) - dx)
def u2(t, y, dy):
    return ddy_d(t) + kpy*(y_d(t) - y) + kdy*(dy_d(t) - dy)

# ----- Vehicle Parameters ----- #
whl_rad = 0.066 # meters
base_line = 0.14089 # meters (dist btw wheels)

# ----- Initialization ----- #
# Servo init
kit = ServoKit(channels=16)
# Start camera
camera = cv2.VideoCapture('/dev/video0')
if not camera.isOpened():
    raise RuntimeError('Could not start camera.')
# Init AprilTag detector
td = TagDetect()
# Init trajectory storage
Traj = np.nan*np.ones((1,21))

# Init control and throttle mapping
speed = speed_d(0)
yaw_rate = yaw_rate_d(0)
rate2throttle = np.load('src/rate2throttle.npy', allow_pickle=True) # load wheel rate calibration
r2t = rate2throttle.item() # scipy Akima1DInterpolator (see sanbox/calib_wheel_spd.py)
right_rate = left_rate = 0

# Initial pose estimate. NOTE: Face camera to tag0
detect_tag0 = False
print('\nLooking for tag0...')
while (not detect_tag0):
    _, img = camera.read()    # Read current camera frame
    tags, _, gray_img = td.DetectTags(img) # Detect AprilTag
    tag_stream.set_tags(tags, gray_img)
    detect_tag0, x_init, y_init, yaw_init = td.InitialPoseEst(tags)
print('Found tag0!')
EKF = ExtendedKalmanFilter(x_init, y_init, yaw_init)
Traj = EKF.GetEKFState().T

tag_row = np.empty((0,13), float) # time, R[0,0], ..., R[2,2], t1, t2, t3
Tags = {'tag0':tag_row, 'tag1':tag_row, 'tag2':tag_row, 'tag3':tag_row, 'tag4':tag_row, 'tag5':tag_row}

# Init timing
prev_t = temp_t = 0
accel = 0
print_hz = 20
start_t = time.time()

try:
    # ----- Control Loop ----- #
    while True:
        # Update times
        curr_t = time.time() - start_t
        dt = curr_t - prev_t
        prev_t = curr_t

        EKF.Propagate(right_rate, left_rate, dt) # Tell state estimator control inputs

        # EKF Measurement/Printing/Logging
        if (curr_t - temp_t > 1.0/print_hz):
            temp_t = curr_t
            # EKF Measurement Step
            _, img = camera.read()    # Read current camera frame
            tags, detect_time, gray_img = td.DetectTags(img) # Detect AprilTag
            EKF.ProcessTagData(tags, detect_time)  # Load tag pose data into EKF
            # Save trajectory for analysis
            Traj = np.vstack((Traj, X_est.T))
            # Save tags for analysis
            for tag in tags:
                tag_id = tag.tag_id
                pose_t = tag.pose_t.reshape(-1, 1)
                pose_R = tag.pose_R
                pose_flat = np.hstack(( tag.pose_R.reshape(1,-1), tag.pose_t.reshape(1,-1) ))[0] # recover R with (pose_flat[0:9]).reshape(3,3) and t with (pose_flat[9:]).reshape(-1,1)
                tag_name = 'tag' + str(tag_id)
                Tags[tag_name] = np.vstack((Tags[tag_name], np.hstack((detect_time, pose_flat)) ))
            # Update plot stream
            plt_stream.set_plot(X_est[0,0], X_est[1,0], X_est[2,0])
            # Update camera stream
            tag_stream.set_tags(tags, gray_img)

        # Apply control to system
        scl = (1/0.45)
        scl_yaw_l = 2*speed/(base_line*np.pi/3)*0.9*0.7
        scl_yaw_r = 2*speed/(base_line*np.pi/3)*0.8*0.68
        left_rate = np.clip((2*speed - yaw_rate*base_line)/(2*whl_rad), 0, 14.1)  # original left wheel rate, provided to EKF
        left_rate_adjusted = np.clip(scl*(2*speed - scl_yaw_l*yaw_rate*base_line)/(2*whl_rad), 0, 14.1)  # left wheel rate applied to system
        right_rate = np.clip((2*speed + yaw_rate*base_line)/(2*whl_rad), 0, 14.1) # original right wheel rate, provided to EKF
        right_rate_adjusted = np.clip(scl*(2*speed + scl_yaw_r*yaw_rate*base_line)/(2*whl_rad), 0, 14.1) # right wheel rate applied to system
        left_throttle = np.clip(r2t(left_rate_adjusted), 0, 1)
        right_throttle = -np.clip(r2t(right_rate_adjusted), 0, 1) # (-) due to flipped motor

        kit.continuous_servo[7].throttle = left_throttle  # left wheel
        kit.continuous_servo[8].throttle = right_throttle # right wheel

        # Get state estimate
        X_est = EKF.GetEKFState()
        x_est = X_est[0,0]
        y_est = X_est[1,0]
        yaw_est = X_est[2,0]
        true_spd_est = X_est[3,0]
        dx_est = true_spd_est*np.cos(yaw_est) # estimated for control law
        dy_est = true_spd_est*np.sin(yaw_est) # estimated for control law

        # Update control input (dynamic feedback linearization)
        u1_ = u1(curr_t, x_est, dx_est)
        #print(u1_)
        u2_ = u2(curr_t, y_est, dy_est)
        speed = speed + dt*accel
        accel = u1_*cos(yaw_est) + u2_*sin(yaw_est)
        yaw_rate = (u2_*cos(yaw_est) - u1_*sin(yaw_est))/speed
        print("speed: " + str(speed))
        print("yaw_rate: " + str(yaw_rate))

except KeyboardInterrupt:
    # shut off servos
    kit.continuous_servo[7].throttle = 0
    kit.continuous_servo[8].throttle = 0
    # Save last EKF state
    np.savetxt("traj.txt", Traj, fmt='%.5f', delimiter=",")
    P_last = EKF.GetEKFCov()
    np.savetxt("logs/err_cov.txt", P_last, fmt='%.5f', delimiter=",")
    for i in range(6):
        np.savetxt("logs/tag"+str(i)+".txt", Tags['tag'+str(i)], fmt='%.5f', delimiter=",")
    # Stop video capture
    camera.release()
    sys.exit()
