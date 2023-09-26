import time
import sys
import numpy as np
import cv2
from numpy import sin, cos, sqrt, pi
from adafruit_servokit import ServoKit
from kalman_filter import ExtendedKalmanFilter
from fiducial_detect import TagDetect
from sshkeyboard import listen_keyboard

kit = ServoKit(channels=16)
camera = cv2.VideoCapture('/dev/video0')
if not camera.isOpened():
    raise RuntimeError('Could not start camera.')
#Traj = np.nan*np.ones((1,21))

global global_tag_img

class KeyPress:
    def __init__(self):
        self.yaw_rate = 0.0
    def press(key):
        rate = np.pi/6
        if key == "a":
            self.yaw_rate = rate
        elif key == "d":
            self.yaw_rate = -rate
    def release(key):
        self.yaw_rate = 0.0
kp = KeyPress()

try:
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
        global_tag_img = td.GetTagImage(tags)
        detect_tag0, x_init, y_init, yaw_init = td.InitialPoseEst(tags)
    print('Found tag0!')
    EKF = ExtendedKalmanFilter(x_init, y_init, yaw_init)
    Traj = EKF.GetEKFState().T

    # Init timing
    prev_t = temp_t = 0
    print_hz = 10
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
        right_rate = (2*speed + yaw_rate*base_line)/(2*whl_rad) # right wheel rate
        left_throttle = np.clip(r2t(left_rate), 0, 1)
        right_throttle = -np.clip(r2t(right_rate), 0, 1) # (-) due to flipped motor
        kit.continuous_servo[7].throttle = left_throttle  # left wheel
        kit.continuous_servo[8].throttle = right_throttle # right wheel

        # Get state estimate
        X_est = EKF.GetEKFState()

        # Update control input (user input)
        listen_keyboard(on_press=kp.press, on_release=kp.release)
        speed = 0.015
        yaw_rate = kp.yaw_rate

        # Printing/Logging
        if (curr_t - temp_t > 1.0/print_hz):
            temp_t = curr_t
            #print(dt)
            # Save to trajectory for analysis
            Traj = np.vstack((Traj, X_est.T))
            # Update stream image
            global_tag_img = td.GetTagImage(tags)

except KeyboardInterrupt:
    # shut off servos
    kit.continuous_servo[7].throttle = 0
    kit.continuous_servo[8].throttle = 0
    # Save last EKF state
    np.savetxt("traj.txt", Traj, fmt='%.5f', delimiter=",")
    # Stop video capture
    camera.release()
    exit()
