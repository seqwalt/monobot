import time
import sys
import os
import numpy as np
import cv2
from numpy import sin, cos, sqrt, pi
from adafruit_servokit import ServoKit
from sshkeyboard import listen_keyboard, stop_listening
from flask import Flask, render_template, Response
import multiprocessing # for flask streaming
import threading       # for key press listener

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/src')
from kalman_filter import ExtendedKalmanFilter
from fiducial_detect import TagDetect
from flask_generators import PlotlyTrajectory, TagImage
from key_press import KeyPress

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

# ----- Vehicle Parameters ----- #
whl_rad = 0.066 # meters
base_line = 0.14089 # meters (dist btw wheels)

# Init control and throttle mapping
speed = 0
yaw_rate = 0
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

kp = KeyPress()
key_thrd = threading.Thread(target=listen_keyboard, name="keyboard listener", kwargs={'on_press': kp.press})
key_thrd.daemon = True
key_thrd.start()
yaw_rate = 0
tag_row = np.empty((0,13), float) # time, R[0,0], ..., R[2,2], t1, t2, t3
Tags = {'tag0':tag_row, 'tag1':tag_row, 'tag2':tag_row, 'tag3':tag_row, 'tag4':tag_row, 'tag5':tag_row}

# Init timing
prev_t = temp_t = 0
print_hz = 10
start_t = time.time()

try:
    # ----- Control Loop ----- #
    while True:
        # Update times
        curr_t = time.time() - start_t
        dt = curr_t - prev_t
        prev_t = curr_t

        # Printing/Logging/EKF Measurement
        if (curr_t - temp_t > 1.0/print_hz):
            temp_t = curr_t
            #print("Left wheel (rad/s): " + str(left_rate))
            #print(dt)
            # EKF Measurement Step
            _, img = camera.read()    # Read current camera frame
            tags, detect_time, gray_img = td.DetectTags(img) # Detect AprilTag
            EKF.ProcessTagData(tags, detect_time)  # Load tag pose data into EKF
            # Save trajectory for analysis
            Traj = np.vstack((Traj, X_est.T))
            # Save tags for analysis
            for (tag in tags):
                tag_id = tag.tag_id
                pose_t = tag.pose_t.reshape(-1, 1)
                pose_R = tag.pose_R
                pose_flat = np.hstack(( tag.pose_R.reshape(1,-1), tag.pose_t.reshape(1,-1) ))[0] # recover R with (pose_flat[0:9]).reshape(3,3) and t with (pose_flat[9:]).reshape(-1,1)
                tag_name = 'tag' + str(tag_id)
                Tags[tag_id] = np.vstack((Tags[tag_id], np.hstack((detect_time, pose_flat)) ))
            # Update plot stream
            plt_stream.set_plot(X_est[0,0], X_est[1,0], X_est[2,0])
            # Update camera stream
            tag_stream.set_tags(tags, gray_img)

        # EKF Propagation Step
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
        speed = 0.3
        if (not yaw_rate == kp.yaw_rate):
            yaw_rate = kp.yaw_rate
            if (not kp.command == None):
                print('\n'+kp.command)
                print('yaw rate: ' + str(yaw_rate))

except KeyboardInterrupt:
    # shut off servos
    kit.continuous_servo[7].throttle = 0
    kit.continuous_servo[8].throttle = 0
    # Save EKF states and tag detections
    np.savetxt("logs/traj.txt", Traj, fmt='%.5f', delimiter=",")
    for (i in range(6)):
        np.savetxt("logs/tag"+str(i)+".txt", Tags['tag'+str(i)], fmt='%.5f', delimiter=",")
    # Stop video capture
    camera.release()
    # Stop keyboard listener
    stop_listening()
    # Exit the program
    sys.exit()
