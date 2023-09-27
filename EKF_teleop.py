import time
import sys
import numpy as np
import cv2
from numpy import sin, cos, sqrt, pi
from adafruit_servokit import ServoKit
from kalman_filter import ExtendedKalmanFilter
from fiducial_detect import TagDetect
from sshkeyboard import listen_keyboard, stop_listening
from flask import Flask, render_template, Response
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import pyplot as plt
import io
import threading

# -------------------- Flask setup -------------------- #
class PlotTrajectory:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.img = np.zeros((480, 640), dtype=np.uint8)
    def set_plot(self, traj, yaw):
        self.ax.plot(traj[:,0], traj[:,1], 'b', linewidth=2, label='Trajectory', zorder=5)
        curr_x = traj[-1,0]
        curr_y = traj[-1,1]
        arrow_len = 0.1
        tip = np.array(([cos(yaw), -sin(yaw)],[sin(yaw), cos(yaw)])) @ np.array((arrow_len,0)).reshape(-1,1)
        plt.arrow(curr_x, curr_y, tip[0,0], tip[1,0], width=0,
                  length_includes_head=True, head_width=0.3,
                  head_starts_at_zero=True, color='k', label='Pose',zorder=10)
        self.ax.plot(curr_x, curr_y, 'bo', markersize=4, zorder=15)
        self.ax.set_title('Trajectory')
        self.ax.set_xlabel('x (m)')
        self.ax.set_ylabel('y (m)')
        #self.ax.set_xlim(0, 8.3)
        #self.ax.set_ylim(-1.2, 3.5)
        self.ax.set_aspect('equal')
        self.ax.legend()
        img = io.BytesIO()
        canvas = FigureCanvasAgg(self.fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        self.img = np.asarray(buf) # convert to a NumPy array
        plt.cla()
    def gen(self):
        yield b'--frame\r\n'
        while True:
            frame = cv2.imencode('.jpg', self.img)[1].tobytes()
            yield b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n--frame\r\n'

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

plt_stream = PlotTrajectory()
cam_stream = Camera()
app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(cam_stream.gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/plot_feed')
def plot_feed():
    """Plotly streaming route"""
    return Response(plt_stream.gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# -------------------- Done Flask setup -------------------- #

stream_thrd = threading.Thread(target=app.run, name="Flask video stream", kwargs={'host': '0.0.0.0', 'threaded': True})
stream_thrd.daemon = True
stream_thrd.start()

class KeyPress:
    def __init__(self):
        self.yaw_rate = 0.0
    def press(self, key):
        rate = np.pi/2
        if key == "a":
            print('left turn')
            self.yaw_rate = rate
        elif key == "d":
            print('right turn')
            self.yaw_rate = -rate
        elif key == "s":
            print('stop turning')
            self.yaw_rate = 0
kp = KeyPress()

kit = ServoKit(channels=16)
camera = cv2.VideoCapture('/dev/video0')
if not camera.isOpened():
    raise RuntimeError('Could not start camera.')
Traj = np.nan*np.ones((1,21))

try:
    # ----- Vehicle Parameters ----- #
    whl_rad = 0.066 # meters
    base_line = 0.14089 # meters (dist btw wheels)

    # ----- Initialization ----- #

    # Init control and throttle mapping
    speed = 0
    yaw_rate = 0
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
        cam_stream.set_img(td.GetTagImage(tags))
        detect_tag0, x_init, y_init, yaw_init = td.InitialPoseEst(tags)
        #time.sleep(0.1)
    print('Found tag0!')
    EKF = ExtendedKalmanFilter(x_init, y_init, yaw_init)
    Traj = EKF.GetEKFState().T

    # Init timing
    prev_t = temp_t = 0
    print_hz = 10
    start_t = time.time()

    key_thrd = threading.Thread(target=listen_keyboard, name="keyboard listener", kwargs={'on_press': kp.press})
    key_thrd.daemon = True
    key_thrd.start()


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
            tags, detect_time = td.DetectTags(img) # Detect AprilTag
            EKF.ProcessTagData(tags, detect_time)  # Load tag pose data into EKF
            # Save to trajectory for analysis
            Traj = np.vstack((Traj, X_est.T))
            # Update plot stream
            plt_stream.set_plot(Traj[:,0:2], yaw_est)
            # Update camera stream
            cam_stream.set_img(td.GetTagImage(tags))

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
        yaw_est = X_est[2,0]

        # Update control input (user input)
        speed = 0.3
        yaw_rate = kp.yaw_rate

except KeyboardInterrupt:
    # shut off servos
    kit.continuous_servo[7].throttle = 0
    kit.continuous_servo[8].throttle = 0
    # Save last EKF state
    np.savetxt("traj.txt", Traj, fmt='%.5f', delimiter=",")
    # Stop video capture
    camera.release()
    # Stop keyboard listener
    stop_listening()
    # Exit the program
    exit()
