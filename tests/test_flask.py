import time
import sys
import os
import cv2
import numpy as np
from numpy import sin, cos, sqrt, pi
import multiprocessing
from flask import Flask, render_template, Response
from perlin_noise import generate_perlin_noise_2d

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../src')
from fiducial_detect import TagDetect
from flask_generators import PlotlyTrajectory, TagImage, Camera

# Reference trajectory
a = 4.1148
b = 1.1557
c = 3.5814
d = 1.8034
s = 0.97    # trajectory corner sharpness, orig: 0.97
T = 20      # total trajectory time
def x_d(t):
    return a + c*( (1/(2*s))*sqrt(2+2*s*sqrt(2)*cos((pi + 2*pi*t/T))+s*s*cos(2*(pi + 2*pi*t/T))) - (1/(2*s))*sqrt(2-2*s*sqrt(2)*cos((pi + 2*pi*t/T))+s*s*cos(2*(pi + 2*pi*t/T))) )
def y_d(t):
    return b + d*( (1/(2*s))*sqrt(2+2*s*sqrt(2)*sin((pi + 2*pi*t/T))-s*s*cos(2*(pi + 2*pi*t/T))) - (1/(2*s))*sqrt(2-2*s*sqrt(2)*sin((pi + 2*pi*t/T))-s*s*cos(2*(pi + 2*pi*t/T))) )

# -------------------- Flask setup -------------------- #
max_queue_sz = 1
plt_queue = multiprocessing.Queue()
cam_queue = multiprocessing.Queue(maxsize=max_queue_sz)
tags_queue = multiprocessing.Queue(maxsize=max_queue_sz)

# t = np.linspace(0,T,200)
# plt_stream = MatplotlibTrajectory(plt_queue,
#                             x_ref=x_d(t), y_ref=y_d(t),
#                             x_lim=(0, 8.3), y_lim=(-1.2, 3.5))
plt_stream = PlotlyTrajectory(plt_queue)
tag_stream = TagImage(tags_queue, cam_queue)
app = Flask(__name__, template_folder='../templates')

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

# Start camera video capture
camera = cv2.VideoCapture('/dev/video0')
if not camera.isOpened():
    raise RuntimeError('Could not start camera.')

# Perlin noise for fake trajectory
np.random.seed(0)
noise = generate_perlin_noise_2d((256, 256), (8, 8))

# Initialize tag detection
td = TagDetect()
print_hz = 15
plot_hz = 0.333
temp_t = temp2_t = 0
start_t = time.time()
traj = np.array((x_d(0), y_d(0))).reshape(1, 2)
curr_pos = traj

try:
    while True:
        curr_t = time.time() - start_t

        # Printing/Logging
        if (curr_t - temp_t > 1.0/print_hz):
            temp_t = curr_t
            # Plot streaming
            dt = 1.0/print_hz
            n = 0.05

            # AprilTag detection
            _, img = camera.read()    # Read current camera frame
            tags, detect_time, gray_img = td.DetectTags(img) # Detect AprilTag
            tag_stream.set_tags(tags, gray_img)

            x_prev = curr_pos[0,0]
            y_prev = curr_pos[0,1]
            curr_pos[0,0] = x_d(curr_t) + 0.2*noise[int(60*cos(2*pi*curr_t/30) + 255/2), int(60*sin(2*pi*curr_t/30) + 255/2)]
            curr_pos[0,1] = y_d(curr_t) + 0.2*noise[int(50*cos(2*pi*curr_t/30) + 255/2), int(50*sin(2*pi*curr_t/30) + 255/2)]
            traj = np.vstack((traj, curr_pos))
            rows, _ = traj.shape
            if (rows > 400):
                traj = np.delete(traj, obj=0, axis=0) # delete top row (oldest)
            yaw = np.arctan2(curr_pos[0,1]-y_prev, curr_pos[0,0]-x_prev)
            plt_stream.set_plot(curr_pos[0,0], curr_pos[0,1], yaw)


except KeyboardInterrupt:
    camera.release()
    sys.exit()
