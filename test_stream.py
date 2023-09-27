import time
import sys
import numpy as np
from numpy import sin, cos, sqrt, pi
import cv2
from fiducial_detect import TagDetect
from flask import Flask, render_template, Response
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import pyplot as plt
import io
import threading

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

#Perlin noise https://github.com/pvigier/perlin-numpy/blob/master/perlin_numpy/perlin2d.py
def interpolant(t):
    return t*t*t*(t*(t*6 - 15) + 10)
def generate_perlin_noise_2d(
        shape, res, tileable=(False, False), interpolant=interpolant
):
    """Generate a 2D numpy array of perlin noise.

    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multiple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of shape shape with the generated noise.

    Raises:
        ValueError: If shape is not a multiple of res.
    """
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]]\
             .transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    if tileable[0]:
        gradients[-1,:] = gradients[0,:]
    if tileable[1]:
        gradients[:,-1] = gradients[:,0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[    :-d[0],    :-d[1]]
    g10 = gradients[d[0]:     ,    :-d[1]]
    g01 = gradients[    :-d[0],d[1]:     ]
    g11 = gradients[d[0]:     ,d[1]:     ]
    # Ramps
    n00 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]  )) * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]  )) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = interpolant(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)

np.random.seed(0)
noise = generate_perlin_noise_2d((256, 256), (8, 8))

# -------------------- Flask setup -------------------- #
class PlotTrajectory:
    def __init__(self):
        self.ref_x, self.ref_y = self.init_ref_traj()
        self.fig, self.ax = plt.subplots()
        self.img = None
    def init_ref_traj(self):
        # reference trajectory data
        t = np.linspace(0,T,200)
        X = x_d(t).reshape(-1,1)
        Y = y_d(t).reshape(-1,1)
        return (X, Y)
    def set_plot(self, traj, yaw):
        self.ax.plot(self.ref_x, self.ref_y, 'g', label='Reference', zorder=0)
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
        self.ax.set_xlim(0, 8.3)
        self.ax.set_ylim(-1.2, 3.5)
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

# run Flask app in a thread
stream_thrd = threading.Thread(target=app.run, name="Flask app", kwargs={'host': '0.0.0.0', 'threaded': True})
stream_thrd.daemon = True
stream_thrd.start()

# Start camera video capture
camera = cv2.VideoCapture('/dev/video0')
if not camera.isOpened():
    raise RuntimeError('Could not start camera.')

# Initialize tag detection
td = TagDetect()
print_hz = 15
temp_t = 0
start_t = time.time()
traj = np.array((x_d(0), y_d(0))).reshape(1, 2)
curr_pos = traj
i = 0

while True:
    curr_t = time.time() - start_t

    # AprilTag detection
    _, img = camera.read()    # Read current camera frame
    tags, detect_time = td.DetectTags(img) # Detect AprilTag
    cam_stream.set_img(td.GetTagImage(tags))

    # Printing/Logging
    if (curr_t - temp_t > 1.0/print_hz):
        temp_t = curr_t
        # Plot streaming
        dt = 1.0/print_hz
        n = 0.05

        x_prev = curr_pos[0,0]
        y_prev = curr_pos[0,1]
        curr_pos[0,0] = x_d(curr_t) + 0.2*noise[int(60*cos(2*pi*curr_t/30) + 255/2), int(60*sin(2*pi*curr_t/30) + 255/2)]
        curr_pos[0,1] = y_d(curr_t) + 0.2*noise[int(50*cos(2*pi*curr_t/30) + 255/2), int(50*sin(2*pi*curr_t/30) + 255/2)]
        traj = np.vstack((traj, curr_pos))
        rows, _ = traj.shape
        if (rows > 400):
            traj = np.delete(traj, obj=0, axis=0) # delete top row (oldest)
        yaw = np.arctan2(curr_pos[0,1]-y_prev, curr_pos[0,0]-x_prev)
        plt_stream.set_plot(traj, yaw)

        i += 1
