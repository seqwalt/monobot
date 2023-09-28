# Flask generator classes

import numpy as np
import io
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import pyplot as plt
from fiducial_detect import TagDetect

class Camera:
    def __init__(self, frame_queue):
        self.frame_queue = frame_queue
    def set_img(self, img):
        if (not self.frame_queue.full()): # check if queue has room for an img
            self.frame_queue.put_nowait(img)
    def gen(self):
        yield b'--frame\r\n'
        while True:
            cam_img = self.frame_queue.get()
            frame = cv2.imencode('.jpg', cam_img)[1].tobytes()
            yield b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n--frame\r\n'

class TagImage:
    def __init__(self, tags_queue, frame_queue):
        self.tags_queue = tags_queue
        self.frame_queue = frame_queue
        self.td = TagDetect()
    def set_tags(self, tags, gray_img):
        if (not self.tags_queue.full() and not self.frame_queue.full()): # check if queues have space
            self.tags_queue.put_nowait(tags)
            self.frame_queue.put_nowait(gray_img)
    def gen(self):
        yield b'--frame\r\n'
        while True:
            tags = self.tags_queue.get()
            gray_img = self.frame_queue.get()
            tag_img = self.td.GetTagImage(tags, gray_img)
            frame = cv2.imencode('.jpg', tag_img)[1].tobytes()
            yield b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n--frame\r\n'

class PlotTrajectory:
    def __init__(self, frame_queue,
                       x_ref=None, y_ref=None,
                       x_lim=None, y_lim=None):
        self.is_ref = False
        if (isinstance(x_ref, np.ndarray) and isinstance(y_ref, np.ndarray)):
            self.is_ref = True # refernce traj exists
            self.ref_x = x_ref.reshape(-1,1)
            self.ref_y = y_ref.reshape(-1,1)
        self.is_x_lim = False
        if (isinstance(x_lim, tuple)):
            self.is_x_lim = True
            self.x_lim = x_lim
        self.is_y_lim = False
        if (isinstance(y_lim, tuple)):
            self.is_y_lim = True
            self.y_lim = y_lim
        self.fig, self.ax = plt.subplots()
        self.frame_queue = frame_queue
    def set_plot(self, traj, yaw):
        if (self.is_ref):
            self.ax.plot(self.ref_x, self.ref_y, 'g', label='Reference', zorder=0)
        self.ax.plot(traj[:,0], traj[:,1], 'b', linewidth=2, label='Trajectory', zorder=5)
        curr_x = traj[-1,0]
        curr_y = traj[-1,1]
        arrow_len = 0.1
        tip = np.array(([np.cos(yaw), -np.sin(yaw)],[np.sin(yaw), np.cos(yaw)])) @ np.array((arrow_len,0)).reshape(-1,1)
        plt.arrow(curr_x, curr_y, tip[0,0], tip[1,0], width=0,
                  length_includes_head=True, head_width=0.3,
                  head_starts_at_zero=True, color='k', label='Pose',zorder=10)
        self.ax.plot(curr_x, curr_y, 'bo', markersize=4, zorder=15)
        self.ax.set_title('Trajectory')
        self.ax.set_xlabel('x (m)')
        self.ax.set_ylabel('y (m)')
        if (self.is_x_lim):
            self.ax.set_xlim(self.x_lim[0], self.x_lim[1])
        if (self.is_y_lim):
            self.ax.set_ylim(self.y_lim[0], self.y_lim[1])
        self.ax.set_aspect('equal')
        self.ax.legend()
        img = io.BytesIO()
        canvas = FigureCanvasAgg(self.fig)
        canvas.draw()
        img = np.asarray(canvas.buffer_rgba())
        if (not self.frame_queue.full()): # check if queue has room for an img
            self.frame_queue.put_nowait(img)     # put img in queue
        plt.cla() # clear plot axes
    def gen(self):
        yield b'--frame\r\n'
        while True:
            plt_img = self.frame_queue.get()
            frame = cv2.imencode('.jpg', plt_img)[1].tobytes()
            yield b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n--frame\r\n'
