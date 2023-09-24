import os
import cv2
from base_camera import BaseCamera
import sys
sys.path.insert(0, '/home/monobot')
from fiducial_detect import TagDetect

class Camera(BaseCamera):
    video_source = 0

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        td = TagDetect()
        while True:
            # read current frame
            _, img = camera.read()

            #print(img.shape) # width: 640, height: 480
            img_tag = td.AprilTagDetect(img)

            # encode as a jpeg image and return it
            #yield cv2.imencode('.jpg', img)[1].tobytes() # original
            yield cv2.imencode('.jpg', img_tag)[1].tobytes()
