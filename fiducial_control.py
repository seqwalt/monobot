import time
import cv2
from dt_apriltags import Detector
from adafruit_servokit import ServoKit

#kit = ServoKit(channels=16)

#kit.continuous_servo[7].throttle = 0.1
#kit.continuous_servo[8].throttle = -0.1

class TagDetect:
    def __init__(self):
        self.detector = Detector(families='tag36h11',
                                 nthreads=1,
                                 quad_decimate=4.0,
                                 quad_sigma=0.0,
                                 refine_edges=1,
                                 decode_sharpening=0.25,
                                 debug=0)

    def AprilTagDetect(self, color_img):
        gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

        fx = 606.55
        fy = 606.22
        cx = 314.20
        cy = 240.28
        sz = 0.10745 # meters

        tags = self.detector.detect(img=gray_img, estimate_tag_pose=True, camera_params=(fx,fy,cx,cy), tag_size=sz)
        tag_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

        for tag in tags:
            #print(f"Tag ID: {tag.tag_id}")
            #print(f"Tag corners: {tag.corners}")
            print(f"Tag pose: {tag.pose_t}, {tag.pose_R}")

            cv2.polylines(tag_img, [tag.corners.astype(int)], True, (0, 255, 0), 2)
            for pt in tag.corners:
                pt = tuple(map(int, pt))
                cv2.circle(tag_img, pt, 5, (0, 0, 255), -1)

        return tag_img
