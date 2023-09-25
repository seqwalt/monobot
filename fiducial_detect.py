import time
import cv2
import numpy as np
from dt_apriltags import Detector

class TagDetect:
    def __init__(self):
        self.detector = Detector(families='tag36h11',
                                 nthreads=1,
                                 quad_decimate=4.0,
                                 quad_sigma=0.0,
                                 refine_edges=1,
                                 decode_sharpening=0.25,
                                 debug=0)
        self.gray_img = np.zeros((640,480), np.uint8)

    def DetectTags(self, color_img):
        self.gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

        fx = 606.55
        fy = 606.22
        cx = 314.20
        cy = 240.28
        sz = 0.10745 # tag size (m)

        tags = self.detector.detect(img=self.gray_img, estimate_tag_pose=True, camera_params=(fx,fy,cx,cy), tag_size=sz)

        return tags

    def GetTagImage(self, tags):
        tag_img = cv2.cvtColor(self.gray_img, cv2.COLOR_GRAY2BGR)

        for tag in tags:
            print(f"Tag ID: {tag.tag_id}")
            #print(f"Tag corners: {tag.corners}")
            #print(f"Tag pose: {tag.pose_t}, {tag.pose_R}")

            cv2.polylines(tag_img, [tag.corners.astype(int)], True, (0, 255, 0), 2)
            for pt in tag.corners:
                pt = tuple(map(int, pt))
                cv2.circle(tag_img, pt, 5, (0, 0, 255), -1)

        return tag_img
