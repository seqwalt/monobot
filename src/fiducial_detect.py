import time
import cv2
import numpy as np
from dt_apriltags import Detector

class TagDetect:
    def __init__(self):
        self.detector = Detector(families='tag36h11',
                                 nthreads=2,
                                 quad_decimate=2.0,
                                 quad_sigma=0.0,
                                 refine_edges=1,
                                 decode_sharpening=0.25,
                                 debug=0)

    def DetectTags(self, color_img):
        gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

        fx = 606.55
        fy = 606.22
        cx = 314.20
        cy = 240.28
        sz = 0.10745 # tag size (m)

        tags = self.detector.detect(img=gray_img, estimate_tag_pose=True, camera_params=(fx,fy,cx,cy), tag_size=sz)
        detect_time = time.time()

        return (tags, detect_time, gray_img)

    def GetTagImage(self, tags, gray_img):
        tag_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

        for tag in tags:
            cv2.polylines(tag_img, [tag.corners.astype(int)], True, (0, 255, 0), 2)
            for pt in tag.corners:
                pt = tuple(map(int, pt))
                cv2.circle(tag_img, pt, 5, (0, 0, 255), -1)

        return tag_img

    def InitialPoseEst(self, tags):
        detect_tag0 = False
        for tag in tags:
            if (tag.tag_id == 0):
                R_TC = tag.pose_R
                p_TC = tag.pose_t.reshape(-1, 1)
                detect_tag0 = True
                break
        else:
            # tag0 not found
            return (detect_tag0, 0, 0, 0)

        # Compute estimates
        R_CB = np.array([[0, 0, 1],[-1, 0, 0],[0, -1, 0]]) # rotates vectors from cam frame to body frame
        p_CB = np.array((0.0325, 0, 0)).reshape(-1, 1)     # position of camera frame in body frame
        R_WT = np.array([[0, 1, 0],[0, 0, -1],[-1, 0, 0]]) # rotates vectors from world frame to tag0 frame
        p_WT = np.array((0, 0, 0)).reshape(-1, 1)          # tag0 is at the origin
        p_WB = R_CB @ R_TC @ p_WT + R_CB @ p_TC + p_CB
        R_WB = R_CB @ R_TC @ R_WT

        p_BW = -R_WB.T @ p_WB
        R_BW = R_WB.T

        x_est = p_BW[0,0]
        y_est = p_BW[1,0]
        yaw_est = np.arctan2(R_BW[1,0], R_BW[0,0]) # assuming only rotation is about z-axis

        return (detect_tag0, x_est, y_est, yaw_est)













#
