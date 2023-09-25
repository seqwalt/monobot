import numpy as np
from fiducial_detect import TagDetect

class EKF:
    def __init__(self):
        # Initialize state
        self.state = 0
        # Initialize covariance matrix
