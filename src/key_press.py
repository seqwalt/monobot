import numpy as np

class KeyPress:
    def __init__(self):
        self.yaw_rate = 0.0
        self.command = None
    def press(self, key):
        rate = np.pi/2
        if key == "a":
            self.command = 'Left turn'
            self.yaw_rate = rate
        elif key == "d":
            self.command = 'Right turn'
            self.yaw_rate = -rate
        elif key == "s":
            self.command = 'Go straight'
            self.yaw_rate = 0
