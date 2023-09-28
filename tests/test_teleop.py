import time
import sys
import os
import numpy as np
import threading
from sshkeyboard import listen_keyboard, stop_listening

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../src')
from key_press import KeyPress

# run keyboard listener in a process
kp = KeyPress()
key_proc = threading.Thread(target=listen_keyboard, name="keyboard listener", kwargs={'on_press': kp.press})
key_proc.daemon = True
key_proc.start()

# Initialize yaw_rate
yaw_rate = -1

print("Type: 'a' for left\n      'd' for right\n      's' for straight")
try:
    while True:
        if (not yaw_rate == kp.yaw_rate):
            yaw_rate = kp.yaw_rate
            if (not kp.command == None):
                print('\n'+kp.command)
                print('yaw rate: ' + str(yaw_rate))
except KeyboardInterrupt:
    stop_listening()
    sys.exit()
