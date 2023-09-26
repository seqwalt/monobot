import time
import sys
from adafruit_servokit import ServoKit


kit = ServoKit(channels=16)

if len(sys.argv) != 2:
    print('Usage: python3 test_speed.py <speed>')
    exit()

spd = float(sys.argv[1])

if (spd == 0.0):
    kit.continuous_servo[7].throttle = 0
    kit.continuous_servo[8].throttle = 0
    exit()

kit.continuous_servo[7].throttle = spd
kit.continuous_servo[8].throttle = -spd
time.sleep(2)
kit.continuous_servo[7].throttle = 0
kit.continuous_servo[8].throttle = 0
