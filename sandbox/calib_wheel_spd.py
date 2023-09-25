#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
import scipy.interpolate as interp

# Measurement data: row are [servo input, initial time (s), final time (s), num full rotations]
data = np.array([ [0.0, 0, 1, 0],
                  [0.01, 9.17, 21.4, 1],
                  [0.015, 3.87, 12.71, 2],
                  [0.02, 5.33, 8.02, 1],
                  [0.03, 4.93, 8.725, 2],
                  [0.04, 5.39, 8.08, 2],
                  [0.05, 4.29, 7.70, 3],
                  [0.06, 4.94, 7.653, 3],
                  [0.07, 6.08, 9.685, 5],
                  [0.08, 4.67, 6.69, 3],
                  [0.09, 4.40, 6.80, 4],
                  [0.1, 3.955, 6.23, 4],
                  [0.15, 4.03, 6.393, 5],
                  [0.2, 5.20, 7.898, 6],
                  #[0.3, 4.76, 7.43, 6],
                  # [0.5, 4.15, 6.82, 6],
                  # [0.7, 4.69, 7.38, 6],
                  [1.0, 4.165, 6.83, 6]
                  ])

# Get wheel rate map
# angle_rate = 2*pi*rotations/(time_final - time_initial)
ang_rate = 2*np.pi*data[:,3]/(data[:,2] - data[:,1])
servo_in = data[:,0]
rate2throttle = interp.Akima1DInterpolator(ang_rate, servo_in) # input: angular rate

# Plot
fig, ax = plt.subplots()
x = np.linspace(0.0,np.max(ang_rate),100)
ax.plot(rate2throttle(x), x)
ax.plot(servo_in, ang_rate, '.')
ax.set_title('Rate map')
ax.set_xlabel('Servo input')
ax.set_ylabel('Angular rate (rad/s)')
plt.show()

# Save wheel_rate_map
np.save('rate2throttle.npy', rate2throttle, allow_pickle=True)

# Load wheel_rate_map
# r2t = np.load('rate2throttle.npy', allow_pickle=True)
# rate2throttle = r2t.item()
# fig, ax = plt.subplots()
# x = np.linspace(0.0,1.0,100)
# ax.plot(x, rate2throttle(x))
# ax.set_title('Rate map')
# ax.set_xlabel('Servo input')
# ax.set_ylabel('Angular rate (rad/s)')
# plt.show()
