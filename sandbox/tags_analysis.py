import sys
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print('Usage: python3 tags_analysis.py <path/to/tag0.txt>')
    exit()

# Define the CSV file path
csv_file = sys.argv[1]
print("\n---- Log data for tag " + csv_file + " ----\n")

# load data
tag_data = genfromtxt(csv_file, delimiter=',')
times = tag_data[:,0]     # time stamps
R_flat_arr = tag_data[:,1:10] # rotation matrix data
p_arr = tag_data[:,10:]       # position data

x_BW_arr = np.empty(len(times))
y_BW_arr = np.empty(len(times))
yaw_BW_arr = np.empty(len(times))

x_TC_arr = np.empty(len(times))
y_TC_arr = np.empty(len(times))
yaw_TC_arr = np.empty(len(times))

R_CB = np.array([[0, 0, 1],[-1, 0, 0],[0, -1, 0]]) # rotates vectors from cam frame to body frame
p_CB = np.array((0.0325, 0, 0)).reshape(-1, 1)     # position of camera frame in body frame
R_WT = np.array([[0, 1, 0],[0, 0, -1],[-1, 0, 0]]) # rotates vectors from world frame to tag0 frame
p_WT = np.array((0, 0, 0)).reshape(-1, 1)          # tag0 is at the origin
for i in range(len(times)):
    R_TC = (R_flat_arr[i,:]).reshape(3,3)
    p_TC = (p_arr[i,:]).reshape(-1,1)
    p_WB = R_CB @ R_TC @ p_WT + R_CB @ p_TC + p_CB
    R_WB = R_CB @ R_TC @ R_WT
    p_BW = -R_WB.T @ p_WB
    R_BW = R_WB.T
    # Compute robot position and yaw estimate in world frame
    # !!!!currently only for tag0!!!!
    x_BW_arr[i] = p_BW[0,0]
    y_BW_arr[i] = p_BW[1,0]
    yaw_BW_arr[i] = np.arctan2(R_BW[1,0], R_BW[0,0]) # assuming only rotation is about z-axis
    # unwrap yaw values!
    diff = yaw_BW_arr[i] - yaw_BW_arr[i-1]
    if (i > 0 and np.abs(diff) > np.pi):
        yaw_BW_arr[i] -= np.sign(diff)*2*np.pi

    # store pose of tag in camera frame
    x_TC_arr[i] = p_TC[0,0]
    y_TC_arr[i] = p_TC[1,0]
    yaw_TC_arr[i] = np.arctan2(R_TC[1,0], R_TC[0,0]) # assuming only rotation is about z-axis

# Plotting
fig1, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
fig1.suptitle('Pose of body w.r.t world')

ax1[0].plot(times, x_BW_arr, label='X')
ax1[0].plot(times, y_BW_arr, label='Y')
ax1[0].set_xlabel('Time (s)')
ax1[0].set_ylabel('Position (m)')
ax1[0].legend()

ax1[1].plot(times, yaw_BW_arr, label='Yaw')
ax1[1].set_xlabel('Time (s)')
ax1[1].set_ylabel('yaw (rad)')
ax1[1].legend()

fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
fig2.suptitle('Pose of tag w.r.t camera')

ax2[0].plot(times, x_TC_arr, label='X')
ax2[0].plot(times, y_TC_arr, label='Y')
ax2[0].set_xlabel('Time (s)')
ax2[0].set_ylabel('Position (m)')
ax2[0].legend()

ax2[1].plot(times, yaw_TC_arr, label='Yaw')
ax2[1].set_xlabel('Time (s)')
ax2[1].set_ylabel('yaw (rad)')
ax2[1].legend()

plt.show()
