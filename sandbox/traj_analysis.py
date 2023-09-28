import csv
import numpy as np
import matplotlib.pyplot as plt

# Define the CSV file path
csv_file = 'traj0.txt'  # Replace with the path to your CSV file

# Initialize empty lists to store data
data = {
    'x': [], 'y': [], 'yaw': [], 'speed': [], 'yaw_rate': [],
    'cr1': [], 'cr2': [], 'cr3': [], 'cl1': [], 'cl2': [], 'cl3': [],
    'x_tag_1': [], 'y_tag_1': [], 'x_tag_2': [], 'y_tag_2': [],
    'x_tag_3': [], 'y_tag_3': [], 'x_tag_4': [], 'y_tag_4': [],
    'x_tag_5': [], 'y_tag_5': []
}

# Read the CSV file and extract data
with open(csv_file, 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        if len(row) >= 21:  # Ensure at least 21 columns of data are present
            for key, value in data.items():
                value.append(float(row.pop(0)))

# Create Figure 1 with subplots
fig1, axes1 = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
fig1.suptitle('Odometry')

# Subplot 1
axes1[0].plot(data['x'], label='X')
axes1[0].plot(data['y'], label='Y')
axes1[0].plot(data['yaw'], label='Yaw')
axes1[0].set_xlabel('Time')
axes1[0].legend()

# Subplot 2
axes1[1].plot(data['speed'], label='Speed')
axes1[1].plot(data['yaw_rate'], label='Yaw Rate')
axes1[1].set_xlabel('Time')
axes1[1].legend()

# Create Figure 2 with subplots
fig2, axes2 = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
fig2.suptitle('Control offsets for right and left wheels')

# Subplot 1
axes2[0].plot(data['cr1'], label='CR1')
axes2[0].plot(data['cr2'], label='CR2')
axes2[0].plot(data['cr3'], label='CR3')
axes2[0].set_xlabel('Time')
axes2[0].legend()

# Subplot 2
axes2[1].plot(data['cl1'], label='CL1')
axes2[1].plot(data['cl2'], label='CL2')
axes2[1].plot(data['cl3'], label='CL3')
axes2[1].set_xlabel('Time')
axes2[1].legend()

# Create Figure 3 with subplots
fig3, axes3 = plt.subplots(nrows=5, ncols=1, figsize=(8, 18))
fig3.suptitle('Tag Positions')

# Subplots 1 to 5
for i in range(5):
    axes3[i].plot(data[f'x_tag_{i + 1}'], label=f'X_TAG_{i + 1}')
    axes3[i].plot(data[f'y_tag_{i + 1}'], label=f'Y_TAG_{i + 1}')
    axes3[i].set_ylabel('Tag Positions')
    axes3[i].legend()
axes3[4].set_xlabel('Time')

# Adjust layout and display figures
plt.tight_layout()
plt.show()
