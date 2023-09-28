import numpy as np

class ExtendedKalmanFilter:
    def __init__(self, x, y, yaw):
        # Initialize state
        # x, y, yaw, speed, yaw_rate, cr1, cr2, cr3, cl1, cl2, cl3, x_tag_1, y_tag_1, x_tag_2, y_tag_2, x_tag_3, y_tag_3, x_tag_4, y_tag_4, x_tag_5, y_tag_5
        self.X = np.array((x, y, yaw, 0, 0, 1, 0, 0, 1, 0, 0, 6.40969, -0.6223, 0.635, -1.0795, 3.683, 1.9304, 1.1684, 2.032, 1.71069, -1.0795)).reshape(-1,1)
        # Initialize covariance matrix
        X_sz, _ = self.X.shape
        self.P = np.eye(X_sz)
        # Tag data history
        self.prev_tag_data = {'t':[], 'tag0':[], 'tag1':[], 'tag2':[], 'tag3':[], 'tag4':[], 'tag5':[]}

    def A_func(self, dt, sin_yaw, cos_yaw, v, r, w_r, w_l, bl):
        A_top = np.array([[1, 0, -dt*v*sin_yaw, dt*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, dt*v*cos_yaw, dt*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, r*w_r/2, r*w_r**2/2, r*w_r**3/2, r*w_l/2, r*w_l**2/2, r*w_l**3/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, r*w_r/bl, r*w_r**2/bl, r*w_r**3/bl, -r*w_l/bl, -r*w_l**2/bl, -r*w_l**3/bl, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        A_bot = np.hstack((np.zeros((16,5)), np.eye(16)))
        return np.vstack((A_top, A_bot))

    def W_func(self):
        W_top = np.eye(11)
        W_bot = np.zeros((10,11))
        return np.vstack((W_top, W_bot))

    def Q_func(self):
        # x, y, yaw, speed, yaw_rate, cr1, cr2, cr3, cl1, cl2, cl3
        Q_diag = np.array([0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.0, 0.0, 0.01, 0.0, 0.0])
        return np.diag(Q_diag)

    # H_i matrix for tag i
    def H_i_func(self, x, y, sin_yaw, cos_yaw, yaw_rate, tag_id, x_tag_i, y_tag_i):
        # tag 0 (origin)
        if (tag_id == 0):
            H_i = np.array([[-cos_yaw, -sin_yaw, x*sin_yaw - y*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [sin_yaw, -cos_yaw, x*cos_yaw + y*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [yaw_rate*sin_yaw, -yaw_rate*cos_yaw, yaw_rate*(x*cos_yaw + y*sin_yaw), -1, x*sin_yaw - y*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [yaw_rate*cos_yaw, yaw_rate*sin_yaw, -yaw_rate*(x*sin_yaw - y*cos_yaw), 0, x*cos_yaw + y*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        # tag 1
        elif (tag_id == 1):
            H_i = np.array([[-cos_yaw, -sin_yaw, -(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0, cos_yaw, sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0],
                            [sin_yaw, -cos_yaw, (x - x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0, -sin_yaw, cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [yaw_rate*sin_yaw, -yaw_rate*cos_yaw, yaw_rate*((x - x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw), -1, -(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw, 0, 0, 0, 0, 0, 0, -yaw_rate*sin_yaw, yaw_rate*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0],
                            [yaw_rate*cos_yaw, yaw_rate*sin_yaw, -yaw_rate*(-(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw), 0, -(-x + x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw, 0, 0, 0, 0, 0, 0, -yaw_rate*cos_yaw, -yaw_rate*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        # tag 2
        elif (tag_id == 2):
            H_i = np.array([[-cos_yaw, -sin_yaw, -(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, cos_yaw, sin_yaw, 0, 0, 0, 0, 0, 0],
                            [sin_yaw, -cos_yaw, (x - x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sin_yaw, cos_yaw, 0, 0, 0, 0, 0, 0],
                            [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [yaw_rate*sin_yaw, -yaw_rate*cos_yaw, yaw_rate*((x - x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw), -1, -(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0, -yaw_rate*sin_yaw, yaw_rate*cos_yaw, 0, 0, 0, 0, 0, 0],
                            [yaw_rate*cos_yaw, yaw_rate*sin_yaw, -yaw_rate*(-(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw), 0, -(-x + x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0, -yaw_rate*cos_yaw, -yaw_rate*sin_yaw, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        # tag 3
        elif (tag_id == 3):
            H_i = np.array([[-cos_yaw, -sin_yaw, -(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, cos_yaw, sin_yaw, 0, 0, 0, 0],
                            [sin_yaw, -cos_yaw, (x - x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sin_yaw, cos_yaw, 0, 0, 0, 0],
                            [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [yaw_rate*sin_yaw, -yaw_rate*cos_yaw, yaw_rate*((x - x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw), -1, -(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -yaw_rate*sin_yaw, yaw_rate*cos_yaw, 0, 0, 0, 0],
                            [yaw_rate*cos_yaw, yaw_rate*sin_yaw, -yaw_rate*(-(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw), 0, -(-x + x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -yaw_rate*cos_yaw, -yaw_rate*sin_yaw, 0, 0, 0, 0],
                            [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        # tag 4
        elif (tag_id == 4):
            H_i = np.array([[-cos_yaw, -sin_yaw, -(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, cos_yaw, sin_yaw, 0, 0],
                            [sin_yaw, -cos_yaw, (x - x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sin_yaw, cos_yaw, 0, 0],
                            [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [yaw_rate*sin_yaw, -yaw_rate*cos_yaw, yaw_rate*((x - x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw), -1, -(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -yaw_rate*sin_yaw, yaw_rate*cos_yaw, 0, 0],
                            [yaw_rate*cos_yaw, yaw_rate*sin_yaw, -yaw_rate*(-(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw), 0, -(-x + x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -yaw_rate*cos_yaw, -yaw_rate*sin_yaw, 0, 0],
                            [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        # tag 5
        elif (tag_id == 5):
            H_i = np.array([[-cos_yaw, -sin_yaw, -(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, cos_yaw, sin_yaw],
                            [sin_yaw, -cos_yaw, (x - x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sin_yaw, cos_yaw],
                            [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [yaw_rate*sin_yaw, -yaw_rate*cos_yaw, yaw_rate*((x - x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw), -1, -(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -yaw_rate*sin_yaw, yaw_rate*cos_yaw],
                            [yaw_rate*cos_yaw, yaw_rate*sin_yaw, -yaw_rate*(-(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw), 0, -(-x + x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -yaw_rate*cos_yaw, -yaw_rate*sin_yaw],
                            [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        else:
            print("Invalid tag number.")
            exit()
        return H_i

    # Get current EKF state
    def GetEKFState(self):
        return self.X

    # Get measurments ready EKF measurment step
    def ProcessTagData(self, tags, detect_time):
        num_tags = len(tags)
        if (num_tags != 0):
            tag_ids = -1*np.ones(num_tags)
            z = np.zeros((6*num_tags, 1))
            R_CB = np.array([[0, 0, 1],[-1, 0, 0],[0, -1, 0]]) # rotates vectors from cam frame to body frame
            p_CB = np.array((0.0325, 0, 0)).reshape(-1, 1)     # position of camera frame in body frame
            iter = 0
            valid_tag_ids = [] # ids for valid measurements
            valid_measurements = True

            for tag in tags:
                # Init current measurment vector
                z_i = np.zeros((6, 1))

                # Tag ID (0, 1, 2, 3, 4 or 5)
                tag_ids[iter] = tag.tag_id

                # Positon of Tag frame in Body frame
                p_TC = tag.pose_t.reshape(-1,1) # position of tag frame in camera frame
                R_TC = tag.pose_R               # rotates vectors from tag frame to camera frame
                p_TB = R_CB @ p_TC + p_CB       # position of tag frame in body frame
                z_i[0,0] = p_TB[0,0] # x_tag position in monobot body frame
                z_i[1,0] = p_TB[1,0] # y_tag position in monobot body frame

                # Yaw of Tag frame w.r.t Body frame
                R_TB = R_CB @ R_TC                        # rotates vectors from tag frame to body frame
                yaw_TB = np.pi + np.arctan2(R_TC[2,0], R_TC[0,0]) # where "yaw_TB" is a rotation about the body frame z axis, and equal rotation about camera frame. Calculated using euler order: YZX (see sandbox/symbolic_sandbox_rotation_mat.py)
                z_i[2,0] = yaw_TB # yaw (about gravity axis) from tag frame to body frame

                # Velocity and Yaw rate of Tag frame in Body frame
                tag_name = 'tag'+str(tag.tag_id)
                prev_meas_exist = len(self.prev_tag_data[tag_name]) != 0
                if (not prev_meas_exist):
                    print('Rejected measurment due no velocity measured')
                    valid_measurements = False
                else:
                    prev_time = self.prev_tag_data['t']
                    prev_x_TB = self.prev_tag_data[tag_name][0]
                    prev_y_TB = self.prev_tag_data[tag_name][1]
                    prev_yaw_TB = self.prev_tag_data[tag_name][2]

                    # Unwrap yaw!
                    diff = yaw_TB - prev_yaw_TB
                    if (np.abs(diff) > np.pi):
                        yaw_TB -= np.sign(diff)*2*np.pi

                    # "Measure" velocity
                    dt = detect_time - prev_time
                    dx_TB = (p_TB[0,0] - prev_x_TB)/dt    # x vel "measurement"
                    dy_TB = (p_TB[1,0] - prev_y_TB)/dt    # y vel "measurement"
                    dyaw_TB = (yaw_TB - prev_yaw_TB)/dt # yaw rate "measurement"
                    z_i[3,0] = dx_TB
                    z_i[4,0] = dy_TB
                    z_i[5,0] = dyaw_TB

                    # store ids of valid measurements
                    valid_tag_ids.append(tag.tag_id)

                # Update prev_tag_data for curr tag
                self.prev_tag_data[tag_name] = [p_TB[0,0], p_TB[1,0], yaw_TB]

                # Load measurement vector
                z[6*iter : 6*iter+6] = z_i

                iter += 1

            # Update prev_tag_data detect time
            self.prev_tag_data['t'] = detect_time

            if (valid_measurements):
                self.Measurement(z, valid_tag_ids)

            # Remove prev_tag_data elements for tag_ids that are not currently visible
            for i in range(6):
                if (i not in tag_ids):
                    self.prev_tag_data['tag'+str(i)] = [] # set dict element to empty
        else:
            # No visible tags, reset prev_tag_data dict
            self.prev_tag_data = {'t':[], 'tag0':[], 'tag1':[], 'tag2':[], 'tag3':[], 'tag4':[], 'tag5':[]}

    def ProcessDyn(self, X_):
        x = X_[0,0]
        y = X_[1,0]
        yaw = X_[2,0]
        speed = X_[3,0]
        yaw_rate = X_[4,0]

        dx = speed*np.cos(yaw)
        dy = speed*np.sin(yaw)
        dyaw = yaw_rate

        dX = np.array((dx, dy, dyaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).reshape(-1,1)
        return dX

    def MeasureDyn_i(self, tag_id_):
        num_rots = np.round(-np.pi/4 + self.X[2,0]/(2*pi)) # round to int nearest to pi/4 + yaw/2pi (heuristic)
        rad_rots = 2*np.pi*num_rots
        tag_id = int(tag_id_)
        if (tag_id == 0):
            x_tag = 0
            y_tag = 0
            yaw_tag = 0 + rad_rots
        else:
            x_tag = self.X[9 + 2*tag_id,0]
            y_tag = self.X[10 + 2*tag_id,0]
            if (tag_id == 1):
                yaw_tag = np.pi + rad_rots
            elif (tag_id == 2):
                yaw_tag = np.pi/2 + rad_rots
            elif (tag_id == 3):
                yaw_tag = 3*np.pi/2 + rad_rots
            elif (tag_id == 4):
                yaw_tag = 2*np.pi + rad_rots
            elif (tag_id == 5):
                yaw_tag = np.pi/2 +rad_rots
        x = self.X[0,0]
        y = self.X[1,0]
        yaw = self.X[2,0]
        speed = self.X[3,0]
        yaw_rate = self.X[4,0]
        dx = speed*np.cos(yaw)
        dy = speed*np.sin(yaw)

        cos = np.cos(yaw)
        sin = np.sin(yaw)
        p_tag_body    = np.array([cos*(x_tag - x) + sin*(y_tag - y), cos*(y_tag - y) - sin*(x_tag - x)]).reshape(-1,1)
        yaw_tag_body  = yaw_tag - yaw
        dp_tag_body   = np.array([yaw_rate*(cos*(y_tag - y) - sin*(x_tag - x)) - speed, -yaw_rate*(sin*(y_tag - y) + cos*(x_tag - x))]).reshape(-1,1)
        dyaw_tag_body = -yaw_rate
        h = np.vstack((p_tag_body, yaw_tag_body, dp_tag_body, dyaw_tag_body))

        return h

    def Propagate(self, w_r, w_l, dt):
        whl_rad = 0.066 # meters
        base_line = 0.14089 # meters (dist btw wheels)

        # Update speed and yaw_rate
        w_r_true = self.X[5,0]*w_r + self.X[6,0]*w_r**2 + self.X[7,0]*w_r**3
        w_l_true = self.X[8,0]*w_l + self.X[9,0]*w_l**2 + self.X[10,0]*w_l**3
        self.X[3,0] = (whl_rad/2)*(w_r_true + w_l_true)         # update speed
        self.X[4,0] = (whl_rad/base_line)*(w_r_true - w_l_true) # update yaw_rate

        # RK-4 propagation
        k1 = self.ProcessDyn(self.X)
        k2 = self.ProcessDyn(self.X+dt/2*k1)
        k3 = self.ProcessDyn(self.X+dt/2*k2)
        k4 = self.ProcessDyn(self.X+dt*k3)
        k = (k1+2*k2+2*k3+k4)/6
        self.X = self.X + dt*k

        # Propogate error covariance
        yaw = self.X[2,0]
        speed = self.X[3,0]
        A = self.A_func(dt, np.sin(yaw), np.cos(yaw), speed, whl_rad, w_r, w_l, base_line)
        W = self.W_func()
        Q = self.Q_func()
        self.P = A @ self.P @ A.T + W @ Q @ W.T

    # input: measurements, visible tag numbers
    def Measurement(self, z, tag_ids):
        # Get some states
        x = self.X[0,0]
        y = self.X[1,0]
        yaw = self.X[2,0]
        speed = self.X[3,0]
        yaw_rate = self.X[4,0]

        # Generate H matrix, measurment dynamics vector, and R diagonal
        iter = 0
        for tag_id_ in tag_ids:
            tag_id = int(tag_id_)
            if (tag_id == 0):
                x_tag = 0
                y_tag = 0
            else:
                x_tag = self.X[9 + 2*tag_id,0]
                y_tag = self.X[10 + 2*tag_id,0]
            tag_dist = np.sqrt(z[0 + 6*iter,0]**2 + z[1 + 6*iter,0]**2)
            if (tag_id == tag_ids[0]):
                # first iteration
                H = self.H_i_func(x, y, np.sin(yaw), np.cos(yaw), yaw_rate, tag_id, x_tag, y_tag)
                MeasDyn = self.MeasureDyn_i(tag_id)
                R_diag = 0.08 + (0.5*tag_dist + 5.0*tag_dist**2)*np.ones(6)
            else:
                H_i = self.H_i_func(x, y, np.sin(yaw), np.cos(yaw), yaw_rate, tag_id, x_tag, y_tag)
                H = np.vstack((H, H_i))
                MeasDyn_i = self.MeasureDyn_i(tag_id)
                MeasDyn = np.vstack((MeasDyn, MeasDyn_i))
                R_diag_i = 0.08 + (0.5*tag_dist + 5.0*tag_dist**2)*np.ones(6)
                R_diag = np.hstack((R_diag, R_diag_i))
            iter += 1

        # Generate R matrix (square of size rows of H)
        # larger R values for further-away tags
        R = np.diag(R_diag)

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + R)

        # Update state est with measurement
        self.X = self.X + K @ (z - MeasDyn)

        # Update error covariance
        X_sz, _ = self.X.shape
        self.P = (np.eye(X_sz) - K @ H) @ self.P
