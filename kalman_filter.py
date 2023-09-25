import numpy as np

class ExtendedKalmanFilter:
    def __init__(self):
        # Initialize state
        self.X = np.array([x, y, yaw, speed, yaw_rate, cr1, cr2, cr3, cl1, cl2, cl3, x_tag_1, y_tag_1, x_tag_2, y_tag_2, x_tag_3, y_tag_3, x_tag_4, y_tag_4, x_tag_5, y_tag_5]).reshape(-1,1)
        # Initialize covariance matrix
        X_sz, _ = self.X.shape
        self.P = np.eye(X_sz)

    def A_func(dt, sin_yaw, cos_yaw, v, r, w_r, w_l, bl):
        A_top = np.array([[1, 0, -dt*v*sin_yaw, dt*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, dt*v*cos_yaw, dt*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, r*w_r/2, r*w_r**2/2, r*w_r**3/2, r*w_l/2, r*w_l**2/2, r*w_l**3/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, r*w_r/bl, r*w_r**2/bl, r*w_r**3/bl, -r*w_l/bl, -r*w_l**2/bl, -r*w_l**3/bl, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        A_bot = np.hstack((np.zeros((16,5)), np.eye(16)))
        return np.vstack((A_top, A_bot))

    def W_func():
        W_top = np.eye(11)
        W_bot = np.zeros((10,11))
        return np.vstack((W_top, W_bot))

    def Q_func():
        Q_diag = 0.1*np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        return np.diag(Q_diag)

    # H_i matrix for tag i
    def H_i_func(x, y, sin_yaw, cos_yaw, yaw_rate, tag_num, x_tag_i, y_tag_i):
        # tag 0 (origin)
        if (tag_num == 0):
            H_i = np.array([[-cos_yaw, -sin_yaw, x*sin_yaw - y*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [sin_yaw, -cos_yaw, x*cos_yaw + y*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [yaw_rate*sin_yaw, -yaw_rate*cos_yaw, yaw_rate*(x*cos_yaw + y*sin_yaw), -1, x*sin_yaw - y*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [yaw_rate*cos_yaw, yaw_rate*sin_yaw, -yaw_rate*(x*sin_yaw - y*cos_yaw), 0, x*cos_yaw + y*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        # tag 1
        elif (tag_num == 1):
            H_i = np.array([[-cos_yaw, -sin_yaw, -(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0, cos_yaw, sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0],
                            [sin_yaw, -cos_yaw, (x - x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0, -sin_yaw, cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [yaw_rate*sin_yaw, -yaw_rate*cos_yaw, yaw_rate*((x - x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw), -1, -(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw, 0, 0, 0, 0, 0, 0, -yaw_rate*sin_yaw, yaw_rate*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0],
                            [yaw_rate*cos_yaw, yaw_rate*sin_yaw, -yaw_rate*(-(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw), 0, -(-x + x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw, 0, 0, 0, 0, 0, 0, -yaw_rate*cos_yaw, -yaw_rate*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        # tag 2
        elif (tag_num == 2):
            H_i = np.array([[-cos_yaw, -sin_yaw, -(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, cos_yaw, sin_yaw, 0, 0, 0, 0, 0, 0],
                            [sin_yaw, -cos_yaw, (x - x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sin_yaw, cos_yaw, 0, 0, 0, 0, 0, 0],
                            [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [yaw_rate*sin_yaw, -yaw_rate*cos_yaw, yaw_rate*((x - x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw), -1, -(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0, -yaw_rate*sin_yaw, yaw_rate*cos_yaw, 0, 0, 0, 0, 0, 0],
                            [yaw_rate*cos_yaw, yaw_rate*sin_yaw, -yaw_rate*(-(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw), 0, -(-x + x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0, -yaw_rate*cos_yaw, -yaw_rate*sin_yaw, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        # tag 3
        elif (tag_num == 3):
            H_i = np.array([[-cos_yaw, -sin_yaw, -(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, cos_yaw, sin_yaw, 0, 0, 0, 0],
                            [sin_yaw, -cos_yaw, (x - x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sin_yaw, cos_yaw, 0, 0, 0, 0],
                            [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [yaw_rate*sin_yaw, -yaw_rate*cos_yaw, yaw_rate*((x - x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw), -1, -(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -yaw_rate*sin_yaw, yaw_rate*cos_yaw, 0, 0, 0, 0],
                            [yaw_rate*cos_yaw, yaw_rate*sin_yaw, -yaw_rate*(-(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw), 0, -(-x + x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -yaw_rate*cos_yaw, -yaw_rate*sin_yaw, 0, 0, 0, 0],
                            [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        # tag 4
        elif (tag_num == 4):
            H_i = np.array([[-cos_yaw, -sin_yaw, -(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, cos_yaw, sin_yaw, 0, 0],
                            [sin_yaw, -cos_yaw, (x - x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sin_yaw, cos_yaw, 0, 0],
                            [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [yaw_rate*sin_yaw, -yaw_rate*cos_yaw, yaw_rate*((x - x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw), -1, -(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -yaw_rate*sin_yaw, yaw_rate*cos_yaw, 0, 0],
                            [yaw_rate*cos_yaw, yaw_rate*sin_yaw, -yaw_rate*(-(-x + x_tag_i)*sin_yaw + (-y + y_tag_i)*cos_yaw), 0, -(-x + x_tag_i)*cos_yaw - (-y + y_tag_i)*sin_yaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -yaw_rate*cos_yaw, -yaw_rate*sin_yaw, 0, 0],
                            [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        # tag 5
        elif (tag_num == 5):
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

    def ProcessTagData(tags):
        num_tags = len(tags)
        if (num_tags != 0):
            z = np.zeros((6*num_tags, 1))
            for tag in tags:
                # TODO
                id = tag.tag_id
                pose_t = tag.pose_t
                pose_ind =
                z[pose_ind]
                pose_R = tag.pose_R
            Measurement(z, )

    def ProcessDyn(X_):
        x = X_[0]
        y = X_[1]
        yaw = X_[2]
        speed = X_[3]
        yaw_rate = X_[4]

        dx = speed*np.cos(yaw)
        dy = speed*np.sin(yaw)
        dyaw = yaw_rate

        dX = np.array(([dx, dy, dyaw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])).reshape(-1,1)
        return dX

    def MeasureDyn_i(tag_num):
        if (tag_num == 0):
            x_tag = 0
            y_tag = 0
            yaw_tag = 0
        else:
            x_tag = self.X[9 + 2*tag_num]
            y_tag = self.X[10 + 2*tag_num]
            if (tag_num == 1):
                #TODO yaw_tag =
            elif (tag_num == 2):
                #TODO yaw_tag =
            elif (tag_num == 3):
                #TODO yaw_tag =
            elif (tag_num == 4):
                #TODO yaw_tag =
            elif (tag_num == 5):
                #TODO yaw_tag =
        x = self.X[0]
        y = self.X[1]
        yaw = self.X[2]
        speed = self.X[3]
        yaw_rate = self.X[4]
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

    def Propagate(self, wr, wl, dt):
        whl_rad = 0.066 # meters
        base_line = 0.14089 # meters (dist btw wheels)

        # Update speed and yaw_rate
        w_r_true = self.X[5]*w_r + self.X[6]*w_r**2 + self.X[7]*w_r**3
        w_l_true = self.X[8]*w_l + self.X[9]*w_l**2 + self.X[10]*w_l**3
        self.X[3] = (whl_rad/2)*(w_r_true + w_l_true)         # update speed
        self.X[4] = (whl_rad/base_line)*(w_r_true - w_l_true) # update yaw_rate

        # RK-4 propagation
        k1 = ProcessDyn(X)
        k2 = ProcessDyn(X+dt/2*k1)
        k3 = ProcessDyn(X+dt/2*k2)
        k4 = ProcessDyn(X+dt*k3)
        k = (k1+2*k2+2*k3+k4)/6
        X = X + dt*k

        # Propogate error covariance
        yaw = X[2]
        speed = X[3]
        A = A_func(dt, np.sin(yaw), np.cos(yaw), speed, whl_rad, wr, wl, base_line)
        W = W_func()
        Q = Q_func()
        self.P = A @ self.P @ A.T + W @ Q @ W.T

    # input: measurements, visible tag numbers
    def Measurement(self, z, tag_nums):
        # Get some states
        x = self.X[0]
        y = self.X[1]
        yaw = self.X[2]
        speed = self.X[3]
        yaw_rate = self.X[4]

        # Generate H matrix, measurment dynamics vector, and R diagonal
        iter = 0
        for tag_num in tag_nums:
            if (tag_num == 0):
                x_tag = 0
                y_tag = 0
            else:
                x_tag = self.X[9 + 2*tag_num]
                y_tag = self.X[10 + 2*tag_num]
            tag_dist = np.sqrt(z[0 + 6*iter]**2 + z[1 + 6*iter]**2)
            if (tag_num == tag_nums[0]):
                # first iteration
                H = H_i_func(x, y, np.sin(yaw), np.cos(yaw), yaw_rate, tag_num, x_tag, y_tag)
                MeasDyn = MeasureDyn_i(tag_num)
                R_diag = 0.1*tag_dist*np.ones(6)
            else:
                H_i = H_i_func(x, y, np.sin(yaw), np.cos(yaw), yaw_rate, tag_num, x_tag, y_tag)
                H = np.vstack((H, H_i))
                MeasDyn_i = MeasureDyn_i(tag_num)
                MeasDyn = np.vstack((MeasDyn, MeasDyn_i))
                R_diag_i = 0.1*tag_dist*np.ones(6)
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
