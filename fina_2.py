import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2

r = []
el = []
az = []
hypotheses = []
meas_tgt_pair = []

class CVFilter:
    def __init__(self):
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.Sp = np.zeros((6, 1))
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)
        self.S = np.eye((3))  # Measurement noise covariance
        self.S[0, 0] = 0.1
        self.S[1, 1] = 0.1
        self.S[2, 2] = 0.1
        self.Meas_Time = 0  # Measured time
        self.Z = np.zeros((3, 1))

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time

    def InitializeMeasurementForFiltering(self, x, y, z, vx, vy, vz, mt):
        Z = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = mt

    def predict_step(self, current_time):
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * self.plant_noise
        self.Sp = np.dot(Phi, self.Sf)
        self.Pp = np.dot(np.dot(Phi, self.Pf), Phi.T) + Q

    def update_step(self):
        Inn = self.Z - np.dot(self.H, self.Sf)
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pf, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sf + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pf)

def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    el = math.atan(z/np.sqrt(x**2 + y**2))*180/3.14
    az = math.atan(y/x)
    if x > 0.0:
        az = 3.14/2 - az
    else:
        az = 3*3.14/2 - az
    az = az*180/3.14
    if az < 0.0:
        az = (360 + az)
    if az > 360:
        az = (az - 360)
    return r, az, el

def cart2sph2(x:float, y:float, z:float, filtered_values_csv):
    for i in range(len(filtered_values_csv)):
        r.append(np.sqrt(x[i]**2 + y[i]**2 + z[i]**2))
        el.append(math.atan(z[i]/np.sqrt(x[i]**2 + y[i]**2))*180/3.14)
        az.append(math.atan(y[i]/x[i]))
        if x[i] > 0.0:
            az[i] = 3.14/2 - az[i]
        else:
            az[i] = 3*3.14/2 - az[i]
        az[i] = az[i]*180/3.14
        if az[i] < 0.0:
            az[i] = (360 + az[i])
        if az[i] > 360:
            az[i] = (az[i] - 360)
    return r, az, el

def read_measurements_from_csv(file_path):
    measurements = []
    meas_tgt_pair = []
    dt1 = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            tid = float(row[0])
            mr = float(row[7])  # MR column
            ma = float(row[8])  # MA column
            me = float(row[9])  # ME column
            mt = float(row[10])  # MT column
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            r, az, el = cart2sph(x, y, z)  # Convert Cartesian to spherical coordinates
            meas_tgt_pair.append((tid, r, az, el, mt))
            dt1.append(mt)
            measurements.append((r, az, el, mt))
    return measurements, meas_tgt_pair

kalman_filter = CVFilter()
csv_file_path = 'ttk_84.csv'
A = read_measurements_from_csv(csv_file_path)
measurements = A[0]
meas_tgt_pair = A[1]

csv_file_predicted = "ttk_84.csv"
df_predicted = pd.read_csv(csv_file_predicted)
filtered_values_csv = df_predicted[['F_TIM', 'F_X', 'F_Y', 'F_Z']].values

A = cart2sph2(filtered_values_csv[:, 1], filtered_values_csv[:, 2], filtered_values_csv[:, 3], filtered_values_csv)
number = 1000
result = np.divide(A[0], number)

time_list = []
r_list = []
az_list = []
el_list = []

for i, (r, az, el, mt) in enumerate(measurements):
    if i == 0:
        kalman_filter.initialize_filter_state(r, az, el, 0, 0, 0, mt)
    elif i == 1:
        prev_r, prev_az, prev_el = measurements[i - 1][:3]
        dt = mt - measurements[i - 1][3]
        vx = (r - prev_r) / dt
        vy = (az - prev_az) / dt
        vz = (el - prev_el) / dt
        kalman_filter.initialize_filter_state(r, az, el, vx, vy, vz, mt)
    else:
        chi2_threshold = chi2.ppf(0.95, df=3)
        cov_inv = np.linalg.inv(kalman_filter.S)

        def mahalanobis_distance(track_pp, report_pos, cov_inv):
            delta = report_pos - track_pp
            return np.dot(delta.T, np.dot(cov_inv, delta)).item()

        # Group Formation
        group_associations = []
        kalman_filter.predict_step(mt)
        A = np.dot(kalman_filter.H, kalman_filter.Sp)

        for tid, track in enumerate(meas_tgt_pair):
            T = np.array([track[1], track[2], track[3]]).reshape((3, 1))
            distance = mahalanobis_distance(A, T, cov_inv)
            if distance <= chi2_threshold:
                group_associations.append(tid)

        print(f"Groups formed at step {i}: {group_associations}")

        # Cluster Formation
        clusters = {}
        for tid in group_associations:
            track = meas_tgt_pair[tid]
            cluster_key = (track[1], track[2], track[3])  # Cluster based on position (r, az, el)
            if cluster_key not in clusters:
                clusters[cluster_key] = []
            clusters[cluster_key].append(tid)

        print(f"Clusters formed at step {i}: {clusters}")

        # Hypothesis Formation
        hypotheses = []
        for cluster_key, cluster_members in clusters.items():
            prob = 1 / len(cluster_members)
            for member in cluster_members:
                hypotheses.append((member, prob))

        print(f"Hypotheses formed at step {i}: {hypotheses}")

        probabilities = [hypothesis[1] for hypothesis in hypotheses]
        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()

        associated_measurements = {}
        for (measurement_id, prob) in hypotheses:
            associated_measurements[measurement_id] = prob

        kalman_filter.Z = np.array([r, az, el]).reshape((3, 1))
        kalman_filter.update_step()
        print(f"Updated state at step {i}: {kalman_filter.Sf.flatten()}")

    time_list.append(mt)
    # r_list.append(kalman_filter.Sf[0, 0])
    # az_list.append(kalman_filter.Sf[1, 0])
    # el_list.append(kalman_filter.Sf[2, 0])
    r_list.append(r)
    az_list.append(az)
    el_list.append(el)

plt.figure(figsize=(12, 6))
plt.subplot(facecolor ="white")
#plt.plot(time_list, r_list, color='green', linewidth=2)
plt.scatter(time_list, r_list, label='filtered range (code)', color='green', marker='*')
#plt.plot(filtered_values_csv[:, 0], filtered_values_csv[:, 1], color='red', linestyle='--')
plt.scatter(filtered_values_csv[:, 0], result, label='filtered range (track id 31)', color='red', marker='*')
#plt.scatter(closest_measurement[3], closest_measurement[0], label='associated range', color='blue', marker='*')
plt.xlabel('Time', color='black')
plt.ylabel('Range (r)', color='black')
plt.title('Range vs. Time', color='black')
plt.grid(color='gray', linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()

# Plot azimuth (az) vs. time
plt.figure(figsize=(12, 6))
plt.subplot(facecolor ="white")
#plt.plot(time_list, az_list, color='green', linewidth=2)
plt.scatter(time_list, az_list, label='filtered azimuth (code)', color='green', marker='*')
#plt.plot(filtered_values_csv[:, 0], filtered_values_csv[:, 2], color='red', marker='*', linestyle='--')
plt.scatter(filtered_values_csv[:, 0], A[1], label='filtered azimuth (track id 31)', color='red', marker='*')
#plt.scatter(closest_measurement[3], closest_measurement[1], label='associated azimuth', color='blue', marker='*')
plt.xlabel('Time', color='black')
plt.ylabel('Azimuth (az)', color='black')
plt.title('Azimuth vs. Time', color='black')
plt.grid(color='gray', linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()

# Plot elevation (el) vs. time
plt.figure(figsize=(12, 6))
plt.subplot(facecolor ="white")
#plt.plot(time_list, el_list, color='green', linewidth=2)
plt.scatter(time_list, el_list, label='filtered elevation (code)', color='green', marker='*')
#plt.plot(filtered_values_csv[:, 0], filtered_values_csv[:, 3], color='red')
plt.scatter(filtered_values_csv[:, 0], A[2], label='filtered elevation (track id 31)', color='red', marker='*')
#plt.scatter(closest_measurement[3], closest_measurement[2], label='associated elevation', color='blue', marker='x')
plt.xlabel('Time', color='black')
plt.ylabel('Elevation (el)', color='black')
plt.title('Elevation vs. Time', color='black')
plt.grid(color='gray', linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()
