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
dt1 = []
dt2 = np.array(dt1)

updatecycle = 50

class CVFilter:
    def __init__(self):
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.Sp = np.zeros((6, 1))
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)
        self.S = np.eye(3)  # Measurement noise covariance
        self.S[0, 0] = 0.1
        self.S[1, 1] = 0.1
        self.S[2, 2] = 0.1
        self.Meas_Time = 0  # Measured time
        self.Z = np.zeros((3, 1))
        self.updatecycle = 50

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time

    def initialize_measurement_for_filtering(self, x, y, z, vx, vy, vz, mt):
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
        Inn = self.Z - np.dot(self.H, self.Sf)  # Calculate innovation directly
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
    el = math.atan(z / np.sqrt(x**2 + y**2)) * 180 / 3.14
    az = math.atan(y / x)

    if x > 0.0:
        az = 3.14 / 2 - az
    else:
        az = 3 * 3.14 / 2 - az

    az = az * 180 / 3.14

    if az < 0.0:
        az = (360 + az)

    if az > 360:
        az = (az - 360)

    return r, az, el

def cart2sph2(x, y, z, filtered_values_csv):
    for i in range(len(filtered_values_csv)):
        r.append(np.sqrt(x[i]**2 + y[i]**2 + z[i]**2))
        el.append(math.atan(z[i] / np.sqrt(x[i]**2 + y[i]**2)) * 180 / 3.14)
        az.append(math.atan(y[i] / x[i]))

        if x[i] > 0.0:
            az[i] = 3.14 / 2 - az[i]
        else:
            az[i] = 3 * 3.14 / 2 - az[i]

        az[i] = az[i] * 180 / 3.14

        if az[i] < 0.0:
            az[i] = (360 + az[i])

        if az[i] > 360:
            az[i] = (az[i] - 360)

    return r, az, el

def read_measurements_from_csv(file_path):
    measurements = []
    meas_tgt_pair = []
    dt1 = []
    updatecycle = 50
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

csv_file_path = 'ttk_84.csv'  # Provide the path to your CSV file
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
        prev_r, prev_az, prev_el = measurements[i-1][:3]
        dt = mt - measurements[i-1][3]
        dt2 = updatecycle  # Initialize dt2
        if dt2 < updatecycle:
            vx = 150
            vy = 150
            vz = 0
            kalman_filter.initialize_filter_state(r, az, el, vx, vy, vz, mt)
    else:
        chi2_threshold = chi2.ppf(0.95, df=3)
        cov_inv = kalman_filter.S

        def mahalanobis_distance(track_pp, report_pos, cov_inv):
            delta = report_pos - track_pp
            delta1 = np.transpose(delta)
            return np.dot((np.dot(delta1, cov_inv)), delta)

        association_list = []
        kalman_filter.predict_step(mt)
        A = np.dot(kalman_filter.H, kalman_filter.Sp)
        tid = np.array([track[0] for track in meas_tgt_pair])
        for tid_index, track in enumerate(meas_tgt_pair):
            for j, report in enumerate(measurements):
                temp = np.array([measurements[j]])
                temp1 = temp.transpose()
                x = temp1[0:3, :]
                y = kalman_filter.Sp[0:3, :]
                distance = mahalanobis_distance(y, x, kalman_filter.S)
                if distance < chi2_threshold:
                    association_list.append((tid_index, j))

        # Print association list
        print(f"\nTime {mt}: Association List:")
        print(association_list)

        clusters = []
        while association_list:
            cluster_tracks = set()
            cluster_reports = set()
            stack = [association_list.pop(0)]
            while stack:
                track_idx, report_idx = stack.pop()
                cluster_tracks.add(track_idx)
                cluster_reports.add(report_idx)
                new_assoc = [(t, r) for t, r in association_list if t == track_idx or r == report_idx]
                for assoc in new_assoc:
                    stack.append(assoc)
                    association_list.remove(assoc)
            clusters.append((cluster_tracks, cluster_reports))

        # Print clusters
        print(f"\nTime {mt}: Clusters:")
        for cluster in clusters:
            print(f"Cluster Tracks: {cluster[0]}, Cluster Reports: {cluster[1]}")

        for cluster_tracks, cluster_reports in clusters:
            m = len(cluster_reports)
            n = len(cluster_tracks)
            if m == 1 and n == 1:
                report = measurements[list(cluster_reports)[0]]
                kalman_filter.Z = np.array([[report[0]], [report[1]], [report[2]]])
                kalman_filter.update_step()
            else:
                num_hypotheses = 2 ** m - 1
                hypotheses = []
                for i in range(num_hypotheses):
                    hypothesis = []
                    for j in range(m):
                        if (i >> j) & 1:
                            hypothesis.append(measurements[list(cluster_reports)[j]])
                    hypotheses.append(hypothesis)

                # Print hypotheses
                print(f"\nTime {mt}: Hypotheses for Cluster (Tracks: {cluster_tracks}, Reports: {cluster_reports}):")
                for hypothesis in hypotheses:
                    print(hypothesis)

                for hypothesis in hypotheses:
                    if len(hypothesis) == 0:
                        continue
                    else:
                        innovation_sum = np.zeros((3, 1))
                        for report in hypothesis:
                            innovation_sum += report
                        innovation_mean = innovation_sum / len(hypothesis)
                        kalman_filter.Z = innovation_mean
                        kalman_filter.update_step()

    time_list.append(mt)
    r_list.append(kalman_filter.Sf[0, 0])
    az_list.append(kalman_filter.Sf[1, 0])
    el_list.append(kalman_filter.Sf[2, 0])

plt.subplot(3, 1, 1)
plt.plot(time_list, r_list)
plt.xlabel('Time')
plt.ylabel('r')

plt.subplot(3, 1, 2)
plt.plot(time_list, az_list)
plt.xlabel('Time')
plt.ylabel('az')

plt.subplot(3, 1, 3)
plt.plot(time_list, el_list)
plt.xlabel('Time')
plt.ylabel('el')

plt.tight_layout()
plt.show()
