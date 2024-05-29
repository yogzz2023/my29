import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2

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
        self.Z = np.array([[x], [y], [z], [vx], [vy], [vz]])
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
        Inn = self.Z[:3] - np.dot(self.H, self.Sf)
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

def read_measurements_from_csv(file_path):
    measurements = []
    meas_tgt_pair = []
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
            measurements.append((r, az, el, mt))
    return measurements, meas_tgt_pair

def process_measurements(measurements, kalman_filter):
    time_list, r_list, az_list, el_list = [], [], [], []
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
            kalman_filter.predict_step(mt)
            predicted_pos = np.dot(kalman_filter.H, kalman_filter.Sp)
            cov_inv = np.linalg.inv(kalman_filter.S)

            def mahalanobis_distance(track_pp, report_pos, cov_inv):
                delta = report_pos - track_pp
                return np.dot(delta.T, np.dot(cov_inv, delta)).item()

            group_associations = []
            for tid, track in enumerate(meas_tgt_pair):
                T = np.array([track[1], track[2], track[3]]).reshape((3, 1))
                distance = mahalanobis_distance(predicted_pos, T, cov_inv)
                if distance <= chi2_threshold:
                    group_associations.append(tid)

            clusters = {}
            for tid in group_associations:
                track = meas_tgt_pair[tid]
                cluster_key = (track[1], track[2], track[3])
                if cluster_key not in clusters:
                    clusters[cluster_key] = []
                clusters[cluster_key].append(tid)

            hypotheses = []
            for cluster_key, cluster_members in clusters.items():
                prob = 1 / len(cluster_members)
                for member in cluster_members:
                    hypotheses.append((member, prob))

            probabilities = np.array([hypothesis[1] for hypothesis in hypotheses])
            probabilities /= probabilities.sum()
            associated_measurements = {measurement_id: prob for (measurement_id, prob) in hypotheses}

            kalman_filter.Z = np.array([r, az, el]).reshape((3, 1))
            kalman_filter.update_step()

        time_list.append(mt)
        r_list.append(r)
        az_list.append(az)
        el_list.append(el)

    return time_list, r_list, az_list, el_list

def plot_results(time_list, r_list, az_list, el_list, filtered_values_csv):
    plt.figure(figsize=(12, 6))
    plt.subplot(facecolor="white")
    plt.scatter(time_list, r_list, label='Filtered range (code)', color='green', marker='*')
    plt.scatter(filtered_values_csv[:, 0], filtered_values_csv[:, 1], label='Filtered range (track id 31)', color='red', marker='*')
    plt.xlabel('Time', color='black')
    plt.ylabel('Range (r)', color='black')
    plt.title('Range vs. Time', color='black')
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.subplot(facecolor="white")
    plt.scatter(time_list, az_list, label='Filtered azimuth (code)', color='green', marker='*')
    plt.scatter(filtered_values_csv[:, 0], filtered_values_csv[:, 2], label='Filtered azimuth (track id 31)', color='red', marker='*')
    plt.xlabel('Time', color='black')
    plt.ylabel('Azimuth (az)', color='black')
    plt.title('Azimuth vs. Time', color='black')
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.subplot(facecolor="white")
    plt.scatter(time_list, el_list, label='Filtered elevation (code)', color='green', marker='*')
    plt.scatter(filtered_values_csv[:, 0], filtered_values_csv[:, 3], label='Filtered elevation (track id 31)', color='red', marker='*')
    plt.xlabel('Time', color='black')
    plt.ylabel('Elevation (el)', color='black')
    plt.title('Elevation vs. Time', color='black')
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Main code execution
kalman_filter = CVFilter()
csv_file_path = 'ttk_84.csv'
measurements, meas_tgt_pair = read_measurements_from_csv(csv_file_path)

csv_file_predicted = "ttk_84.csv"
df_predicted = pd.read_csv(csv_file_predicted)
filtered_values_csv = df_predicted[['F_TIM', 'F_X', 'F_Y', 'F_Z']].values

time_list, r_list, az_list, el_list = process_measurements(measurements, kalman_filter)
plot_results(time_list, r_list, az_list, el_list, filtered_values_csv)
