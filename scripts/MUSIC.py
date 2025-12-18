import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA

# ---------------------------------------------------------
# Parameters and Constants
# ---------------------------------------------------------
C = 299792458               # Speed of light (m/s)
FREQ = 2.4e9 + 2.5e6        # BLE frequency (Hz)
LAMBDA = C / FREQ           # BLE Wavelength (m)
WAVE_NUMBER = 2 * np.pi / LAMBDA  # Wavenumber (k)

N_ELEMENTS = 8              # Number of antenna elements (8-element UCA)
N_SOURCES = 1               # Number of signal sources (beacons)

D_SPACING = 0.0456     # Antenna spacing in meters (4.56 cm)
RADIUS = D_SPACING / (2 * np.sin(np.pi / N_ELEMENTS)) # isosceles triangle : R = d / (2 * sin(pi / N))
# RADIUS = 0.065               # Alternative fixed radius (6.5 cm) based on literature

# ---------------------------------------------------------
# Load CSV file as DataFrame
# ---------------------------------------------------------
def load_csv(filepath):
    print(f"Loading {filepath}...")
    
    try:
        df = pd.read_csv(filepath, header=None)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None

    return df

# ---------------------------------------------------------
# MUSIC Alogirithm Implementation
# ---------------------------------------------------------
def estimate_aoa_music(phase_data_row):
    complex_signal = np.exp(1j * np.deg2rad(phase_data_row)) # Euler : e^(j*theta) = cos(theta) + j*sin(theta)

    # Average phase per slot (3 samples per slot)
    # .reshape(-1, 3) : (,111) -> (37, 3)
    # .mean(axis=1) : (37, 3) -> (37,)
    average_signal = complex_signal.reshape(-1, 3).mean(axis=1)

    # 37 // 8 = 4 (full cycles) 
    cycles = len(average_signal) // N_ELEMENTS
    
    # Reshape to (Antennas x Snapshots)
    # (37,) -> (32,) -> (4, 8) -> Transpose -> (8, 4)
    snapshots = average_signal[:cycles * N_ELEMENTS].reshape(cycles, N_ELEMENTS).T

    # Covariance Matrix R (N x N)
    R = snapshots @ snapshots.conj().T
    R = R / cycles  # Normalize by number of snapshots

    # Eigen Decomposition (Eigenvalues and Eigenvectors)
    w, v = LA.eigh(R) 
    
    # Sort Eigenvalues and Eigenvectors in descending order
    idx = w.argsort()[::-1]
    w = w[idx] 
    v = v[:, idx]

    # Noise Subspace (En)
    En = v[:, N_SOURCES:]

    # Physical angles for UCA
    # The reference angle for calculating steering vectors(phase shifts) of each element
    phi_n = 2 * np.pi * np.arange(N_ELEMENTS) / N_ELEMENTS

    spectrum = []

    for theta in np.arange(0, 361, 1):
        theta_rad = np.deg2rad(theta)
        
        # UCA Steering Vector : (8, )
        # a(theta) = exp(j * k * R * cos(theta - phi_n))
        a = np.exp(1j * WAVE_NUMBER * RADIUS * np.cos(theta_rad - phi_n))
        
        # MUSIC Spectrum: P = 1 / (a^H * En * En^H * a)
        # Project steering vector onto noise subspace
        metric = np.abs(a.conj().T @ En @ En.conj().T @ a)
        spectrum.append(1.0 / metric)

    spectrum = np.array(spectrum)
    spectrum = spectrum / np.max(spectrum)  # Normalize spectrum for visualization
    
    return spectrum


# ---------------------------------------------------------
# Visualize Results
# ---------------------------------------------------------
def visualize_results(results_df):
    """
    Required Columns in results_df: ['timestamp', 'beacon_id', 'aoa']
    """
    plt.figure(figsize=(12, 6))

    beacons = results_df['beacon_id'].unique()
    
    for bid in beacons:
        beacon_df = results_df[results_df['beacon_id'] == bid]
        plt.scatter(beacon_df['timestamp'], beacon_df['aoa'], label=f'Beacon {bid}', alpha=0.6, s=15)
        
        # Print mean AoA for each beacon
        mean_aoa = beacon_df['aoa'].mean()
        print(f"Beacon {bid}: Mean AoA = {mean_aoa:.2f} degrees")

    plt.title(f'AoA Estimation over Time (MUSIC Algorithm) - 8 Element UCA')
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Estimated Angle of Arrival (Degree)')
    plt.ylim(-180, 180)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import os
    
    csv_name = '../denoised/mapSmall_x0y1_diffusion.csv'
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(cur_dir, csv_name)
    
    df = load_csv(csv_path)
    if df is None:
        exit(1)

    results = []
    for index, row in df.iterrows():
        timestamp = row[0]
        beacon_id = row[1].astype(int)
        phase_data = row[2:].values

        spectrum = estimate_aoa_music(phase_data)
        aoa = np.argmax(spectrum)
        if aoa > 180:
            aoa -= 360  # Convert to -180 ~ 180 degree range

        results.append({
            'timestamp': timestamp,
            'beacon_id': beacon_id,
            'aoa': aoa
        })

    results_df = pd.DataFrame(results)
    visualize_results(results_df)