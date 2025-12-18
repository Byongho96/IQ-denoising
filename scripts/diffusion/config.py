import numpy as np
import os

class Config:
    # --- Physics Constants ---
    C = 299792458          # Speed of light (m/s)
    FREQ = 2.4e9 + 2.5e6        # BLE frequency (Hz)
    LAMBDA = C / FREQ           # BLE Wavelength (m)
    WAVE_NUMBER = 2 * np.pi / LAMBDA  # Wavenumber (k)
    
    # --- Hardware Geometry (8-element UCA) ---    
    N_ELEMENTS = 8              # Number of antenna elements (8-element UCA)
    N_SOURCES = 1               # Number of signal sources (beacons)
    N_SAMPLES_PER_ANT = 3         # Number of samples per antenna in the switching pattern

    PHI_N = 2 * np.pi * np.arange(N_ELEMENTS) / N_ELEMENTS  # Antenna angles in radians
    D_SPACING = 0.0456     # Antenna spacing in meters (4.56 cm)
    RADIUS = D_SPACING / (2 * np.sin(np.pi / N_ELEMENTS)) # isosceles triangle : R = d / (2 * sin(pi / N))
    # RADIUS = 0.065               # Alternative fixed radius (6.5 cm) based on literature

    TX_POSITIONS = {
        2: (0, 4),
        5: (4, 4),
        4: (0, 0),
        1: (4, 0)
    }

    # --- Data Specs ---
    SEQ_LENGTH = 111
    
    # --- Training Hyperparameters ---
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    EPOCHS = 20
    Val_SPLIT = 0.2          # train/validation split ratio
    
    # --- Paths ---
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = '../../data/'     # Path to CSV data files
    SAVE_DIR = './checkpoints/'  # Path to save model checkpoints

    @staticmethod
    def get_data_dir():
        return os.path.join(Config.CURRENT_DIR, Config.DATA_DIR)
    
    @staticmethod
    def get_save_dir():
        return os.path.join(Config.CURRENT_DIR, Config.SAVE_DIR)
    
# Create directories if they don't exist
os.makedirs(Config.SAVE_DIR, exist_ok=True)