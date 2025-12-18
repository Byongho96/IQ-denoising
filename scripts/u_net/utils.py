import pandas as pd
import numpy as np
import glob
import os
import re
from config import Config

def load_all_csvs_and_gt_aoa(data_dir):
    """
    Load all CSV files from the specified directory and GT AoA based on Tx/Rx positions.
    """
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    print(f"Found {len(all_files)} CSV files.")
    
    if len(all_files) == 0:
        raise ValueError("No CSV files found in the specified directory.")
    
    row_list = []
    aoa_list = []
    for filename in all_files:
        df = pd.read_csv(filename, header=None)

        # Parse receiver coordinates from filename
        match = re.search(r'x(\d+)y(\d+)', filename)
        if not match:
            print(f"[Warning] Could not find coordinates in filename: {filename}. Skipping.")
            continue
        rx_x = float(match.group(1))
        rx_y = 4 - float(match.group(2))  # Y-axis inversion as

        for _, row in df.iterrows():
            beacon_id = int(row.iloc[1])
            if beacon_id in Config.TX_POSITIONS:
                tx_x, tx_y = Config.TX_POSITIONS[beacon_id]
                delta_x = abs(tx_x - rx_x)
                delta_y = abs(tx_y - rx_y)
                aoa_rad = np.arctan2(delta_y, delta_x)
                aoa_deg = np.rad2deg(aoa_rad)
                
                # Adjust AoA based on beacon position
                if beacon_id == 2:      # Tx at (0,4)
                    aoa_deg = 90 + aoa_deg
                elif beacon_id == 5:    # Tx at (4,4)
                    aoa_deg = 270 - aoa_deg
                elif beacon_id == 4:    # Tx at (0,0)
                    aoa_deg = 90 - aoa_deg
                elif beacon_id == 1:    # Tx at (4,0)
                    aoa_deg = 270 + aoa_deg
                
                row_list.append(row)
                aoa_list.append(aoa_deg)
            else:
                print(f"[Warning] Beacon ID {beacon_id} not found in TX_POSITIONS.")
                
    combined_df = pd.DataFrame(row_list)
    aoa_df = pd.DataFrame(aoa_list)
    return combined_df, aoa_df
