import numpy as np

class Config:
    C = 3e8 
    FREQ = 2.4e9 + 250e3
    LAMBDA = 0.125
    
    NUM_ANTENNAS = 8
    ANTENNA_SPACING = 0.0456 
    RADIUS = ANTENNA_SPACING / (2 * np.sin(np.pi / NUM_ANTENNAS))

    SEQ_LEN = 111 # 패킷 당 샘플 수
    
    ANTENNA_PATTERN = [0, 1, 2, 3, 4, 5, 6, 7] 
    SAMPLES_PER_ANT = 3 
    
    TX_POSITIONS = {
        1: (4.0, 0.0),
        2: (0.0, 4.0),
        4: (0.0, 0.0),
        5: (4.0, 4.0)
    }

    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 100

    LAMBDA_PHYSICS = 0.5
