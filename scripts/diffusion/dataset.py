from config import Config
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

class IQDenoisingDataset(Dataset):
    def __init__(self, dataframe, gt_aoas, augment=False, use_cache=True):
        """
        dataframe: Pandas DataFrame containing the dataset : Timestamp, Beacon ID, Phase Data...
        gt_aoas: Ground Truth AoA list corresponding to each row (numpy array recommended)
        augment: Whether to apply data augmentation
        use_cache: Whether to use caching for GT sequences
        """
        self.augment = augment
        self.use_cache = use_cache

        self.k = Config.WAVE_NUMBER
        self.num_cycles = Config.SEQ_LENGTH // Config.N_ELEMENTS
        self.remainder = Config.SEQ_LENGTH % Config.N_ELEMENTS

        # Preprocessed Data
        self.noisy_iq_list = []
        self.clean_iq_list = []

        print("Pre-processing dataset and generating aligned GT...")
        for idx in tqdm(range(len(dataframe))):
            # Input Preprocess
            row = dataframe.iloc[idx].values
            noisy_phase = self._preprocess_input_data(row)
            noisy_iq = np.exp(1j * np.deg2rad(noisy_phase))

            # GT Generation & Alignment
            gt_aoa = gt_aoas[idx]
            base_clean_iq = self._get_gt_sequence(gt_aoa)
            aligned_clean_iq = self._align_phase(noisy_iq, base_clean_iq)

            self.noisy_iq_list.append(noisy_iq)
            self.clean_iq_list.append(aligned_clean_iq)

    def __len__(self):
        return len(self.noisy_iq_list)

    def __getitem__(self, idx):
        # Get Preprocessed I/Q Data
        noisy_iq = self.noisy_iq_list[idx]
        clean_iq = self.clean_iq_list[idx]
        
        # Augmentation (Phase Rotation)
        if self.augment:
            noisy_iq, clean_iq = self.apply_phase_rotation(noisy_iq, clean_iq)
            
        # Convert to Tensor
        noisy_tensor = self.complex_to_2ch_tensor(noisy_iq)
        clean_tensor = self.complex_to_2ch_tensor(clean_iq)
        
        return noisy_tensor, clean_tensor
    
    def _preprocess_input_data(self, row_data):
        """
        [Timestamp, ID, Phase1, Phase2, ...]
        -> [Phase1, Phase2, ...] (numpy array of CONFIG.SEQ_LENGTH)
        """
        raw_phase = row_data[2:].astype(np.float32)
        
        if len(raw_phase) > Config.SEQ_LENGTH:
            # Truncate if too long
            raw_phase = raw_phase[:Config.SEQ_LENGTH]
        elif len(raw_phase) < Config.SEQ_LENGTH:
            # Pad if too short
            pad_width = Config.SEQ_LENGTH - len(raw_phase)
            raw_phase = np.pad(raw_phase, (0, pad_width), mode='edge')
        
        return raw_phase

    def _get_gt_sequence(self, aoa_degree):
        """
        AoA에 해당하는 Ideal I/Q 시퀀스를 생성하거나 캐시에서 반환
        """
        # Spatial phase difference
        # UCA Steering vector : a(theta) = exp(j * k * r * cos(theta - phi))
        aoa_rad = np.deg2rad(aoa_degree)
        spatial_phase = np.exp(1j * self.k * Config.RADIUS * np.cos(aoa_rad - Config.PHI_N))
        
        # Temporal phase difference (2, 3, 4 samples per slot)
        # 250kHz rotate 90degree per 1us
        # Sample 2 @ 1.0us = 90 deg
        # Sample 3 @ 1.5us = 135 deg
        # Sample 4 @ 2.0us = 180 deg
        temporal_phase = np.deg2rad(np.array([90.0, 135.0, 180.0]))
        
        # Spatial + Temporal phase combination
        # Shape: (8, 3)
        combined_phase = spatial_phase[:, np.newaxis] + temporal_phase[np.newaxis, :]
        
        # Flatten to 1D sequence according to antenna switching pattern
        # One cycle block length = 8 antennas * samples per antenna = 24
        one_cycle_block = np.exp(1j * combined_phase).flatten()
        
        num_cycles = Config.SEQ_LENGTH // len(one_cycle_block) # 111 // 24 = 4
        remainder = Config.SEQ_LENGTH % len(one_cycle_block)   # 111 % 24 = 15
        
        full_sequence = np.tile(one_cycle_block, num_cycles)
        if remainder > 0:
            full_sequence = np.concatenate([full_sequence, one_cycle_block[:remainder]])
            
        return full_sequence

    def _align_phase(self, noisy_input, clean_target):
        """
        Align clean_target phase to noisy_input phase
        By using correlation to find optimal [global phase shift].
        """
        # Correlation(Dot Product) = sum(Input * conj(Target))
        correlation = np.sum(noisy_input * np.conj(clean_target))
        
        # Average Phase Difference
        phase_diff = np.angle(correlation)
        
        # Rotate Target (Aligned to Input)
        aligned_target = clean_target * np.exp(1j * phase_diff)
        
        return aligned_target

    def apply_phase_rotation(self, noisy, clean):
        """
        Data Augmentation: Random Phase Rotation
        """
        # Phase Rotation Angle
        rotation_angle = np.random.uniform(-np.pi, np.pi)
        rotator = np.exp(1j * rotation_angle)
        noisy = noisy * rotator
        clean = clean * rotator

        # Add AWGN
        # noise_power = np.random.uniform(0.01, 0.05) # 노이즈 강도 조절 필요
        # complex_noise = (np.random.randn(*noisy.shape) + 1j * np.random.randn(*noisy.shape)) * noise_power
        # noisy = noisy + complex_noise

        return noisy, clean

    def complex_to_2ch_tensor(self, complex_data):
        real = torch.from_numpy(complex_data.real).float()
        imag = torch.from_numpy(complex_data.imag).float()
        return torch.stack([real, imag], dim=0)