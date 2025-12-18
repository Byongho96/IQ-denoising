import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
from model import DiffusionUNet1D
from diffusion import Diffusion
from config import Config
from utils import load_all_csvs_and_gt_aoa
from dataset import IQDenoisingDataset

# 위에서 정의한 클래스들 임포트 가정
# from diffusion_model import DiffusionUNet1D
# from diffusion_utils import Diffusion

def train_diffusion():
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Data
    df, aoa = load_all_csvs_and_gt_aoa(Config.get_data_dir())
    dataset = IQDenoisingDataset(df, aoa.values, augment=True) # Augmentation 권장
    
    val_size = int(len(dataset) * Config.Val_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Initialize Model & Diffusion
    model = DiffusionUNet1D(in_channels=4, out_channels=2).to(device)
    diffusion = Diffusion(noise_steps=1000, device=device)
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    mse = torch.nn.MSELoss()

    print("Start Diffusion Training...")
    
    best_loss = float('inf')
    
    for epoch in range(Config.EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        train_loss = 0.0
        
        for noisy_input, clean_target in pbar:
            # noisy_input: Condition (Measurement)
            # clean_target: Ground Truth (Start for x0)
            noisy_input = noisy_input.to(device)
            clean_target = clean_target.to(device)
            
            # 1. Sample Timesteps t
            t = diffusion.sample_timesteps(clean_target.shape[0])
            
            # 2. Add Noise to Clean Target (Forward Process) -> x_t
            x_t, noise = diffusion.noise_images(clean_target, t)
            
            # 3. Model Prediction (Input: x_t concat with Condition)
            # We want the model to look at the noisy measurement and the current latent
            # to predict the noise added to the clean signal.
            model_input = torch.cat([x_t, noisy_input], dim=1) # (B, 4, L)
            predicted_noise = model(model_input, t)
            
            # 4. Loss (Simple MSE on Noise)
            loss = mse(noise, predicted_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix(MSE=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        
        # Validation (Loss Check Only, Generation is slow)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy_input, clean_target in val_loader:
                noisy_input = noisy_input.to(device)
                clean_target = clean_target.to(device)
                
                t = diffusion.sample_timesteps(clean_target.shape[0])
                x_t, noise = diffusion.noise_images(clean_target, t)
                
                model_input = torch.cat([x_t, noisy_input], dim=1)
                predicted_noise = model(model_input, t)
                
                loss = mse(noise, predicted_noise)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        
        # Save Best Model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), f"{Config.SAVE_DIR}/best_diffusion_model.pth")
            print("Model Saved!")

if __name__ == "__main__":
    train_diffusion()