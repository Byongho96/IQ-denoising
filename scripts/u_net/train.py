import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm

from config import Config
from utils import load_all_csvs_and_gt_aoa
from dataset import IQDenoisingDataset
from model import ComplexUNet1D
from loss import PhysicsInformedLoss

def main():
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Data with GT AoA
    df, aoa = load_all_csvs_and_gt_aoa(Config.get_data_dir())

    # Create Dataset and DataLoaders
    dataset = IQDenoisingDataset(df, aoa.values, augment=False, use_cache=True)
    
    # Train/Validation Split
    val_size = int(len(dataset) * Config.Val_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True) # Shuffle for training
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Model, Loss, Optimizer
    model = ComplexUNet1D().to(device)
    criterion = PhysicsInformedLoss(lambda_mse=0.3, lambda_cosine=0.5, lambda_cov=0.2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Training Loop
    patience = 5  # Early stopping patience
    counter = 0
    best_loss = float('inf')
    
    print("Start Training...")
    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        for noisy, clean in loop:
            noisy, clean = noisy.to(device), clean.to(device)
            
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                output = model(noisy)
                loss = criterion(output, clean)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        
        # Checkpoint
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), f"{Config.SAVE_DIR}/best_u_net_model.pth")
            print("Model Saved!")
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter} out of {patience}")
            if counter >= patience:
                print("Early Stopping!")
                break

if __name__ == "__main__":
    main()