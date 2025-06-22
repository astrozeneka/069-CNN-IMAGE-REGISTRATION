
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tifffile
import os
from pathlib import Path
import matplotlib.pyplot as plt
from AffineNet import AffineNet

class AffineDataset(Dataset):
    """ Custom dataset for loading tile pairs and affine parameter """

    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform

        # Load affine parameters
        self.affine_params = np.load(self.data_dir / "affine_parameters.npy")
        self.num_samples = len(self.affine_params)

        # Verify all required files exist
        for i in range(self.num_samples):
            original_path = self.data_dir / f"original_tile_{i:02d}.tif"
            transformed_path = self.data_dir / f"transformed_tile_{i:02d}.tif"

            if not original_path.exists():
                raise FileNotFoundError(f"Missing original tile: {original_path}")
            if not transformed_path.exists():
                raise FileNotFoundError(f"Missing transformed tile: {transformed_path}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Load images
        original_path = self.data_dir / f"original_tile_{idx:02d}.tif"
        transformed_path = self.data_dir / f"transformed_tile_{idx:02d}.tif"

        # Open using tifffile library
        original_img = tifffile.imread(original_path)
        transformed_img = tifffile.imread(transformed_path)

        # Normalize
        original_img = self._normalize_image(original_img)
        transformed_img = self._normalize_image(transformed_img)

        # Convert to tensors and add channel dimension
        original_tensor = torch.tensor(original_img).unsqueeze(0).float()  # Shape: (1, H, W)
        transformed_tensor = torch.tensor(transformed_img).unsqueeze(0).float()

        # Get affine parameters
        affine_params = torch.from_numpy(self.affine_params[idx]).float()  # Shape: (6,)

        return original_tensor, transformed_tensor, affine_params

    def _normalize_image(self, img):
        """ Normalize image to range [0, 1] """
        img = img.astype(np.float32)
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        return img

class AffineTrainer:
    """ Training class for Affinenet """

    def __init__(self, model, device, learning_rate=1e-4):
        self.model = model.to(device)
        self.device = device

        # Loss function and optimizer
        self.criterion = nn.MSELoss() # other options: nn.L1Loss(), nn.SmoothL1Loss(), etc.
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.8)

        # Training history
        self.train_losses = []
        self.best_loss = float('inf')

    def train_epoch(self, dataloader):
        """ Train for one epoch """
        self.model.train()
        epoch_loss = 0.0

        for batch_idx, (original, transformed, target_params) in enumerate(dataloader):
            # Move data to device
            original = original.to(self.device)
            transformed = transformed.to(self.device)
            target_params = target_params.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predicted_params = self.model(original, transformed)

            # Calculate loss
            loss = self.criterion(predicted_params, target_params)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            # Print batch details
            print(f" Batch {batch_idx + 1}: Loss = {loss.item():.6f}")
            print(f"    Predicted: {predicted_params[0].detach().cpu().numpy()}")
            print(f"    Target:    {target_params[0].cpu().numpy()}")

        return epoch_loss / len(dataloader)

    def evaluate(self, dataloader):
        """Evaluate model on validation data"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for original, transformed, target_params in dataloader:
                original = original.to(self.device)
                transformed = transformed.to(self.device)
                target_params = target_params.to(self.device)

                predicted_params = self.model(original, transformed)
                loss = self.criterion(predicted_params, target_params)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def train(self, train_loader, val_loader=None, num_epochs=100, save_dir="models"):
        """ Full training loop calling 'train_epoch' above (not used)"""
        os.makedirs(save_dir, exist_ok=True)

        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        print(f"Training samples: {len(train_loader.dataset)}")

        # if val_loader is provided
        if val_loader:
            print(f"Validation samples: {len(val_loader.dataset)}")

        val_losses = []

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validate if validation loader provider
            val_loss = None
            if val_loader:
                val_loss = self.evaluate(val_loader)
                val_losses.append(val_loss)


            # Update learning rate
            self.scheduler.step()

            # Print epoch summary
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch { epoch + 1 } Summary: ")
            print(f"    Training Loss: {train_loss:.6f}")
            if val_loss is not None:
                print(f"    Validation Loss: {val_loss:.6f}")
            print(f"    Learning Rate: {current_lr:.6f}")

            # Save the best model based on the validation loss (if available) or training loss
            current_loss = val_loss if val_loss is not None else train_loss
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.save_model(os.path.join(save_dir, "best_model.pth"))
                loss_type = "validation" if val_loss is not None else "training"
                print(f" New Best model saved! {loss_type.capitalize()} Loss: {current_loss:.6f}")

            # Save checkpoint every 25 epochs
            if (epoch + 1) % 25 == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pth")
                self.save_model(checkpoint_path)
                print(f"Checkpoint saved at: {checkpoint_path}")

            # Save best model (validation is used instead of training step for saving)
            #if train_loss < self.best_loss:
            #    self.best_loss = train_loss
            #    self.save_model(os.path.join(save_dir, "best_model.pth"))
            #    print(f" New Best model saved! Loss: {train_loss:.6f}")

        print(f"\nTraining completed! Best loss: {self.best_loss:.6f}")
        return self.train_losses, val_losses

    def save_model(self, path):
        """ Save model state """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'train_losses': self.train_losses
        }, path)

    def load_model(self, path):
        """Load model state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_loss = checkpoint['best_loss']
        self.train_losses = checkpoint['train_losses']

def plot_training_curve(losses, save_path="training_curve.png"):
    """ Plot and save training curve """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', linewidth=2, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('AffineNet Training Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training curve saved to: {save_path}")

if __name__ == '__main__':

    # Variables
    DATA_DIR = "training_data"
    BATCH_SIZE = 8
    NUM_EPOCHS = 4
    LEARNING_RAGE = 1e-4
    TRAIN_SPLIT = 0.8  # 80% for training, 20% for validation

    # Device setup (cuda doesn't work on mac)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataset and dataloader
    print("Loading dataset...")
    full_dataset = AffineDataset(DATA_DIR)
    print(f"Dataset size: {len(full_dataset)} samples")

    # Split dataset into training and validation sets
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # Split dataset into train and validation
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator = torch.Generator().manual_seed(42)  # For reproducibility
    )

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # The dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Create model
    print("Initializing model...")
    model = AffineNet(image_size=(128, 128))

    # Create trainer with validation capability
    trainer = AffineTrainer(model, device, learning_rate=LEARNING_RAGE)

    # Train model with validation in one go
    train_losses, val_losses = trainer.train(train_loader, val_loader, num_epochs=NUM_EPOCHS)

    # Plot training curves
    plt.figure(figsize=(12, 5))

    # Training curve
    # TODO: SHOULD BE REFACTORED TO A NEW FUNCTION
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', linewidth=2, label='Training Loss')
    if val_losses:
        plt.plot(val_losses, 'r-', linewidth=2, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Loss comparison (if validation available)
    if val_losses:
        plt.subplot(1, 2, 2)
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', alpha=0.7, label='Training')
        plt.plot(epochs, val_losses, 'r-', alpha=0.7, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale for better visualization

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Training curves saved to: training_curves.png")


    # ====
    # Detailed evaluation on validation set
    print("\nEvaluating model on validation data:")
    model.eval()
    total_error = np.zeros(6)
    sample_count = 0

    with torch.no_grad():
        for batch_idx, (original, transformed, target_params) in enumerate(val_loader):
            original = original.to(device)
            transformed = transformed.to(device)
            target_params = target_params.to(device)

            predicted_params = model(original, transformed)

            # Calculate errors for statistics
            for j in range(len(predicted_params)):
                pred = predicted_params[j].cpu().numpy()
                target = target_params[j].cpu().numpy()
                error = np.abs(pred - target)
                total_error += error
                sample_count += 1

                if batch_idx == 0:  # Show details for first batch only
                    print(f"\nValidation Sample {j + 1}:")
                    print(f"  Predicted: [{', '.join([f'{x:.4f}' for x in pred])}]")
                    print(f"  Target:    [{', '.join([f'{x:.4f}' for x in target])}]")
                    print(f"  Error:     [{', '.join([f'{x:.4f}' for x in error])}]")

    # Print average errors
    avg_error = total_error / sample_count
    param_names = ['scale_x', 'shear_x', 'trans_x', 'shear_y', 'scale_y', 'trans_y']

    print(f"\nAverage Absolute Errors across {sample_count} validation samples:")
    for i, (name, error) in enumerate(zip(param_names, avg_error)):
        print(f"  {name}: {error:.4f}")
    print(f"  Overall Mean Error: {np.mean(avg_error):.4f}")

    # Test model on a few training samples for comparison
    print("\nEvaluating model on training data (first batch):")
    model.eval()
    with torch.no_grad():
        for i, (original, transformed, target_params) in enumerate(train_loader):
            if i > 0:  # Only show first batch
                break

            original = original.to(device)
            transformed = transformed.to(device)
            target_params = target_params.to(device)

            predicted_params = model(original, transformed)

            for j in range(min(3, len(predicted_params))):  # Show max 3 samples
                pred = predicted_params[j].cpu().numpy()
                target = target_params[j].cpu().numpy()
                error = np.abs(pred - target)

                print(f"\nTraining Sample {j + 1}:")
                print(f"  Predicted: [{', '.join([f'{x:.4f}' for x in pred])}]")
                print(f"  Target:    [{', '.join([f'{x:.4f}' for x in target])}]")
                print(f"  Error:     [{', '.join([f'{x:.4f}' for x in error])}]")

    print("\nTraining completed successfully!")
    print("Model saved in 'models/' directory")
    print("Best model based on validation loss" if val_losses else "Best model based on training loss")