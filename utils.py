import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import yaml
from pathlib import Path
from datetime import datetime

def setup_logging(log_dir):
    """Setup logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(
                log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            ),
            logging.StreamHandler()
        ]
    )

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return Config(config)

class Config:
    """Configuration class to store training parameters"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)
        
        # Create directories
        self.model_dir = Path(self.model_dir)
        self.log_dir = Path(self.log_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

def save_checkpoint(model, filepath):
    """Save model checkpoint"""
    torch.save({
        'model_state_dict': model.state_dict(),
    }, filepath)
    logging.info(f"Model saved to {filepath}")

def load_checkpoint(model, filepath, device):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Model loaded from {filepath}")
    return model

def plot_metrics(train_loss, test_rmse, test_mae, save_path=None):
    """Plot training metrics"""
    plt.style.use('seaborn')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1.plot(train_loss, 'b-', label='Training Loss', linewidth=2)
    ax1.set_title('Training Loss over Epochs', fontsize=14, pad=15)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=12)
    
    ax2.plot(test_rmse, 'r-', label='Test RMSE', linewidth=2)
    ax2.plot(test_mae, 'g-', label='Test MAE', linewidth=2)
    ax2.set_title('Test Metrics over Epochs', fontsize=14, pad=15)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Error', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=12)
    
    min_train_loss = min(train_loss)
    min_test_rmse = min(test_rmse)
    min_test_mae = min(test_mae)
    
    stats_text = (
        f'Best Metrics:\n'
        f'Train Loss: {min_train_loss:.4f}\n'
        f'Test RMSE: {min_test_rmse:.4f}\n'
        f'Test MAE: {min_test_mae:.4f}'
    )
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logging.info(f"Metrics plot saved to {save_path}")
    
    plt.show()

def print_training_summary(train_loss, test_rmse, test_mae):
    """Print training summary statistics"""
    print("\nTraining Analysis:")
    print(f"Initial Train Loss: {train_loss[0]:.4f}")
    print(f"Final Train Loss: {train_loss[-1]:.4f}")
    print(f"Loss Improvement: {((train_loss[0] - train_loss[-1]) / train_loss[0] * 100):.2f}%")
    print(f"\nBest Test RMSE: {min(test_rmse):.4f} at Epoch {np.argmin(test_rmse)+1}")
    print(f"Best Test MAE: {min(test_mae):.4f} at Epoch {np.argmin(test_mae)+1}")