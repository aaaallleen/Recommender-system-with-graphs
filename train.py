import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import logging
import argparse
from pathlib import Path

import numpy as np

from models import RatingPredictionGNN
from dataloader import DataPreprocessor, get_dataloaders
from utils import plot_metrics, setup_logging, save_checkpoint, load_checkpoint, load_config
from sklearn.metrics import mean_squared_error, mean_absolute_error

class RatingTrainer:
    def __init__(
        self,
        model,
        train_df,
        test_df,
        edge_index,
        n_users,
        device,
        batch_size=1024,
        lr=0.001,
        weight_decay=1e-4
    ):
        self.model = model
        self.train_df = train_df
        self.test_df = test_df
        self.edge_index = edge_index
        self.n_users = n_users
        self.device = device
        self.batch_size = batch_size
        
        # Create data loaders
        self.train_loader, self.test_loader = get_dataloaders(
            train_df, 
            test_df, 
            batch_size
        )
        
        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            patience=5, 
            factor=0.5, 
            verbose=True
        )
        
        # Keep track of best model
        self.best_rmse = float('inf')
        self.best_state_dict = None
        
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch_users, batch_items, batch_ratings in tqdm(self.train_loader, desc="Training"):
            batch_users = batch_users.to(self.device)
            batch_items = batch_items.to(self.device)
            batch_ratings = batch_ratings.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Get embeddings
            _, final_embeddings = self.model(self.edge_index)
            
            # Predict ratings
            predicted_ratings = self.model.predict_ratings(
                batch_users,
                batch_items,
                final_embeddings
            )
            
            # Calculate loss
            loss = torch.nn.functional.mse_loss(predicted_ratings, batch_ratings)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        predictions = []
        actuals = []
        
        for batch_users, batch_items, batch_ratings in tqdm(self.test_loader, desc="Evaluating"):
            batch_users = batch_users.to(self.device)
            batch_items = batch_items.to(self.device)
            
            _, final_embeddings = self.model(self.edge_index)
            pred_ratings = self.model.predict_ratings(
                batch_users,
                batch_items,
                final_embeddings
            )
            
            predictions.extend(pred_ratings.cpu().numpy())
            actuals.extend(batch_ratings.numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        
        return rmse, mae
    
    def train(self, num_epochs=50, early_stopping_patience=10):
        no_improve = 0
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch()
            
            # Evaluate
            test_rmse, test_mae = self.evaluate()
            
            # Learning rate scheduling
            self.scheduler.step(test_rmse)
            
            # Save best model
            if test_rmse < self.best_rmse:
                self.best_rmse = test_rmse
                self.best_state_dict = self.model.state_dict()
                no_improve = 0
            else:
                no_improve += 1
            
            print(f"Epoch {epoch + 1}:")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Test RMSE: {test_rmse:.4f}")
            print(f"Test MAE: {test_mae:.4f}")
            print("-" * 50)
            
            # Early stopping
            if no_improve >= early_stopping_patience:
                print("Early stopping triggered!")
                break
        
        # Load best model
        self.model.load_state_dict(self.best_state_dict)
        return self.model

class RatingInference:
    def __init__(self, model, edge_index, n_users, device):
        self.model = model
        self.edge_index = edge_index
        self.n_users = n_users
        self.device = device
        
        # Compute embeddings once
        self.model.eval()
        with torch.no_grad():
            _, self.final_embeddings = self.model(self.edge_index)
    
    @torch.no_grad()
    def predict_single_rating(self, user_idx, item_idx):
        """Predict rating for a single user-item pair"""
        user_tensor = torch.LongTensor([user_idx]).to(self.device)
        item_tensor = torch.LongTensor([item_idx]).to(self.device)
        
        pred_rating = self.model.predict_ratings(
            user_tensor,
            item_tensor,
            self.final_embeddings
        )
        
        return pred_rating.item()
    
    @torch.no_grad()
    def predict_top_k_items(self, user_idx, k=10, exclude_rated=True, rated_items=None):
        """Predict top K items for a user"""
        user_tensor = torch.LongTensor([user_idx]).to(self.device)
        
        # Predict ratings for all items
        all_items = torch.arange(self.n_users).to(self.device)
        pred_ratings = self.model.predict_ratings(
            user_tensor.repeat(len(all_items)),
            all_items,
            self.final_embeddings
        )
        
        # Exclude already rated items if requested
        if exclude_rated and rated_items is not None:
            pred_ratings[rated_items] = float('-inf')
        
        # Get top K items
        top_k_ratings, top_k_indices = torch.topk(pred_ratings, k)
        
        return top_k_indices.cpu().numpy(), top_k_ratings.cpu().numpy()


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_data()
        self.setup_model()
        
    def setup_data(self):
        """Initialize data preprocessing and loading"""
        self.preprocessor = DataPreprocessor(self.config.base_path)
        self.user_df, self.book_df, self.ratings_df = self.preprocessor.load_all_data()
        self.train_df, self.test_df = self.preprocessor.prepare_train_test_data()
        
        # Get dataloaders
        self.train_loader, self.test_loader = get_dataloaders(
            self.train_df, 
            self.test_df, 
            self.config.batch_size
        )
        
        # Create edge index
        self.n_users = len(self.preprocessor.user_features_df)
        self.n_books = len(self.preprocessor.book_features_df)
        self.edge_index = self.preprocessor.create_edge_index(
            self.train_df,
            self.n_users,
            self.device
        )

    def setup_model(self):
        """Initialize model, optimizer, and trainer"""
        self.model = RatingPredictionGNN(
            latent_dim=self.config.latent_dim,
            num_layers=self.config.num_layers,
            num_users=self.n_users,
            num_books=self.n_books,
            model=self.config.model_type,
            user_features=self.user_features,
            book_numerical_features=self.book_numerical_features,
            book_genre_features=self.book_genre_features,
            dropout=self.config.dropout
        ).to(self.device)
        
        self.trainer = RatingTrainer(
            model=self.model,
            train_df=self.train_df,
            test_df=self.test_df,
            edge_index=self.edge_index,
            n_users=self.n_users,
            device=self.device,
            batch_size=self.config.batch_size,
            lr=self.config.learning_rate
        )

    def train(self):
        """Main training loop"""
        logging.info("Starting training...")
        trained_model = self.trainer.train(
            num_epochs=self.config.epochs,
            early_stopping_patience=self.config.patience
        )
        
        # Save the trained model
        save_checkpoint(
            trained_model,
            self.config.model_dir / f"{self.config.model_type}_model.pt"
        )
        
        return trained_model

def main():
    parser = argparse.ArgumentParser(description='Train recommendation model')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='path to config file')
    parser.add_argument('--model_type', type=str, default='LightGCN',
                      choices=['LightGCN', 'NGCF'],
                      help='type of model to train')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config.model_type = args.model_type
    
    # Setup logging
    setup_logging(config.log_dir)
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Train model
    trained_model = trainer.train()
    
    # Create inference engine
    inference_engine = RatingInference(
        model=trained_model,
        edge_index=trainer.edge_index,
        n_users=trainer.n_users,
        device=trainer.device
    )
    
    return trained_model, inference_engine

if __name__ == "__main__":
    trained_model, inference_engine = main()