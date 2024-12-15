import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset as TensorDataset
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split

class RatingDataset(Dataset):
    """Rating Dataset for training and evaluation"""
    def __init__(self, user_ids, item_ids, ratings):
        self.user_ids = torch.LongTensor(user_ids)
        self.item_ids = torch.LongTensor(item_ids)
        self.ratings = torch.FloatTensor(ratings)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return (
            self.user_ids[idx],
            self.item_ids[idx],
            self.ratings[idx]
        )

def load_user_stats(filepath):
    """Load user statistics from JSON file"""
    with open(filepath, 'r') as f:
        user_stats = json.load(f)
    
    user_features_df = pd.DataFrame.from_dict(user_stats, orient='index')
    user_features_df = user_features_df.reset_index().rename(columns={'index': 'user_id'})
    
    return user_features_df

def load_book_data(filepath):
    """Load and preprocess book features"""
    book_features_df = pd.read_csv(filepath).fillna(0)
    
    book_features_df.drop(columns=[
        'title', 'publication_year', 'ratings_count', 'text_reviews_count'
    ], inplace=True, axis=1)
    
    return book_features_df

def load_ratings(filepath):
    """Load ratings data"""
    ratings_df = pd.read_csv(filepath).fillna(0)
    ratings_df.drop(columns=['review_text', 'n_votes'], inplace=True, axis=1)
    return ratings_df

def create_dataloader(data, batch_size, shuffle=True):
    """Create DataLoader for training/evaluation"""
    users = torch.LongTensor(data['user_id_idx'].values)
    items = torch.LongTensor(data['book_id_idx'].values)
    ratings = torch.FloatTensor(data['rating'].values.astype(np.float32))
    
    dataset = RatingDataset(users, items, ratings)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def process_features(user_features_df, book_features_df, device):
    """Process user and book features"""
    user_feature_cols = ['mean', 'variance']
    book_numerical_cols = ['num_pages', 'average_rating', 'rating_variance']
    genre_cols = [
        'Children', 'Biography', 'Comics', 'Fantasy Paranormal',
        'Mystery Thriller Crime', 'Poetry', 'Young Adult', 'Romance'
    ]
    
    user_features = torch.FloatTensor(
        user_features_df[user_feature_cols].values
    ).to(device)

    book_numerical_features = torch.FloatTensor(
        book_features_df[book_numerical_cols].values
    ).to(device)

    book_genre_features = torch.FloatTensor(
        book_features_df[genre_cols].values
    ).to(device)
    
    return user_features, book_numerical_features, book_genre_features

class DataPreprocessor:
    """Class to handle all data preprocessing steps"""
    def __init__(self, base_path):
        self.base_path = base_path
        self.le_user = pp.LabelEncoder()
        self.le_item = pp.LabelEncoder()

    def load_all_data(self):
        """Load all required data"""
        # Load user stats
        self.user_features_df = load_user_stats(f"{self.base_path}user_stats.json")
        
        # Load book features
        self.book_features_df = load_book_data(f"{self.base_path}books_data.csv")
        
        # Load ratings
        self.ratings_df = load_ratings(f"{self.base_path}dataset.csv")
        
        return self.user_features_df, self.book_features_df, self.ratings_df

    def prepare_train_test_data(self, test_size=0.1, random_state=42):
        """Prepare training and test datasets"""
        # Split train/test
        train, test = train_test_split(
            self.ratings_df.values, 
            test_size=test_size, 
            random_state=random_state
        )
        
        train_df = pd.DataFrame(train, columns=self.ratings_df.columns)
        test_df = pd.DataFrame(test, columns=self.ratings_df.columns)

        train_df['user_id_idx'] = self.le_user.fit_transform(train_df['user_id'].values)
        train_df['book_id_idx'] = self.le_item.fit_transform(train_df['book_id'].values)

        train_user_ids = train_df['user_id'].unique()
        train_book_ids = train_df['book_id'].unique()

        self.user_features_df = self.user_features_df[
            self.user_features_df['user_id'].isin(train_user_ids)
        ]
        self.book_features_df = self.book_features_df[
            self.book_features_df['book_id'].isin(train_book_ids)
        ]

        test_df = test_df[
            (test_df['user_id'].isin(train_user_ids)) & 
            (test_df['book_id'].isin(train_book_ids))
        ]

        self.user_features_df['user_id_idx'] = self.le_user.transform(
            self.user_features_df['user_id'].values
        )
        self.book_features_df['book_id_idx'] = self.le_item.transform(
            self.book_features_df['book_id'].values
        )
        test_df['user_id_idx'] = self.le_user.transform(test_df['user_id'].values)
        test_df['book_id_idx'] = self.le_item.transform(test_df['book_id'].values)

        return train_df, test_df

    def create_edge_index(self, train_df, n_users, device):
        """Create edge index for graph"""
        u_t = torch.FloatTensor(train_df.user_id_idx)
        b_t = torch.FloatTensor(train_df.book_id_idx) + n_users

        train_edge_index = torch.stack((
            torch.cat([u_t, b_t]),
            torch.cat([b_t, u_t])
        )).to(device)
        
        return train_edge_index.type(torch.long)

def get_dataloaders(train_df, test_df, batch_size):
    """Create train and test dataloaders"""
    train_loader = create_dataloader(train_df, batch_size)
    test_loader = create_dataloader(test_df, batch_size, shuffle=False)
    return train_loader, test_loader