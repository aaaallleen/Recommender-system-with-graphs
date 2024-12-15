# Book Rating Prediction with Graph Neural Networks

This project implements a book recommendation system using Graph Neural Networks (GNN). It uses both LightGCN and NGCF architectures to predict user ratings for books based on user-item interactions and features.

This project is modified and rewritten based on [Recommender Systems with GNNs in PyG](https://medium.com/stanford-cs224w/recommender-systems-with-gnns-in-pyg-d8301178e377) written by Derrick Li.

I rewritten my implementation into python files for easier training, but the correctness isn't guaranteed. The .ipynb file provided is a working notebook which I used to test the implementation.
## Project Structure
```
book-rating-prediction/
│
├── models/                 # Model checkpoint directory
├── logs/                  # Training logs directory
├── plots/                 # Training visualization plots
│
├── dataloader.py          # Data loading and preprocessing
├── models.py              # GNN model architectures
├── train.py              # Training pipeline
├── utils.py              # Utility functions
├── requirements.txt      # Project dependencies
├── config.yaml           # Configuration file
│
└── dataset/              # Dataset directory
├── books_data.csv
├── dataset.csv
└── user_stats.json
```
## Dataset
The dataset used for my testing could be found on my repository under "GOODREADS dataset"

## Models 
The project implements two GNN architectures:

1. LightGCN: A lightweight version of GCN for recommendation
2. NGCF: Neural Graph Collaborative Filtering

## Usage 
### Training 
1. Configure training parameters in config.yaml
```
# Model parameters
model_type: "LightGCN"  # or "NGCF"
latent_dim: 64
num_layers: 3
dropout: 0.1

# Training parameters
batch_size: 1024
learning_rate: 0.001
epochs: 50
patience: 10
```
2. Run training 
```
python train.py --config config.yaml --model_type LightGCN
```

### Making predictions 
```
from models import RatingPredictionGNN
from utils import load_checkpoint

# Load trained model
model = load_checkpoint(
    model=RatingPredictionGNN(...),
    filepath="models/lightgcn_model.pt",
    device=device
)

# Create inference engine
inference_engine = RatingInference(
    model=model,
    edge_index=edge_index,
    n_users=n_users,
    device=device
)

# Make predictions
predicted_rating = inference_engine.predict_single_rating(user_idx, item_idx)
```
### Training Visualization
The training process generates plots showing:

* Training loss over epochs
* Test RMSE over epochs
* Test MAE over epochs

Plots are saved in the plots/ directory.
### Results
Performance metrics for the models:

LightGCN:
Best Test RMSE: 1.1185
Best Test MAE: 0.8154
