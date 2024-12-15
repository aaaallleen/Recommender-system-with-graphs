import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree

class LightGCNConv(MessagePassing):
  def __init__(self, **kwargs):
    super().__init__(aggr='add')

  def forward(self, x, edge_index):
    from_, to_ = edge_index
    deg = degree(to_, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]
    return self.propagate(edge_index, x=x, norm=norm)

  def message(self, x_j, norm):
    return norm.view(-1, 1) * x_j
  
class NGCFConv(MessagePassing):
  def __init__(self, latent_dim, dropout, bias=True, **kwargs):
    super(NGCFConv, self).__init__(aggr='add', **kwargs)

    self.dropout = dropout

    self.lin_1 = nn.Linear(latent_dim, latent_dim, bias=bias)
    self.lin_2 = nn.Linear(latent_dim, latent_dim, bias=bias)

    self.init_parameters()


  def init_parameters(self):
    nn.init.xavier_uniform_(self.lin_1.weight)
    nn.init.xavier_uniform_(self.lin_2.weight)


  def forward(self, x, edge_index):
    # Compute normalization
    from_, to_ = edge_index
    deg = degree(to_, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]

    # Start propagating messages
    out = self.propagate(edge_index, x=(x, x), norm=norm)

    # Perform update after aggregation
    out += self.lin_1(x)
    out = F.dropout(out, self.dropout, self.training)
    return F.leaky_relu(out)


  def message(self, x_j, x_i, norm):
    return norm.view(-1, 1) * (self.lin_1(x_j) + self.lin_2(x_j * x_i))
  
class RatingPredictionGNN(nn.Module):
    def __init__(
            self,
            latent_dim,
            num_layers,
            num_users,
            num_books,
            model,
            user_features,
            book_numerical_features,
            book_genre_features,
            dropout=0.1
    ):
        super(RatingPredictionGNN, self).__init__()
        assert (model == 'NGCF' or model == 'LightGCN'), 'Model must be NGCF or LightGCN'
        self.model = model

        # Keep the same feature processing and embedding parts
        self.user_features = user_features
        self.book_numerical_features = book_numerical_features
        self.book_genre_features = book_genre_features

        self.user_feature_proj = nn.Linear(user_features.shape[1], latent_dim)
        self.book_numerical_features_proj = nn.Linear(book_numerical_features.shape[1], latent_dim)
        self.book_genre_features_proj = nn.Linear(book_genre_features.shape[1], latent_dim)

        self.embedding = nn.Embedding(num_users + num_books, latent_dim)

        if self.model == 'NGCF':
            self.convs = nn.ModuleList(
                NGCFConv(latent_dim, dropout=dropout) for _ in range(num_layers)
            )
        else:
            self.convs = nn.ModuleList(LightGCNConv() for _ in range(num_layers))

        # Add prediction layers for ratings
        self.prediction = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 1)
        )

        self.init_parameters()
    
    def init_parameters(self):
        if self.model == 'NGCF':
            nn.init.xavier_uniform_(self.embedding.weight, gain=1)
            nn.init.xavier_uniform_(self.user_feature_proj.weight)
            nn.init.xavier_uniform_(self.book_numerical_features_proj.weight)
            nn.init.xavier_uniform_(self.book_genre_features_proj.weight)
        else:
            nn.init.normal_(self.embedding.weight, std=0.1)
            nn.init.normal_(self.user_feature_proj.weight, std=0.1)
            nn.init.normal_(self.book_numerical_features_proj.weight, std=0.1)
            nn.init.normal_(self.book_genre_features_proj.weight, std=0.1)
            
    def forward(self, edge_index):
        # Same as before until obtaining embeddings
        user_proj = self.user_feature_proj(self.user_features)
        book_num_proj = self.book_numerical_features_proj(self.book_numerical_features)
        book_genre_proj = self.book_genre_features_proj(self.book_genre_features)

        book_proj = book_num_proj + book_genre_proj

        emb0 = self.embedding.weight.clone()
        emb0[:self.user_features.shape[0]] += user_proj
        emb0[self.user_features.shape[0]:] += book_proj

        embs = [emb0]
        emb = emb0

        for conv in self.convs:
            emb = conv(x=emb, edge_index=edge_index)
            embs.append(emb)

        out = (
            torch.cat(embs, dim=-1) if self.model == 'NGCF'
            else torch.mean(torch.stack(embs, dim=0), dim=0)
        )

        return emb0, out

    def predict_ratings(self, user_indices, item_indices, final_embeddings):
        """Predict ratings for given user-item pairs"""
        user_emb = final_embeddings[user_indices]
        item_emb = final_embeddings[item_indices]
        
        # Concatenate user and item embeddings
        combined = torch.cat([user_emb, item_emb], dim=1)
        
        # Predict rating
        rating = self.prediction(combined)
        return rating.squeeze()