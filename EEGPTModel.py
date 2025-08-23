import torch
import torch.nn as nn

class EEGPTModel(nn.Module):
    """EEGPT model architecture matching the pretrained model"""

    def __init__(self, d_model=512, n_heads=8, n_layers=12, patch_size=64, 
                 n_channels=58, sequence_length=4000):
        super().__init__()
        
        self.d_model = d_model
        self.patch_size = patch_size
        self.n_channels = n_channels
        
        # Patch embedding
        self.patch_embed = nn.Conv1d(n_channels, d_model, kernel_size=patch_size, stride=patch_size)
        
        # Positional encoding
        max_patches = sequence_length // patch_size
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches, d_model) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """Forward pass"""
        # x shape: (batch, channels, time)
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch, d_model, n_patches)
        x = x.transpose(1, 2)    # (batch, n_patches, d_model)
        
        # Add positional encoding
        if x.size(1) <= self.pos_embed.size(1):
            x = x + self.pos_embed[:, :x.size(1), :]
        
        # Transformer
        x = self.transformer(x)
        x = self.norm(x)
        
        # Global average pooling
        features = x.mean(dim=1)  # (batch, d_model)
        
        # Return features and attention (simplified attention for demo)
        attention_weights = x.transpose(1, 2)  # For interpretability
        
        return features, attention_weights