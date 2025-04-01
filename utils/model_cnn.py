### New model architecture based on CNN

### Author: Soham Datta
### Date: 2025-04-01

####

import torch
import torch.nn as nn
import torch.nn.functional as F



# -------------------------------------------------------------------------------------------
# CNN Encoder Layer
# -------------------------------------------------------------------------------------------
class CNNEncoderLayer(nn.Module):
    """
    Single CNN-based layer:
      - Conv1d(hidden_dim -> hidden_dim, kernel=3, padding=1)
      - ReLU
      - Dropout
      - Residual connection
      - LayerNorm
    """
    def __init__(self, hidden_dim, dropout=0.0):
        super(CNNEncoderLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels=hidden_dim,
                              out_channels=hidden_dim,
                              kernel_size=3,
                              padding=1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        x is assumed to be (B, hidden_dim, seq_len) for the convolution.
        We'll transpose back to (B, seq_len, hidden_dim) for layer norm.
        """
        residual = x                        # (B, hidden_dim, seq_len)
        out = self.conv(x)                  # (B, hidden_dim, seq_len)
        out = F.relu(out)
        out = self.dropout(out)
        
        # Transpose for layer norm
        out = out.transpose(1, 2)           # (B, seq_len, hidden_dim)
        out = self.layer_norm(out)          # apply LN over last dimension
        out = out.transpose(1, 2)           # (B, hidden_dim, seq_len)

        out = out + residual                # add residual
        return out


# -------------------------------------------------------------------------------------------
# CNN Decoder Layer
# -------------------------------------------------------------------------------------------
class CNNDecoderLayer(nn.Module):
    """
    Single CNN-based layer for decoder side:
      - Conv1d(hidden_dim -> hidden_dim, kernel=3, padding=1)
      - ReLU
      - Dropout
      - Residual connection
      - LayerNorm
    """
    def __init__(self, hidden_dim, dropout=0.0):
        super(CNNDecoderLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels=hidden_dim,
                              out_channels=hidden_dim,
                              kernel_size=3,
                              padding=1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        x is (B, hidden_dim, seq_len).
        Similar flow as the encoder layer.
        """
        residual = x
        out = self.conv(x)
        out = F.relu(out)
        out = self.dropout(out)

        out = out.transpose(1, 2)
        out = self.layer_norm(out)
        out = out.transpose(1, 2)

        out = out + residual
        return out


# -------------------------------------------------------------------------------------------
# CNN Encoder
# -------------------------------------------------------------------------------------------
class EncoderCNN(nn.Module):
    """
    CNN-based Encoder:
     1) Linear embedding from input_dim -> hidden_dim
     2) Stack of CNNEncoderLayer
     3) Optional final LayerNorm
    """
    def __init__(self, input_dim, hidden_dim, n_layers, dropout=0.0, use_norm=True):
        super(EncoderCNN, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            CNNEncoderLayer(hidden_dim, dropout) for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_dim) if use_norm else None

    def forward(self, x):
        """
        x shape: (B, seq_len, input_dim)
        We embed -> transpose to (B, hidden_dim, seq_len) -> pass layers -> transpose back.
        """
        # 1) Embed
        x = self.embedding(x)               # (B, seq_len, hidden_dim)
        # 2) Transpose for Conv1D
        x = x.transpose(1, 2)              # (B, hidden_dim, seq_len)

        # 3) Pass through N encoder layers
        for layer in self.layers:
            x = layer(x)

        # 4) Optionally apply final LN in (B, seq_len, hidden_dim) shape
        x = x.transpose(1, 2)              # (B, seq_len, hidden_dim)
        if self.layer_norm:
            x = self.layer_norm(x)
        return x                            # (B, seq_len, hidden_dim)


# -------------------------------------------------------------------------------------------
# CNN Decoder
# -------------------------------------------------------------------------------------------
class DecoderCNN(nn.Module):
    """
    CNN-based Decoder:
     1) Stack of CNNDecoderLayer
     2) Final linear layer from hidden_dim -> output_dim
     3) Optional final LayerNorm
    """
    def __init__(self, output_dim, hidden_dim, n_layers, dropout=0.0, use_norm=True):
        super(DecoderCNN, self).__init__()
        self.layers = nn.ModuleList([
            CNNDecoderLayer(hidden_dim, dropout) for _ in range(n_layers)
        ])
        self.fc_output = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim) if use_norm else None

    def forward(self, encoder_outputs):
        """
        encoder_outputs shape: (B, seq_len, hidden_dim)
        We'll feed them directly through decoder CNN layers.
        """
        x = encoder_outputs                  # (B, seq_len, hidden_dim)
        
        # Convert to (B, hidden_dim, seq_len) for conv
        x = x.transpose(1, 2)               # (B, hidden_dim, seq_len)

        # Pass through CNN decoder layers
        for layer in self.layers:
            x = layer(x)

        # Transpose back for final linear
        x = x.transpose(1, 2)               # (B, seq_len, hidden_dim)

        # Optional LN
        if self.layer_norm:
            x = self.layer_norm(x)

        # Final projection: hidden_dim -> output_dim
        x = self.fc_output(x)               # (B, seq_len, output_dim)
        return x


# -------------------------------------------------------------------------------------------
# CNN-based Seq2Seq
# -------------------------------------------------------------------------------------------
class Seq2Seq_CNN(nn.Module):
    def __init__(self, encoder, decoder, device):
        """
        encoder: EncoderCNN instance
        decoder: DecoderCNN instance
        """
        super(Seq2Seq_CNN, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src):
        """
        src shape: (B, seq_len, input_dim)
        """
        # 1) Encode
        encoder_outputs = self.encoder(src)   # (B, seq_len, hidden_dim)

        # 2) Decode
        output = self.decoder(encoder_outputs)  # (B, seq_len, output_dim)
        return output
