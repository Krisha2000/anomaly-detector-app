import torch
import torch.nn as nn

# --- PyTorch LSTM Autoencoder Model ---
class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=128):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )
        self.decoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )
        self.output_layer = nn.Linear(embedding_dim, n_features)

    def forward(self, x):
        # Encoder
        _, (hidden, _) = self.encoder(x)
        
        # Decoder
        # Use the last hidden state of the encoder as the input for the decoder
        decoder_input = hidden.repeat(self.seq_len, 1, 1).permute(1, 0, 2)
        decoder_output, _ = self.decoder(decoder_input)
        
        return self.output_layer(decoder_output)
