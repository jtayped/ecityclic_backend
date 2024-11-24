import torch.nn as nn


class TramitPredictor(nn.Module):
    def __init__(self, num_tramits, embedding_dim=32, hidden_dim=64, context_size=3):
        super().__init__()
        self.embedding = nn.Embedding(num_tramits, embedding_dim)

        # LSTM layer to process the context
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )

        # Prediction layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, num_tramits)

    def forward(self, x):
        # x shape: (batch_size, context_size)
        embedded = self.embedding(x)  # (batch_size, context_size, embedding_dim)

        lstm_out, _ = self.lstm(embedded)  # (batch_size, context_size, hidden_dim)

        # Take the last output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)

        # Final prediction
        x = self.fc1(last_output)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
