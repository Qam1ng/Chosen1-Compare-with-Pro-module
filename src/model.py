import torch.nn as nn

class BCModel(nn.Module):
    def __init__(self, input_size, num_actions):
        super(BCModel, self).__init__()
        # <<< CUSTOMIZE HERE >>>
        # This is where you can experiment with your model's architecture.
        # You can add more layers, change the number of neurons, or add dropout.
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        return self.network(x)
