import torch
import torch.nn as nn

class TrajectoryEncoder(nn.Module):
    """
    A simple feed-forward neural network that encodes features for a single time step.
    This model processes each time step of a trajectory independently.
    """
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.2):
        """
        Initializes the TrajectoryEncoder.

        Args:
            input_dim (int): The number of features for each time step in the input trajectory.
                             (e.g., player position, velocity, view angles, etc.)
            output_dim (int): The dimension of the compressed feature representation (embedding).
            dropout (float): The dropout rate for regularization.
        """
        super(TrajectoryEncoder, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass for the encoder.

        Args:
            x (torch.Tensor): The input tensor representing a batch of trajectories.
                              Shape: (batch_size, sequence_length, input_dim)

        Returns:
            torch.Tensor: The encoded tensor.
                          Shape: (batch_size, sequence_length, output_dim)
        """
        # The network is applied to the last dimension (the features) of the input tensor.
        # PyTorch's nn.Linear can handle the batch and sequence dimensions automatically.
        return self.network(x)

# --- This block is for testing and demonstration purposes ---
if __name__ == '__main__':
    # --- Simulation Parameters ---
    BATCH_SIZE = 16
    SEQ_LENGTH = 50
    # This must match the number of features in your 'secondrecord.csv'
    INPUT_FEATURE_DIM = 24
    ENCODER_OUTPUT_DIM = 128

    # --- Check for available device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Instantiate the Model ---
    print("\nInitializing TrajectoryEncoder...")
    encoder = TrajectoryEncoder(
        input_dim=INPUT_FEATURE_DIM,
        output_dim=ENCODER_OUTPUT_DIM
    ).to(device)

    print(encoder)

    # --- Create a Dummy Input Tensor ---
    # This simulates a batch of trajectory data.
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LENGTH, INPUT_FEATURE_DIM).to(device)
    print(f"\nShape of dummy input tensor: {dummy_input.shape}")

    # --- Perform a Forward Pass Test ---
    try:
        output = encoder(dummy_input)
        print(f"Shape of encoder output tensor: {output.shape}")

        # Verify that the output shape is correct
        assert output.shape == (BATCH_SIZE, SEQ_LENGTH, ENCODER_OUTPUT_DIM)
        print("\nEncoder forward pass successful!")

    except Exception as e:
        print(f"\nAn error occurred during the forward pass: {e}")