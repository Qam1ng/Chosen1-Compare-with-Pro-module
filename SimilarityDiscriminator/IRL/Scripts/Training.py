import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import our custom modules 
import sys
sys.path.insert(1, 'C:\\Users\\19716\\Desktop\\chosen1-Compare-with-Pro-module\\SimilarityDiscriminator\\IRL\\src')

from creatingDataset import TrajectoryDatasetCSVMultiFile
from encoder import TrajectoryEncoder
from discriminator import StyleDiscriminator

def train():
    """Main function to run the training process."""
    # --- 1. Hyperparameters and Configuration ---
    print("--- Setting up configuration ---")
    NUM_EPOCHS = 200
    LEARNING_RATE = 0.001
    BATCH_SIZE = 64 # You can increase this if you have more data and memory
    SEQUENCE_LENGTH = 128 # Must match the dataset configuration
    
    # Model dimensions
    # This MUST match the number of columns in your '..._secondrecord.csv' file
    TRAJECTORY_FEATURE_DIM = 11
    ENCODER_OUTPUT_DIM = 128 # The size of the feature vector after encoding
    DISCRIMINATOR_HIDDEN_DIM = 256 # LSTM hidden size
    NUM_LSTM_LAYERS = 2
    
    # Dataset path
    DATA_ROOT_DIR = "raw_csv_data"
    
    # --- 2. Setup Device, Dataset, and DataLoader ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    print("--- Initializing Dataset and DataLoader ---")
    dataset = TrajectoryDatasetCSVMultiFile(
        data_root_dir=DATA_ROOT_DIR,
        sequence_length=SEQUENCE_LENGTH
    )
    
    if len(dataset) == 0:
        print("\nERROR: Dataset is empty. No complete demo groups were found.")
        print("Please check your file names and the output of creatingDataset.py.")
        return # Exit if no data is available

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    # The number of unique players found by the dataset
    NUM_CLASSES = dataset.label_counter
    print(f"Found {NUM_CLASSES} unique player styles to classify.\n")

    # --- 3. Initialize Models, Loss, and Optimizer ---
    print("--- Initializing models ---")
    encoder = TrajectoryEncoder(
        input_dim=TRAJECTORY_FEATURE_DIM,
        output_dim=ENCODER_OUTPUT_DIM
    ).to(device)

    discriminator = StyleDiscriminator(
        input_dim=ENCODER_OUTPUT_DIM,
        hidden_dim=DISCRIMINATOR_HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        num_layers=NUM_LSTM_LAYERS
    ).to(device)

    print("\nEncoder Architecture:")
    print(encoder)
    print("\nDiscriminator Architecture:")
    print(discriminator)

    # Combine parameters from both models for the optimizer
    combined_params = list(encoder.parameters()) + list(discriminator.parameters())
    
    # Loss function for classification
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer to update model weights
    optimizer = optim.Adam(combined_params, lr=LEARNING_RATE)
    
    print("\n--- Starting Training ---")
    # --- 4. The Training Loop ---
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Set models to training mode
        encoder.train()
        discriminator.train()

        for i, (trajectories, labels) in enumerate(data_loader):
            # Move data to the selected device
            trajectories = trajectories.to(device)
            labels = labels.to(device)

            # --- Forward Pass ---
            # 1. Pass trajectories through the encoder
            encoded_trajectories = encoder(trajectories)
            
            # 2. Pass the encoded sequence to the discriminator
            logits = discriminator(encoded_trajectories)
            
            # --- Calculate Loss ---
            loss = criterion(logits, labels)

            # --- Backward Pass and Optimization ---
            # 1. Clear previous gradients
            optimizer.zero_grad()
            # 2. Calculate gradients
            loss.backward()
            # 3. Update weights
            optimizer.step()

            # --- Logging and Metrics ---
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        # --- Print Epoch Statistics ---
        avg_loss = running_loss / len(data_loader)
        accuracy = (correct_predictions / total_samples) * 100
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    print("\n--- Training Finished ---")

if __name__ == '__main__':
    train()