import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

# Import from our other source files
from data_loader import PlayerDemosDataset, prepare_encoders
from model import BCModel

# --- Main Training Function ---
def train_model(model, dataloader, epochs=25, model_save_path="models/star_player_policy.pth"):
    # <<< CUSTOMIZE HERE >>>
    # You can experiment with different optimizers (e.g., SGD) or learning rates.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\n--- Starting Training on Star Player ---")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for states, actions in dataloader:
            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    # Save the trained model
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(model.state_dict(), model_save_path)
    print(f"--- Training Finished. Model saved to {model_save_path} ---")

# --- Execution Block ---
if __name__ == "__main__":
    # Define data directories
    STAR_PLAYER_DIR = "data/raw/star_player_demos"
    OTHER_PLAYERS_DIR = "data/raw/other_players_demos"

    if not os.path.exists(STAR_PLAYER_DIR):
        print(f"Error: Star player data not found at '{STAR_PLAYER_DIR}'.")
        print("Please place your star player's JSON demo files in that directory.")
        exit()
    
    # Prepare encoders using data from ALL players for consistency
    print("\n--- Preparing Data Encoders ---")
    all_files = []
    all_files.extend([os.path.join(STAR_PLAYER_DIR, f) for f in os.listdir(STAR_PLAYER_DIR)])
    
    if os.path.exists(OTHER_PLAYERS_DIR):
        # Assumes other players are in subdirectories, e.g., other_players_demos/player_A/
        for player_subdir in os.listdir(OTHER_PLAYERS_DIR):
            player_path = os.path.join(OTHER_PLAYERS_DIR, player_subdir)
            if os.path.isdir(player_path):
                all_files.extend([os.path.join(player_path, f) for f in os.listdir(player_path)])
        
    state_encoder, action_encoder = prepare_encoders(all_files)

    # Create dataset and dataloader ONLY for the star player
    star_demo_files = [os.path.join(STAR_PLAYER_DIR, f) for f in os.listdir(STAR_PLAYER_DIR)]
    star_dataset = PlayerDemosDataset(star_demo_files, state_encoder, action_encoder)
    star_dataloader = DataLoader(star_dataset, batch_size=32, shuffle=True)
    
    # Get model dimensions from the encoders
    input_size = state_encoder.transform([star_dataset.states[0]]).shape[1]
    num_actions = len(action_encoder.classes_)
    print(f"State vector size: {input_size}")
    print(f"Number of actions: {num_actions}")

    # Initialize and train the model
    star_player_model = BCModel(input_size, num_actions)
    train_model(star_player_model, star_dataloader)
