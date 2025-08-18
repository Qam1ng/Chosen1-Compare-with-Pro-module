import os
import torch
import json

# Import from our other source files
from data_loader import prepare_encoders
from model import BCModel

# --- Main Evaluation Function ---
def calculate_similarity_score(star_model, player_demo_files, state_encoder, action_encoder):
    star_model.eval()
    total_events, matches = 0, 0
    
    for file_path in player_demo_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            for event in data:
                total_events += 1
                # <<< CUSTOMIZE HERE >>>
                # This logic must match the data loading in data_loader.py
                actual_state_values = list(event["game_state"].values())
                actual_action_label = event["player_action"]["tactic"]
                
                state_np = state_encoder.transform([actual_state_values]).toarray()
                state_tensor = torch.FloatTensor(state_np)
                
                with torch.no_grad():
                    outputs = star_model(state_tensor)
                    predicted_idx = torch.argmax(outputs, dim=1).item()
                    predicted_action_label = action_encoder.inverse_transform([predicted_idx])[0]
                
                if predicted_action_label == actual_action_label:
                    matches += 1
                    
    return (matches / total_events) * 100 if total_events > 0 else 0

# --- Execution Block ---
if __name__ == "__main__":
    MODEL_PATH = "models/star_player_policy.pth"
    STAR_PLAYER_DIR = "data/raw/star_player_demos"
    OTHER_PLAYERS_DIR = "data/raw/other_players_demos"

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Please run train.py first.")
        exit()

    # We need to initialize the encoders the same way we did for training
    print("\n--- Preparing Data Encoders for Evaluation ---")
    all_files = []
    all_files.extend([os.path.join(STAR_PLAYER_DIR, f) for f in os.listdir(STAR_PLAYER_DIR)])
    for player_subdir in os.listdir(OTHER_PLAYERS_DIR):
        player_path = os.path.join(OTHER_PLAYERS_DIR, player_subdir)
        if os.path.isdir(player_path):
            all_files.extend([os.path.join(player_path, f) for f in os.listdir(player_path)])
            
    state_encoder, action_encoder = prepare_encoders(all_files)

    # Load the trained model
    # <<< CUSTOMIZE HERE >>>
    # This calculation of input_size is based on the mock data.
    # It sums the number of categories for each state feature.
    # You must update this if you change your state representation.
    input_size = sum(len(cat) for cat in state_encoder.categories_)
    num_actions = len(action_encoder.classes_)
    
    star_player_model = BCModel(input_size, num_actions)
    star_player_model.load_state_dict(torch.load(MODEL_PATH))
    print("Star player model loaded successfully.")

    # Evaluate all other players
    print("\n--- Calculating Similarity Scores for Other Players ---")
    # Assumes other players are in subdirectories within OTHER_PLAYERS_DIR
    other_player_folders = [os.path.join(OTHER_PLAYERS_DIR, d) for d in os.listdir(OTHER_PLAYERS_DIR) if os.path.isdir(os.path.join(OTHER_PLAYERS_DIR, d))]
    similarity_scores = {}

    for player_folder in other_player_folders:
        player_files = [os.path.join(player_folder, f) for f in os.listdir(player_folder)]
        score = calculate_similarity_score(star_player_model, player_files, state_encoder, action_encoder)
        folder_name = os.path.basename(player_folder)
        similarity_scores[folder_name] = score
        print(f"Player '{folder_name}' Similarity Score: {score:.2f}%")

    # Rank the players
    ranked_players = sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)
    
    print("\n--- Player Similarity Ranking ---")
    for i, (player, score) in enumerate(ranked_players):
        print(f"{i+1}. {player}: {score:.2f}%")
        
    if ranked_players:
        print(f"\nThe player most similar to the star player is: '{ranked_players[0][0]}'")
