import json
import os
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class PlayerDemosDataset(Dataset):
    def __init__(self, demo_files, state_encoder, action_encoder):
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.states = []
        self.actions = []

        for file_path in demo_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                for event in data:
                    # <<< CUSTOMIZE HERE >>>
                    # This part depends entirely on your VLM's JSON structure.
                    # You need to extract the features that represent the "state".
                    state_values = list(event["game_state"].values())
                    self.states.append(state_values)
                    
                    # <<< CUSTOMIZE HERE >>>
                    # This part extracts the "action" the player took.
                    action_value = event["player_action"]["tactic"]
                    self.actions.append(action_value)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state_np = self.state_encoder.transform([self.states[idx]]).toarray()
        state_tensor = torch.FloatTensor(state_np).squeeze(0)
        
        action_label = self.action_encoder.transform([self.actions[idx]])[0]
        action_tensor = torch.LongTensor([action_label]).squeeze(0)
        
        return state_tensor, action_tensor

def prepare_encoders(all_demo_files):
    """
    Scans all data from ALL players to create consistent encoders.
    """
    all_states = []
    all_actions = []
    for file_path in all_demo_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            for event in data:
                # <<< CUSTOMIZE HERE >>>
                # Make sure this matches the logic in the PlayerDemosDataset class.
                all_states.append(list(event["game_state"].values()))
                all_actions.append(event["player_action"]["tactic"])

    state_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    state_encoder.fit(all_states)
    
    action_encoder = LabelEncoder()
    action_encoder.fit(all_actions)
    
    return state_encoder, action_encoder
