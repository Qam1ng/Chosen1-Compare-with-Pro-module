import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
raw_data_dir = "raw_csv_data"
pd_data_dir = "raw_pd_data"
os.makedirs(raw_data_dir, exist_ok=True)
os.makedirs(pd_data_dir, exist_ok=True)

# --- 能够处理多个文件的 Dataset 类 ---
try:
    # Load your uploaded files into pandas DataFrames
    # Note: I've corrected the filenames to match the ones you just uploaded.
    df_second_record = pd.read_csv(f"{raw_data_dir}/donk_dust2_1_secondrecord.csv")
    df_info_when_hurt = pd.read_csv(f"{raw_data_dir}/donk_dust2_1_infowhenhurt.csv")
    df_start_info = pd.read_csv(f"{raw_data_dir}/donk_dust2_1_startinfo.csv")

    # Save these DataFrames into the data directory with the correct names
    # This makes the process scalable for when you add more demos later.
    df_second_record.to_csv(os.path.join(pd_data_dir, "donk_dust2_1_secondrecord.csv"), index=False)
    df_info_when_hurt.to_csv(os.path.join(pd_data_dir, "donk_dust2_1_infowhenhurt.csv"), index=False)
    df_start_info.to_csv(os.path.join(pd_data_dir, "donk_dust2_1_startinfo.csv"), index=False)

    print(f"Successfully loaded and saved your data to the '{pd_data_dir}/' directory.")

except FileNotFoundError as e:
    print(f"ERROR: Could not find an uploaded CSV file. Please ensure all files are uploaded.")
    print(f"Details: {e}")
    # Exit if files aren't found, as the rest of the script cannot run.
    exit()

print("-" * 20)

class TrajectoryDatasetCSVMultiFile(Dataset):
    """
    Uses pandas.read_csv for robust loading of mixed data types from a directory.
    It correctly parses filenames like 'donk_dust2_1_secondrecord.csv'.
    """
    def __init__(self, data_root_dir: str, sequence_length: int):
        self.data_root_dir = data_root_dir
        self.sequence_length = sequence_length
        self.demos = []
        self.player_to_label = {}
        self.label_counter = 0

        print("--- Initializing Dataset ---")
        file_groups = defaultdict(dict)
        # These types match your filenames: startinfo, infowhenhurt, secondrecord
        known_types = ['startinfo', 'infowhenhurt', 'secondrecord']

        for filename in sorted(os.listdir(self.data_root_dir)):
            if not filename.endswith(".csv"): continue
            base_name = filename.replace(".csv", "")
            for file_type in known_types:
                suffix = f'_{file_type}'
                if base_name.endswith(suffix):
                    prefix = base_name[:-len(suffix)]
                    file_groups[prefix][file_type] = filename
                    break
        
        print("\n--- File Grouping Report ---")
        for prefix, files in file_groups.items():
            print(f"Prefix '{prefix}': Found {len(files)} files -> {list(files.keys())}")
        print("--------------------------\n")

        for prefix, files in file_groups.items():
            if all(key in files for key in known_types):
                player_name = prefix.split('_')[0]
                if player_name not in self.player_to_label:
                    self.player_to_label[player_name] = self.label_counter
                    self.label_counter += 1
                label = self.player_to_label[player_name]
                self.demos.append({'prefix': prefix, 'files': files, 'label': label})

        print("--- Initialization Complete ---")
        print("Player to label mapping:", self.player_to_label)
        print(f"Found {len(self.demos)} complete demos.")
        print("-----------------------------\n")

    def __len__(self) -> int:
        return len(self.demos)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        demo_info = self.demos[idx]
        label = demo_info['label']
        # The main trajectory comes from the 'secondrecord' file
        main_trajectory_file = os.path.join(self.data_root_dir, demo_info['files']['secondrecord'])
        
        try:
            # Use pandas to correctly handle boolean values ('True'/'False')
            df = pd.read_csv(main_trajectory_file)
            full_trajectory = df.to_numpy().astype(np.float32)

        except Exception as e:
            print(f"ERROR: Could not load or process file {main_trajectory_file}. Error: {e}")
            return torch.zeros(self.sequence_length, 1), torch.tensor(-1, dtype=torch.long)
        
        if full_trajectory.shape[0] == 0:
             return torch.zeros(self.sequence_length, 1), torch.tensor(label, dtype=torch.long)
        if full_trajectory.ndim == 1:
            full_trajectory = full_trajectory.reshape(1, -1)

        trajectory_len, feature_dim = full_trajectory.shape
        start_idx = 0
        if trajectory_len > self.sequence_length:
            start_idx = np.random.randint(0, trajectory_len - self.sequence_length)
        
        end_idx = start_idx + self.sequence_length
        sampled_trajectory = full_trajectory[start_idx:end_idx]

        if sampled_trajectory.shape[0] < self.sequence_length:
            padding_needed = self.sequence_length - sampled_trajectory.shape[0]
            padding = np.zeros((padding_needed, feature_dim), dtype=np.float32)
            sampled_trajectory = np.vstack([sampled_trajectory, padding])

        trajectory_tensor = torch.from_numpy(sampled_trajectory)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return trajectory_tensor, label_tensor

# --- Part 3: Instantiate and Use the Dataset ---
SEQUENCE_LENGTH = 128
BATCH_SIZE = 1 # Set to 1 since we only have one demo for now

# The Dataset will read from the 'my_game_demos' directory we created in Part 1
dataset = TrajectoryDatasetCSVMultiFile(
    data_root_dir=raw_data_dir,
    sequence_length=SEQUENCE_LENGTH
)

if len(dataset) > 0:
    data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("--- DataLoader Test ---")
    try:
        sample_batch, labels_batch = next(iter(data_loader))
        print(f"Successfully fetched one batch.")
        print(f"  - Sample batch shape: {sample_batch.shape}")
        print(f"  - Labels in batch: {labels_batch.numpy()}")
    except Exception as e:
        print(f"An error occurred while fetching a batch: {e}")
    print("-----------------------\n")
else:
    print("--- DataLoader Creation Failed ---")
    print("Could not create DataLoader because no complete demo groups were found.")
    print("Please check the 'File Grouping Report' above to diagnose your filenames.")
    print("----------------------------------\n")
