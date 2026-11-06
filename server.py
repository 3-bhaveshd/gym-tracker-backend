from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
from scipy.interpolate import interp1d

app = FastAPI()

@app.get("/")
async def wake_up():
    return {"status": "Server is awake and ready!"}

# 1. Define the CNN Architecture (copied exactly from your script)
class GymTrackerCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(GymTrackerCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), 
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), 
        )
        self.fc_encoder = nn.Sequential(
            nn.Linear(64 * 25, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        features = self.conv_block(x)
        features = features.view(features.size(0), -1) 
        encoding = self.fc_encoder(features)
        output = self.classifier(encoding)
        return output

# Load Model Configuration
EXERCISE_CLASSES = ["bayesian_curl", "bicep_curl", "overhead_dumbbell_extension", "single_arm_extension", "skull_crusher", "tricep_pushdown"]
model = GymTrackerCNN(num_classes=6)
try:
    model.load_state_dict(torch.load("tracker_model.pth"))
    model.eval()
except FileNotFoundError:
    print("Warning: tracker_model.pth not found. Using untrained weights for testing.")

# 2. Server State and Configuration Variables
class ServerState:
    def __init__(self):
        self.columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        self.buffer = pd.DataFrame(columns=self.columns)
        self.current_exercise = "Identifying..."
        self.rep_count = 0
        self.locked_in = False
        self.max_buffer_size = 500 
        
        # --- NEW TRACKING VARIABLES ---
        self.total_samples = 0
        self.last_peak_index = -1

state = ServerState()

class IMUData(BaseModel):
    imu_data: list[list[float]]

# 3. Helper Functions from your Script
def butter_lowpass_filter(data, cutoff=3.0, fs=100.0, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Add output='ba' and # type: ignore to bypass Pylance
    b, a = butter(order, normal_cutoff, btype='low', analog=False, output='ba') # type: ignore
    return filtfilt(b, a, data)

def interpolate_rep(rep_df):
    TARGET_LENGTH = 100
    original_length = len(rep_df)
    original_indices = np.linspace(0, 1, original_length)
    target_indices = np.linspace(0, 1, TARGET_LENGTH)
    
    interpolated_features = []
    for col in state.columns:
        f = interp1d(original_indices, rep_df[col].values, kind='linear')
        interpolated_features.append(f(target_indices))
    return np.array(interpolated_features)

# 4. API Endpoint
@app.post("/predict")
async def process_stream(data: IMUData):
    new_data = pd.DataFrame(data.imu_data, columns=state.columns, dtype=float)
    state.total_samples += len(new_data)
    state.buffer = pd.concat([state.buffer, new_data], ignore_index=True)
    
    if len(state.buffer) > state.max_buffer_size:
        state.buffer = state.buffer.iloc[-state.max_buffer_size:].reset_index(drop=True)

    if len(state.buffer) > 100:
        smoothed_signal = butter_lowpass_filter(state.buffer['acc_x'].to_numpy(dtype=float))
        peaks, _ = find_peaks(smoothed_signal, distance=50, prominence=5.0)
        
        if len(peaks) >= 2:
            latest_start = peaks[-2]
            latest_end = peaks[-1]
            absolute_latest_peak = (state.total_samples - len(state.buffer)) + latest_end
            
            # Check if this is a NEW repetition (using the 40-sample cooldown)
            if absolute_latest_peak - state.last_peak_index > 40:
                
                # Extract the latest rep data and run the CNN
                rep_data = state.buffer.iloc[latest_start:latest_end].copy()
                
                if len(rep_data) > 5:
                    standardized_rep = interpolate_rep(rep_data)
                    input_tensor = torch.tensor(standardized_rep, dtype=torch.float32).unsqueeze(0)
                    
                    with torch.no_grad():
                        logits = model(input_tensor)
                        probabilities = F.softmax(logits, dim=1).numpy()[0]
                        
                    max_prob_idx = np.argmax(probabilities)
                    max_conf = probabilities[max_prob_idx]
                    pred_exercise = EXERCISE_CLASSES[max_prob_idx]
                    
                    # If model is confident, decide whether to increment or reset
                    if max_conf >= 0.70:
                        # Check if the exercise has changed
                        if state.current_exercise != "Identifying..." and state.current_exercise != pred_exercise:
                            state.rep_count = 1  # Reset counter for the new exercise
                            print(f"\n*** SWITCHED EXERCISE -> {pred_exercise} ***")
                        else:
                            state.rep_count += 1 # Same exercise, just increment
                            
                        state.current_exercise = pred_exercise
                        print(f"Rep {state.rep_count} | Identified: {state.current_exercise} (Conf: {max_conf:.2f})")
                        
                    # If model is NOT confident, just add a rep to the current exercise
                    else:
                        if state.current_exercise != "Identifying...":
                            state.rep_count += 1
                            print(f"Rep {state.rep_count} counted. CNN confidence too low ({max_conf:.2f}) to switch exercise.")
                            
                # Update the peak tracker so we don't double-count this peak
                state.last_peak_index = absolute_latest_peak

    return {
        "exercise": state.current_exercise,
        "reps": state.rep_count
    }

@app.post("/reset")
async def reset_tracker():
    state.buffer = pd.DataFrame(columns=state.columns)
    state.current_exercise = "Identifying..."
    state.rep_count = 0
    state.locked_in = False
    state.total_samples = 0
    state.last_peak_index = -1
    return {"status": "reset successful"}