import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import sounddevice as sd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import warnings
import os
warnings.filterwarnings('ignore')

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class AudioClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.features_list = []
        self.labels = []
        self.filenames = []
        
    def extract_features(self, audio_file):
        """Extract audio features from a file."""
        try:
            # Load audio file
            y, sr = librosa.load(audio_file)
            
            # Feature 1: Spectral Centroid (frequency domain)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_centroid_mean = np.mean(spectral_centroids)
            
            # Feature 2: Zero Crossing Rate (time domain)
            zero_crossing = librosa.feature.zero_crossing_rate(y)[0]
            zero_crossing_mean = np.mean(zero_crossing)
            
            # Feature 3: MFCC (frequency domain)
            mfccs = librosa.feature.mfcc(y=y, sr=sr)
            mfcc_mean = np.mean(mfccs[0])
            
            # Feature 4: RMS Energy (time domain)
            rms = librosa.feature.rms(y=y)[0]
            rms_mean = np.mean(rms)
            
            return [spectral_centroid_mean, zero_crossing_mean, mfcc_mean, rms_mean]
        except Exception as e:
            print(f"Error processing file {audio_file}: {str(e)}")
            return None
    
    def load_files_from_folders(self):
        """Load audio files from music and speech folders."""
        self.features_list = []
        self.labels = []
        self.filenames = []
        
        # Process music files
        music_folder = os.path.join(SCRIPT_DIR, "music")
        for file in os.listdir(music_folder):
            if file.endswith(('.wav', '.mp3')):
                file_path = os.path.join(music_folder, file)
                features = self.extract_features(file_path)
                if features:
                    self.features_list.append(features)
                    self.labels.append("yes")  # yes for music
                    self.filenames.append(file_path)
        
        # Process speech files
        speech_folder = os.path.join(SCRIPT_DIR, "speech")
        for file in os.listdir(speech_folder):
            if file.endswith(('.wav', '.mp3')):
                file_path = os.path.join(speech_folder, file)
                features = self.extract_features(file_path)
                if features:
                    self.features_list.append(features)
                    self.labels.append("no")   # no for speech
                    self.filenames.append(file_path)

    def train_model(self):
        """Train the SVM model using 2/3 of the data."""
        if len(self.features_list) < 3:
            raise ValueError("Not enough samples to train the model. Need at least 3 samples per class.")
            
        X = np.array(self.features_list)
        y = np.array(self.labels)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data into training and testing sets
        try:
            X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(
                X_scaled, y, self.filenames, test_size=0.33, random_state=42
            )
        except ValueError:
            # If stratification fails, try without stratification
            X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(
                X_scaled, y, self.filenames, test_size=0.33, random_state=42, stratify=None
            )
        
        # Train model
        self.model = SVC(kernel='rbf')
        self.model.fit(X_train, y_train)
        
        return X_test, y_test, files_test
    
    def test_model(self, X_test, y_test, files_test):
        """Test the model and return results."""
        predictions = self.model.predict(X_test)
        results = []
        
        for i in range(len(predictions)):
            results.append({
                'filename': files_test[i],
                'prediction': predictions[i],
                'ground_truth': y_test[i]
            })
            
        return results

class AudioPlayerGUI:
    def __init__(self, classifier):
        self.classifier = classifier
        self.current_audio = None
        self.playing = False
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Audio Classifier")
        self.root.geometry("800x600")
        
        # Create GUI elements
        self.create_widgets()
    
    def create_widgets(self):
        # Training frame
        train_frame = ttk.LabelFrame(self.root, text="Training", padding=10)
        train_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Button(train_frame, text="Train Model", 
                  command=self.train_and_test).pack(side="left", padx=5)
        
        # Results frame
        self.results_frame = ttk.LabelFrame(self.root, text="Results", padding=10)
        self.results_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create treeview for results
        self.tree = ttk.Treeview(self.results_frame, 
                                columns=("Filename", "Prediction", "Ground Truth"),
                                show="headings")
        
        self.tree.heading("Filename", text="Filename")
        self.tree.heading("Prediction", text="Prediction")
        self.tree.heading("Ground Truth", text="Ground Truth")
        
        # Configure column widths
        self.tree.column("Filename", width=300)
        self.tree.column("Prediction", width=100)
        self.tree.column("Ground Truth", width=100)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Playback frame
        playback_frame = ttk.LabelFrame(self.root, text="Playback", padding=10)
        playback_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Button(playback_frame, text="Play Selected", 
                  command=self.play_selected).pack(side="left", padx=5)
        ttk.Button(playback_frame, text="Stop", 
                  command=self.stop_playback).pack(side="left", padx=5)
    
    def train_and_test(self):
        try:
            # Clear previous results
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # Load and prepare dataset
            self.classifier.load_files_from_folders()
            
            # Train and test model
            X_test, y_test, files_test = self.classifier.train_model()
            results = self.classifier.test_model(X_test, y_test, files_test)
            
            # Display results
            for result in results:
                self.tree.insert("", "end", values=(
                    result['filename'],
                    "Music" if result['prediction'] == "yes" else "Speech",
                    "Music" if result['ground_truth'] == "yes" else "Speech"
                ))
            
            messagebox.showinfo("Success", "Model training and testing completed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def play_selected(self):
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Warning", "Please select a file to play.")
            return
        
        filename = self.tree.item(selected_item[0])['values'][0]
        
        def play_audio():
            try:
                y, sr = librosa.load(filename)
                self.playing = True
                sd.play(y, sr)
                sd.wait()
                self.playing = False
            except Exception as e:
                messagebox.showerror("Error", f"Error playing audio: {str(e)}")
                self.playing = False
        
        if not self.playing:
            threading.Thread(target=play_audio, daemon=True).start()
    
    def stop_playback(self):
        if self.playing:
            sd.stop()
            self.playing = False
    
    def run(self):
        self.root.mainloop()

def main():
    # Check if required folders exist using absolute paths
    music_folder = os.path.join(SCRIPT_DIR, "music")
    speech_folder = os.path.join(SCRIPT_DIR, "speech")
    
    if not os.path.exists(music_folder) or not os.path.exists(speech_folder):
        print(f"Error: 'music' and 'speech' folders must exist in {SCRIPT_DIR}")
        return
        
    classifier = AudioClassifier()
    gui = AudioPlayerGUI(classifier)
    gui.run()

if __name__ == "__main__":
    main()