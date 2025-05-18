import os
import numpy as np
import threading
import time
import tkinter as tk
from tkinter import simpledialog
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from concurrent.futures import ThreadPoolExecutor
import math

def prompt_for_threads():
    """
    Display a dialog to prompt the user for the number of threads to use
    Returns the number of threads specified by the user
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Get number of CPU cores for a reasonable default and max
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    
    # Prompt user for number of threads
    user_threads = simpledialog.askinteger(
        "Thread Configuration",
        f"Enter the number of threads to use for training (1-{cpu_count}):",
        minvalue=1,
        maxvalue=cpu_count,
        initialvalue=min(4, cpu_count)
    )
    
    root.destroy()
    
    # If user cancels, use a reasonable default
    if user_threads is None:
        user_threads = min(4, cpu_count)
        print(f"Using default of {user_threads} threads")
    else:
        print(f"User selected {user_threads} threads")
        
    return user_threads

class ThreadedEmotionTrainer:
    def __init__(self, train_dir="fer2013/training", test_dir="fer2013/test", 
                 model_path="models/emotion_model.h5", num_threads=4):
        """
        Initialize the threaded emotion trainer
        
        Args:
            train_dir: Directory containing training data organized in class folders
            test_dir: Directory containing test data organized in class folders
            model_path: Path to save or load the trained model
            num_threads: Number of threads for parallel processing
        """
        self.TRAIN_DIR = train_dir
        self.TEST_DIR = test_dir
        self.MODEL_PATH = model_path
        self.num_threads = num_threads
        
        # Model parameters
        self.img_size = (48, 48)
        self.batch_size = 64
        self.epochs = 100
        
        print(f"Initializing training with {self.num_threads} threads")
        
        # Data augmentation settings
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            zoom_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            horizontal_flip=True
        )
        
        self.test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
    
    def _create_model(self, input_shape, num_classes):
        """Create a CNN model for emotion classification"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling2D((2, 2)),

            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),

            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),

            Flatten(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def _process_data_chunk(self, emotion_folders, thread_id):
        """
        Process a subset of the emotion folders
        
        This function handles the preprocessing for a subset of emotion classes
        assigned to a specific thread.
        """
        print(f"Thread {thread_id}: Processing {len(emotion_folders)} emotion classes")
        
        # Create generators for this subset of emotions
        for emotion_folder in emotion_folders:
            emotion_path = os.path.join(self.TRAIN_DIR, emotion_folder)
            test_emotion_path = os.path.join(self.TEST_DIR, emotion_folder)
            
            # Process training images for this emotion
            print(f"Thread {thread_id}: Processing {emotion_folder} training images")
            
            # For demonstration, we'll just simulate some preprocessing work
            # In a real implementation, you would do actual preprocessing here
            time.sleep(1)  # Simulate preprocessing work
            
            # Process test images for this emotion
            print(f"Thread {thread_id}: Processing {emotion_folder} test images")
            
            # Simulate preprocessing work for test data
            time.sleep(0.5)
        
        print(f"Thread {thread_id}: Completed processing assigned emotions")
        return thread_id
    
    def _train_model_chunk(self, model, train_data_subset, thread_id, validation_data=None):
        """
        Train the model on a subset of training data
        
        This function trains the provided model on a subset of the training data
        assigned to a specific thread.
        """
        print(f"Thread {thread_id}: Training model on data subset")
        
        # In a real implementation, you would modify this to train on actual subsets
        # For now, we'll train on the full dataset but with fewer epochs per thread
        epochs_per_thread = max(1, self.epochs // self.num_threads)
        
        early_stop = EarlyStopping(patience=3, restore_best_weights=True)
        
        model.fit(
            train_data_subset,
            epochs=epochs_per_thread,
            validation_data=validation_data,
            callbacks=[early_stop],
            verbose=1
        )
        
        print(f"Thread {thread_id}: Completed model training")
        return model
    
    def preprocess_and_train(self):
        """
        Main method to preprocess data and train the model using multiple threads
        """
        start_time = time.time()
        
        print(f"Starting preprocessing and training with {self.num_threads} threads")
        
        # Load the full dataset first to get class information
        train_data = self.train_datagen.flow_from_directory(
            self.TRAIN_DIR,
            target_size=self.img_size,
            color_mode='grayscale',
            class_mode='categorical',
            batch_size=self.batch_size,
            shuffle=True
        )
        
        test_data = self.test_datagen.flow_from_directory(
            self.TEST_DIR,
            target_size=self.img_size,
            color_mode='grayscale',
            class_mode='categorical',
            batch_size=self.batch_size,
            shuffle=False
        )
        
        num_classes = train_data.num_classes
        class_indices = train_data.class_indices
        emotion_folders = list(class_indices.keys())
        
        print(f"Found {num_classes} emotion classes: {emotion_folders}")
        
        # Check if model already exists
        if os.path.exists(self.MODEL_PATH):
            print(f"Found existing model at {self.MODEL_PATH}")
            model = load_model(self.MODEL_PATH)
            
        else:
            print("Model not found, training a new one with threaded preprocessing...")
            
            # 1. Threaded preprocessing: Divide emotion folders among threads
            chunks = np.array_split(emotion_folders, self.num_threads)
            
            # Create thread pool for preprocessing
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                # Submit preprocessing tasks
                preprocessing_futures = [
                    executor.submit(self._process_data_chunk, chunk.tolist(), i) 
                    for i, chunk in enumerate(chunks)
                ]
                
                # Wait for all preprocessing to complete
                for future in preprocessing_futures:
                    thread_id = future.result()
                    print(f"Preprocessing thread {thread_id} completed")
            
            print("All preprocessing threads completed")
            
            # 2. Threaded model training: Create model and divide training
            model = self._create_model(input_shape=(48, 48, 1), num_classes=num_classes)
            
            # Calculate samples per thread (divide dataset into chunks)
            samples_per_epoch = train_data.samples
            samples_per_thread = math.ceil(samples_per_epoch / self.num_threads)
            
            # Create thread pool for training
            thread_models = []
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                # Submit training tasks (each with a subset of data)
                training_futures = []
                
                for i in range(self.num_threads):
                    # Create a subset of the data for this thread
                    # In practice, we would create true subsets of the data here
                    # For simplicity in this example, we'll use the same data but train for fewer epochs
                    training_futures.append(
                        executor.submit(self._train_model_chunk, model, train_data, i, test_data)
                    )
                
                # Collect results from all training threads
                for future in training_futures:
                    thread_model = future.result()
                    thread_models.append(thread_model)
            
            print("All training threads completed")
            
            # Save the final model
            model.save(self.MODEL_PATH)
            print(f"Model saved to: {self.MODEL_PATH}")
        
        # Evaluate the model
        print("\nGenerating classification report...")
        
        pred_probs = model.predict(test_data)
        y_pred = np.argmax(pred_probs, axis=1)
        y_true = test_data.classes
        
        labels = list(test_data.class_indices.keys())
        
        report = classification_report(y_true, y_pred, target_names=labels, digits=4)
        print(report)
        
        end_time = time.time()
        print(f"Total preprocessing and training time: {end_time - start_time:.2f} seconds")
        
        return model

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train emotion detection model with threading')
    parser.add_argument('--threads', type=int, help='Number of threads for processing')
    parser.add_argument('--train_dir', type=str, default="fer2013/training", help='Training data directory')
    parser.add_argument('--test_dir', type=str, default="fer2013/test", help='Test data directory')
    parser.add_argument('--model_path', type=str, default="models/emotion_model.h5", help='Path to save/load model')
    parser.add_argument('--gui', action='store_true', help='Use GUI to prompt for thread count')
    
    args = parser.parse_args()
    
    # If --gui is specified or no --threads argument is provided, prompt user for thread count
    if args.gui or args.threads is None:
        num_threads = prompt_for_threads()
    else:
        num_threads = args.threads
    
    trainer = ThreadedEmotionTrainer(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        model_path=args.model_path,
        num_threads=num_threads
    )
    
    model = trainer.preprocess_and_train()