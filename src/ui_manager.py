import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
import threading
import time
import os
import cv2

class UIManager:
    def __init__(self, emotion_detector, music_player):
        self.emotion_detector = emotion_detector
        self.music_player = music_player
        
        self.app = None
        self.camera_frame = None
        self.status_label = None
        self.emotion_label = None
        self.music_label = None
        self.camera_label = None
        self.confidence_bar = None
        self.control_frame = None
        
        self.is_running = False
        self.camera_thread = None
        self.update_interval = 100  # milliseconds
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the main UI components"""
        ctk.set_appearance_mode("System")  
        ctk.set_default_color_theme("blue")  
        
        self.app = ctk.CTk()
        self.app.title("Emotion-Based Music Player")
        self.app.geometry("1000x700")
        self.app.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        title_frame = ctk.CTkFrame(self.app)
        title_frame.pack(fill="x", padx=20, pady=10)
        
        title_label = ctk.CTkLabel(
            title_frame, 
            text="Emotion-Based Music Player",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=10)
        
        content_frame = ctk.CTkFrame(self.app)
        content_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.camera_frame = ctk.CTkFrame(content_frame)
        self.camera_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        camera_title = ctk.CTkLabel(
            self.camera_frame, 
            text="Emotion Detection",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        camera_title.pack(pady=10)
        
        # Camera display
        self.camera_label = ctk.CTkLabel(self.camera_frame, text="")
        self.camera_label.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.control_frame = ctk.CTkFrame(content_frame)
        self.control_frame.pack(side="right", fill="both", padx=10, pady=10, expand=True)
        
        controls_title = ctk.CTkLabel(
            self.control_frame, 
            text="Controls & Status",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        controls_title.pack(pady=10)
        
        emotion_frame = ctk.CTkFrame(self.control_frame)
        emotion_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(emotion_frame, text="Current Emotion:").pack(side="left", padx=10)
        self.emotion_label = ctk.CTkLabel(emotion_frame, text="None", width=100)
        self.emotion_label.pack(side="left", padx=10)
        
        confidence_frame = ctk.CTkFrame(self.control_frame)
        confidence_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(confidence_frame, text="Confidence:").pack(side="left", padx=10)
        self.confidence_bar = ctk.CTkProgressBar(confidence_frame, width=200)
        self.confidence_bar.pack(side="left", padx=10)
        self.confidence_bar.set(0)
        
        music_frame = ctk.CTkFrame(self.control_frame)
        music_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(music_frame, text="Current Music:").pack(side="left", padx=10)
        self.music_label = ctk.CTkLabel(music_frame, text="None", width=200)
        self.music_label.pack(side="left", padx=10)
        
        volume_frame = ctk.CTkFrame(self.control_frame)
        volume_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(volume_frame, text="Volume:").pack(side="left", padx=10)
        volume_slider = ctk.CTkSlider(
            volume_frame, 
            from_=0, to=1, 
            command=self.on_volume_change
        )
        volume_slider.pack(side="left", padx=10, fill="x", expand=True)
        volume_slider.set(0.7)  # Default volume
        
        button_frame = ctk.CTkFrame(self.control_frame)
        button_frame.pack(fill="x", padx=20, pady=10)
        
        start_button = ctk.CTkButton(
            button_frame, 
            text="Start",
            command=self.start
        )
        start_button.pack(side="left", padx=10, pady=10, fill="x", expand=True)
        
        stop_button = ctk.CTkButton(
            button_frame, 
            text="Stop",
            command=self.stop
        )
        stop_button.pack(side="left", padx=10, pady=10, fill="x", expand=True)
        
        status_frame = ctk.CTkFrame(self.app)
        status_frame.pack(fill="x", padx=20, pady=5)
        
        self.status_label = ctk.CTkLabel(status_frame, text="Ready")
        self.status_label.pack(pady=5)
        
        if not self.music_player.is_music_available():
            self.show_music_setup_instructions()
    
    def show_music_setup_instructions(self):
        """Show instructions for setting up music files"""
        instructions_window = ctk.CTkToplevel(self.app)
        instructions_window.title("Music Setup Required")
        instructions_window.geometry("600x400")
        
        frame = ctk.CTkFrame(instructions_window)
        frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        title = ctk.CTkLabel(
            frame,
            text="🎵 Music Setup Required 🎵",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title.pack(pady=10)
        
        instructions = ctk.CTkTextbox(frame)
        instructions.pack(fill="both", expand=True, padx=20, pady=10)
        
        instructions.insert("1.0", "No music files were found in the data directory!\n\n")
        instructions.insert("end", "Please add Egyptian music files to the following directories:\n\n")
        
        for emotion in self.music_player.emotions:
            instructions.insert("end", f"• data/music/{emotion}/\n")
        
        instructions.insert("end", "\nAdd MP3, WAV, or OGG files to these directories based on their emotional content.\n")
        instructions.insert("end", "For example, put upbeat songs in the 'happy' folder and melancholic ones in the 'sad' folder.\n\n")
        instructions.insert("end", "Once you've added the music files, restart the application.")
        
        instructions.configure(state="disabled")
        
        ok_button = ctk.CTkButton(
            frame,
            text="OK",
            command=instructions_window.destroy
        )
        ok_button.pack(pady=10)
    
    def update_camera(self):
        """Update the camera feed and detect emotions"""
        if not self.is_running:
            return
        
        try:
            emotion, confidence, frame = self.emotion_detector.detect_emotion()
            
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                img = self._resize_image(img, self.camera_frame.winfo_width() - 40, 
                                         self.camera_frame.winfo_height() - 100)
                
                photo = ImageTk.PhotoImage(image=img)
                self.camera_label.configure(image=photo)
                self.camera_label.image = photo
                
                if emotion and emotion != "no_face":
                    self.emotion_label.configure(text=emotion.capitalize())
                    self.confidence_bar.set(confidence)
                    
                    if confidence > 0.4 and emotion != self.music_player.get_current_emotion():
                        self.update_status(f"Detected {emotion} emotion, changing music...")
                        if self.music_player.play(emotion):
                            self.music_label.configure(text=self.music_player.get_current_track_name())
                
                elif emotion == "no_face":
                    self.emotion_label.configure(text="No Face")
                    self.confidence_bar.set(0)
            
            self.app.after(self.update_interval, self.update_camera)
        
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            self.app.after(1000, self.update_camera)  
    
    def _resize_image(self, img, width, height):
        """Resize image maintaining aspect ratio"""
        img_width, img_height = img.size
        
        if img_width > img_height:
            new_width = min(img_width, width)
            new_height = int(img_height * (new_width / img_width))
        else:
            new_height = min(img_height, height)
            new_width = int(img_width * (new_height / img_height))
        
        return img.resize((new_width, new_height), Image.LANCZOS)
    
    def update_status(self, message):
        """Update the status bar message"""
        self.status_label.configure(text=message)
    
    def on_volume_change(self, value):
        """Handle volume slider change"""
        self.music_player.set_volume(float(value))
    
    def start(self):
        """Start the camera and emotion detection"""
        if not self.is_running:
            try:
                self.emotion_detector.start_camera()
                self.is_running = True
                self.update_status("Camera started. Detecting emotions...")
                self.update_camera()
            except Exception as e:
                self.update_status(f"Error starting camera: {str(e)}")
    
    def stop(self):
        """Stop the camera and emotion detection"""
        if self.is_running:
            self.is_running = False
            self.emotion_detector.close()
            self.music_player.stop()
            self.update_status("Stopped")
            self.emotion_label.configure(text="None")
            self.music_label.configure(text="None")
            self.confidence_bar.set(0)
    
    def on_closing(self):
        """Handle window closing event"""
        self.stop()

        if self.app:
            self.app.destroy()
    
    def run(self):
        """Start the UI main loop"""
        self.update_status("Ready to start. Click 'Start' to begin.")
        self.app.mainloop()

if __name__ == "__main__":
    from emotion_detector import EmotionDetector
    from music_player import MusicPlayer
    
    detector = EmotionDetector()
    player = MusicPlayer()
    ui = UIManager(detector, player)
    ui.run()