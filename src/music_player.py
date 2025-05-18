import os
import random
import pygame
import threading
import time

class MusicPlayer:
    def __init__(self, music_root="data/music"):
        pygame.mixer.init()
        
        self.music_root = music_root
        
        os.makedirs(self.music_root, exist_ok=True)
        
        self.emotions = ['happy', 'sad', 'neutral', 'angry', 'surprised']
        for emotion in self.emotions:
            os.makedirs(os.path.join(self.music_root, emotion), exist_ok=True)
        
        self.emotion_map = {
            'angry': 'angry',
            'disgust': 'angry',
            'fear': 'sad',
            'happy': 'happy',
            'sad': 'sad',
            'surprised': 'surprised',
            'neutral': 'neutral'
        }
        
        self.current_track = None
        self.current_emotion = None
        self.is_playing = False
        self._fade_thread = None
        
        self._check_sample_music()
    
    def _check_sample_music(self):
        """Check if there are any music files; if not, display a message"""
        has_music = False
        for emotion in self.emotions:
            dir_path = os.path.join(self.music_root, emotion)
            if os.path.exists(dir_path) and len(os.listdir(dir_path)) > 0:
                has_music = True
                break
        
        if not has_music:
            print("\nIMPORTANT: No music files found.")
            print(f"Please add Egyptian music files to the '{self.music_root}' directory, organized by emotion.")
            print("For example:")
            for emotion in self.emotions:
                print(f"  - {self.music_root}/{emotion}/")
            print("You'll need to add MP3 files to these directories to match the detected emotions.\n")
    
    def get_music_for_emotion(self, emotion):
        """
        Get a random music file for the detected emotion
        Returns the path to the music file or None if not found
        """
        mapped_emotion = self.emotion_map.get(emotion, 'neutral')
        
        emotion_dir = os.path.join(self.music_root, mapped_emotion)
        
        if not os.path.exists(emotion_dir):
            print(f"Warning: No music directory found for {mapped_emotion}")
            return None
        
        music_files = [f for f in os.listdir(emotion_dir) 
                      if f.endswith(('.mp3', '.wav', '.ogg'))]
        
        if not music_files:
            print(f"Warning: No music files found for {mapped_emotion}")
            return None
        
        selected_music = random.choice(music_files)
        return os.path.join(emotion_dir, selected_music)
    
    def play(self, emotion=None, fade=True):
        """
        Play music matching the given emotion
        If emotion is None, continue with current emotion
        """
        if emotion:
            self.current_emotion = emotion
        
        if not self.current_emotion:
            return False
        
        music_path = self.get_music_for_emotion(self.current_emotion)
        if not music_path:
            return False
        
        if self.is_playing and fade:
            self._fade_out()
        
        try:
            pygame.mixer.music.load(music_path)
            pygame.mixer.music.play(-1)  # Loop indefinitely
            self.current_track = music_path
            self.is_playing = True
            print(f"Now playing: {os.path.basename(music_path)} ({self.current_emotion})")
            return True
        except Exception as e:
            print(f"Error playing music: {e}")
            return False
    
    def stop(self, fade=True):
        if not self.is_playing:
            return
        
        if fade:
            self._fade_out()
        else:
            pygame.mixer.music.stop()
            self.is_playing = False
    
    def _fade_out(self, duration=1.0):
        if self._fade_thread and self._fade_thread.is_alive():
            return
        
        def fade():
            volume = 1.0
            decrement = 0.1
            steps = int(duration / 0.1)
            
            for _ in range(steps):
                volume = max(0.0, volume - decrement)
                pygame.mixer.music.set_volume(volume)
                time.sleep(0.1)
            
            pygame.mixer.music.stop()
            pygame.mixer.music.set_volume(1.0)
            self.is_playing = False
        
        self._fade_thread = threading.Thread(target=fade)
        self._fade_thread.daemon = True
        self._fade_thread.start()
    
    def set_volume(self, volume):
        pygame.mixer.music.set_volume(max(0.0, min(1.0, volume)))
    
    def get_current_track_name(self):
        if self.current_track:
            return os.path.basename(self.current_track)
        return "None"
    
    def get_current_emotion(self):
        return self.current_emotion
    
    def is_music_available(self):
        for emotion in self.emotions:
            dir_path = os.path.join(self.music_root, emotion)
            if os.path.exists(dir_path) and len([f for f in os.listdir(dir_path) 
                                              if f.endswith(('.mp3', '.wav', '.ogg'))]) > 0:
                return True
        return False
    
    def close(self):
        """Clean up resources"""
        self.stop(fade=False)
        pygame.mixer.quit()

if __name__ == "__main__":
    player = MusicPlayer()
    
    if player.is_music_available():
        print("Testing music player...")
        
        for emotion in ['happy', 'sad', 'neutral', 'angry', 'surprised']:
            print(f"Playing {emotion} music...")
            player.play(emotion)
            time.sleep(3)  
        
        player.stop()
        print("Music player test complete")
    else:
        print("No music files found for testing")
    
    player.close()