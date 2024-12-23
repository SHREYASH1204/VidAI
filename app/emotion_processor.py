import cv2
from diffusers import StableDiffusionPipeline
from PIL import Image
import torch
import streamlit as st
from fer import FER
import numpy as np
import pygame
import os
import tempfile
import shutil
from datetime import datetime
from moviepy.editor import ImageClip, AudioFileClip
from scipy.io import wavfile
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time

# Initialize Spotify client
client_credentials_manager = SpotifyClientCredentials(
    client_id='647d6d6fd0af403bbeb245171f80505f',
    client_secret='97d22dd241984ea48eb2ab65067180fc'
)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def detect_emotion_from_image(img):
    try:
        if isinstance(img, Image.Image):
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        detector = FER(mtcnn=True)
        result = detector.detect_emotions(img)
        if result and len(result) > 0:
            emotions = result[0]['emotions']
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            return dominant_emotion
        return "neutral"
    except Exception as e:
        st.error(f"Error detecting emotion: {e}")
        return "neutral"

def detect_emotion_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    emotions = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 10 == 0:
            try:
                emotion = detect_emotion_from_image(frame)
                emotions.append(emotion)
            except Exception as e:
                st.error(f"Error processing frame {frame_count}: {e}")
        frame_count += 1
    cap.release()
    if emotions:
        return max(set(emotions), key=emotions.count)
    return "neutral"

def generate_background(emotion):
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe = pipe.to("cpu")
    emotion_prompts = {
        "happy": "a bright sunny day in a beautiful garden with blooming flowers, cheerful atmosphere",
        "sad": "a rainy day with gray clouds, melancholic atmosphere, gentle rain falling",
        "angry": "dramatic stormy sky with dark clouds and lightning, intense atmosphere",
        "fear": "dark misty forest with fog, mysterious atmosphere",
        "surprise": "magical starry night sky with aurora borealis, wonderful atmosphere",
        "neutral": "calm serene landscape with soft clouds, peaceful atmosphere",
        "disgust": "abstract dark pattern with moody lighting",
    }
    prompt = emotion_prompts.get(emotion.lower(), "beautiful landscape with natural lighting")
    with torch.no_grad():
        image = pipe(prompt, num_inference_steps=20).images[0]
    return image

def generate_music(emotion, duration=5):
    try:
        pygame.mixer.init(frequency=44100)
        sample_rate = 44100
        def generate_tone(frequency, duration, volume=0.5):
            t = np.linspace(0, duration, int(sample_rate * duration))
            tone = np.sin(2 * np.pi * frequency * t)
            return (tone * volume * 32767).astype(np.int16)
        def save_temp_music(audio_data, filename="temp_music.wav"):
            wavfile.write(filename, sample_rate, audio_data)
            return filename
        emotion_music = {
            "happy": {"frequencies": [392, 440, 494, 523], "rhythm": 0.2, "volume": 0.6},
            "sad": {"frequencies": [440, 415, 392, 370], "rhythm": 0.4, "volume": 0.4},
            "angry": {"frequencies": [277, 311, 370, 415], "rhythm": 0.15, "volume": 0.7},
            "fear": {"frequencies": [233, 277, 311, 370], "rhythm": 0.3, "volume": 0.5},
            "surprise": {"frequencies": [523, 587, 659, 698], "rhythm": 0.25, "volume": 0.6},
            "neutral": {"frequencies": [349, 392, 440, 494], "rhythm": 0.3, "volume": 0.5},
            "disgust": {"frequencies": [466, 415, 392, 370], "rhythm": 0.2, "volume": 0.5}
        }
        params = emotion_music.get(emotion.lower(), emotion_music["neutral"])
        music_data = np.array([], dtype=np.int16)
        for _ in range(int(duration / params["rhythm"])):
            freq = np.random.choice(params["frequencies"])
            tone = generate_tone(freq, params["rhythm"], params["volume"])
            music_data = np.concatenate([music_data, tone])
        timestamp = int(time.time())
        temp_file = f"temp_music_{timestamp}.wav"
        save_temp_music(music_data, temp_file)
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()
        return temp_file
    except Exception as e:
        st.error(f"Error generating music: {e}")
        return None

def get_spotify_recommendations(emotion):
    try:
        emotion_params = {
            "happy": {"seed_genres": ["pop", "dance", "disco"], "target_valence": 0.8, "target_energy": 0.8, "limit": 5},
            "sad": {"seed_genres": ["classical", "piano", "indie"], "target_valence": 0.2, "target_energy": 0.3, "limit": 5},
            "angry": {"seed_genres": ["metal", "rock", "punk"], "target_valence": 0.3, "target_energy": 0.9, "limit": 5},
            "fear": {"seed_genres": ["ambient", "classical", "instrumental"], "target_valence": 0.3, "target_energy": 0.4, "limit": 5},
            "surprise": {"seed_genres": ["electronic", "dance", "pop"], "target_valence": 0.7, "target_energy": 0.7, "limit": 5},
            "neutral": {"seed_genres": ["indie", "alternative", "folk"], "target_valence": 0.5, "target_energy": 0.5, "limit": 5},
            "disgust": {"seed_genres": ["industrial", "electronic", "rock"], "target_valence": 0.3, "target_energy": 0.6, "limit": 5}
        }
        params = emotion_params.get(emotion.lower(), emotion_params["neutral"])
        seed_genres = ",".join(params["seed_genres"])
        recommendations = sp.recommendations(
            seed_genres=seed_genres,
            target_valence=params["target_valence"],
            target_energy=params["target_energy"],
            limit=params["limit"]
        )
        songs = []
        for track in recommendations['tracks']:
            artists = ", ".join([artist['name'] for artist in track['artists']])
            song_info = f"{track['name']} - {artists}"
            song_url = track['external_urls']['spotify']
            songs.append({'name': song_info, 'url': song_url})
        return songs
    except Exception as e:
        st.error(f"Error getting Spotify recommendations: {e}")
        return []

def create_audiovisual_experience(background_path, audio_path, emotion):
    try:
        video_clip = ImageClip(background_path)
        audio_clip = AudioFileClip(audio_path)
        video_clip = video_clip.set_duration(audio_clip.duration)
        final_clip = video_clip.set_audio(audio_clip)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"emotion_experience_{emotion}_{timestamp}.mp4"
        final_clip.write_videofile(output_path, fps=24)
        video_clip.close()
        audio_clip.close()
        return output_path
    except Exception as e:
        st.error(f"Error creating audiovisual experience: {e}")
        return None

def blend_images(foreground, background):
    """Blend the foreground image with the background image"""
    try:
        # Convert PIL images to numpy arrays if needed
        if isinstance(foreground, Image.Image):
            foreground = np.array(foreground)
        if isinstance(background, Image.Image):
            background = np.array(background)
        
        # Resize background to match foreground dimensions
        background = cv2.resize(np.array(background), (foreground.shape[1], foreground.shape[0]))
        
        # Convert to float32 for alpha blending
        foreground = foreground.astype(float)
        background = background.astype(float)
        
        # Alpha blending parameters
        alpha = 0.7  # Adjust this value to control blending (0.0 to 1.0)
        beta = 1.0 - alpha
        
        # Perform the blending
        blended = cv2.addWeighted(foreground, alpha, background, beta, 0)
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        # Convert back to PIL Image
        return Image.fromarray(blended)
    except Exception as e:
        st.error(f"Error blending images: {e}")
        return None

def segment_person(image):
    """Segment person from image using MediaPipe"""
    try:
        import mediapipe as mp
        
        # Initialize MediaPipe
        mp_selfie_segmentation = mp.solutions.selfie_segmentation
        selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        
        # Convert PIL to cv2 format
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Process the image
        results = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Get segmentation mask
        mask = results.segmentation_mask
        mask = np.stack((mask,) * 3, axis=-1) > 0.1
        
        return mask, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        st.error(f"Error in person segmentation: {e}")
        return None, None

def blend_with_background(foreground, background, mask):
    """Replace background while keeping the person"""
    try:
        # Convert PIL images to numpy arrays if needed
        if isinstance(foreground, Image.Image):
            foreground = np.array(foreground)
        if isinstance(background, Image.Image):
            background = np.array(background)
            
        # Ensure background is in RGB
        if background.shape[-1] == 4:  # If RGBA
            background = background[:, :, :3]
            
        # Resize background to match foreground dimensions
        background = cv2.resize(background, (foreground.shape[1], foreground.shape[0]))
        
        # Create the composite image
        composite = np.where(mask, foreground, background)
        
        # Optional: Add slight blending at the edges
        kernel = np.ones((5,5), np.float32)/25
        mask_blurred = cv2.filter2D(mask.astype(np.float32), -1, kernel)
        mask_blurred = np.stack((mask_blurred,) * 3, axis=-1)
        
        blended = composite * mask_blurred + background * (1 - mask_blurred)
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        return Image.fromarray(blended)
    except Exception as e:
        st.error(f"Error blending with background: {e}")
        return None

def process_media(media_path):
    music_file = None
    background_path = None
    final_output = None
    try:
        # Load the input image
        input_image = Image.open(media_path)
        
        # Detect emotion
        emotion = detect_emotion_from_image(input_image)
        st.write(f"Detected emotion: {emotion}")
        
        # Generate background image
        background = generate_background(emotion)
        if background is None:
            st.error("Failed to generate background. Please check your API key and connection.")
            return emotion, None
        
        # Segment person from image
        mask, foreground = segment_person(input_image)
        if mask is None:
            st.error("Failed to segment person from image")
            return emotion, None
        
        # Blend person with new background
        result_image = blend_with_background(foreground, background, mask)
        if result_image is None:
            st.error("Failed to blend images")
            return emotion, None
            
        # Save result image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"background_changed_{emotion}_{timestamp}.png"
        result_image.save(output_path, format="PNG")
        
        # Show the result
        st.image(output_path, caption="Result with Changed Background", use_column_width=True)
        
        # Rest of the function remains the same...
        music_file = generate_music(emotion)
        if music_file:
            st.write("Playing generated music...")
            
        songs = get_spotify_recommendations(emotion)
        if songs:
            st.write("Spotify Recommendations for your mood:")
            for song in songs:
                st.write(f"🎵 {song['name']}")
                st.write(f"   Listen here: {song['url']}")
        
        if music_file:
            st.write("Creating audiovisual experience...")
            final_output = create_audiovisual_experience(output_path, music_file, emotion)
            
        return emotion, final_output or output_path
        
    except Exception as e:
        st.error(f"Error processing media: {e}")
        return None, None
    finally:
        # Clean up temporary files
        for file_path in [music_file, background_path]:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    st.error(f"Error cleaning up file {file_path}: {e}")

def show_emotion_processor():
    st.title("Emotion-Based Media Processor 🎭")
    st.write("Upload an image or video to detect emotions and create an audiovisual experience.")
    
    uploaded_file = st.file_uploader("Upload your media file", type=["png", "jpg", "jpeg", "mp4", "avi"])
    
    if uploaded_file:
        # Create a temp directory for processing
        temp_dir = tempfile.mkdtemp()
        try:
            # Save uploaded file
            media_path = os.path.join(temp_dir, uploaded_file.name)
            with open(media_path, "wb") as f:
                f.write(uploaded_file.read())
            
            # Display the uploaded media
            if media_path.lower().endswith(('.mp4', '.avi')):
                st.video(media_path)
            else:
                # For images, use st.image with caption
                st.image(media_path, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Process Media"):
                with st.spinner("Processing your media..."):
                    emotion, output_path = process_media(media_path)
                    
                    if emotion and output_path:
                        st.success(f"Processing complete! Detected emotion: {emotion}")
                        
                        # Display the generated background image
                        if os.path.exists(output_path):
                            if output_path.endswith(('.mp4')):
                                st.video(output_path)
                            else:
                                st.image(output_path, caption="Generated Experience", use_column_width=True)
                            
                            # Create download button
                            with open(output_path, "rb") as file:
                                btn = st.download_button(
                                    label="Download Audiovisual Experience",
                                    data=file,
                                    file_name=os.path.basename(output_path),
                                    mime="video/mp4" if output_path.endswith('.mp4') else "image/png"
                                )
                    else:
                        st.error("Failed to process the media. Please try again.")
        
        finally:
            # Clean up temp directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                st.error(f"Error cleaning up temporary files: {e}") 