import cv2
import os
import numpy as np
import streamlit as st
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import tempfile
import shutil
import time

def extract_highlight(video_path, highlight_duration):
    """
    Extracts the most interesting part of the video based on audio intensity and scene changes.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"The file '{video_path}' does not exist.")

    temp_audio_path = None
    video = None
    
    try:
        video = VideoFileClip(video_path)
        
        # Adjust highlight duration if video is too short
        if video.duration < highlight_duration:
            st.warning(f"Requested highlight duration ({highlight_duration}s) exceeds video duration ({video.duration}s).")
            highlight_duration = video.duration
            st.warning(f"Highlight duration adjusted to {highlight_duration:.2f} seconds.")

        # Extract audio and analyze nonsilent parts
        temp_audio_path = tempfile.mktemp(suffix=".wav")
        video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
        video.close()  # Close video after audio extraction

        audio = AudioSegment.from_file(temp_audio_path, format="wav")
        silence_thresh = audio.dBFS - 10  # Dynamic silence threshold
        nonsilent_ranges = detect_nonsilent(audio, min_silence_len=300, silence_thresh=silence_thresh)

        if not nonsilent_ranges:
            raise ValueError("No significant audio activity detected in the video.")

        # Reopen video for frame analysis
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        nonsilent_ranges_frames = [(int(start / 1000 * fps), int(end / 1000 * fps)) 
                                 for start, end in nonsilent_ranges]

        frame_diffs = []
        prev_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame is not None:
                diff = cv2.absdiff(gray_frame, prev_frame)
                frame_diffs.append(np.sum(diff))
            prev_frame = gray_frame

        cap.release()
        frame_diffs = frame_diffs[:total_frames]

        if not frame_diffs:
            raise ValueError("No significant scene changes detected in the video.")

        # Combine scores
        frame_scores = np.zeros(total_frames)
        for start, end in nonsilent_ranges_frames:
            frame_scores[start:end] += 1

        for i, diff in enumerate(frame_diffs):
            if i < len(frame_scores):
                frame_scores[i] += diff

        # Adjust highlight duration
        highlight_frames = int(highlight_duration * fps)
        if len(frame_scores) < highlight_frames:
            highlight_duration = len(frame_scores) / fps
            st.warning(f"Highlight duration adjusted to {highlight_duration:.2f} seconds.")

        highlight_frames = int(highlight_duration * fps)
        start_frame = np.argmax(
            [np.sum(frame_scores[i:i + highlight_frames]) 
             for i in range(len(frame_scores) - highlight_frames)]
        )
        end_frame = start_frame + highlight_frames

        # Extract and save the highlight
        output_path = tempfile.mktemp(suffix=".mp4")
        highlight_clip = VideoFileClip(video_path).subclip(start_frame / fps, end_frame / fps)
        highlight_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", 
                                     verbose=False, logger=None)
        highlight_clip.close()

        return output_path

    except Exception as e:
        raise e

    finally:
        # Cleanup resources
        if video is not None and video.reader:
            video.close()
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except:
                pass

def show_highlight_extractor():
    st.title("Highlight Extractor 🎬")
    st.write("Upload a video to extract the most interesting highlights based on audio intensity and scene changes.")

    uploaded_file = st.file_uploader("Upload your video", type=["mp4", "avi"])
    if uploaded_file:
        # Create a unique temporary directory
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, "input_video.mp4")
        
        try:
            # Save uploaded file
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())

            st.video(video_path)

            highlight_duration = st.number_input(
                "Enter the desired highlight duration in seconds", 
                min_value=1, 
                value=10
            )

            if st.button("Extract Highlight"):
                try:
                    with st.spinner("Processing video..."):
                        output_path = extract_highlight(video_path, highlight_duration)
                        
                        st.success("Highlight extraction completed successfully.")
                        st.video(output_path)
                        
                        # Read the file content before offering download
                        with open(output_path, "rb") as f:
                            video_bytes = f.read()
                        
                        st.download_button(
                            "Download Highlight",
                            video_bytes,
                            file_name="highlight.mp4",
                            mime="video/mp4"
                        )
                        
                        # Clean up the output file
                        try:
                            os.unlink(output_path)
                        except:
                            pass

                except Exception as e:
                    st.error(f"Error: {str(e)}")

        finally:
            # Ensure all files are closed before cleanup
            time.sleep(0.5)  # Small delay to ensure files are released
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                st.warning(f"Could not clean up temporary files: {str(e)}") 