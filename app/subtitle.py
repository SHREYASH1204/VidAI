import os
import subprocess
import streamlit as st
from datetime import timedelta
import whisper
import pysrt
from pysrt import SubRipItem

def extract_audio(video_path, output_audio_path):
    if os.path.exists(output_audio_path):
        os.remove(output_audio_path)  # Remove existing file
    command = [
        'ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', output_audio_path
    ]
    subprocess.run(command, check=True)

def transcribe_hindi_audio_to_english_segments(audio_path, model_name='medium'):
    # Load Whisper model
    model = whisper.load_model(model_name)
    # Transcribe with translation to English
    result = model.transcribe(audio_path, language='hi', task='translate')
    return result['segments']  # Use segments for timing

def format_timedelta_to_srt_time(td):
    total_seconds = int(td.total_seconds())
    milliseconds = int((td.total_seconds() - total_seconds) * 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def create_srt_file_from_segments(segments, output_srt_path):
    if os.path.exists(output_srt_path):
        os.remove(output_srt_path)  # Remove existing file
    subs = pysrt.SubRipFile()

    for i, segment in enumerate(segments):
        start_time = timedelta(seconds=segment['start'])
        end_time = timedelta(seconds=segment['end'])
        text = segment['text'].strip()

        sub = SubRipItem(
            index=i + 1,
            start=format_timedelta_to_srt_time(start_time),
            end=format_timedelta_to_srt_time(end_time),
            text=text
        )
        subs.append(sub)

    subs.save(output_srt_path, encoding='utf-8')

def add_subtitles_to_video(video_path, srt_path, output_video_path):
    if os.path.exists(output_video_path):
        os.remove(output_video_path)  # Remove existing file
    command = [
        'ffmpeg',
        '-i', video_path,  # Input video
        '-vf', f"subtitles={srt_path}",  # Add subtitles
        '-c:v', 'libx264',  # Video codec
        '-c:a', 'copy',  # Copy audio without re-encoding
        '-strict', '-2',  # Compatibility flag
        output_video_path  # Output video file
    ]
    subprocess.run(command, check=True)

def show_subtitle():
    # Streamlit app
    st.title("Hindi Audio to English Subtitle Generator")

    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mkv"])
    if uploaded_video:
        with open(uploaded_video.name, "wb") as f:
            f.write(uploaded_video.getbuffer())
        
        st.video(uploaded_video)

        if st.button("Generate Subtitles"):
            try:
                audio_file = "extracted_audio.mp3"
                srt_file = "subtitles.srt"
                output_video_file = "output_video_with_subtitles.mp4"

                # Step 1: Extract audio from video
                st.text("Extracting audio from video...")
                extract_audio(uploaded_video.name, audio_file)

                # Step 2: Transcribe Hindi audio to English segments
                st.text("Transcribing Hindi audio to English segments...")
                segments = transcribe_hindi_audio_to_english_segments(audio_file)

                # Step 3: Create SRT file from transcription segments
                st.text("Creating SRT file...")
                create_srt_file_from_segments(segments, srt_file)

                # Step 4: Burn subtitles into the video
                st.text("Adding subtitles to video...")
                add_subtitles_to_video(uploaded_video.name, srt_file, output_video_file)

                st.success("Process complete! Check the output video with subtitles.")
                st.video(output_video_file)

                with open(output_video_file, "rb") as f:
                    st.download_button(
                        label="Download Video with Subtitles",
                        data=f,
                        file_name="video_with_subtitles.mp4",
                        mime="video/mp4"
                    )
            except Exception as e:
                st.error(f"An error occurred: {e}")