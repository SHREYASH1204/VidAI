import streamlit as st
from moviepy.editor import VideoFileClip, concatenate_videoclips
import tempfile
import os

def trim_video_segments(video_path, trim_times):
    """
    Trim multiple segments from a video and concatenate them.
    
    Args:
        video_path (str): Path to the input video
        trim_times (list): List of tuples containing (start_time, end_time) for each segment
    
    Returns:
        str: Path to the output video
    """
    try:
        # Load video
        video = VideoFileClip(video_path)
        
        # Create clips for each segment
        clips = []
        for start_time, end_time in trim_times:
            if start_time < end_time and end_time <= video.duration:
                clip = video.subclip(start_time, end_time)
                clips.append(clip)
        
        if not clips:
            raise ValueError("No valid segments to trim")
            
        # Concatenate all clips
        final_clip = concatenate_videoclips(clips, method="compose")
        
        # Save the final video
        output_path = "temp_trimmed.mp4"
        final_clip.write_videofile(output_path)
        
        # Cleanup
        video.close()
        for clip in clips:
            clip.close()
        final_clip.close()
        
        return output_path
        
    except Exception as e:
        raise Exception(f"Error processing video: {str(e)}")

def show_trim_video():
    st.title("Trim Video ✂️")
    
    uploaded_file = st.file_uploader("Upload your video", type=['mp4', 'mov', 'avi'])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()
        
        # Load video
        video = VideoFileClip(tfile.name)
        duration = video.duration
        video.close()
        
        st.video(uploaded_file)
        
        # Trim controls
        st.markdown("### Trim Settings")
        
        # Initialize trim times in session state if not exists
        if 'trim_times' not in st.session_state:
            st.session_state.trim_times = [(0, duration)]
        
        # Add segment button
        if st.button("Add Another Segment"):
            st.session_state.trim_times.append((0, duration))
        
        # Create sliders for each segment
        updated_trim_times = []
        for i, (start, end) in enumerate(st.session_state.trim_times):
            st.markdown(f"#### Segment {i+1}")
            col1, col2 = st.columns(2)
            with col1:
                new_start = st.slider(
                    f"Start Time {i+1} (seconds)", 
                    0, 
                    int(duration), 
                    int(start)
                )
            with col2:
                new_end = st.slider(
                    f"End Time {i+1} (seconds)", 
                    0, 
                    int(duration), 
                    int(end)
                )
            updated_trim_times.append((new_start, new_end))
        
        # Update session state
        st.session_state.trim_times = updated_trim_times
        
        if st.button("Trim Video"):
            try:
                if len(st.session_state.trim_times) > 0:
                    output_path = trim_video_segments(tfile.name, st.session_state.trim_times)
                    
                    st.success("Video trimmed successfully!")
                    st.video(output_path)
                    
                    # Add download button
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="Download Trimmed Video",
                            data=f,
                            file_name="trimmed_video.mp4",
                            mime="video/mp4"
                        )
                    
                    # Cleanup
                    os.unlink(output_path)
                else:
                    st.error("Please add at least one segment to trim!")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
            
            finally:
                # Cleanup
                os.unlink(tfile.name)
