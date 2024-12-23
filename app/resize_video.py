import streamlit as st
from moviepy.editor import VideoFileClip
import tempfile
import os

def resize_video_file(video_path, width, height):
    """
    Resize a video to specified dimensions.
    
    Args:
        video_path (str): Path to the input video
        width (int): Target width
        height (int): Target height
    
    Returns:
        str: Path to the output video
    """
    try:
        # Load video
        video = VideoFileClip(video_path)
        
        # Resize video
        resized_video = video.resize(newsize=(width, height))
        
        # Save the resized video
        output_path = "temp_resized.mp4"
        resized_video.write_videofile(output_path)
        
        # Cleanup
        video.close()
        resized_video.close()
        
        return output_path
        
    except Exception as e:
        raise Exception(f"Error processing video: {str(e)}")

def show_resize_video():
    st.title("Resize Video for Social Media 🎥")
    
    uploaded_file = st.file_uploader("Upload your video", type=['mp4', 'mov', 'avi'])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()
        
        try:
            # Show original video
            st.video(uploaded_file)
            
            # Resize controls
            st.markdown("### Resize Settings")
            
            # Platform selection
            platform = st.selectbox(
                "Choose a platform", 
                [
                    "YouTube (16:9)", 
                    "Instagram Post (1:1)", 
                    "Instagram Story (9:16)", 
                    "Instagram Reels (9:16)", 
                    "Custom"
                ]
            )
            
            # Initialize dimensions based on platform
            if platform == "YouTube (16:9)":
                width, height = 1920, 1080
            elif platform == "Instagram Post (1:1)":
                width, height = 1080, 1080
            elif platform in ["Instagram Story (9:16)", "Instagram Reels (9:16)"]:
                width, height = 1080, 1920
            else:  # Custom
                st.markdown("### Custom Dimensions")
                col1, col2 = st.columns(2)
                with col1:
                    width = st.number_input(
                        "Width",
                        min_value=100,
                        max_value=4000,
                        value=1080,
                        help="Enter target width in pixels"
                    )
                with col2:
                    height = st.number_input(
                        "Height",
                        min_value=100,
                        max_value=4000,
                        value=1920,
                        help="Enter target height in pixels"
                    )
            
            # Show current dimensions
            st.info(f"Output dimensions: {width}x{height} pixels")
            
            if st.button("Resize Video"):
                try:
                    with st.spinner("Resizing video..."):
                        output_path = resize_video_file(tfile.name, width, height)
                    
                    st.success(f"Video resized successfully for {platform}!")
                    st.video(output_path)
                    
                    # Add download button
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="Download Resized Video",
                            data=f,
                            file_name=f"resized_video_{width}x{height}.mp4",
                            mime="video/mp4"
                        )
                    
                    # Cleanup output file
                    os.unlink(output_path)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        finally:
            # Cleanup input file
            try:
                os.unlink(tfile.name)
            except Exception as e:
                st.error(f"Error cleaning up temporary file: {str(e)}")
    
    else:
        # Show instructions when no file is uploaded
        st.markdown("""
        ### How to use:
        1. Upload your video using the file uploader above
        2. Choose a target platform or custom dimensions
        3. Click "Resize Video" to process
        4. Download the resized video
        
        ### Available Platforms:
        - YouTube (1920x1080)
        - Instagram Post (1080x1080)
        - Instagram Story/Reels (1080x1920)
        - Custom (specify your own dimensions)
        """)

if __name__ == "__main__":
    show_resize_video() 