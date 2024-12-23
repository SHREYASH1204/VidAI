import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path

def apply_super_resolution(frame, scale_factor=2):
    """Apply super resolution to a frame"""
    return cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

def apply_sharpening(frame, kernel_size=3, strength=1.0):
    """Apply sharpening with adjustable strength"""
    kernel = np.array([
        [-1, -1, -1],
        [-1, 9 + strength, -1],
        [-1, -1, -1]
    ])
    return cv2.filter2D(frame, -1, kernel)

def apply_brightness(frame, value):
    """Apply brightness adjustment"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = cv2.add(hsv[:,:,2], value)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_contrast(frame, value):
    """Apply contrast adjustment"""
    return cv2.convertScaleAbs(frame, alpha=value, beta=0)

def process_video(input_path, enhancement_params):
    """Process video with selected enhancement options"""
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    output_path = str(temp_dir / "enhanced_video.mp4")
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video at path: {input_path}")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Adjust output dimensions if super resolution is enabled
    if enhancement_params.get('super_resolution', False):
        frame_width *= 2
        frame_height *= 2
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update progress
            progress = int((frame_count / total_frames) * 100)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_count}/{total_frames}")
            
            # Apply selected enhancements
            if enhancement_params.get('super_resolution'):
                frame = apply_super_resolution(frame)
            
            if enhancement_params.get('sharpening'):
                frame = apply_sharpening(frame, strength=enhancement_params['sharpening_strength'])
            
            if enhancement_params.get('brightness'):
                frame = apply_brightness(frame, enhancement_params['brightness_value'])
            
            if enhancement_params.get('contrast'):
                frame = apply_contrast(frame, enhancement_params['contrast_value'])
            
            out.write(frame)
            frame_count += 1
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return output_path
        
    finally:
        cap.release()
        out.release()

def show_enhance_video():
    st.title("Enhance Video 🎨")
    
    uploaded_file = st.file_uploader(
        "Upload your video", 
        type=['mp4', 'mov', 'avi'],
        help="Upload a video file to enhance"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()
        
        try:
            # Show original video
            st.markdown("### Original Video")
            st.video(uploaded_file)
            
            # Enhancement options with detailed controls
            st.markdown("### Enhancement Options")
            
            enhancement_params = {}
            
            # Super Resolution
            if st.checkbox("Super Resolution (2x)", help="Increase video resolution"):
                enhancement_params['super_resolution'] = True
            
            # Sharpening
            if st.checkbox("Sharpening", help="Enhance video sharpness"):
                enhancement_params['sharpening'] = True
                enhancement_params['sharpening_strength'] = st.slider(
                    "Sharpening Strength",
                    0.0, 2.0, 1.0,
                    help="Adjust the strength of sharpening effect"
                )
            
            # Brightness
            if st.checkbox("Brightness", help="Adjust video brightness"):
                enhancement_params['brightness'] = True
                enhancement_params['brightness_value'] = st.slider(
                    "Brightness Adjustment",
                    -50, 50, 0,
                    help="Adjust the brightness level"
                )
            
            # Contrast
            if st.checkbox("Contrast", help="Adjust video contrast"):
                enhancement_params['contrast'] = True
                enhancement_params['contrast_value'] = st.slider(
                    "Contrast Adjustment",
                    0.5, 2.0, 1.0, 0.1,
                    help="Adjust the contrast level"
                )
            
            if enhancement_params and st.button("Enhance Video", type="primary"):
                try:
                    with st.spinner("Processing video... This may take a while."):
                        output_path = process_video(tfile.name, enhancement_params)
                        
                        st.success("Video enhanced successfully! 🎉")
                        
                        # Show enhanced video
                        st.markdown("### Enhanced Video")
                        with open(output_path, 'rb') as f:
                            st.video(f)
                        
                        # Add download button
                        with open(output_path, 'rb') as f:
                            st.download_button(
                                label="Download Enhanced Video",
                                data=f.read(),
                                file_name="enhanced_video.mp4",
                                mime="video/mp4"
                            )
                        
                        # Cleanup enhanced video
                        os.unlink(output_path)
                
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")
        
        finally:
            # Cleanup input file
            try:
                os.unlink(tfile.name)
            except:
                pass
    
    else:
        st.markdown("""
        ### How to enhance your video:
        1. Upload your video using the file uploader above
        2. Select the enhancement options you want to apply:
           - **Super Resolution**: Increase video resolution (2x)
           - **Sharpening**: Enhance video sharpness with adjustable strength
           - **Brightness**: Fine-tune video brightness
           - **Contrast**: Adjust video contrast levels
        3. Click "Enhance Video" to process your video
        4. Download the enhanced video when processing is complete
        """)
