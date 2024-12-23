import warnings

from app.sceneoptimizer import detect_scenes
warnings.filterwarnings('ignore')

import streamlit as st
from app.home import show_home
from app.trim_video import show_trim_video
from app.enhance_video import show_enhance_video
from app.sceneoptimizer import show_sceneoptimizer
from app.highlight_extractor import show_highlight_extractor
from app.emotion import show_emotion_based_highlight_reel
from app.transition import show_transition
from app.subtitle import show_subtitle
from app.resize_video import show_resize_video
st.set_page_config(
    page_title="VidAI - AI Video Editor",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
with open('assets/styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main():
    # Sidebar
    st.sidebar.title("VidAI")
    st.sidebar.markdown("---")
    
    # Navigation
    pages = {
        "Home": show_home,
        "Trim Video": show_trim_video,
        "Enhance Video": show_enhance_video,
        "Highlight Reel": show_highlight_extractor,
        "Scene Optimizer":show_sceneoptimizer,
        "Emotion-Based Highlight Reel": show_emotion_based_highlight_reel,
        "Transition": show_transition,
        "Subtitle": show_subtitle,
        "Resize Video": show_resize_video,
    }
    
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    
    # Display selected page
    pages[selection]()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "VidAI is an AI-powered video editing tool that helps you "
        "create professional videos with ease."
    )

if __name__ == "__main__":
    main()
