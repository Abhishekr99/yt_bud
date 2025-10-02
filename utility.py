import time

from dotenv import load_dotenv
import re
import streamlit as st

from youtube_transcript_api import YouTubeTranscriptApi

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Function to extract video ID from a YouTube URL (Helper Function)
def extract_video_id(url):
    """
    Extracts the YouTube video ID from any valid YouTube URL.
    """
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if match:
        return match.group(1)
    st.error("Invalid YouTube URL. Please enter a valid video link.")
    return None

# function to get transcript from the video.
def get_transcript(video_id, language):
    ytt_api= YouTubeTranscriptApi()
    try:
        transcript= ytt_api.fetch(video_id, languages=[language])
        full_transcript= " ".join([i.text for i in transcript])
        time.sleep(10) # To avoid hitting rate limits
        return full_transcript
    except Exception as e:
        st.error(f"Error fething video {e}")
