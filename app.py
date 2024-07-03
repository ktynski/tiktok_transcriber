import streamlit as st
import pandas as pd
import os
import logging
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyktok as pyk
import re
import glob


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set OpenAI API key
openai.api_key = st.secrets['OPENAI_API_KEY']

# Set Streamlit page configuration
st.set_page_config(page_title="TikTok Video Transcriptionasdfasdf", layout="wide")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
def transcribe_audio_with_whisper(audio_path):
    logger.info(f"Transcribing audio from {audio_path}")
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        logger.info(f"Transcription completed for {audio_path}")
        return transcript
    except Exception as e:
        logger.error(f"Error transcribing audio with Whisper: {e}")
        return ""

def get_latest_video_file():
    list_of_files = glob.glob('*.mp4')
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def download_and_process_video(url, browser_name='chrome'):
    try:
        # Specify browser for Pyktok
        pyk.specify_browser(browser_name)

        # Download video using Pyktok
        video_data_path = 'video_data.csv'
        pyk.save_tiktok(url, True, video_data_path, browser_name)
        
        # Determine video path based on the most recently created .mp4 file
        video_path = get_latest_video_file()
        
        if not video_path or not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found after download for URL: {url}")

        # Extract audio
        logger.info(f"Extracting audio from {video_path}")
        video = mp.VideoFileClip(video_path)
        audio_path = video_path.replace('.mp4', '.wav')
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)

        # Transcribe audio
        transcription = transcribe_audio_with_whisper(audio_path)

        # Clean up temporary files
        video.close()
        os.remove(audio_path)
        logger.info(f"Temporary files cleaned up for video {url}")

        return {"URL": url, "Transcript": transcription}
    except Exception as e:
        logger.error(f"Error processing video {url}: {str(e)}")
        return {"URL": url, "Transcript": f"Error: {str(e)}"}

def process_videos_concurrently(urls, browser_name='chrome'):
    logger.info(f"Starting concurrent processing of {len(urls)} videos")
    results = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(download_and_process_video, url, browser_name) for url in urls]

        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Completed processing video {i + 1}/{len(urls)}")
            except Exception as e:
                logger.error(f"Error processing video {i + 1}/{len(urls)}: {str(e)}")

    logger.info(f"Concurrent processing completed. Total results: {len(results)}")
    return results

def main():
    st.title("TikTok Video Transcription")
    st.markdown("Provide the TikTok URLs to transcribe.")

    urls_input = st.text_area("TikTok URLs (one per line)", height=200)
    urls = [url.strip() for url in urls_input.split('\n') if url.strip()]

    if st.button("Transcribe Videos"):
        if not urls:
            st.warning("Please enter at least one URL.")
            return

        logger.info(f"Starting transcription process for {len(urls)} videos")
        st.write(f"Processing {len(urls)} videos. This may take a while...")

        transcripts = process_videos_concurrently(urls)

        if transcripts:
            df = pd.DataFrame(transcripts)
            st.subheader("Transcription Results")
            st.dataframe(df)
            logger.info("Transcription results displayed in Streamlit app")
        else:
            st.warning("No transcripts were generated. Please check the URLs and try again.")
            logger.warning("No transcripts generated")

if __name__ == "__main__":
    logger.info("Application started")
    main()
    logger.info("Application finished")
