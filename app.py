import streamlit as st
import pandas as pd
import os
import logging
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyktok as pyk
import re
import glob
from tenacity import retry, stop_after_attempt, wait_exponential


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set OpenAI API key
openai.api_key = st.secrets['OPENAI_API_KEY']

# Set Streamlit page configuration
st.set_page_config(page_title="TikTok Video Transcription", layout="wide")

# Workaround for DBUS_SESSION_BUS_ADDRESS error
if 'DBUS_SESSION_BUS_ADDRESS' not in os.environ:
    os.environ['DBUS_SESSION_BUS_ADDRESS'] = 'unix:path=/run/user/1000/bus'

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
def transcribe_audio_with_whisper(audio_path):
    logger.info(f"Transcribing audio from {audio_path}")
    try:
        with open(audio_path, "rb") as audio_file:
            response = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        logger.info(f"Transcription completed for {audio_path}")
        return response['text']
    except Exception as e:
        logger.error(f"Error transcribing audio with Whisper: {e}")
        return ""

def download_video_in_memory(url):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        video_data = io.BytesIO(response.content)
        return video_data
    else:
        logger.error(f"Error downloading video from URL: {url}")
        return None

def process_video_in_memory(video_data):
    video = mp.VideoFileClip(video_data)
    audio_data = io.BytesIO()
    video.audio.write_audiofile(audio_data, verbose=False, logger=None)
    return audio_data

def download_and_process_video(url, browser_name='chrome'):
    try:
        # Specify browser for Pyktok
        pyk.specify_browser(browser_name)

        # Download video using Pyktok and get video URL
        video_metadata = pyk.get_tiktok_json(url)
        video_url = video_metadata['videoData']['itemInfos']['video']['urls'][0]

        # Download video in memory
        video_data = download_video_in_memory(video_url)
        if video_data is None:
            raise FileNotFoundError(f"Video file not found after download for URL: {url}")

        # Process video in memory
        audio_data = process_video_in_memory(video_data)

        # Transcribe audio
        transcription = transcribe_audio_with_whisper(audio_data)

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
