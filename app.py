import streamlit as st
import pandas as pd
import json
import os
from urllib.request import Request, urlopen, urlretrieve
from apify_client import ApifyClient
import base64
import requests
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import moviepy.editor as mp

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set OpenAI API key
openai.api_key = st.secrets['OPENAI_API_KEY']
APIFY_API_KEY = st.secretes['APIFY_API_KEY']

a_client = ApifyClient(APIFY_API_KEY)

# Set Streamlit page configuration
st.set_page_config(page_title="TikTok Video Transcription", layout="wide")

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
def fetch_video_metadata(keyword, num_videos, publish_time, extra_videos=2):
    try:
        with st.spinner("Fetching video metadata..."):
            run_input = {
                "keyword": keyword,
                "limit": num_videos + extra_videos,
                "sortType": 1,
                "region": "US",
                "publishTime": publish_time,
                "proxyConfiguration": {"useApifyProxy": True},
                "type": "SEARCH",
            }
            run = a_client.actor("nCNiU9QG1e0nMwgWj").call(run_input=run_input)
            url = f'https://api.apify.com/v2/actor-runs/{run["id"]}?waitForFinish='

            request = Request(url)
            response_body = urlopen(request, timeout=60).read()
            run_details = json.loads(response_body)

            if "defaultDatasetId" not in run:
                logger.error("No dataset ID found in the run details.")
                st.error("No dataset ID found in the run details.")
                return None

            total_items = run_details.get("itemCount", 1)

            items = list(a_client.dataset(run["defaultDatasetId"]).iterate_items())
            if not items:
                logger.warning("No items found in the dataset.")
                st.warning("No items found in the dataset.")
                return None

            progress_bar = st.progress(0)
            data = []
            for idx, item in enumerate(items):
                progress = (idx + 1) / total_items
                progress_bar.progress(progress)
                if isinstance(item, dict):
                    flat_record = flatten_dict(item)
                    data.append(flat_record)
                else:
                    logger.warning(f"Item {idx + 1} is not a dictionary and will be skipped.")
                    st.warning(f"Item {idx + 1} is not a dictionary and will be skipped.")

            df = pd.DataFrame(data)
            df['original_index'] = df.index
            return df.head(num_videos)  # Return only the desired number of videos
    except Exception as e:
        logger.error(f"Error fetching video metadata: {e}")
        st.error(f"Error fetching video metadata. Please try again.")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
def download_video(video_url, video_path):
    try:
        if video_url and isinstance(video_url, str):
            response = requests.get(video_url, stream=True)
            response.raise_for_status()
            with open(video_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Failed to download video from {video_url}")
        else:
            logger.warning(f"Invalid video URL: {video_url}")
            raise ValueError(f"Invalid video URL: {video_url}")
    except Exception as e:
        logger.error(f"Error downloading video: {e}")
        raise e

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
def transcribe_audio_with_whisper(audio_path):
    try:
        logger.info(f"Transcribing audio from {audio_path}")
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

def get_video_url(video_data):
    video_url = video_data.get('aweme_info_video_download_addr_url_list')
    if isinstance(video_url, list):
        video_url = video_url[0]
    elif isinstance(video_url, str):
        video_url = json.loads(video_url)[0] if video_url.startswith('[') else video_url
    return video_url

def process_video(video_data):
    try:
        logger.info(f"Processing video: {video_data['original_index']}")
        video_url = get_video_url(video_data)
        if not video_url:
            raise ValueError("Video URL not found.")

        video_path = f"video_{video_data['original_index']}.mp4"
        
        # Download video
        download_video(video_url, video_path)
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video path does not exist after download: {video_path}")

        # Extract audio
        video = mp.VideoFileClip(video_path)
        audio_path = f"audio_{video_data['original_index']}.wav"
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)

        # Transcribe audio
        transcription = transcribe_audio_with_whisper(audio_path)

        # Clean up temporary files
        video.close()
        os.remove(video_path)
        os.remove(audio_path)

        logger.info(f"Video processing completed for {video_data['original_index']}")
        return {"URL": video_url, "Transcript": transcription}
    except Exception as e:
        logger.error(f"Error processing video {video_data['original_index']}: {str(e)}")
        return {"URL": video_url if 'video_url' in locals() else "Unknown", "Transcript": f"Error: {str(e)}"}

def process_videos_concurrently(metadata_df):
    results = []
    progress_bar = st.progress(0)
    total_videos = len(metadata_df)
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_video, row) for _, row in metadata_df.iterrows()]
        
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing video: {str(e)}")
            
            # Update progress
            progress = (i + 1) / total_videos
            progress_bar.progress(progress)
            
    return results

def main():
    st.title("TikTok Video Transcription")
    st.markdown("Enter the keyword and number of videos to transcribe.")

    keyword = st.text_input("Keyword", value="")
    num_videos = st.number_input("Number of Videos", min_value=1, max_value=100, value=3)
    publish_time = st.selectbox("Publish Time", options=["WEEK", "MONTH", "YEAR"])

    if st.button("Transcribe Videos"):
        if not keyword.strip():
            st.warning("Please enter a keyword.")
            return

        metadata_df = fetch_video_metadata(keyword, num_videos, publish_time)
        if metadata_df is None or metadata_df.empty:
            return

        st.subheader("Video Metadata")
        relevant_columns = [
            'aweme_info_desc',
            'aweme_info_author_nickname',
            'aweme_info_video_duration',
            'aweme_info_statistics_play_count',
            'original_index'
        ]
        display_names = {
            'aweme_info_desc': 'Description',
            'aweme_info_author_nickname': 'Author',
            'aweme_info_video_duration': 'Duration',
            'aweme_info_statistics_play_count': 'Play Count'
        }
        st.dataframe(metadata_df[relevant_columns].rename(columns=display_names))

        st.write(f"Processing {len(metadata_df)} videos. This may take a while...")
        transcripts = process_videos_concurrently(metadata_df)

        if transcripts:
            df = pd.DataFrame(transcripts)
            st.subheader("Transcription Results")
            st.dataframe(df)
            logger.info("Transcription results displayed")
        else:
            st.warning("No transcripts were generated. Please check the URLs and try again.")
            logger.warning("No transcripts generated")

if __name__ == "__main__":
    main()
