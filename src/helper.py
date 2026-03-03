from langchain_core.callbacks import BaseCallbackHandler
import re
import logging
import os
import pandas as pd

logger = logging.getLogger("helper_debugger")

def extract_match_score(response_text):
    # Search for a number between 0 and 100
    match = re.search(r'\b(100|[1-9]?[0-9])\b', response_text)
    if match:
        return int(match.group(0))
    return 0


def load_tracker_data():
    """Loads the job tracker data from a CSV, or creates an empty DataFrame if it doesn't exist."""
    TRACKER_FILE = "job_tracker.csv"

    if os.path.exists(TRACKER_FILE):
        return pd.read_csv(TRACKER_FILE)
    else:
        # Define the columns for a new tracker
        df = pd.DataFrame(columns=[
            "Date Applied", "Company", "Job Title", "Match Score", "Status", "URL", "Notes"
        ])
        return df

def save_tracker_data(df):
    TRACKER_FILE = "job_tracker.csv"
    """Saves the DataFrame to the CSV file."""
    df.to_csv(TRACKER_FILE, index=False)


class DebugCallbackHandler(BaseCallbackHandler):

    def on_llm_start(self, serialized, prompts, **kwargs):
        """Run when LLM starts running. This gives us the FINAL prompt sent to the LLM."""
        logger.info("============== 📤 PROMPT SENT TO MODEL ==============")
        # prompts[0] is the final string with context and question injected
        logger.info(prompts[0])
        logger.info("=====================================================")

    def on_llm_end(self, response, **kwargs):
        """Run when LLM ends running."""
        logger.info("============== 📥 RESPONSE FROM MODEL ===============")
        # Access the actual text generated
        logger.info(response.generations[0][0].text)
        logger.info("=====================================================")