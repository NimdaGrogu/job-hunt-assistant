# Libraries

from ingestion import get_jd_from_url, get_pdf_text_pypdf, get_pdf_text_pdfplumber
from prompt_eng_recruiter import get_prompt_ver, jd_as_context
from rag_implementation import get_rag_chain
from helper import extract_match_score
from css_template import sidebar_footer_style
from dotenv import load_dotenv
import streamlit as st
import os
import logging
from rich.logging import RichHandler


# Configure basic Logging config with RichHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s", # Rich handles the timestamp and level separately
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("app")

# load the env variables
load_dotenv()
open_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit Configuration
st.set_page_config(page_title="AI Job Hunt Assistant", page_icon="🚀", layout='wide')
# Main Streamlit
st.title("👔 AI Job Hunt Assistant")
st.markdown("**Provide a job description URL and a candidate resume to get a comprehensive analysis.**")
# Sidebar for Inputs
with st.sidebar:
    st.header("Input Data")
    # Input 1: Web Page Link (Job Description)
    jd_url = st.text_input(placeholder="https://linkedin.com/jobs/",
                           max_chars=5000,
                           label="Job Description URL ")
    # Input 2: Raw text (Job Description)
    jd_text = st.text_input("Job Description Raw Text", max_chars=5000)
    # Input 3: Upload the PDF
    uploaded_resume = st.file_uploader("Upload Candidate Resume (PDF)", type=["pdf"])
    # Button to trigger analysis
    submit = st.button("Analyse Candidate Resume")

    # --- Reset Button ---
    if st.button("Reset Analysis"):
        # Clear the specific session state keys
        if 'analysis_results' in st.session_state:
            del st.session_state['analysis_results']
        if 'full_report' in st.session_state:
            del st.session_state['full_report']

        # Force the app to rerun immediately
        st.rerun()
    # -------------------------

    # 4. Add your footer content (LAST thing in the sidebar)
    st.markdown("---")  # Optional horizontal rule
    st.link_button("Visit GitHub Repo", "https://github.com/NimdaGrogu/job-hunt-assistant.git")
    st.caption("© 2026 Grogus")

    # 3. Inject the CSS
    st.markdown(sidebar_footer_style, unsafe_allow_html=True)

# --- MAIN SECTION ---

# 1. Initialize Session State
# A place to store the data so it survives when the clicked 'Download' is true
if 'analysis_results' not in st.session_state:
    st.session_state['analysis_results'] = None
if 'full_report' not in st.session_state:
    st.session_state['full_report'] = None

# 2. Trigger Analysis (COMPUTATION LAYER)
if submit:
    # --- Validations ---
    if not open_api_key:
        st.error("⚠️ OpenAI API Key is missing. Please check your .env file.")
        st.stop()
    if not uploaded_resume:
        st.warning("⚠️ Please provide Resume PDF ...")
        st.stop()
    if not jd_url and not jd_text:
        st.error("⚠️ Please provide Job Description ...")
        st.stop()

    # --- Processing ---
    if jd_url:
        job_description = get_jd_from_url(jd_url)
        if job_description is None:
            st.error("❌ Something went wrong accessing the URL.")
            st.stop()
    else:
        job_description = jd_text

    with st.spinner("Analysing Candidate Resume and Job Description.."):
        try:
            resume_text = get_pdf_text_pdfplumber(uploaded_resume)
            qa_chain = get_rag_chain(resume_text, uploaded_resume.name)
            questions = get_prompt_ver(version="v2")
            query = jd_as_context(jd=job_description)

            # --- EXECUTE CHAINS & STORE IN DICTIONARY ---
            # We store the RESULTS, not the widgets.
            results = {}

            # Q3 Match Score
            q3_ans = qa_chain.invoke({"query": f"{query}\n\n{questions['q3']}"})
            results['q3'] = q3_ans['result']
            results['score'] = extract_match_score(q3_ans['result'])

            # Q1 Skills
            q1_ans = qa_chain.invoke({"query": f"{query}\n\n{questions['q1']}"})
            results['q1'] = q1_ans['result']

            # Q2 Fit Check
            q2_ans = qa_chain.invoke({"query": f"{query}\n\n{questions['q2']}"})
            results['q2'] = q2_ans['result']

            # Q4, Q5, Q6 SWOT
            q4_ans = qa_chain.invoke({"query": f"{query}\n\n{questions['q4']}"})
            results['q4'] = q4_ans['result']

            q5_ans = qa_chain.invoke({"query": f"{query}\n\n{questions['q5']}"})
            results['q5'] = q5_ans['result']

            q6_ans = qa_chain.invoke({"query": f"{query}\n\n{questions['q6']}"})
            results['q6'] = q6_ans['result']

            # Q7, Q8, Q9 App Kit
            q7_ans = qa_chain.invoke({"query": f"{query}\n\n{questions['q7']}"})
            results['q7'] = q7_ans['result']

            q8_ans = qa_chain.invoke({"query": f"{query}\n\n{questions['q8']}"})
            results['q8'] = q8_ans['result']

            q9_ans = qa_chain.invoke({"query": f"{query}\n\n{questions['q9']}"})
            results['q9'] = q9_ans['result']

            # Build the Report String
            report = f"# Candidate Analysis Report\n"
            report += f"**Job Description:** {jd_url or 'Provided Text'}\n\n---\n\n"
            report += f"## Match Score: {results['score']}%\n\n"
            report += f"### Skills Check\n{results['q1']}\n\n"
            report += f"### Fit Conclusion\n{results['q2']}\n\n"
            report += f"### Strengths\n{results['q4']}\n\n"
            report += f"### Opportunities\n{results['q5']}\n\n"
            report += f"### Red Flags\n{results['q6']}\n\n"
            report += f"### Cover Letter\n{results['q7']}\n\n"
            report += f"### Differentiators\n{results['q8']}\n\n"
            report += f"### Elevator Pitch\n{results['q9']}\n\n"

            # SAVE TO SESSION STATE
            st.session_state['analysis_results'] = results
            st.session_state['full_report'] = report

            logger.info("✅ Analysis stored in Session State.")

        except Exception as e:
            st.error(f"☠️ An error occurred: {e}")
            st.stop()

# 3. Render Results (DISPLAY LAYER)
# This block runs if 'analysis_results' exists in memory, regardless of button clicks.
if st.session_state['analysis_results']:
    results = st.session_state['analysis_results']

    st.markdown("---")
    st.subheader("📊 Analysis Results")

    tabs = st.tabs(["Fit Analysis", "Strengths & Weaknesses", "Cover Letter & Tips", "Interview Tips"])

    with tabs[0]:
        st.markdown("### 🎯 Fit Assessment")

        # Display Score
        st.metric(label="Match Score:", value=f"{results['score']}%")
        st.progress(results['score'] / 100)

        if results['score'] < 50:
            st.error("Low Match - Missing critical skills.")
        elif results['score'] < 80:
            st.warning("Good Match - Some gaps identified.")
        else:
            st.success("High Match - Strong candidate!")

        with st.expander("**Skills Check:**"):
            st.write(results['q1'])

        with st.expander("**Fit Check:**"):
            st.write(results['q2'])

    with tabs[1]:
        st.markdown("### 📈 SWOT Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("Strengths", icon="💪")
            st.write(results['q4'])
        with col2:
            st.warning("Opportunities", icon="🌤️")
            st.write(results['q5'])
        with col3:
            st.error("Weaknesses", icon="🚨")
            st.write(results['q6'])

    with tabs[2]:
        st.markdown("### 📝 Application Kit")
        with st.expander("Draft Cover Letter"):
            st.write(results['q7'])
        with st.expander("**How to Stand Out:**"):
            st.write(results['q8'])

    with tabs[3]:
        st.subheader("🎤 Interview Elevator Pitch")
        st.info(results['q9'])

    # --- EXPORT BUTTON ---
    st.divider()
    st.subheader("📥 Export Report")

    # This button will now work because 'full_report' is read from session_state
    st.download_button(
        label="Download Full Analysis (Markdown/Text)",
        data=st.session_state['full_report'],
        file_name="candidate_analysis.md",
        mime="text/markdown"
    )