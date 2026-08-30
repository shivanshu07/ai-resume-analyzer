"""
Streamlit frontend for the AI Resume Analyzer API.

This is a THIN CLIENT: all the actual work (PDF parsing,
embedding, hybrid scoring, LLM gap explanation) happens in the
deployed FastAPI service. This file only collects input, calls
the API over HTTP, and renders the JSON response in a readable
way -- so a recruiter or interviewer doesn't have to read raw
JSON off Swagger UI to understand what the tool does.

Run locally:
    streamlit run streamlit_app.py

Deploy free on Streamlit Community Cloud:
    share.streamlit.io -> New app -> point at this file in
    your GitHub repo. No server to manage.
"""

import requests
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

# Change this to your actual Render URL if it differs.
DEFAULT_API_URL = "https://ai-resume-analyzer.onrender.com"

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

ASSESSMENT_STYLE = {
    "STRONG_ALIGNMENT": ("🟢", "Strong"),
    "PARTIAL_ALIGNMENT": ("🟡", "Partial"),
    "WEAK_ALIGNMENT": ("🟠", "Weak"),
    "NO_ALIGNMENT": ("🔴", "Missing"),
}


# ============================================================
# HELPERS
# ============================================================

def interpretation_color(score: float) -> str:

    if score >= 80:
        return "green"

    if score >= 65:
        return "blue"

    if score >= 50:
        return "orange"

    return "red"


def render_matched_missing(container, title, block):

    with container:

        st.markdown(f"**{title}**")

        matched = block.get("matched", [])
        missing = block.get("missing", [])

        if matched:
            st.markdown("✅ " + ", ".join(matched))

        if missing:
            st.markdown("❌ " + ", ".join(missing))

        if not matched and not missing:
            st.caption("No data")


def render_results(payload: dict) -> None:

    ats = payload.get("ats_analysis", payload)

    score = ats.get("overall_ats_score", 0.0)
    interpretation = ats.get("score_interpretation", "")

    color = interpretation_color(score)

    st.markdown(
        f"### Overall Match: "
        f":{color}[{score:.1f} / 100 -- {interpretation}]"
    )
    st.progress(min(max(score / 100, 0.0), 1.0))

    # --------------------------------------------------------
    # Category breakdown
    # --------------------------------------------------------

    category_summary = ats.get("category_summary", {})

    if category_summary:

        st.markdown("#### By Category")

        cols = st.columns(len(category_summary))

        for col, (category, info) in zip(
            cols,
            category_summary.items()
        ):

            col.metric(
                label=category.replace("_", " ").title(),
                value=f"{info.get('score', 0):.1f}"
            )

    # --------------------------------------------------------
    # Requirement summary badges
    # --------------------------------------------------------

    summary = ats.get("requirement_summary", {})

    if summary:

        st.markdown("#### Requirement Breakdown")

        badge_cols = st.columns(4)

        badge_cols[0].metric(
            "🟢 Strong", summary.get("strong", 0)
        )
        badge_cols[1].metric(
            "🟡 Partial", summary.get("partial", 0)
        )
        badge_cols[2].metric(
            "🟠 Weak", summary.get("weak", 0)
        )
        badge_cols[3].metric(
            "🔴 Missing", summary.get("no_alignment", 0)
        )

    # --------------------------------------------------------
    # LLM summary, if present and non-null
    # --------------------------------------------------------

    llm_summary = ats.get("llm_summary")

    if llm_summary:

        st.markdown("#### 💡 Suggested Improvements")
        st.info(llm_summary)

    # --------------------------------------------------------
    # Skills / concepts / education matched vs missing
    # --------------------------------------------------------

    st.markdown("#### Skills & Concepts")

    skill_col, concept_col, edu_col = st.columns(3)

    render_matched_missing(
        skill_col, "Skills", ats.get("skills", {})
    )
    render_matched_missing(
        concept_col, "Concepts", ats.get("concepts", {})
    )
    render_matched_missing(
        edu_col, "Education", ats.get("education", {})
    )

    # --------------------------------------------------------
    # Priority gaps
    # --------------------------------------------------------

    priority_gaps = ats.get("priority_gaps", [])

    if priority_gaps:

        st.markdown("#### Top Priority Gaps")

        for gap in priority_gaps[:5]:

            emoji, label = ASSESSMENT_STYLE.get(
                gap.get("assessment", ""),
                ("⚪", "Unknown")
            )

            requirement_preview = gap.get("requirement", "")[:90]

            with st.expander(
                f"{emoji} [{label}] {requirement_preview}"
            ):

                st.write(
                    f"**Category:** {gap.get('category', 'N/A')}"
                )
                st.write(
                    f"**Importance:** "
                    f"{gap.get('importance', 'N/A')}"
                )
                st.write(
                    f"**Hybrid score:** "
                    f"{gap.get('hybrid_score', 0):.2f}"
                )

    with st.expander("Raw JSON response"):
        st.json(payload)


# ============================================================
# SIDEBAR -- API CONFIG
# ============================================================

st.sidebar.header("⚙️ Settings")

api_url = st.sidebar.text_input(
    "API base URL",
    value=DEFAULT_API_URL,
    help=(
        "Your deployed FastAPI service. Change this to "
        "http://localhost:8000 to test against a local run "
        "instead of the live deployment."
    )
)

include_llm_summary = st.sidebar.checkbox(
    "Generate AI improvement summary",
    value=True,
    help=(
        "Uses Groq to write a short natural-language summary "
        "of the biggest gaps. Adds a few seconds of latency."
    )
)

st.sidebar.caption(
    "Note: the free-hosted API sleeps after 15 minutes of "
    "inactivity. The first request after a while may take "
    "30-60 seconds to wake it up."
)


# ============================================================
# MAIN LAYOUT
# ============================================================

st.title("📄 AI Resume Analyzer")
st.caption(
    "Upload a resume and a job description to see how well "
    "they align, requirement by requirement."
)

left, right = st.columns([1, 1.4])

with left:

    st.markdown("#### 1. Upload your resume")

    resume_file = st.file_uploader(
        "Resume (PDF)",
        type=["pdf"]
    )

    st.markdown("#### 2. Add the job description")

    jd_mode = st.radio(
        "How would you like to provide the JD?",
        ["Paste text", "Upload a .txt file"],
        horizontal=True
    )

    jd_text = None
    jd_file = None

    if jd_mode == "Paste text":

        jd_text = st.text_area(
            "Job description",
            height=250,
            placeholder="Paste the full job description here..."
        )

    else:

        jd_file = st.file_uploader(
            "Job description (.txt)",
            type=["txt"]
        )

    analyze_clicked = st.button(
        "🔍 Analyze",
        type="primary",
        use_container_width=True
    )

with right:

    if not analyze_clicked:

        st.info(
            "Upload a resume and a job description, then click "
            "**Analyze** to see your results here."
        )

    else:

        if resume_file is None:

            st.error("Please upload a resume PDF first.")

        elif jd_mode == "Paste text" and not jd_text:

            st.error("Please paste a job description.")

        elif jd_mode == "Upload a .txt file" and jd_file is None:

            st.error("Please upload a job description file.")

        else:

            files = {
                "resume": (
                    resume_file.name,
                    resume_file.getvalue(),
                    "application/pdf"
                )
            }

            if jd_file is not None:

                files["job_description_file"] = (
                    jd_file.name,
                    jd_file.getvalue(),
                    "text/plain"
                )

            form_data = {
                "include_llm_summary": str(
                    include_llm_summary
                ).lower()
            }

            if jd_text:
                form_data["job_description"] = jd_text

            with st.spinner(
                "Analyzing... this can take 10-30 seconds, "
                "longer if the server was asleep."
            ):

                try:

                    response = requests.post(
                        f"{api_url.rstrip('/')}/analyze",
                        files=files,
                        data=form_data,
                        timeout=120
                    )

                    if response.status_code == 200:

                        render_results(response.json())

                    else:

                        st.error(
                            f"API returned an error "
                            f"({response.status_code}): "
                            f"{response.text}"
                        )

                except requests.exceptions.Timeout:

                    st.error(
                        "The request timed out. If the server "
                        "was asleep (free-tier services sleep "
                        "after inactivity), try again -- it "
                        "should be faster the second time."
                    )

                except requests.exceptions.ConnectionError:

                    st.error(
                        "Couldn't reach the API. Check that the "
                        "API base URL in the sidebar is correct "
                        "and the service is running."
                    )