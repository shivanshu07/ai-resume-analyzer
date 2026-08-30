"""
FastAPI wrapper around ResumeAnalysisPipeline.

Run locally with:

    uvicorn app.api:app --reload

Then either open http://127.0.0.1:8000/docs for the
auto-generated Swagger UI, or:

    curl -X POST http://127.0.0.1:8000/analyze \\
        -F "resume=@data/raw/resume/resume.pdf" \\
        -F "job_description_file=@data/raw/job_description/job_description.txt" \\
        -F "include_llm_summary=true"
"""

import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse

from src.pipeline import ResumeAnalysisPipeline
from src.llm.gap_explainer import LLMGapExplainer


app = FastAPI(
    title="AI Resume Analyzer API",
    description=(
        "Upload a resume (PDF) and a job description to get a "
        "hybrid semantic + rule-based ATS-style alignment "
        "analysis, with an optional LLM-generated improvement "
        "summary."
    ),
    version="1.0.0"
)

# Loaded once at startup, not per-request -- the embedding
# model is expensive to load and should not be reloaded on
# every call.
pipeline = ResumeAnalysisPipeline()
gap_explainer = LLMGapExplainer()


@app.get("/", include_in_schema=False)
def root():

    # Anyone opening the bare URL (e.g. a recruiter clicking a
    # link on your resume) gets sent straight to Swagger UI
    # instead of landing on a blank 404 page.
    return RedirectResponse(url="/docs")


@app.get("/health")
def health_check():

    return {
        "status": "ok",
        "llm_summary_available": gap_explainer.is_available()
    }


@app.post("/analyze")
async def analyze(
    resume: UploadFile = File(
        ...,
        description="Resume as a PDF file."
    ),
    job_description: Optional[str] = Form(
        None,
        description="Job description as raw text."
    ),
    job_description_file: Optional[UploadFile] = File(
        None,
        description="Job description as a .txt file."
    ),
    include_llm_summary: bool = Form(
        False,
        description=(
            "If true, also generate a short natural-language "
            "improvement summary via LLMGapExplainer. Adds "
            "latency and requires OPENAI_API_KEY to be set."
        )
    ),
    include_raw_results: bool = Form(
        False,
        description=(
            "If true, include the full per-requirement hybrid "
            "results alongside the summarized ats_analysis."
        )
    )
):

    if not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Resume must be a PDF file."
        )

    if not job_description and job_description_file is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Provide either 'job_description' text or a "
                "'job_description_file'."
            )
        )

    with tempfile.TemporaryDirectory() as tmp_dir_name:

        tmp_dir = Path(tmp_dir_name)

        resume_path = tmp_dir / "resume.pdf"

        with resume_path.open("wb") as buffer:
            shutil.copyfileobj(resume.file, buffer)

        jd_path = tmp_dir / "job_description.txt"

        if job_description_file is not None:

            with jd_path.open("wb") as buffer:
                shutil.copyfileobj(
                    job_description_file.file,
                    buffer
                )

        else:

            jd_path.write_text(
                job_description,
                encoding="utf-8"
            )

        try:

            result = pipeline.run(
                str(resume_path),
                str(jd_path),
                persist=False
            )

        except ValueError as exc:

            # Bad/empty input (unreadable PDF, empty JD, etc.)
            raise HTTPException(
                status_code=422,
                detail=str(exc)
            )

        except Exception as exc:

            raise HTTPException(
                status_code=500,
                detail=f"Pipeline failed: {exc}"
            )

    ats_analysis = result["ats_analysis"]

    if include_llm_summary:

        ats_analysis["llm_summary"] = gap_explainer.explain(
            ats_analysis
        )

    response = {"ats_analysis": ats_analysis}

    if include_raw_results:

        response["hybrid_results"] = result["hybrid_results"]

    return JSONResponse(content=response)