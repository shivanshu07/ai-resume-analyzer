"""
Evaluation tests for the LLM-generated gap-explanation summary
(src/llm/gap_explainer.py's output).

Why this file exists
---------------------
Every other test in this repo checks deterministic code: given
input X, does the function return exactly Y? That doesn't work
for LLM output -- the same ats_analysis could produce many
differently-worded summaries that are all equally good. What
CAN be checked, and what actually matters for credibility, is
whether the summary:

  1. Is grounded in the real analysis (doesn't invent skills,
     degrees, or gaps that were never actually identified)
  2. Is actionable for the candidate's RESUME specifically
     (not generic career advice, not commentary on the job
     itself)

This uses deepeval's GEval metric -- an LLM-as-a-judge that
scores natural-language output against a written rubric. The
judge model is Groq (GroqEvalModel), not OpenAI, so running
this evaluation suite doesn't reintroduce the OpenAI cost this
project deliberately moved away from.

IMPORTANT: the test case's `input` is built by calling the
REAL production LLMGapExplainer.build_prompt() method against
a real, full ats_analysis payload -- not a hand-written
approximation of it. An early version of this file used a
short, hand-picked subset of the real context (only "missing
skills", none of the JD's actual requirement text) and got
correctly penalized by the judge for looking ungrounded, when
the real production prompt actually included plenty of
grounding the test simply hadn't given it. Reusing the real
method guarantees the test can't drift out of sync with what
the LLM actually sees.

Requires GROQ_API_KEY to be set (same as gap_explainer.py
itself) -- these tests are skipped, not failed, if it's
missing, since CI/local runs without a key shouldn't be
treated as a code failure.
"""

import os

import pytest

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from src.llm.groq_eval_model import GroqEvalModel
from src.llm.gap_explainer import LLMGapExplainer


requires_groq_key = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set -- skipping LLM-judge evaluation tests"
)


def _judge():

    return GroqEvalModel()


REAL_ATS_ANALYSIS = {
    "overall_ats_score": 45.13,
    "score_interpretation": "Weak Match",
    "skills": {
        "matched": [
            "Statistics", "Python", "R", "SQL", "Data Analytics",
            "Machine Learning", "Artificial Intelligence"
        ],
        "missing": [
            "Data Science", "Statistical Analysis", "Database",
            "MATLAB", "Statistical Methods", "Marketing Analytics",
            "Modeling", "Problem Scoping", "Marketing Effectiveness"
        ]
    },
    "concepts": {
        "matched": [
            "Business Insights", "Stakeholder Management",
            "Business Processes", "Product Engineering Collaboration"
        ],
        "missing": [
            "Business Problems", "Client Engagement",
            "Marketing Portfolio Management", "Customer Collaboration",
            "Proof of Concept", "Decision Making", "Strategic Insights",
            "Innovation", "Data Structures", "Metrics"
        ]
    },
    "education": {
        "matched": ["Statistics", "Data Science", "Engineering"],
        "missing": [
            "Mathematics", "Physics", "Economics",
            "Operations Research", "Bioinformatics"
        ]
    },
    "priority_gaps": [
        {
            "requirement_id": 3,
            "requirement": (
                "Master's degree in Statistics, Mathematics, "
                "Bioinformatics, Economics, another quantitative "
                "field, or equivalent practical experience."
            ),
            "category": "required",
            "importance": "high",
            "assessment": "NO_ALIGNMENT",
            "hybrid_score": 0.1962,
        },
        {
            "requirement_id": 1,
            "requirement": (
                "Bachelor's degree in Statistics, Data Science, "
                "Mathematics, Physics, Economics, Operations "
                "Research, Engineering, or a related quantitative "
                "field, or equivalent practical experience."
            ),
            "category": "required",
            "importance": "high",
            "assessment": "WEAK_ALIGNMENT",
            "hybrid_score": 0.3874,
        },
        {
            "requirement_id": 5,
            "requirement": (
                "Experience with statistical software (e.g., "
                "Python, R or MATLAB) and database languages "
                "(e.g., SQL) with a good understanding of the "
                "AI/ML and statistical methods typically used in "
                "marketing analytics."
            ),
            "category": "required",
            "importance": "high",
            "assessment": "WEAK_ALIGNMENT",
            "hybrid_score": 0.4447,
        },
        {
            "requirement_id": 11,
            "requirement": (
                "Develop comprehensive understanding of Google "
                "data structures, and metrics, advocating for "
                "product changes where needed."
            ),
            "category": "responsibility",
            "importance": "medium",
            "assessment": "NO_ALIGNMENT",
            "hybrid_score": 0.1075,
        },
        {
            "requirement_id": 7,
            "requirement": (
                "Lead data science aspects of client engagements "
                "in the area of marketing effectiveness and "
                "marketing portfolio management - with deep "
                "knowledge of ML and statistics."
            ),
            "category": "responsibility",
            "importance": "medium",
            "assessment": "NO_ALIGNMENT",
            "hybrid_score": 0.1641,
        },
    ],
}

REAL_LLM_SUMMARY = (
    "Focus on making your quantitative background unmistakable: add "
    "a dedicated \u201cEducation\u201d line that lists your degree, major, "
    "and any relevant coursework (e.g., statistical modeling, "
    "econometrics, operations research, bioinformatics) and, if you "
    "have completed any certifications in data science or "
    "analytics, list those right under the degree. Expand the "
    "\u201cTechnical Skills\u201d section to explicitly name Python, R, "
    "MATLAB, SQL, and any ML libraries you\u2019ve used, and pair each "
    "with a brief bullet that quantifies your experience (e.g., "
    "\u201cBuilt and deployed a churn-prediction model in Python using "
    "scikit-learn on a 2M-record SQL database\u201d). Add one or two "
    "project bullets that showcase marketing-focused analytics."
)


def _build_real_input_and_context():
    """
    Reconstructs the same prompt build_prompt() actually sends
    to the LLM (used as `input`), plus a structured context
    list drawn from the same real data (used for the
    groundedness check).
    """

    explainer = LLMGapExplainer()  # build_prompt() makes no API call

    real_prompt = explainer.build_prompt(REAL_ATS_ANALYSIS)

    context = [
        "Matched skills: "
        + ", ".join(REAL_ATS_ANALYSIS["skills"]["matched"]),
        "Missing skills: "
        + ", ".join(REAL_ATS_ANALYSIS["skills"]["missing"]),
        "Missing concepts: "
        + ", ".join(REAL_ATS_ANALYSIS["concepts"]["missing"]),
        "Missing education fields: "
        + ", ".join(REAL_ATS_ANALYSIS["education"]["missing"]),
    ] + [
        f"Requirement (JD text, {gap['assessment']}): "
        f"{gap['requirement']}"
        for gap in REAL_ATS_ANALYSIS["priority_gaps"]
    ]

    return real_prompt, context


@requires_groq_key
def test_gap_summary_is_grounded_in_the_real_analysis():
    """
    The summary should only reference gaps, skills, or missing
    fields that actually appear in the analysis -- not invent
    requirements the JD never mentioned. Skills explicitly
    named in a JD requirement's own text (e.g. "Python, R or
    MATLAB" in requirement 5) count as grounded, since they
    genuinely came from the real analysis, not just the
    "missing skills" list alone.
    """

    real_input, real_context = _build_real_input_and_context()

    groundedness = GEval(
        name="Groundedness",
        model=_judge(),
        evaluation_steps=[
            "Check whether the actual output makes any FALSE or "
            "MISLEADING factual claim about the candidate's resume "
            "or the job's requirements -- for example, asserting "
            "the candidate already has a skill, degree, or years "
            "of experience they don't, or asserting a job "
            "requirement exists that isn't listed in the context.",

            "Do NOT penalize the output for offering generic, "
            "clearly-hedged illustrative examples or suggestions "
            "(phrases like 'e.g.', 'for example', 'such as', or "
            "'if you have completed') as long as they are "
            "reasonable, relevant advice tied to the gaps "
            "described in the context -- even if the exact "
            "example terms used don't appear verbatim in the "
            "context. Suggesting illustrative coursework or "
            "certification examples is expected, useful advice, "
            "not a factual claim.",

            "Score high if the output stays truthful about what "
            "the candidate does and doesn't have, and stays "
            "relevant to the actual gaps identified. Score low "
            "only if it asserts something false or contradicts "
            "the context.",
        ],
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.CONTEXT,
        ],
        threshold=0.5
    )

    test_case = LLMTestCase(
        input=real_input,
        actual_output=REAL_LLM_SUMMARY,
        context=real_context
    )

    groundedness.measure(test_case)

    print(
        f"\n[Groundedness] score={groundedness.score:.2f} "
        f"threshold={groundedness.threshold} "
        f"reason={groundedness.reason}"
    )

    assert groundedness.success, (
        f"Groundedness scored {groundedness.score:.2f} "
        f"(threshold {groundedness.threshold}): {groundedness.reason}"
    )


@requires_groq_key
def test_gap_summary_is_actionable_for_the_resume():
    """
    The summary should give the candidate specific, concrete
    things to change on their OWN resume -- not generic career
    advice, and not commentary aimed at the job posting itself.
    """

    real_input, _ = _build_real_input_and_context()

    actionability = GEval(
        name="Resume Actionability",
        model=_judge(),
        criteria=(
            "Determine whether the actual output gives the "
            "candidate specific, concrete suggestions for what to "
            "add or change on their own resume, in light of the "
            "gaps described in the input. Penalize output that is "
            "vague, generic career advice, or that talks about "
            "the job/company rather than the candidate's resume."
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],
        threshold=0.5
    )

    test_case = LLMTestCase(
        input=real_input,
        actual_output=REAL_LLM_SUMMARY
    )

    actionability.measure(test_case)

    print(
        f"\n[Resume Actionability] score={actionability.score:.2f} "
        f"threshold={actionability.threshold} "
        f"reason={actionability.reason}"
    )

    assert actionability.success, (
        f"Resume Actionability scored {actionability.score:.2f} "
        f"(threshold {actionability.threshold}): "
        f"{actionability.reason}"
    )