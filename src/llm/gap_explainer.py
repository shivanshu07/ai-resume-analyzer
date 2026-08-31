"""
Optional LLM-based layer that turns the rule-based gap
analysis into a short, natural-language explanation.

This is intentionally a thin, isolated add-on:

    - It reads ONLY from the already-computed ats_analysis
      dict (skills, concepts, education, experience,
      priority_gaps). It does not re-run any matching itself.
    - If no API key is configured, or the request fails for
      any reason, explain() returns None instead of raising --
      the rest of the pipeline's output is unaffected either
      way.

This is what actually earns "GenAI" on your resume: the rest
of the project is embeddings + rule-based scoring, not
generative AI. This module is the one place that calls a
generative model.
"""

from typing import Any, Dict, List, Optional

from openai import OpenAI

from config.settings import GROQ_API_KEY, GROQ_BASE_URL, MODEL_NAME
from src.utils.logger import get_logger

logger = get_logger()


class LLMGapExplainer:
    """
    Uses Groq's free tier (https://console.groq.com) rather
    than OpenAI. Groq's API is OpenAI-compatible, so this is
    still the `openai` Python client -- just pointed at Groq's
    base_url with a Groq API key. No new dependency needed.

    Free tier is roughly 30 requests/minute and ~1,000
    requests/day per Groq's published limits -- more than
    enough for interactive, one-resume-at-a-time use. No
    credit card required to get a key.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        max_gaps: int = 5,
        max_tokens: int = 500
    ) -> None:

        self.api_key = api_key or GROQ_API_KEY
        self.model_name = model_name or MODEL_NAME
        self.base_url = base_url or GROQ_BASE_URL
        self.max_gaps = max_gaps
        self.max_tokens = max_tokens

        self.client = (
            OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            if self.api_key
            else None
        )

        if self.client is None:
            logger.warning(
                "LLMGapExplainer: GROQ_API_KEY is not set -- "
                "LLM summaries will be skipped (explain() "
                "returns None). Get a free key at "
                "https://console.groq.com/keys"
            )

    def is_available(self) -> bool:

        return self.client is not None

    # ========================================================
    # PROMPT
    # ========================================================

    def build_prompt(
        self,
        ats_analysis: Dict[str, Any]
    ) -> str:

        score = ats_analysis.get("overall_ats_score", 0.0)

        interpretation = ats_analysis.get(
            "score_interpretation", ""
        )

        top_gaps: List[Dict[str, Any]] = ats_analysis.get(
            "priority_gaps", []
        )[: self.max_gaps]

        if top_gaps:

            gap_lines = "\n".join(
                f"- [{str(gap.get('importance', 'medium')).upper()}] "
                f"{gap.get('requirement', '')} "
                f"(assessment: {gap.get('assessment', 'N/A')}, "
                f"score: {gap.get('hybrid_score', 0.0)})"
                for gap in top_gaps
            )

        else:

            gap_lines = (
                "None -- strong alignment across requirements."
            )

        missing_skills = ", ".join(
            ats_analysis.get("skills", {}).get("missing", [])[:10]
        ) or "None listed"

        missing_concepts = ", ".join(
            ats_analysis.get("concepts", {})
            .get("missing", [])[:10]
        ) or "None listed"

        missing_education = ", ".join(
            ats_analysis.get("education", {})
            .get("missing", [])[:10]
        ) or "None listed"

        return (
            "You are helping a candidate improve their resume "
            "for a specific job application. Below is an "
            "automated requirement-by-requirement analysis "
            "comparing their resume against a job description.\n"
            "\n"
            f"Overall alignment score: {score}/100 "
            f"({interpretation})\n"
            "\n"
            "Top unmet or weakly-met requirements:\n"
            f"{gap_lines}\n"
            "\n"
            f"Missing skills: {missing_skills}\n"
            f"Missing concepts: {missing_concepts}\n"
            f"Missing education fields: {missing_education}\n"
            "\n"
            "In 3-5 sentences, written directly to the "
            "candidate, explain the most important, realistic "
            "improvements they could make to their RESUME "
            "(wording, emphasis, or projects to add) to close "
            "these gaps. Be specific and actionable. Do not "
            "repeat the raw score back to them, and do not "
            "suggest adding anything false or fabricated to "
            "the resume."
        )

    # ========================================================
    # EXPLAIN
    # ========================================================

    def explain(
        self,
        ats_analysis: Dict[str, Any]
    ) -> Optional[str]:

        if not self.is_available():
            return None

        prompt = self.build_prompt(ats_analysis)

        try:

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=0.4
            )

            content = response.choices[0].message.content

            return content.strip() if content else None

        except Exception as exc:

            logger.error(
                f"LLMGapExplainer request failed: {exc}"
            )

            return None