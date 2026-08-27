"""
Standalone Groq connectivity test. Run this directly, separate
from the FastAPI app, to see the RAW exception that
LLMGapExplainer.explain() is currently swallowing and logging
(rather than raising).

Usage (from the project root, same place you run uvicorn from):

    python test_groq.py
"""

from src.llm.gap_explainer import LLMGapExplainer


def main():

    explainer = LLMGapExplainer()

    print("API key loaded:", bool(explainer.api_key))
    print("Base URL:", explainer.base_url)
    print("Model name:", explainer.model_name)
    print("is_available():", explainer.is_available())

    if not explainer.is_available():
        print(
            "\nNo API key detected. Check that GROQ_API_KEY is "
            "set in your .env, and that .env sits in the same "
            "directory you're running this script (and uvicorn) "
            "from."
        )
        return

    fake_ats_analysis = {
        "overall_ats_score": 50.0,
        "score_interpretation": "Moderate Match",
        "priority_gaps": [
            {
                "importance": "high",
                "requirement": (
                    "Bachelor's degree in a quantitative field"
                ),
                "assessment": "WEAK_ALIGNMENT",
                "hybrid_score": 0.4
            }
        ],
        "skills": {"missing": ["FastAPI"]},
        "concepts": {"missing": ["Client Engagement"]},
        "education": {"missing": ["Mathematics"]},
    }

    print(
        "\nCalling Groq directly (bypassing the try/except so "
        "errors are visible)...\n"
    )

    try:

        response = explainer.client.chat.completions.create(
            model=explainer.model_name,
            messages=[
                {
                    "role": "user",
                    "content": explainer.build_prompt(
                        fake_ats_analysis
                    )
                }
            ],
            max_tokens=explainer.max_tokens,
            temperature=0.4
        )

        print("SUCCESS. Groq responded:\n")
        print(response.choices[0].message.content)

    except Exception as exc:

        print("FAILED. Raw exception:\n")
        print(repr(exc))


if __name__ == "__main__":
    main()