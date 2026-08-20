import json
from pathlib import Path

from src.evaluation.ats_scorer import ATSScorer
from src.evaluation.gap_analyzer import ResumeGapAnalyzer
from src.evaluation.analysis import ResumeAnalysisEngine


HYBRID_RESULTS_PATH = Path(
    "data/processed/hybrid_results.json"
)


def load_results():

    with open(
        HYBRID_RESULTS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def test_ats_score():

    results = load_results()

    scorer = ATSScorer()

    score = scorer.calculate_score(
        results
    )

    assert 0 <= score <= 100


def test_assessment_summary():

    results = load_results()

    scorer = ATSScorer()

    summary = (
        scorer.summarize_assessments(
            results
        )
    )

    assert summary["total"] == len(
        results
    )

    assert (
        summary["strong"]
        + summary["partial"]
        + summary["weak"]
        + summary["no_alignment"]
        == summary["total"]
    )


def test_skill_analysis():

    results = load_results()

    analyzer = ResumeGapAnalyzer()

    analysis = analyzer.analyze(
        results
    )

    assert "skills" in analysis

    assert "matched" in analysis[
        "skills"
    ]

    assert "missing" in analysis[
        "skills"
    ]


def test_concept_analysis():

    results = load_results()

    analyzer = ResumeGapAnalyzer()

    analysis = analyzer.analyze(
        results
    )

    assert "concepts" in analysis


def test_full_analysis():

    results = load_results()

    engine = ResumeAnalysisEngine()

    analysis = engine.analyze(
        results
    )

    assert (
        "overall_ats_score"
        in analysis
    )

    assert (
        "requirement_summary"
        in analysis
    )

    assert (
        "skills"
        in analysis
    )

    assert (
        "concepts"
        in analysis
    )

    assert (
        "education"
        in analysis
    )

    assert (
        "experience"
        in analysis
    )

    assert (
        "priority_gaps"
        in analysis
    )