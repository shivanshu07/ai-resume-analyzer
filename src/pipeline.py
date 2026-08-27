"""
Reusable, callable version of the pipeline currently
implemented as a top-level script in app/main.py.

Why this exists
----------------
app/main.py is a script: it runs once, top to bottom, and
exits. That's fine for the CLI, but an API needs to run the
same logic per-request, and needs the expensive parts (loading
the sentence-transformer model) done ONCE at startup rather
than on every request.

ResumeAnalysisPipeline wraps the exact same steps main.py
performs -- extraction, cleaning, chunking, embedding,
matching, hybrid scoring, ATS/gap analysis -- as a class you
instantiate once and call .run() on repeatedly.

app/main.py is not required to change for this to work, but if
you want the CLI and the API to guarantee identical behavior
(recommended), replace main.py's body with a call to
ResumeAnalysisPipeline().run(PDF_PATH, JD_PATH, persist=True)
and print the returned dict's fields.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.extraction.pdf_parser import PDFParser
from src.extraction.jd_parser import JobDescriptionParser
from src.extraction.requirement_extractor import RequirementExtractor

from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.chunker import SemanticChunker
from src.preprocessing.requirement_normalizer import (
    RequirementNormalizer
)

from src.llm.embedder import TextEmbedder
from src.llm.matcher import ResumeJDMatcher

from src.evaluation.hybrid_scorer import HybridMatcher
from src.evaluation.analysis import ResumeAnalysisEngine

# NOTE: matches main.py's actual import. See the flag above
# this file about the utils/ vs src/utils/ discrepancy --
# if that import is wrong in main.py, it's wrong here too,
# and fixing one fixes both.
from src.utils.file_handler import FileHandler


PathLike = Union[str, Path]

# Default output locations, mirroring main.py's constants.
# Only used when run(..., persist=True).
DEFAULT_PATHS = {
    "chunks": "data/processed/resume_chunks.json",
    "jd_requirements": "data/processed/jd_requirements.json",
    "match_results": "data/processed/match_results.json",
    "hybrid_results": "data/processed/hybrid_results.json",
    "ats_analysis": "data/processed/ats_analysis.json",
}


class ResumeAnalysisPipeline:
    """
    End-to-end resume/JD analysis pipeline.

    Instantiate once (this loads the embedding model), then
    call .run(pdf_path, jd_path) as many times as needed --
    once per incoming API request, for example.
    """

    def __init__(self) -> None:

        self.parser = PDFParser()
        self.cleaner = TextCleaner()
        self.chunker = SemanticChunker(max_characters=1800)
        self.handler = FileHandler()

        self.requirement_extractor = RequirementExtractor()
        self.requirement_normalizer = RequirementNormalizer()

        # Expensive: loads the sentence-transformer model.
        # This is why the pipeline is a class instantiated
        # once, not a plain function.
        self.embedder = TextEmbedder()

        self.matcher = ResumeJDMatcher(
            similarity_threshold=0.35
        )

        self.hybrid_scorer = HybridMatcher()

        self.analysis_engine = ResumeAnalysisEngine()

    # ========================================================
    # HELPERS (identical logic to main.py)
    # ========================================================

    @staticmethod
    def calculate_semantic_scores(
        requirement_embedding,
        resume_embeddings
    ) -> List[float]:

        query = np.asarray(
            requirement_embedding,
            dtype=np.float32
        ).reshape(-1)

        matrix = np.asarray(
            resume_embeddings,
            dtype=np.float32
        )

        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)

        if matrix.ndim != 2:
            raise ValueError(
                "Resume embeddings must be a 2D matrix."
            )

        if matrix.shape[1] != query.shape[0]:
            raise ValueError(
                "Embedding dimension mismatch: resume "
                f"embeddings have dimension {matrix.shape[1]}, "
                f"while JD embedding has dimension "
                f"{query.shape[0]}."
            )

        query_norm = np.linalg.norm(query)

        matrix_norms = np.linalg.norm(matrix, axis=1)

        if query_norm == 0:
            return [0.0 for _ in range(len(matrix))]

        matrix_norms = np.where(
            matrix_norms == 0,
            1e-12,
            matrix_norms
        )

        scores = (matrix @ query) / (matrix_norms * query_norm)

        scores = np.clip(scores, -1.0, 1.0)

        return [round(float(score), 4) for score in scores]

    @staticmethod
    def build_resume_evidence_chunks(
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        output = []

        for index, chunk in enumerate(chunks):

            if not isinstance(chunk, dict):
                raise TypeError(
                    f"Resume chunk {index} is not a dictionary."
                )

            output.append({
                **chunk,
                "section": str(chunk.get("section", "UNKNOWN")),
                "text": str(chunk.get("text", "")),
                "chunk_id": str(
                    chunk.get(
                        "chunk_id",
                        f"resume_{index + 1:03d}"
                    )
                ),
            })

        return output

    @staticmethod
    def validate_pipeline_inputs(
        requirements,
        jd_embeddings,
        chunks,
        resume_embeddings
    ) -> None:

        if not requirements:
            raise ValueError(
                "No normalized JD requirements were generated."
            )

        if not chunks:
            raise ValueError(
                "No resume chunks were generated."
            )

        if jd_embeddings is None:
            raise ValueError(
                "JD embeddings could not be generated."
            )

        if resume_embeddings is None:
            raise ValueError(
                "Resume embeddings could not be generated."
            )

        if len(requirements) != len(jd_embeddings):
            raise ValueError(
                "Requirement/embedding mismatch: "
                f"{len(requirements)} requirements but "
                f"{len(jd_embeddings)} JD embeddings."
            )

        if len(chunks) != len(resume_embeddings):
            raise ValueError(
                "Resume chunk/embedding mismatch: "
                f"{len(chunks)} chunks but "
                f"{len(resume_embeddings)} resume embeddings."
            )

    # ========================================================
    # RUN
    # ========================================================

    def run(
        self,
        pdf_path: PathLike,
        jd_path: PathLike,
        persist: bool = False,
        output_paths: Optional[Dict[str, PathLike]] = None
    ) -> Dict[str, Any]:
        """
        Run the full pipeline for one resume/JD pair.

        Parameters
        ----------
        pdf_path : str or Path
            Path to the resume PDF.
        jd_path : str or Path
            Path to the job description text file.
        persist : bool
            If True, write intermediate/final JSON outputs to
            disk (same behavior as app/main.py). Defaults to
            False, which is what you want for an API handling
            concurrent requests.
        output_paths : dict, optional
            Override any of DEFAULT_PATHS's destinations.

        Returns
        -------
        dict with keys: chunks, normalized_requirements,
        match_results, hybrid_results, ats_analysis
        """

        paths = dict(DEFAULT_PATHS)

        if output_paths:
            paths.update(output_paths)

        # ----------------------------------------------------
        # Resume processing
        # ----------------------------------------------------

        raw_text = self.parser.extract_text(pdf_path)

        if not raw_text.strip():
            raise ValueError(
                "No text could be extracted from the resume PDF."
            )

        sections = self.cleaner.clean(raw_text)

        if not sections:
            raise ValueError(
                "No resume sections were detected."
            )

        chunks = self.chunker.create_chunks(sections)
        chunks = self.build_resume_evidence_chunks(chunks)

        if not chunks:
            raise ValueError(
                "No resume chunks were created."
            )

        if persist:
            self.handler.save_json(chunks, paths["chunks"])

        # ----------------------------------------------------
        # Resume embeddings
        # ----------------------------------------------------

        resume_texts = [chunk["text"] for chunk in chunks]

        resume_embeddings = self.embedder.generate_embeddings(
            resume_texts
        )

        # ----------------------------------------------------
        # Job description processing
        # ----------------------------------------------------

        parsed_jd = JobDescriptionParser(jd_path).parse()

        jd_sections = self.requirement_extractor.extract(
            parsed_jd["lines"]
        )

        requirements = (
            self.requirement_extractor.build_requirement_objects(
                jd_sections
            )
        )

        normalized_requirements = (
            self.requirement_normalizer.normalize_all(
                requirements
            )
        )

        if not normalized_requirements:
            raise ValueError(
                "No normalized JD requirements were generated."
            )

        if persist:
            self.handler.save_json(
                normalized_requirements,
                paths["jd_requirements"]
            )

        # ----------------------------------------------------
        # JD embeddings
        # ----------------------------------------------------

        jd_texts = [
            requirement.get("original_text", "")
            for requirement in normalized_requirements
        ]

        jd_embeddings = self.embedder.generate_embeddings(
            jd_texts
        )

        self.validate_pipeline_inputs(
            normalized_requirements,
            jd_embeddings,
            chunks,
            resume_embeddings
        )

        # ----------------------------------------------------
        # Semantic matching
        # ----------------------------------------------------

        match_results = self.matcher.match_all(
            normalized_requirements,
            jd_embeddings,
            chunks,
            resume_embeddings
        )

        if persist:
            self.handler.save_json(
                match_results,
                paths["match_results"]
            )

        # ----------------------------------------------------
        # Hybrid matching
        # ----------------------------------------------------

        hybrid_results = []

        for index, requirement in enumerate(
            normalized_requirements
        ):

            requirement_embedding = jd_embeddings[index]

            semantic_scores = self.calculate_semantic_scores(
                requirement_embedding,
                resume_embeddings
            )

            result = self.hybrid_scorer.match_requirement(
                requirement=requirement,
                resume_chunks=chunks,
                resume_sections=chunks,
                semantic_scores=semantic_scores
            )

            result["requirement_id"] = index + 1
            result["requirement"] = requirement.get(
                "original_text", ""
            )
            result["category"] = requirement.get(
                "category", "preferred"
            )
            result["importance"] = requirement.get(
                "importance", "medium"
            )
            result["requirement_type"] = requirement.get(
                "category", "preferred"
            )

            hybrid_results.append(result)

        if persist:
            self.handler.save_json(
                hybrid_results,
                paths["hybrid_results"]
            )

        # ----------------------------------------------------
        # ATS + gap analysis
        # ----------------------------------------------------

        ats_analysis = self.analysis_engine.analyze(
            hybrid_results
        )

        if persist:
            self.handler.save_json(
                ats_analysis,
                paths["ats_analysis"]
            )

        return {
            "chunks": chunks,
            "normalized_requirements": normalized_requirements,
            "match_results": match_results,
            "hybrid_results": hybrid_results,
            "ats_analysis": ats_analysis,
        }