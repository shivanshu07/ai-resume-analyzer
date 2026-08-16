import json
import numpy as np


class ResumeJDMatcher:

    def __init__(
        self,
        similarity_threshold=0.35
    ):

        self.similarity_threshold = (
            similarity_threshold
        )

    # ========================================================
    # Load JSON
    # ========================================================

    def load_json(
        self,
        path
    ):

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    # ========================================================
    # Calculate cosine similarity
    # ========================================================

    def cosine_similarity(
        self,
        vector_a,
        vector_b
    ):

        vector_a = np.array(
            vector_a
        )

        vector_b = np.array(
            vector_b
        )

        denominator = (
            np.linalg.norm(vector_a)
            *
            np.linalg.norm(vector_b)
        )

        if denominator == 0:

            return 0.0

        similarity = (
            np.dot(
                vector_a,
                vector_b
            )
            /
            denominator
        )

        return float(similarity)

    # ========================================================
    # Match one JD requirement against all resume chunks
    # ========================================================

    def match_requirement(
        self,
        requirement,
        jd_embedding,
        resume_chunks,
        resume_embeddings
    ):

        matches = []

        for index, chunk in enumerate(
            resume_chunks
        ):

            similarity = (
                self.cosine_similarity(
                    jd_embedding,
                    resume_embeddings[index]
                )
            )

            matches.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "section": chunk["section"],
                    "text": chunk["text"],
                    "similarity": round(
                        similarity,
                        4
                    )
                }
            )

        # Highest similarity first

        matches.sort(
            key=lambda x: x["similarity"],
            reverse=True
        )

        best_match = matches[0]

        # ----------------------------------------------------
        # Determine match level
        # ----------------------------------------------------

        score = best_match["similarity"]

        if score >= 0.70:

            match_level = "strong"

        elif score >= 0.55:

            match_level = "moderate"

        elif score >= self.similarity_threshold:

            match_level = "weak"

        else:

            match_level = "no_match"

        return {

            "requirement": requirement[
                "original_text"
            ],

            "category": requirement[
                "category"
            ],

            "importance": requirement[
                "importance"
            ],

            "skills": requirement.get(
                "skills",
                []
            ),

            "education_fields": requirement.get(
                "education_fields",
                []
            ),

            "experience": requirement.get(
                "experience",
                []
            ),

            "concepts": requirement.get(
                "concepts",
                []
            ),

            "best_match": best_match,

            "match_level": match_level,

            "all_matches": matches
        }

    # ========================================================
    # Match all JD requirements
    # ========================================================

    def match_all(
        self,
        requirements,
        jd_embeddings,
        resume_chunks,
        resume_embeddings
    ):

        results = []

        for index, requirement in enumerate(
            requirements
        ):

            result = self.match_requirement(

                requirement,

                jd_embeddings[index],

                resume_chunks,

                resume_embeddings
            )

            result["requirement_id"] = (
                index + 1
            )

            results.append(
                result
            )

        return results

    # ========================================================
    # Save results
    # ========================================================

    def save_results(
        self,
        results,
        output_path
    ):

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                results,
                file,
                indent=4,
                ensure_ascii=False
            )