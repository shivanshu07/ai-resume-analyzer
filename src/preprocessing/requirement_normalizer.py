import re


class RequirementNormalizer:

    def __init__(self):

        # ====================================================
        # Technical and analytical skills
        # ====================================================

        self.skill_patterns = {

            "Python": r"\bpython\b",

            "R": r"\bR\b",

            "SQL": r"\bsql\b",

            "MATLAB": r"\bmatlab\b",

            "Machine Learning": (
                r"\b(?:machine learning|ml)\b"
            ),

            "Artificial Intelligence": (
                r"\b(?:artificial intelligence|ai)\b"
            ),

            "Statistics": (
                r"\bstatistics\b"
            ),

            "Statistical Analysis": (
                r"\bstatistical analysis\b"
            ),

            "Statistical Methods": (
                r"\bstatistical methods?\b"
            ),

            "Data Science": (
                r"\bdata science\b"
            ),

            "Data Analytics": (
                r"\b(?:data analytics|analytics)\b"
            ),

            "Database": (
                r"\b(?:database|databases|"
                r"database languages|querying databases)\b"
            ),

            "Marketing Analytics": (
                r"\bmarketing analytics\b"
            ),

            "Marketing Effectiveness": (
                r"\bmarketing effectiveness\b"
            ),

            "Modeling": (
                r"\bmodel(?:ing|elling)\b"
            ),

            "Problem Scoping": (
                r"\bproblem scoping\b"
            ),

            "Model Interpretation": (
                r"\bmodel interpretation\b"
            )
        }

        # ====================================================
        # Education-related fields
        # ====================================================

        self.education_patterns = {

            "Statistics": (
                r"\bstatistics\b"
            ),

            "Data Science": (
                r"\bdata science\b"
            ),

            "Mathematics": (
                r"\bmathematics\b"
            ),

            "Physics": (
                r"\bphysics\b"
            ),

            "Economics": (
                r"\beconomics\b"
            ),

            "Operations Research": (
                r"\boperations research\b"
            ),

            "Engineering": (
                r"\bengineering\b"
            ),

            "Bioinformatics": (
                r"\bbioinformatics\b"
            )
        }

        # ====================================================
        # Experience / responsibility concepts
        # ====================================================

        self.concept_patterns = {

            "Client Engagement": (
                r"\bclient engagements?\b"
            ),

            "Stakeholder Management": (
                r"\bstakeholders?\b"
            ),

            "Customer Collaboration": (
                r"\bcollaborat(?:e|ing|ion)\b"
                r".{0,100}"
                r"\bcustomers?\b"
            ),

            "Business Insights": (
                r"\b(?:business insights|insights)\b"
            ),

            "Decision Making": (
                r"\bdecision[- ]making\b"
            ),

            "Business Problems": (
                r"\bbusiness problems?\b"
            ),

            "Product Problems": (
                r"\bproduct problems?\b"
            ),

            "Proof of Concept": (
                r"\bproof[- ]of[- ]concept\b"
            ),

            "Business Processes": (
                r"\bbusiness processes?\b"
            ),

            "Strategic Insights": (
                r"\bstrategic insights?\b"
            ),

            "Tactical Insights": (
                r"\btactical insights?\b"
            ),

            "Marketing Portfolio Management": (
                r"\bmarketing portfolio management\b"
            ),

            "Product Engineering Collaboration": (
                r"\bproduct/engineering\b"
            ),

            "Innovation": (
                r"\binnovation\b"
            ),

            "Data Structures": (
                r"\bdata structures?\b"
            ),

            "Metrics": (
                r"\bmetrics?\b"
            )
        }

    # ========================================================
    # Extract skills
    # ========================================================

    def extract_skills(self, text):

        found = []

        for skill, pattern in self.skill_patterns.items():

            if re.search(
                pattern,
                text,
                flags=re.IGNORECASE
            ):

                found.append(skill)

        return found

    # ========================================================
    # Extract education fields
    # ========================================================

    def extract_education_fields(self, text):

        found = []

        for field, pattern in (
            self.education_patterns.items()
        ):

            if re.search(
                pattern,
                text,
                flags=re.IGNORECASE
            ):

                found.append(field)

        return found

    # ========================================================
    # Extract experience duration
    # ========================================================

    def extract_experience(self, text):

        patterns = [
            r"\b\d+\+?\s+years?\b",
            r"\b\d+\+?\s+months?\b"
        ]

        results = []

        for pattern in patterns:

            matches = re.findall(
                pattern,
                text,
                flags=re.IGNORECASE
            )

            results.extend(matches)

        return list(dict.fromkeys(results))

    # ========================================================
    # Extract responsibility concepts
    # ========================================================

    def extract_concepts(self, text):

        found = []

        for concept, pattern in (
            self.concept_patterns.items()
        ):

            if re.search(
                pattern,
                text,
                flags=re.IGNORECASE
            ):

                found.append(concept)

        return found

    # ========================================================
    # Normalize one requirement
    # ========================================================

    def normalize_requirement(
        self,
        requirement,
        category,
        importance
    ):

        skills = self.extract_skills(
            requirement
        )

        education_fields = (
            self.extract_education_fields(
                requirement
            )
        )

        experience = self.extract_experience(
            requirement
        )

        concepts = self.extract_concepts(
            requirement
        )

        return {

            "original_text": requirement,

            "category": category,

            "importance": importance,

            "skills": skills,

            "education_fields": education_fields,

            "experience": experience,

            "concepts": concepts
        }

    # ========================================================
    # Normalize all requirements
    # ========================================================

    def normalize_all(
        self,
        requirements
    ):

        normalized = []

        for requirement in requirements:

            normalized.append(
                self.normalize_requirement(
                    requirement["requirement"],
                    requirement["category"],
                    requirement["importance"]
                )
            )

        return normalized