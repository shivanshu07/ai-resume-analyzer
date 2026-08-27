import re


class RequirementExtractor:

    def __init__(self):

        self.ignore_lines = [
    "data scientist"
]

        self.section_keywords = {

            "required_skills": [
                "minimum qualifications",
                "minimum qualification",
                "required skills",
                "required qualifications",
                "basic qualifications",
                "must have",
                "must haves",
                "requirements",
                "required",
                "essential skills",
                "essential requirements",
                "qualifications",
                "what you need",
                "what you'll need",
                "what you will need",
                "your profile",
                "your background",
                "who you are"
            ],

            "preferred_skills": [
                "preferred qualifications",
                "preferred qualification",
                "preferred skills",
                "nice to have",
                "nice to haves",
                "bonus points",
                "good to have",
                "desired skills",
                "desired qualifications",
                "additional qualifications",
                "bonus skills",
                "would be a plus",
                "pluses"
            ],

            "responsibilities": [
                "responsibilities",
                "responsibility",
                "key responsibilities",
                "what you'll do",
                "what you will do",
                "the role",
                "your role",
                "role overview",
                "duties",
                "day to day",
                "in this role"
            ],

            "education": [
                "education",
                "educational qualification",
                "educational qualifications",
                "academic qualifications",
                "academic background"
            ]
        }

        self.ignore_headings = [
            "about the job",
            "job description",
            "overview",
            "introduction",
            "description"
        ]

    # ---------------------------------------------------------
    # Normalize text
    # ---------------------------------------------------------

    def normalize(self, text):

        text = text.lower().strip()

        text = text.replace("’", "'")
        text = text.replace("–", "-")
        text = text.replace("—", "-")

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    # ---------------------------------------------------------
    # Detect section heading
    # ---------------------------------------------------------

    def identify_section(self, line):

        normalized = self.normalize(line)

        # Remove trailing punctuation
        normalized = re.sub(
            r"[:\-]+$",
            "",
            normalized
        ).strip()

        # Section headings are short, standalone lines. Guarding
        # on length avoids misclassifying a long bullet sentence
        # that happens to contain a keyword (e.g. "...meets all
        # requirements for...") as a section heading.
        if len(normalized) > 60:
            return None

        for section, keywords in self.section_keywords.items():

            for keyword in keywords:

                if normalized == keyword:
                    return section

                # Whole-word substring match so headings like
                # "Qualifications:" or "MINIMUM QUALIFICATIONS
                # FOR THIS ROLE" are still caught, not just an
                # exact string match.
                if re.search(
                    rf"\b{re.escape(keyword)}\b",
                    normalized
                ):
                    return section

        return None

    # ---------------------------------------------------------
    # Detect headings that should not become requirements
    # ---------------------------------------------------------

    def is_ignored_heading(self, line):

        normalized = self.normalize(line)

        normalized = re.sub(
            r"[:\-]+$",
            "",
            normalized
        ).strip()

        return normalized in self.ignore_headings

    # ---------------------------------------------------------
    # Clean requirement text
    # ---------------------------------------------------------

    def clean_requirement(self, line):

        line = line.strip()

        # Remove common bullet characters
        line = re.sub(
            r"^[\s•●▪◦‣⁃\-\*]+",
            "",
            line
        )

        # Remove numbered bullets
        line = re.sub(
            r"^\d+\s*[\.\)\-:]\s*",
            "",
            line
        )

        # Normalize whitespace
        line = re.sub(
            r"\s+",
            " ",
            line
        )

        return line.strip()

    # ---------------------------------------------------------
    # Determine whether a line contains useful content
    # ---------------------------------------------------------

    def is_requirement_line(self, line):

        stripped = line.strip()

        if not stripped:
            return False

        # Ignore known headings
        if self.identify_section(stripped):
            return False

        if self.is_ignored_heading(stripped):
            return False

        # Ignore extremely short text
        if len(stripped) < 3:
            return False

        return True

    # ---------------------------------------------------------
    # Extract sections
    # ---------------------------------------------------------

    def extract(self, lines):

        sections = {
            "required_skills": [],
            "preferred_skills": [],
            "responsibilities": [],
            "education": [],
            "other": []
        }

        current_section = None

        for line in lines:

            line = line.strip()

            if not line:
                continue
            
            if self.is_ignored_line(line):
                continue

            # ---------------------------------------------
            # Check whether this line is a section heading
            # ---------------------------------------------

            detected_section = self.identify_section(
                line
            )

            if detected_section:

                current_section = detected_section

                continue

            # ---------------------------------------------
            # Ignore headings such as "About the job"
            # ---------------------------------------------

            if self.is_ignored_heading(line):

                # We don't want the About the Job narrative
                # to be treated as a requirement.
                current_section = None

                continue

            # ---------------------------------------------
            # Extract content
            # ---------------------------------------------

            if not self.is_requirement_line(line):
                continue

            cleaned = self.clean_requirement(line)

            if not cleaned:
                continue

            if current_section is not None:

                sections[current_section].append(
                    cleaned
                )

            else:

                # No recognized heading is currently active --
                # rather than silently dropping this line (which
                # is how JDs with unrecognized heading wording
                # used to end up with zero extracted
                # requirements), keep it as a lower-confidence
                # "other" requirement so the pipeline still has
                # something to work with.
                sections["other"].append(
                    cleaned
                )

        return sections

    # ---------------------------------------------------------
    # Build structured requirement objects
    # ---------------------------------------------------------

    def build_requirement_objects(self, sections):

        requirements = []

        # Required qualifications
        for item in sections["required_skills"]:

            requirements.append({
                "requirement": item,
                "category": "required",
                "importance": "high"
            })

        # Preferred qualifications
        for item in sections["preferred_skills"]:

            requirements.append({
                "requirement": item,
                "category": "preferred",
                "importance": "medium"
            })

        # Responsibilities
        for item in sections["responsibilities"]:

            requirements.append({
                "requirement": item,
                "category": "responsibility",
                "importance": "medium"
            })

        # Education
        for item in sections["education"]:

            requirements.append({
                "requirement": item,
                "category": "education",
                "importance": "high"
            })

        # Fallback bucket -- lines that appeared under no
        # recognized heading. Kept as low-importance "other"
        # requirements rather than dropped, so a JD with
        # unfamiliar section wording still produces something
        # for the pipeline to score instead of failing outright.
        for item in sections["other"]:

            requirements.append({
                "requirement": item,
                "category": "other",
                "importance": "low"
            })

        return requirements

    def classify_requirement(self, text, current_section):

        normalized = text.lower()

        education_keywords = [
            "bachelor",
            "master",
            "phd",
            "degree",
            "doctoral",
            "undergraduate",
            "graduate degree"
        ]

        for keyword in education_keywords:

            if keyword in normalized:

                return "education"

        if current_section == "required_skills":

            return "required"

        if current_section == "preferred_skills":

            return "preferred"

        if current_section == "responsibilities":

            return "responsibility"

        return "other"

    def is_ignored_line(self, line):

        normalized = self.normalize(line)

        normalized = re.sub(
            r"[:\-]+$",
            "",
            normalized
        ).strip()

        return normalized in self.ignore_lines

# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":

    from src.extraction.jd_parser import JobDescriptionParser

    parser = JobDescriptionParser(
        "data/raw/job_description/job_description.txt"
    )

    parsed = parser.parse()

    print("\nRAW JD LINES")
    print("=" * 60)

    for index, line in enumerate(
        parsed["lines"]
    ):

        print(
            f"{index}: {repr(line)}"
        )

    extractor = RequirementExtractor()

    sections = extractor.extract(
        parsed["lines"]
    )

    print("\nEXTRACTED SECTIONS")
    print("=" * 60)

    for section, items in sections.items():

        print("\n" + section.upper())

        for item in items:

            print(
                "-",
                item
            )