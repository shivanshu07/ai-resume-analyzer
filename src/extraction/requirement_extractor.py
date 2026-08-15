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
                "must have"
            ],

            "preferred_skills": [
                "preferred qualifications",
                "preferred qualification",
                "preferred skills",
                "nice to have"
            ],

            "responsibilities": [
                "responsibilities",
                "responsibility",
                "key responsibilities"
            ],

            "education": [
                "education",
                "educational qualification",
                "educational qualifications"
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

        for section, keywords in self.section_keywords.items():

            for keyword in keywords:

                if normalized == keyword:
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

            if (
                current_section is not None
                and self.is_requirement_line(line)
            ):

                cleaned = self.clean_requirement(line)

                if cleaned:

                    sections[current_section].append(
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