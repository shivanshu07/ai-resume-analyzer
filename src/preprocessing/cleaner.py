import re
import unicodedata


class TextCleaner:
    """
    Clean and structure extracted resume text while preserving
    semantic meaning and section boundaries.
    """

    RESUME_SECTIONS = {
        "SUMMARY": "SUMMARY",
        "PROFILE": "SUMMARY",
        "OBJECTIVE": "SUMMARY",

        "SKILLS": "SKILLS",
        "TECHNICAL SKILLS": "SKILLS",

        "WORK EXPERIENCE": "WORK EXPERIENCE",
        "WORK EXPERIENCE & INTERNSHIPS": "WORK EXPERIENCE",
        "EXPERIENCE": "WORK EXPERIENCE",
        "PROFESSIONAL EXPERIENCE": "WORK EXPERIENCE",
        "EMPLOYMENT": "WORK EXPERIENCE",

        "PROJECTS": "PROJECTS",
        "PROJECT": "PROJECTS",

        "EDUCATION": "EDUCATION",

        "CERTIFICATIONS": "CERTIFICATIONS & LANGUAGES",
        "CERTIFICATION": "CERTIFICATIONS & LANGUAGES",
        "CERTIFICATIONS & LANGUAGES": "CERTIFICATIONS & LANGUAGES",

        "LANGUAGES": "CERTIFICATIONS & LANGUAGES",

        "ACHIEVEMENTS": "ACHIEVEMENTS",
        "AWARDS": "AWARDS",
        "PUBLICATIONS": "PUBLICATIONS",
        "COURSEWORK": "COURSEWORK",
        "INTERESTS": "INTERESTS",
    }

    def normalize_unicode(self, text):
        """
        Normalize Unicode characters without removing
        meaningful information.
        """

        return unicodedata.normalize("NFKC", text)

    def normalize_dashes(self, text):
        """
        Normalize different dash characters.
        """

        text = text.replace("–", "-")
        text = text.replace("—", "-")
        text = text.replace("−", "-")

        return text

    def remove_pdf_artifacts(self, text):

        text = text.replace(
            "\ufffd",
            ""
        )

        text = text.replace(
            "\\*",
            ""
        )

        text = re.sub(
            r"(?m)^\s*\*+\s*$",
            "",
            text
        )

        text = re.sub(
            r"(?m)^\s*[•●▪◦■□◆◇]\s*",
            "- ",
            text
        )

        return text

    def normalize_spaces(self, text):

        text = text.replace(
            "\t",
            " "
        )

        text = re.sub(
            r"[ \t]+$",
            "",
            text,
            flags=re.MULTILINE
        )

        text = re.sub(
            r"[ \t]{2,}",
            " ",
            text
        )

        text = re.sub(
            r"\n[ \t]*\n+",
            "\n\n",
            text
        )

        return text

    def normalize_heading(self, line):

        heading = line.strip()

        heading = re.sub(
            r"^[\s|:_\-•●▪◦]+",
            "",
            heading
        )

        heading = re.sub(
            r"[\s|:_\-•●▪◦]+$",
            "",
            heading
        )

        heading = re.sub(
            r"\s+",
            " ",
            heading
        )

        heading = unicodedata.normalize(
            "NFKC",
            heading
        )

        return heading.upper().strip()

    def is_section_heading(self, line):
        """
        Determine whether a line represents a resume section heading.
        """

        normalized = self.normalize_heading(line)

        return normalized in self.RESUME_SECTIONS

    def get_section_name(self, line):
        """
        Return the canonical section name.
        """

        normalized = self.normalize_heading(line)

        return self.RESUME_SECTIONS.get(
            normalized
        )

    def is_bullet(self, line):

        return bool(
            re.match(
                r"^\s*-\s+",
                line
            )
        )

    def is_date_line(self, line):
        """
        Detect common resume date formats.
        """

        month_pattern = (
            r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        )

        month_year = re.search(
            rf"\b{month_pattern}\s+\d{{4}}\b",
            line,
            flags=re.IGNORECASE
        )

        year_range = re.search(
            r"\b\d{4}\s*-\s*(Present|\d{4})\b",
            line,
            flags=re.IGNORECASE
        )

        return bool(
            month_year or year_range
        )

    def is_skill_category(self, line):
        """
        Detect skill category lines such as:

        Programming:
        Machine Learning:
        Deep Learning:
        """

        return bool(
            re.match(
                r"^[A-Za-z][A-Za-z /&-]{1,40}:",
                line.strip()
            )
        )

    def fix_hyphenated_line_breaks(self, text):

        return re.sub(
            r"(\w)-\s*\n\s*(\w)",
            r"\1-\2",
            text
        )

    def clean_basic_text(self, text):
        """
        Perform safe cleaning that does not alter
        section boundaries.
        """

        if not text:
            return ""

        text = self.normalize_unicode(text)

        text = self.normalize_dashes(text)

        text = self.remove_pdf_artifacts(text)

        text = self.fix_hyphenated_line_breaks(text)

        text = self.normalize_spaces(text)

        return text.strip()

    def merge_wrapped_lines(self, lines):
    # Merge PDF visual line wrapping while preserving
    #     semantic boundaries such as:

    #     - section headings
    #     - bullet points
    #     - project titles
    #     - job titles
    #     - dates
    #     - skill categories

    #     A non-bullet line followed by a date is treated as
    #     a title and is therefore NOT merged into the previous
    #     bullet point.

        merged = []

        for index, line in enumerate(lines):

            stripped = line.strip()

            if not stripped:
                continue

            # -----------------------------------------
            # 1. Section headings must always remain
            #    independent.
            # -----------------------------------------

            if self.is_section_heading(stripped):

                merged.append(stripped)

                continue

            # -----------------------------------------
            # 2. First meaningful line
            # -----------------------------------------

            if not merged:

                merged.append(stripped)

                continue

            previous = merged[-1]

            # -----------------------------------------
            # 3. IMPORTANT:
            #    If this line is followed by a date,
            #    it is most likely a project/job/education
            #    title.
            #
            # Example:
            #
            # Number Plate Recognition using CV & OCR
            # Feb 2023 - May 2023
            #
            # Therefore, DO NOT merge it with the
            # previous bullet.
            # -----------------------------------------

            next_line = ""

            if index + 1 < len(lines):

                next_line = lines[index + 1].strip()

            if (
                next_line
                and self.is_date_line(next_line)
            ):

                merged.append(stripped)

                continue

            # -----------------------------------------
            # 4. Date lines remain independent.
            # -----------------------------------------

            if self.is_date_line(stripped):

                merged.append(stripped)

                continue

            # -----------------------------------------
            # 5. Bullet lines start a new semantic
            #    unit.
            # -----------------------------------------

            if self.is_bullet(stripped):

                merged.append(stripped)

                continue

            # -----------------------------------------
            # 6. Skill category lines remain separate.
            #
            # Example:
            #
            # Machine Learning:
            # Deep Learning:
            # -----------------------------------------

            if self.is_skill_category(stripped):

                merged.append(stripped)

                continue

            # -----------------------------------------
            # 7. Contact information remains separate.
            # -----------------------------------------

            if (
                "@" in stripped
                or "linkedin" in stripped.lower()
                or "github" in stripped.lower()
            ):

                merged.append(stripped)

                continue

            # -----------------------------------------
            # 8. If previous line is a bullet,
            #    this line is likely a wrapped continuation
            #    of that bullet.
            # -----------------------------------------

            if self.is_bullet(previous):

                merged[-1] = (
                    previous + " " + stripped
                )

                continue

            # -----------------------------------------
            # 9. Otherwise, treat it as a continuation
            #    of the previous line.
            # -----------------------------------------

            merged[-1] = (
                previous + " " + stripped
            )

        return merged

    def clean(self, text):
        """
        Main cleaning pipeline.

        Important:
        Section detection happens before semantic merging.
        """

        # -----------------------------------
        # Step 1: Safe cleaning
        # -----------------------------------

        text = self.clean_basic_text(
            text
        )

        # -----------------------------------
        # Step 2: Preserve original lines
        # -----------------------------------

        lines = text.splitlines()

        # -----------------------------------
        # Step 3: Detect sections BEFORE
        # aggressive line merging
        # -----------------------------------

        sections = self.detect_sections(
            lines
        )

        # -----------------------------------
        # Step 4: Clean content inside
        # each detected section
        # -----------------------------------

        cleaned_sections = []

        for section in sections:

            merged_content = self.merge_wrapped_lines(
                section["content"]
            )

            cleaned_sections.append(
                {
                    "section": section["section"],
                    "content": merged_content
                }
            )

        return cleaned_sections

    def detect_sections(self, lines):
        """
        Detect resume sections from raw cleaned lines.
        """

        sections = []

        current_section = None

        for line in lines:

            stripped = line.strip()

            if not stripped:
                continue

            # -----------------------------------
            # Section heading
            # -----------------------------------

            if self.is_section_heading(
                stripped
            ):

                current_section = self.get_section_name(
                    stripped
                )

                sections.append(
                    {
                        "section": current_section,
                        "content": []
                    }
                )

                continue

            # -----------------------------------
            # Content belonging to current section
            # -----------------------------------

            if current_section:

                sections[-1]["content"].append(
                    stripped
                )

        return sections