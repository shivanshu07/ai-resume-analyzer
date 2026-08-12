import re


class SemanticChunker:
    """
    Create semantically meaningful chunks from structured
    resume sections.
    """

    def __init__(self, max_characters=1800):

        self.max_characters = max_characters

    def is_bullet(self, line):
        """
        Determine whether a line is a bullet point.
        """

        return bool(
            re.match(
                r"^\s*-\s+",
                line
            )
        )

    def is_date_line(self, line):
        """
        Determine whether a line represents a date/range.
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

    def create_chunks(self, sections):

        chunks = []

        chunk_id = 1

        for section in sections:

            section_name = section["section"]

            content = section["content"]

            if section_name == "PROJECTS":

                section_chunks = self.chunk_projects(
                    section_name,
                    content
                )

            elif section_name in {
                "WORK EXPERIENCE",
                "WORK EXPERIENCE & INTERNSHIPS",
                "EXPERIENCE",
                "PROFESSIONAL EXPERIENCE",
                "EMPLOYMENT",
            }:

                section_chunks = self.chunk_experience(
                    section_name,
                    content
                )

            else:

                section_chunks = self.chunk_generic_section(
                    section_name,
                    content
                )

            for chunk in section_chunks:

                chunk["chunk_id"] = (
                    f"resume_{chunk_id:03d}"
                )

                chunks.append(chunk)

                chunk_id += 1

        return chunks

    def chunk_generic_section(
        self,
        section_name,
        content
    ):
        """
        Create chunks for general resume sections.
        """

        chunks = []

        current_lines = []
        current_length = 0

        for line in content:

            line_length = len(line) + 1

            if (
                current_length + line_length
                > self.max_characters
                and current_lines
            ):

                chunks.append(
                    self.create_chunk(
                        section_name,
                        current_lines
                    )
                )

                current_lines = []
                current_length = 0

            current_lines.append(line)

            current_length += line_length

        if current_lines:

            chunks.append(
                self.create_chunk(
                    section_name,
                    current_lines
                )
            )

        return chunks

    def chunk_experience(
        self,
        section_name,
        content
    ):
        """
        Keep work experience information together
        whenever possible.
        """

        if not content:
            return []

        total_length = sum(
            len(line) + 1
            for line in content
        )

        if total_length <= self.max_characters:

            return [
                self.create_chunk(
                    section_name,
                    content
                )
            ]

        return self.split_large_block(
            section_name,
            content
        )

    def chunk_projects(
        self,
        section_name,
        content
    ):
        """
        Group project title, date, and associated bullets
        into independent semantic units.
        """

        projects = []

        current_project = []

        for index, line in enumerate(content):

            # A non-bullet line followed by a date line
            # is treated as a project title.
            if (
                not self.is_bullet(line)
                and index + 1 < len(content)
                and self.is_date_line(
                    content[index + 1]
                )
            ):

                # Finalize previous project
                if current_project:

                    projects.append(
                        self.create_chunk(
                            section_name,
                            current_project
                        )
                    )

                current_project = [line]

            else:

                current_project.append(line)

        # Finalize final project
        if current_project:

            projects.append(
                self.create_chunk(
                    section_name,
                    current_project
                )
            )

        if not projects:

            return self.chunk_generic_section(
                section_name,
                content
            )

        final_chunks = []

        for project in projects:

            project_lines = project["text"].splitlines()

            text_length = sum(
                len(line) + 1
                for line in project_lines
            )

            if text_length <= self.max_characters:

                final_chunks.append(project)

            else:

                final_chunks.extend(
                    self.split_large_block(
                        section_name,
                        project_lines
                    )
                )

        return final_chunks

    def split_large_block(
        self,
        section_name,
        lines
    ):
        """
        Split an oversized block while trying not to
        separate bullet points unnecessarily.
        """

        chunks = []

        current_lines = []
        current_length = 0

        for line in lines:

            line_length = len(line) + 1

            if (
                current_lines
                and current_length + line_length
                > self.max_characters
                and self.is_bullet(line)
            ):

                chunks.append(
                    self.create_chunk(
                        section_name,
                        current_lines
                    )
                )

                current_lines = []
                current_length = 0

            current_lines.append(line)

            current_length += line_length

        if current_lines:

            chunks.append(
                self.create_chunk(
                    section_name,
                    current_lines
                )
            )

        return chunks

    def create_chunk(
        self,
        section_name,
        lines
    ):
        """
        Create a structured semantic chunk.
        """

        return {
            "section": section_name,
            "text": "\n".join(lines).strip()
        }