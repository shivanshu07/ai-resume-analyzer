class SemanticChunker:

    def __init__(self, max_characters=1500):
        self.max_characters = max_characters

    def create_chunks(self, sections):

        chunks = []

        for section in sections:

            section_name = section["section"]
            content = section["content"]

            current_lines = []
            current_length = 0

            for line in content:

                proposed_length = (
                    current_length
                    + len(line)
                    + 1
                )

                if (
                    proposed_length > self.max_characters
                    and current_lines
                ):

                    chunk = self._create_chunk(
                        section_name,
                        current_lines
                    )

                    chunks.append(chunk)

                    current_lines = []
                    current_length = 0

                current_lines.append(line)

                current_length += len(line) + 1

            if current_lines:

                chunk = self._create_chunk(
                    section_name,
                    current_lines
                )

                chunks.append(chunk)

        return chunks

    def _create_chunk(self, section, lines):

        return {
            "section": section,
            "text": "\n".join(lines)
        }