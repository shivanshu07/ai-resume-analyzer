import re


class JobDescriptionParser:

    def __init__(self, file_path):
        self.file_path = file_path

    def load_text(self):
        with open(self.file_path, "r", encoding="utf-8") as file:
            return file.read()

    def clean_text(self, text):
        text = text.replace("\r", "\n")

        # Remove excessive spaces
        text = re.sub(r"[ \t]+", " ", text)

        # Remove excessive blank lines
        text = re.sub(r"\n+", "\n", text)

        return text.strip()

    def split_into_lines(self, text):
        return [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

    def parse(self):
        text = self.load_text()
        text = self.clean_text(text)
        lines = self.split_into_lines(text)

        return {
            "raw_text": text,
            "lines": lines
        }