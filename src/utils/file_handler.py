import json


class FileHandler:

    def save_text(self, text, path):

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(text)

    def load_text(self, path):

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            return file.read()

    def save_json(self, data, path):

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False
            )

    def load_json(self, path):

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)