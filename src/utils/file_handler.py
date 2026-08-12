import json
from pathlib import Path


class FileHandler:
    """
    Utility class for reading and writing project files.
    """

    def save_text(self, text, path):
        """
        Save text to a UTF-8 encoded file.

        Parameters
        ----------
        text : str
            Text that needs to be saved.

        path : str
            Destination file path.
        """

        path = Path(path)

        # Create parent directories if they don't exist
        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(text)

    def load_text(self, path):
        """
        Load text from a UTF-8 encoded file.

        Parameters
        ----------
        path : str
            Path of the text file.

        Returns
        -------
        str
            File contents.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {path}"
            )

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            return file.read()

    def save_json(self, data, path):
        """
        Save Python data as formatted JSON.

        Parameters
        ----------
        data : dict or list
            Data to be saved.

        path : str
            Destination JSON file path.
        """

        path = Path(path)

        # Create parent directories if they don't exist
        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

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
        """
        Load JSON data from a file.

        Parameters
        ----------
        path : str
            Path of the JSON file.

        Returns
        -------
        dict or list
            Parsed JSON data.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"JSON file not found: {path}"
            )

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)