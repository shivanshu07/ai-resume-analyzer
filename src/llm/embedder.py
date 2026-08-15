import json

from sentence_transformers import SentenceTransformer


class TextEmbedder:

    def __init__(
        self,
        model_name="all-MiniLM-L6-v2"
    ):

        print(
            f"\nLoading embedding model: "
            f"{model_name}"
        )

        self.model = SentenceTransformer(
            model_name
        )

        print(
            "Embedding model loaded successfully."
        )

    # ========================================================
    # Generate embeddings
    # ========================================================

    def generate_embeddings(
        self,
        texts
    ):

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings

    # ========================================================
    # Save embeddings
    # ========================================================

    def save_embeddings(
        self,
        embeddings,
        output_path
    ):

        embeddings_list = (
            embeddings.tolist()
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                embeddings_list,
                file,
                indent=4
            )

    # ========================================================
    # Load embeddings
    # ========================================================

    def load_embeddings(
        self,
        input_path
    ):

        with open(
            input_path,
            "r",
            encoding="utf-8"
        ) as file:

            embeddings = json.load(
                file
            )

        return embeddings