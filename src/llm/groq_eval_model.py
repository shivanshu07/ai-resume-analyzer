"""
A deepeval-compatible LLM judge backed by Groq instead of
OpenAI.

deepeval's metrics (GEval, FaithfulnessMetric, etc.) default to
using OpenAI as the "judge" model that scores your LLM's
output. That would reintroduce the exact OpenAI cost dependency
this project deliberately moved away from in gap_explainer.py.

This class lets deepeval use Groq's free tier as the judge
instead, by subclassing DeepEvalBaseLLM -- the same officially
supported extension point deepeval provides for Anthropic,
Gemini, Ollama, or any other non-OpenAI provider.

Reuses the exact same OpenAI-compatible-client-pointed-at-Groq
pattern already used in gap_explainer.py, rather than
introducing a second, different way of talking to Groq.
"""

from openai import OpenAI

from deepeval.models.base_model import DeepEvalBaseLLM

from config.settings import GROQ_API_KEY, GROQ_BASE_URL, MODEL_NAME


class GroqEvalModel(DeepEvalBaseLLM):

    def __init__(
        self,
        api_key=None,
        model_name=None,
        base_url=None
    ):

        self.api_key = api_key or GROQ_API_KEY
        self.model_name = model_name or MODEL_NAME
        self.base_url = base_url or GROQ_BASE_URL

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def load_model(self):

        return self

    def generate(self, prompt: str) -> str:

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        return response.choices[0].message.content or ""

    async def a_generate(self, prompt: str) -> str:

        # deepeval calls a_generate for async evaluation runs.
        # Groq's free-tier client here is synchronous, so this
        # just wraps the sync call rather than maintaining a
        # separate async client -- fine for CI-scale evaluation
        # volumes, not meant for high-throughput production use.
        return self.generate(prompt)

    def get_model_name(self) -> str:

        return f"groq:{self.model_name}"