"""LangChain LLM wrapper (optional: requires ``langchain``).

Lets you drop lesforge's mock VLM into a LangChain chain as an LLM. The
``_generate`` hook maps LangChain prompts to the backend's ``generate`` call.
"""


class LesForgeLLM:
    """Minimal LangChain-compatible LLM wrapper around a lesforge backend."""

    def __init__(self, backend):
        self.backend = backend
        self._identifying_params = {"backend": getattr(backend, "name", "unknown")}

    @property
    def _llm_type(self):
        return "lesforge"

    def _generate(self, prompts, stop=None, run_manager=None, **kwargs):
        from langchain.schema import LLMResult, Generation  # type: ignore
        generations = []
        for p in prompts:
            text, _ = self.backend.generate(p)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    def __call__(self, prompt, stop=None, **kwargs):
        text, _ = self.backend.generate(prompt)
        return text
