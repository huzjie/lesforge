"""Five inference backends registered behind a common interface.

    * mock         - deterministic, trainable, zero-dependency (default)
    * cpu          - naive CPU reference loop over the same MoE math
    * openai       - delegate to an OpenAI-compatible chat endpoint
    * vllm         - delegate to a vLLM OpenAI-compatible server
    * transformers - delegate to a local HF transformers pipeline

Every backend exposes ``generate(prompt, image=None) -> str`` and ``skill`` (the
trainable capability knob used by the mock/cpu backends). The mock backend is
what the benchmarks and the training loop use.
"""
from .core.registry import BACKENDS


def register_backend(key):
    def deco(cls):
        BACKENDS.register(key)(cls)
        return cls
    return deco


class _FallbackMixin:
    """Common OpenAI-compatible HTTP delegation used by the hosted backends."""

    def __init__(self, base_url, model, api_key=None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def _chat(self, prompt, image=None):
        import json
        import urllib.request

        messages = [{"role": "user", "content": prompt}]
        body = {"model": self.model, "messages": messages, "max_tokens": 256}
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        if self.api_key:
            req.add_header("Authorization", "Bearer " + self.api_key)
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


@register_backend("mock")
class MockBackend:
    name = "mock"

    def __init__(self, cfg):
        from .model.engine import VLMEngine
        self.cfg = cfg
        self.engine = VLMEngine(cfg)
        self.skill = cfg.model.skill

    def generate(self, prompt, image=None, max_tokens=None):
        out = self.engine.generate(image=image, prompt=prompt, skill=self.skill,
                                   max_tokens=max_tokens)
        return out["text"], out

    def stats(self):
        return self.engine.scheduler.stats()


@register_backend("cpu")
class CpuBackend:
    name = "cpu"

    def __init__(self, cfg):
        from .model.engine import VLMEngine
        self.cfg = cfg
        self.engine = VLMEngine(cfg)
        self.skill = cfg.model.skill

    def generate(self, prompt, image=None, max_tokens=None):
        # same reference math, but flagged as the CPU loop
        out = self.engine.generate(image=image, prompt=prompt, skill=self.skill,
                                   max_tokens=max_tokens)
        out["backend"] = "cpu"
        return out["text"], out


@register_backend("openai")
class OpenAIBackend(_FallbackMixin):
    name = "openai"

    def __init__(self, cfg):
        super().__init__("https://api.openai.com/v1", cfg.model.name, None)
        self.skill = 1.0

    def generate(self, prompt, image=None, max_tokens=None):
        return self._chat(prompt, image), {}


@register_backend("vllm")
class VLLMBackend(_FallbackMixin):
    name = "vllm"

    def __init__(self, cfg):
        super().__init__("http://127.0.0.1:8001/v1", cfg.model.name, None)
        self.skill = 1.0

    def generate(self, prompt, image=None, max_tokens=None):
        return self._chat(prompt, image), {}


@register_backend("transformers")
class TransformersBackend:
    name = "transformers"

    def __init__(self, cfg):
        self.model_name = cfg.model.name
        self.skill = 1.0
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            from transformers import pipeline  # type: ignore
            self._pipe = pipeline("text-generation", model=self.model_name)
        return self._pipe

    def generate(self, prompt, image=None, max_tokens=None):
        out = self._load()(prompt, max_new_tokens=max_tokens or 32)
        return out[0]["generated_text"], {}


def get_backend(name, cfg):
    cls = BACKENDS.get(name)
    if cls is None:
        raise KeyError("unknown backend %r (available: %s)" % (name, BACKENDS.keys()))
    return cls(cfg)


def list_backends():
    return BACKENDS.keys()
