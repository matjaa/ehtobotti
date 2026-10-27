import dataclasses
import sys
import types

from ehtobotti.config import ChatModelOptions
from ehtobotti.llm import OllamaClient
from tests.test_generate import SETTINGS


class FakeOllama:
    def __init__(self, host):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))


def client_for(settings, monkeypatch):
    monkeypatch.setitem(sys.modules, "ollama", types.SimpleNamespace(Client=FakeOllama))
    return OllamaClient(settings)


def test_think_flag_only_sent_to_models_that_support_it(monkeypatch):
    settings = dataclasses.replace(SETTINGS, seed=7, chat_models={
        "qwen": ChatModelOptions(think_supported=True, think=True),
        "gemma": ChatModelOptions(think_supported=False),
    })
    client = client_for(settings, monkeypatch)
    client.chat([{"role": "user", "content": "x"}], model="qwen", format={"type": "object"})
    client.chat([{"role": "user", "content": "x"}], model="gemma")
    qwen, gemma = client._client.calls
    assert qwen["think"] is True and qwen["format"] == {"type": "object"} and qwen["options"]["seed"] == 7
    assert "think" not in gemma and "format" not in gemma
