from abc import ABC, abstractmethod
from typing import List

from langchain_openai import ChatOpenAI, AzureChatOpenAI

from Agentless.agentless.util.api_requests import create_chatgpt_config, request_chatgpt_engine
from apps.services.open_ia_llm import OpenIA_LLM


class DecoderBase(ABC):
    def __init__(
        self,
        name: str,
        batch_size: int = 1,
        temperature: float = 0.8,
        max_new_tokens: int = 1024,
    ) -> None:
        print("Initializing a decoder model: {} ...".format(name))
        self.name = name
        self.batch_size = batch_size
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens


    @abstractmethod
    def is_direct_completion(self) -> bool:
        pass

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


class OpenAIChatDecoder(DecoderBase):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)


    def is_direct_completion(self) -> bool:
        return False


def make_model(
    model: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
):
    if OpenIA_LLM.use_azure:
        return AzureChatOpenAI(deployment_name=model, temperature=temperature)
    return ChatOpenAI(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )

