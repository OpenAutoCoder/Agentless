import time
from typing import Dict, Union

import os
import openai
import anthropic
import tiktoken
import google.generativeai as genai



def num_tokens_from_messages(message, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if isinstance(message, list):
        # use last message.
        num_tokens = len(encoding.encode(message[0]["content"]))
    else:
        num_tokens = len(encoding.encode(message))
    return num_tokens


def create_chatgpt_config(
    message: Union[str, list],
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "gpt-3.5-turbo",
) -> Dict:
    if isinstance(message, list):
        config = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": batch_size,
            "messages": [{"role": "system", "content": system_message}] + message,
        }
    else:
        config = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": batch_size,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": message},
            ],
        }
    return config


def handler(signum, frame):
    # swallow signum and frame
    raise Exception("end of time")


def request_chatgpt_engine(config, logger, base_url=None, max_retries=40, timeout=100):
    ret = None
    retries = 0

    client = openai.OpenAI(base_url=base_url)

    while ret is None and retries < max_retries:
        try:
            # Attempt to get the completion
            logger.info("Creating API request")

            ret = client.chat.completions.create(**config)

        except openai.OpenAIError as e:
            if isinstance(e, openai.BadRequestError):
                logger.info("Request invalid")
                print(e)
                logger.info(e)
                raise Exception("Invalid API Request")
            elif isinstance(e, openai.RateLimitError):
                print("Rate limit exceeded. Waiting...")
                logger.info("Rate limit exceeded. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(5)
            elif isinstance(e, openai.APIConnectionError):
                print("API connection error. Waiting...")
                logger.info("API connection error. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(5)
            else:
                print("Unknown error. Waiting...")
                logger.info("Unknown error. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(1)

        retries += 1

    logger.info(f"API response {ret}")
    return ret


def request_gemini_engine(config, logger, base_url=None, max_retries=40, timeout=100):
    ret = None
    try:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model=genai.GenerativeModel(model_name=config["model"],
                                    system_instruction=config["system_message"])
        chat = model.start_chat(history=[])
        ret = chat.send_message(config["message"])
    except Exception as e:
        logger.info("Can't run Gemini engine")
        print(e)
    return ret
        

def request_anthropic_engine(config, logger, base_url=None, max_retries=40, timeout=100):
    ret = None
    retries = 0

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", None))

    while ret is None and retries < max_retries:
        try:
            # Attempt to get the completion
            logger.info("Creating API request")

            ret = client.messages.create(**config)

        except anthropic.AnthropicError as e:
            if isinstance(e, anthropic.BadRequestError):
                logger.info("Request invalid")
                print(e)
                logger.info(e)
                raise Exception("Invalid API Request")
            elif isinstance(e, anthropic.RateLimitError):
                print("Rate limit exceeded. Waiting...")
                logger.info("Rate limit exceeded. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(5)
            elif isinstance(e, anthropic.APIConnectionError):
                print("API connection error. Waiting...")
                logger.info("API connection error. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(5)
            else:
                print("Unknown error. Waiting...")
                logger.info("Unknown error. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(1)

        retries += 1

    logger.info(f"API response {ret}")
    return ret


def create_anthropic_config(
    message: str,
    prefill_message: str,
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "claude-2.1",
) -> Dict:
    if isinstance(message, list):
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system": system_message,
            "messages": message,
        }
    else:
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system": system_message,
            "messages": [
                {"role": "user", "content": message},
                {"role": "assistant", "content": prefill_message},
            ],
        }
    return config


# def request_anthropic_engine(client, config, logger, max_retries=40, timeout=100):
#     ret = None
#     retries = 0

#     while ret is None and retries < max_retries:
#         try:
#             start_time = time.time()
#             ret = client.messages.create(**config)
#         except Exception as e:
#             logger.error("Unknown error. Waiting...", exc_info=True)
#             # Check if the timeout has been exceeded
#             if time.time() - start_time >= timeout:
#                 logger.warning("Request timed out. Retrying...")
#             else:
#                 logger.warning("Retrying after an unknown error...")
#             time.sleep(10)
#         retries += 1

#     return ret
