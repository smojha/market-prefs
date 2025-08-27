from openai import OpenAI
import threading
import os
import json
import time
from typing import List
from mistralai import Mistral
import google.generativeai as genai
import anthropic
import heapq
import tiktoken
from config.constants import SUPPORTED_LLM_MODELS


# Define the constants and locks
TPM = 80000  # Tokens per minute limit
throughput = 0
heap = []
heap_lock = threading.Lock()
throughput_lock = threading.Lock()
heapq.heapify(heap)

def set_throughput(value):
    global throughput
    throughput = value

def get_throughput():
    global throughput
    return throughput

def clean_up_old_requests():
    """Remove requests from the heap that are older than 65 seconds."""
    current_time = time.time()
    with heap_lock:
        while heap and heap[0][0] < current_time - 65:
            _, freed_tokens = heapq.heappop(heap)
            set_throughput(get_throughput() - freed_tokens)

def estimate_claude_token_count(prompt):
    """Estimate token count for the Claude model."""
    import math
    model = "gpt-4o-2024-08-06"
    encoding = tiktoken.encoding_for_model(model)
    token_count = len(encoding.encode(prompt))
    # Add a 25% buffer as we are using Claude but the tokenizer is for GPT-4o
    return math.ceil(token_count * 1.25)

def wait_for_throughput_availability(token_count):
    """Wait until there is enough throughput capacity to make the request."""
    while True:
        with throughput_lock:
            clean_up_old_requests()  # Free up old requests
            if get_throughput() + token_count <= TPM:
                set_throughput(get_throughput() + token_count)
                with heap_lock:
                    heapq.heappush(heap, (time.time(), token_count))
                return
        time.sleep(0.25)  # Avoid busy-waiting, sleep briefly before retrying

def fetch_llm_completion(prompt, model_name):
    # route to the correct model and return the response
    if model_name not in SUPPORTED_LLM_MODELS:
        raise Exception("Model not supported")
    if SUPPORTED_LLM_MODELS[model_name]["service_provider"] == "OpenAI":
        # make call to OpenAI API
        openai_client = OpenAI(api_key= os.getenv('OPENAI_API_KEY'))
        MODEL_SPEC = SUPPORTED_LLM_MODELS[model_name]["model_name"]
        completion_raw = openai_client.chat.completions.create(
                        model=MODEL_SPEC,
                        response_format={"type": "json_object"},
                        messages=[{"role": "user", 
                                    "content": prompt}],
                    )
        # parse + return response
        resp_obj = json.loads(completion_raw.choices[0].message.content)
        return resp_obj
    elif SUPPORTED_LLM_MODELS[model_name]["service_provider"] == "Mistral":
        # make call to Mistral API
        api_key = os.getenv('MISTRAL_API_KEY')
        model = SUPPORTED_LLM_MODELS[model_name]["model_name"]

        client = Mistral(api_key=api_key)
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        chat_response = client.chat.complete(
            model = model,
            messages = messages,
            response_format = {
                "type": "json_object",
            }
        )

        resp_obj = chat_response.choices[0].message.content

        # turn resp_obj into a dictionary
        if isinstance(resp_obj, str):
            resp_obj = json.loads(resp_obj)
        return resp_obj

    elif SUPPORTED_LLM_MODELS[model_name]["service_provider"] == "xAI":
        # make call to xAI API
        xapi_key = os.getenv('XAI_API_KEY')
        model = SUPPORTED_LLM_MODELS[model_name]["model_name"]
        client = OpenAI(
        api_key=xapi_key,
        base_url="https://api.x.ai/v1",
        )

        completion = client.chat.completions.create(
            model=model,
            messages=[
                # {"role": "system", "content": "You are Grok, a chatbot inspired by the Hitchhikers Guide to the Galaxy."},
                {"role": "user", "content": prompt},
            ],
        )
        resp_obj = json.loads(completion.choices[0].message.content)
        return resp_obj
    elif SUPPORTED_LLM_MODELS[model_name]["service_provider"] == "Google":
        # make call to Google API
        api_key = os.getenv('GOOGLEAI_API_KEY')
        genai.configure(api_key=api_key)
        model_name = SUPPORTED_LLM_MODELS[model_name]["model_name"]
        model = genai.GenerativeModel(model_name)
        resp_obj = model.generate_content(prompt).text
        # it often looks like a string with the ```json at the beginning and ``` at the end. we need to remove these before parsing the JSON
        resp_obj = resp_obj.replace('```json', '').replace('```', '')
        # turn resp_obj into a dictionary
        if isinstance(resp_obj, str):
            resp_obj = json.loads(resp_obj)
        return resp_obj
    elif SUPPORTED_LLM_MODELS[model_name]["service_provider"] == "Anthropic":
        # make call to Anthropic API
        api_key = os.getenv('ANTHROPIC_API_KEY')
        model = SUPPORTED_LLM_MODELS[model_name]["model_name"]
        client = anthropic.Anthropic(api_key=api_key)

        # Estimate token count for this request
        token_count = estimate_claude_token_count(prompt)

        # Wait until there is enough throughput capacity
        wait_for_throughput_availability(token_count)
        
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        resp_obj = json.loads(message.content[0].text)
        return resp_obj