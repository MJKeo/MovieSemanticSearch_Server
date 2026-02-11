import os
import time
import json
import re
import sys
import csv
 
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, field_validator, Field, RootModel
from enum import Enum
from typing import List, Optional, Dict, Tuple, Any
from datetime import date
from concurrent import futures
from tqdm import tqdm
from pathlib import Path
from openai.lib._pydantic import to_strict_json_schema

# Load environment variables (for API key)
load_dotenv()


# ===============================
#           Clients
# ===============================

openai_api_key = os.getenv("OPENAI_API_KEY")
kimi_api_key = os.environ.get("MOONSHOT_API_KEY")

openai_client = OpenAI(api_key=openai_api_key)
kimi_client = OpenAI(
    api_key=kimi_api_key,
    base_url="https://api.moonshot.ai/v1",
)


# ===============================
#     Base Generation Methods
# ===============================

def generate_openai_response(
    user_prompt: str,
    system_prompt: str,
    response_format: BaseModel,
    model: str = "gpt-5-mini",
    reasoning_effort: str = "low",
    verbosity: str = "low"
):
    try:
        response = openai_client.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=response_format,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity
        )
        
        # Extract the parsed response - OpenAI automatically validates structure matches response_format
        return response.choices[0].message.parsed
    except Exception as e:
        raise ValueError(f"OpenAI failed to generate response: {e}")

def generate_kimi_response(
    user_prompt: str,
    system_prompt: str,
    response_format: BaseModel,
    enable_thinking: bool = False,
):
    try:
        thinking_type = "enabled" if enable_thinking else "disabled"
        schema = to_strict_json_schema(response_format)

        response = kimi_client.chat.completions.create(
            model="kimi-k2.5",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__class__.__name__,
                    "strict": True,
                    "schema": schema,
                },
            },
            extra_body={
                "thinking": {"type": thinking_type}
            }
        )
        
        # Extract the parsed response and enforce schema structure
        raw = response.choices[0].message.content
        data = json.loads(raw)
        metadata = response_format.model_validate(data)
        return metadata
    except Exception as e:
        raise ValueError(f"Kimi failed to generate response: {e}")