from typing import Dict, Any, AsyncGenerator, List
import asyncio
import threading
from transformers import TextIteratorStreamer
import httpx, json
import unicodedata, time

# logger = get_logger('llm', '🤖')


class LLMGenerator:
    def __init__(self,
                 client: Any = None,  # ollama client
                 model_name: str | None = None
                 ):
        self.model_name = model_name
        self.client = client
                
    # ---------- NON-STREAMING ----------
    def generate(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 20000,
        temperature: float = 0.2,
        top_p: float = 0.95,
        reasoning_effort: str = "medium",   # "low" | "medium" | "high"
        ) -> str:
        
        MAX_CONTEXT = 131072 
        options = {
                "num_ctx": MAX_CONTEXT}
        
        if reasoning_effort:
            options["reasoning_effort"] = reasoning_effort 


        kwargs = dict(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            extra_body=options)
        
        start_time = time.perf_counter()

        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        thinking = hasattr(message, 'reasoning') and message.reasoning

        end_time = time.perf_counter()
        duration = end_time - start_time

        answer = message.content
        if answer:
            answer = answer
            # answer = ''.join(
            #             ch for ch in unicodedata.normalize('NFKC', answer)
            #             if unicodedata.category(ch) != 'Cf')
            
        else:
             raise ValueError("LLM returned empty response")
        
        # # Metrics Extraction
        p_tokens = response.usage.prompt_tokens
        c_tokens = response.usage.completion_tokens
        tps = c_tokens / duration if duration > 0 else 0

        current_metrics = {
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "total_tokens": response.usage.total_tokens,
            "duration": duration,
            "tps": tps,
            "ttft": None # Non-streaming
        }
        print(f"LLM Generation Metrics: {current_metrics}")

        return  answer
              
    # ---------- STREAMING ----------
    async def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 512,
        temperature: float = 0.5,
        top_p: float = 0.95,
        reasoning_effort: str | None = None,   # "low" | "medium" | "high"
        task: str = 'test'
    ) -> AsyncGenerator:
        
        MAX_CONTEXT = 131072 
        options = {
            "num_ctx": MAX_CONTEXT
        }
        if reasoning_effort:
            options["reasoning_effort"] = reasoning_effort


        # start_time = time.perf_counter()
        # ttft = None  # Time to first token

        kwargs = dict(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=options
        )
        stream = self.client.chat.completions.create(**kwargs)
        for chunk in stream:
                # 1. Capture Time to First Token (TTFT)
            # if ttft is None and chunk.choices and chunk.choices[0].delta.content:
            #     ttft = time.perf_counter() - start_time

            # if hasattr(chunk, 'usage') and chunk.usage is not None:
            #     yield {
            #         "type": "usage",
            #         "prompt_tokens": chunk.usage.prompt_tokens,
            #         "completion_tokens": chunk.usage.completion_tokens,
            #         "total_tokens": chunk.usage.total_tokens
            #     }

                # end_time = time.perf_counter()
                # duration = end_time - start_time

                # gen_duration = duration - (ttft or 0)
                # tps = chunk.usage.completion_tokens / gen_duration if gen_duration > 0 else 0

                # metrics = {
                #     "prompt_tokens": chunk.usage.prompt_tokens,
                #     "completion_tokens": chunk.usage.completion_tokens,
                #     "total_tokens": chunk.usage.total_tokens,
                #     "duration": duration,
                #     "ttft": ttft,
                #     "tps": tps
                # }
                # # yield {"type": "usage", **metrics}

                # continue
            if chunk.choices:
                delta = chunk.choices[0].delta

                thinking = hasattr(delta, 'reasoning') and delta.reasoning
                if thinking:
                    yield {"type": "thinking", "text": thinking}

                if getattr(delta, "content", None):
                    yield {"type": "token", "text": delta.content}