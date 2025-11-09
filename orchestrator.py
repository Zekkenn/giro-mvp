# orchestrator.py
"""
Drop-in orchestrator that you can replace with your deep-agent + LangChain stack.

Two main hooks:
- Orchestrator.chat_stream(...) -> yields text chunks for streaming
- Orchestrator.generate_image(...) -> returns list of image file paths
"""
import base64
import io
import os
import time
from dataclasses import dataclass
from typing import Dict, Any, Generator, List, Optional

from dotenv import load_dotenv

load_dotenv()

@dataclass
class ChatMessage:
    role: str
    content: str

class Orchestrator:
    def __init__(self):
        self.model = os.getenv("OPENAI_MODEL", "gpt-5")
        self.api_key = os.getenv("OPENAI_API_KEY", None)
        self._client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except Exception:
                self._client = None

    # ---------------------
    # Chat (streaming)
    # ---------------------
    def chat_stream(
        self,
        subject: str,
        topic_facts: str,
        profile: Dict[str, Any],
        history: List[ChatMessage],
        message: str,
        session_id: str,
    ) -> Generator[str, None, None]:
        """
        Replace this method to call your own deep-agent planner/executor.
        Must yield strings (partial tokens or growing text) for streaming.
        """
        system = profile.get("system_prompt") or (
            "You are a rigorous, friendly AI tutor. Ask a brief guiding question before explaining; "
            "end with one mini exercise; never reveal chain-of-thought."
        )

        if not self._client:
            # Stubbed stream: fake token stream for demos without API key
            fake = (
                f"Okay! Let's explore **{subject}**.\n\n"
                f"First, what do you already know about this topic?\n\n"
                f"Key facts I'm using:\n{topic_facts[:400]}{'...' if len(topic_facts)>400 else ''}\n\n"
                "Now, step 1: "
            )
            for ch in fake:
                time.sleep(0.01)
                yield ch
            return

        # Build messages
        msgs = [{"role":"system","content":system}]
        if topic_facts.strip():
            msgs.append({"role":"system","content":f"TOPIC FACTS (teacher-provided):\n{topic_facts}"})
        for m in history:
            msgs.append({"role":m.role, "content":m.content})
        msgs.append({"role":"user","content":message})

        try:
            # Use Chat Completions streaming (OpenAI SDK >=1.0)
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=msgs,
                stream=True,
            )
            for event in resp:
                if not event.choices: 
                    continue
                delta = event.choices[0].delta.content or ""
                if delta:
                    yield delta
        except Exception as e:
            # Fallback to legacy API if available or emit error
            yield f"\n[Streaming error: {e}]"

    # ---------------------
    # Image generation
    # ---------------------
    def generate_image(self, prompt:str, size:str="768x768", steps:int=28, guidance:float=7.5, seed:Optional[int]=-1)->List[str]:
        """
        Return list of file paths to generated images.
        Replace with your preferred image API (OpenAI, Stability, etc.).
        """
        if self._client:
            try:
                # OpenAI Images API style (gpt-image-1). Adjust to your provider.
                from PIL import Image
                w, h = [int(x) for x in size.split("x")]
                result = self._client.images.generate(
                    model="gpt-image-1",
                    prompt=prompt,
                    size=size,
                    n=1,
                )
                b64 = result.data[0].b64_json
                img_bytes = base64.b64decode(b64)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                out_path = f"/mnt/data/gen_{abs(hash(prompt))%999999}_{w}x{h}.png"
                img.save(out_path, format="PNG")
                return [out_path]
            except Exception as e:
                pass

        # Stubbed placeholder using PIL if no API key
        try:
            from PIL import Image, ImageDraw
            w, h = [int(x) for x in size.split("x")]
            img = Image.new("RGB", (w, h), color=(245, 246, 250))
            d = ImageDraw.Draw(img)
            txt = f"[stub image]\n{prompt[:120]}"
            d.multiline_text((24,24), txt, fill=(50,50,60), spacing=6)
            out_path = f"/mnt/data/stub_{abs(hash(prompt))%999999}_{w}x{h}.png"
            img.save(out_path, format="PNG")
            return [out_path]
        except Exception as e:
            return []
