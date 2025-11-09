# utils/transcript.py
from typing import List
from dataclasses import dataclass, asdict
import json, time

@dataclass
class ChatMessage:
    role: str
    content: str

def now_str() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def export_transcript_json(messages: List[ChatMessage], subject:str):
    payload = {
        "subject": subject,
        "created_at": now_str(),
        "messages": [asdict(m) for m in messages],
    }
    content = json.dumps(payload, ensure_ascii=False, indent=2)
    fname = f"/mnt/data/transcript_{subject.replace(' ','_')}_{now_str()}.json"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(content)
    return content, fname

def export_transcript_md(messages: List[ChatMessage], subject:str):
    lines = [f"# Transcript — {subject}", f"_Exported: {now_str()}_", ""]
    for m in messages:
        prefix = "👩‍🎓" if m.role=="user" else "🧑‍🏫"
        lines.append(f"**{prefix} {m.role.capitalize()}:** {m.content}")
        lines.append("")
    content = "\n".join(lines)
    fname = f"/mnt/data/transcript_{subject.replace(' ','_')}_{now_str()}.md"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(content)
    return content, fname
