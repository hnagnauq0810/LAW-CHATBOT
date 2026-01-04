import json
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
import openai
import config

@dataclass
class Plan:
    legal_question: str
    facts: List[str]
    keywords: List[str]
    filters: Optional[Dict] = None

class QueryPlanner:
    def __init__(self):
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY) if config.OPENAI_API_KEY else None

    def plan(self, query: str) -> Plan:
        if not self.client:
            return self._fallback_plan(query)

        prompt = f"""Phân tích câu hỏi pháp lý sau thành các thành phần để tìm kiếm:
Query: "{query}"

Trả về JSON format gồm:
- "legal_question": câu hỏi pháp lý cốt lõi (ngắn gọn, 1 câu).
- "facts": list[str], 3-8 sự kiện/dữ kiện pháp lý quan trọng (ngắn gọn).
- "keywords": list[str], 0-8 từ khóa quan trọng (tên luật, điều khoản nếu có, thuật ngữ chuyên ngành).
- "filters": dict, các bộ lọc nếu chắc chắn (ví dụ: "time": "2023", "doc_type": "Nghị định").

Output JSON only, no markdown."""

        try:
            response = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a legal query planner helper."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            
            return Plan(
                legal_question=data.get("legal_question", query),
                facts=data.get("facts", []),
                keywords=data.get("keywords", []),
                filters=data.get("filters", {})
            )
        except Exception as e:
            if config.DEBUG_RETRIEVAL:
                print(f"[QueryPlanner] Error: {e}. Using fallback.")
            return self._fallback_plan(query)

    def _fallback_plan(self, query: str) -> Plan:
        """Heuristic fallback when LLM fails."""
        # Simple heuristic:
        # legal_question = last sentence or full query
        sentences = re.split(r'[.?!]\s+', query)
        legal_question = sentences[-1] if sentences else query
        
        # keywords = words starting with uppercase or containing digits (heuristic for laws)
        tokens = query.split()
        keywords = [t for t in tokens if t[0].isupper() or any(c.isdigit() for c in t)]
        # Cap keywords
        keywords = keywords[:5]
        
        # facts = noun chunks? For now just split query into chunks of 5 words
        words = query.split()
        facts = []
        chunk_size = 5
        for i in range(0, len(words), chunk_size):
            facts.append(" ".join(words[i:i+chunk_size]))
            
        return Plan(
            legal_question=legal_question,
            facts=facts[:5],
            keywords=keywords,
            filters={}
        )
