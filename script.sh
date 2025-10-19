python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# (опционально) для LLM-анализа:
export OPENAI_API_KEY=sk-...                           # Windows: set OPENAI_API_KEY=...
# можно и OpenAI-совместимую локальную точку:
# export OPENAI_BASE_URL=http://localhost:8000/v1

python reels_cutter.py
