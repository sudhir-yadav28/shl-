# SHL Conversational Assessment Recommender

A stateless FastAPI service that recommends SHL assessments through dialogue.

## Endpoints

- `GET /health` → `{"status": "ok"}`
- `POST /chat` → conversational recommendation. Request/response schema in `app/schemas.py`.

## Local run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then fill in your keys
uvicorn app.main:app --reload
```

## Deploy

Render free tier — Docker, blueprint in `render.yaml`. Set `GEMINI_API_KEY` and `GROQ_API_KEY` in the Render dashboard.

See `PLAN.md` for architecture and design notes.
