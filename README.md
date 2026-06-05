# Smart Customer Support Ticket Classification System

Production-style machine learning backend for classifying customer support tickets by category and priority.

## Current Phase

Phase 2: Environment Setup

## Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```
