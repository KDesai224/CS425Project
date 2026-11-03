"""Web app: upload a contract, ask questions, get plain-English answers with their evidence.

    uvicorn app:app --host 0.0.0.0 --port 8000     (docker compose up does this)

The models load in a background thread, so the page opens at once and shows when
they are ready. Contracts can be uploaded meanwhile; reading them needs no GPU.

JSON API, for another frontend:
    GET    /api/health                      {"ready", "error", "gemma"}
    GET    /api/topics                      the 10 categories and their display names
    POST   /api/documents                   multipart "file" -> contract summary with "doc_id"
    POST   /api/documents/{doc_id}/ask      {"question": "..."} or {"category": "<CUAD category>"} -> answer
    DELETE /api/documents/{doc_id}
"""
import os
import threading
import traceback
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from jargonaut.config import CATEGORIES, TOPIC_LABELS
from jargonaut.ingest import IngestError
from jargonaut.pipeline import Conversation, Jargonaut, open_contract

STATIC = Path(__file__).resolve().parent / "static"
MAX_CONTRACTS = 20
STATE = {"system": None, "error": None, "status": "starting"}
CONTRACTS = OrderedDict()      # doc_id -> (Contract, Conversation | None), oldest first
LOCK = threading.Lock()


def load_models():
    try:
        STATE["system"] = Jargonaut(load_simplifier=os.environ.get("LOAD_GEMMA", "1") != "0",
                                    log=lambda msg: (STATE.update(status=msg), print(msg)))
        STATE["status"] = "ready"
    except Exception as e:   # shown in the UI; the server keeps running
        traceback.print_exc()
        STATE["error"], STATE["status"] = f"{type(e).__name__}: {e}", "failed"


@asynccontextmanager
async def lifespan(_):
    threading.Thread(target=load_models, daemon=True).start()
    yield


app = FastAPI(title="Jargonaut", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class AskRequest(BaseModel):
    question: str | None = None
    category: str | None = None


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    system = STATE["system"]
    return {"ready": system is not None, "status": STATE["status"], "error": STATE["error"],
            "gemma": bool(system and system.simplifier),
            "calibrated": bool(system and system.extractor.calibration)}


@app.get("/api/topics")
def topics():
    return [{"category": c, "topic": TOPIC_LABELS[c]} for c in CATEGORIES]


@app.post("/api/documents")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    try:
        contract = open_contract(data, file.filename or "contract")
    except IngestError as e:
        raise HTTPException(422, str(e))
    with LOCK:
        CONTRACTS[contract.doc_id] = (contract, None)
        while len(CONTRACTS) > MAX_CONTRACTS:
            CONTRACTS.popitem(last=False)
    summary = contract.summary()
    summary["outline"] = [{"level": h.level, "text": h.text, "page": contract.doc.page_at(h.start)}
                          for h in contract.doc.headings[:300]]
    return summary


@app.post("/api/documents/{doc_id}/ask")
def ask(doc_id: str, req: AskRequest):
    system = STATE["system"]
    if system is None:
        raise HTTPException(503, STATE["error"] or "The models are still loading. Try again in a minute.")
    with LOCK:
        if doc_id not in CONTRACTS:
            raise HTTPException(404, "Unknown contract. Upload it again.")
        contract, conversation = CONTRACTS[doc_id]
        if conversation is None:
            conversation = Conversation(system, contract)
            CONTRACTS[doc_id] = (contract, conversation)
    if req.category:
        if req.category not in CATEGORIES:
            raise HTTPException(422, f"Unknown category: {req.category}")
        return conversation.reply_category(req.category).to_dict()
    if not (req.question or "").strip():
        raise HTTPException(422, "Empty question.")
    return conversation.reply(req.question[:1000]).to_dict()


@app.delete("/api/documents/{doc_id}")
def forget(doc_id: str):
    with LOCK:
        CONTRACTS.pop(doc_id, None)
    return {"deleted": doc_id}
