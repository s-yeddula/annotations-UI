import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import phoenix as px

class QueryRequest(BaseModel):
    apiKey: str
    endpoint: str
    projectName: str | None = "default"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Lovable domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def setup_defaults():
    pass
    # no-op, environment is set dynamically per-request

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Backend is running"}

@app.post("/fetch-spans")
async def fetch_spans(req: QueryRequest):
    os.environ["PHOENIX_API_KEY"] = req.apiKey
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = req.endpoint.rstrip("/")

    try:
        df = px.Client().get_spans_dataframe(project_name=req.projectName)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Phoenix fetch error: {e}")

    if df is None:
        return {"spans": []}

    spans = df.reset_index().to_dict(orient="records")
    return {"spans": spans}
