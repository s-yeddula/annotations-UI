import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from collections import defaultdict
from typing import Any, Dict, Optional
from pydantic import BaseModel
import pandas as pd
import numpy as np
import phoenix as px

class QueryRequest(BaseModel):
    apiKey: str
    endpoint: str
    projectName: str | None = "default"
    
class AnnotationPayload(BaseModel):
    span_id: str
    name: str  # annotation name like "correctness"
    result: Dict[str, Any]  # { "label": str, "score": float, "explanation": Optional[str] }
    identifier: Optional[str] = None  # optional unique identifier 
    

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
        
   
    def replace_missing_with_zero_robust(x):
    if isinstance(x, (list, np.ndarray)):
        return 0 if len(x) == 0 else x
    elif pd.isna(x):
        return 0
    else:
        return x

    df = df.apply(lambda col: col.map(replace_missing_with_zero_robust))

    print("SPANS BELOW:")
    print(df)

    
    records = df.to_dict(orient="records")

    grouped = defaultdict(list)
    for rec in records:
        trace_id = rec.get("context.trace_id")
        grouped[trace_id].append(rec)

    traces = [
        {"trace_id": trace_id, "spans": spans}
        for trace_id, spans in grouped.items()
    ]
    print("MADE IT HERE")
    return_val = {"traces": traces}
    try:
        json.dumps(return_val)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("❌ Your return value is not JSON serializable. Inspect below:")
        import pprint
        pprint.pprint(return_val)
        raise
    
@app.post("/annotate-span", response_model=Any)
async def annotate_span(payload: AnnotationPayload, req: QueryRequest):
    os.environ["PHOENIX_API_KEY"] = req.apiKey
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = req.endpoint.rstrip("/")
    from phoenix.client import Client

    client = Client()
    ann = client.annotations.add_span_annotation(
        annotation_name=payload.name,
        annotator_kind= "HUMAN",
        span_id=payload.span_id,
        label=payload.result.get("label"),
        score=payload.result.get("score"),
        explanation=payload.result.get("explanation"),
        metadata={},
        identifier=payload.identifier, 
    )
