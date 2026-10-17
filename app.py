import uvicorn
from fastapi import FastAPI
from routes import router

app = FastAPI(title="On-Premise Agentic RAG API")

# Include API routes
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)