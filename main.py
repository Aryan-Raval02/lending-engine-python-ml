from fastapi import FastAPI
from api.routes import router
import uvicorn

app = FastAPI(
    title="AI Risk Engine API",
    description="Microservice for B2B Loan Credit Scoring using ML",
    version="1.0.0"
)

# Include REST routes
app.include_router(router)

@app.get("/health")
async def health_check():
    return {"status": "up", "service": "ai-risk-engine"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
