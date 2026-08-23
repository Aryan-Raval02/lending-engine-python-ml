from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from ml.risk_engine import evaluate_risk

router = APIRouter(prefix="/api/v1")

class RiskAssessmentRequest(BaseModel):
    application_id: str
    tenant_id: str
    requested_amount: float = Field(..., gt=0)
    term_months: int = Field(..., gt=0)
    annual_revenue: float = Field(..., ge=0)
    existing_debt: float = Field(..., ge=0)
    loan_purpose: str
    business_description: str

class RiskMetrics(BaseModel):
    dscr: float
    debt_to_revenue_ratio: float
    loan_to_revenue_ratio: float

class RiskAssessmentResponse(BaseModel):
    application_id: str
    score: float
    category: str
    reasons: List[str]
    metrics: RiskMetrics
    feature_importances: Optional[Dict[str, float]] = None

@router.post("/risk-assess", response_model=RiskAssessmentResponse)
async def assess_risk(request: RiskAssessmentRequest):
    """
    Evaluates a loan application and returns a risk score (0-100) and category.
    Higher score means lower risk (safer).
    """
    try:
        # Pass data to the ML engine
        result = evaluate_risk(request.model_dump())
        
        return RiskAssessmentResponse(
            application_id=request.application_id,
            score=result["score"],
            category=result["category"],
            reasons=result["reasons"],
            metrics=RiskMetrics(**result["metrics"]),
            feature_importances=result.get("feature_importances")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk engine failure: {str(e)}")
