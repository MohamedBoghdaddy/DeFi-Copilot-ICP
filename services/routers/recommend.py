# routers/recommend.py

from fastapi import APIRouter, Query
from services.gemini_api import get_investment_advice

router = APIRouter()

@router.get("/generate/investment")
async def generate_investment_advice(
    goal: str = Query(..., description="User financial goal, e.g., 'retirement', 'buy a house'"),
    income: float = Query(..., description="User monthly income"),
    age: int = Query(..., description="User's current age"),
    risk_tolerance: str = Query("medium", description="Risk tolerance: low, medium, or high"),
    lifestyle: str = Query("moderate", description="Spending lifestyle: frugal, moderate, or lavish")
):
    """
    Generate personalized investment advice using Gemini AI model.
    """
    try:
        response = get_investment_advice(
            goal=goal,
            income=income,
            age=age,
            risk_tolerance=risk_tolerance,
            lifestyle=lifestyle
        )
        return {"success": True, "advice": response}
    except Exception as e:
        return {"success": False, "error": str(e)}
