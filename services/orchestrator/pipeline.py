"""
Async inter-service communication pipeline using httpx for the Bharat Orchestrator.
Dispatches requests to recommender, chatbot, and early_warning microservices.
"""

import asyncio
import logging
import os
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger("orchestrator.pipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

SERVICE_URLS = {
    "recommender": "http://recommender:8001",
    "chatbot": "http://chatbot:8002",
    "early_warning": "http://early_warning:8003",
}

SERVICE_URLS_LOCAL = {
    "recommender": os.getenv("RECOMMENDER_URL", "http://127.0.0.1:8001"),
    "chatbot": os.getenv("CHATBOT_URL", "http://127.0.0.1:8002"),
    "early_warning": os.getenv("EARLY_WARNING_URL", "http://127.0.0.1:8003"),
}


def get_service_urls() -> Dict[str, str]:
    """Retrieve service endpoints based on environment setting."""
    use_local_raw = os.getenv("USE_LOCAL_SERVICES", "true").strip().lower()
    use_local = use_local_raw in ("true", "1", "yes")
    return SERVICE_URLS_LOCAL if use_local else SERVICE_URLS


async def _post_with_retry(url: str, payload: Dict[str, Any], timeout: float = 10.0) -> Dict[str, Any]:
    """Execute HTTP POST with one connection retry."""
    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json()
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            if attempt == 0:
                await asyncio.sleep(0.5)
                continue
            logger.error(f"POST {url} failed: {exc}")
            return {"error": str(exc)}
    return {"error": "Request failed after retry"}


async def call_early_warning_check(customer_id: str) -> Dict[str, Any]:
    """Check customer for financial stress and anomalies."""
    urls = get_service_urls()
    url = f"{urls['early_warning']}/check/{customer_id}"
    res = await _post_with_retry(url, {})
    if "error" in res and "stress_level" not in res:
        return {"stress_level": "unknown", "risk_score": 0, "error": res["error"]}
    return res


async def call_early_warning_intervene(customer_id: str, language: str) -> Dict[str, Any]:
    """Fetch compassionate intervention message and relief pathways."""
    urls = get_service_urls()
    url = f"{urls['early_warning']}/intervene/{customer_id}"
    return await _post_with_retry(url, {"language": language})


async def call_recommender(customer_id: str, stress_level: str, language: str) -> Dict[str, Any]:
    """Request personalized product recommendations filtered by stress gate."""
    urls = get_service_urls()
    url = f"{urls['recommender']}/recommend/{customer_id}"
    return await _post_with_retry(url, {"stress_level": stress_level, "language": language})


async def call_chatbot_chat(
    session_id: str,
    message: str,
    language: str,
    customer_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Forward conversation turn to the vernacular chatbot service."""
    urls = get_service_urls()
    url = f"{urls['chatbot']}/chat"
    payload = {
        "session_id": session_id,
        "message": message,
        "language": language,
        "customer_id": customer_id,
    }
    return await _post_with_retry(url, payload)
