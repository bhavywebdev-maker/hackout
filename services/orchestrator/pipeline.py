"""
Async inter-service communication pipeline using httpx for the Bharat Orchestrator.
Dispatches requests to recommender, chatbot, and early_warning microservices.
Supports both HTTP network communication and in-memory ASGI transport (for Vercel / serverless deployments).
"""

import asyncio
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import httpx

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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

_ASGI_APPS: Dict[str, Any] = {}


def get_service_urls() -> Dict[str, str]:
    """Retrieve service endpoints based on environment setting."""
    use_local_raw = os.getenv("USE_LOCAL_SERVICES", "true").strip().lower()
    use_local = use_local_raw in ("true", "1", "yes")
    return SERVICE_URLS_LOCAL if use_local else SERVICE_URLS


def _get_asgi_app(service_name: str) -> Optional[Any]:
    """Lazily load the ASGI app for in-memory dispatch in serverless environments."""
    if service_name not in _ASGI_APPS:
        try:
            if service_name == "recommender":
                from services.recommender.main import app as rec_app
                _ASGI_APPS[service_name] = rec_app
            elif service_name == "early_warning":
                from services.early_warning.main import app as ew_app
                _ASGI_APPS[service_name] = ew_app
            elif service_name == "chatbot":
                from services.chatbot.main import app as chat_app
                _ASGI_APPS[service_name] = chat_app
        except Exception as exc:
            logger.error(f"Error importing ASGI app for {service_name}: {exc}")
            return None
    return _ASGI_APPS.get(service_name)


async def _dispatch_post(service_name: str, path: str, payload: Dict[str, Any], timeout: float = 12.0) -> Dict[str, Any]:
    """
    Execute POST request to a microservice.
    Tries HTTP network URL first (if not in Vercel/serverless mode).
    If network is unavailable or running in serverless, dispatches in-memory via ASGITransport.
    """
    is_serverless = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))
    urls = get_service_urls()
    base_url = urls.get(service_name, "")
    full_url = f"{base_url}{path}"

    if not is_serverless:
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(full_url, json=payload)
                    resp.raise_for_status()
                    return resp.json()
            except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                if attempt == 0:
                    await asyncio.sleep(0.3)
                    continue
                logger.warning(f"Network call to {full_url} failed ({exc}). Falling back to in-memory ASGI dispatch.")

    # In-memory ASGI dispatch
    asgi_app = _get_asgi_app(service_name)
    if asgi_app is not None:
        try:
            transport = httpx.ASGITransport(app=asgi_app)
            async with httpx.AsyncClient(transport=transport, base_url="http://internal") as client:
                resp = await client.post(path, json=payload, timeout=timeout)
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            logger.error(f"In-memory ASGI call to {service_name}{path} failed: {exc}")
            return {"error": str(exc)}

    return {"error": f"Failed to dispatch to {service_name}"}


async def call_early_warning_check(customer_id: str) -> Dict[str, Any]:
    """Check customer for financial stress and anomalies."""
    res = await _dispatch_post("early_warning", f"/check/{customer_id}", {})
    if "error" in res and "stress_level" not in res:
        return {"stress_level": "unknown", "risk_score": 0, "error": res["error"]}
    return res


async def call_early_warning_intervene(customer_id: str, language: str) -> Dict[str, Any]:
    """Fetch compassionate intervention message and relief pathways."""
    return await _dispatch_post("early_warning", f"/intervene/{customer_id}", {"language": language})


async def call_recommender(customer_id: str, stress_level: str, language: str) -> Dict[str, Any]:
    """Request personalized product recommendations filtered by stress gate."""
    return await _dispatch_post("recommender", f"/recommend/{customer_id}", {"stress_level": stress_level, "language": language})


async def call_chatbot_chat(
    session_id: str,
    message: str,
    language: str,
    customer_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Forward conversation turn to the vernacular chatbot service."""
    payload = {
        "session_id": session_id,
        "message": message,
        "language": language,
        "customer_id": customer_id,
    }
    return await _dispatch_post("chatbot", "/chat", payload)
