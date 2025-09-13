#!/usr/bin/env python3
"""
HTTP-to-Unix-Socket Bridge for ArangoDB Graph Operations.

Simple interface that receives HTTP requests and forwards them to the Unix socket,
acting as a transparent bridge to the existing graph infrastructure.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UNIX_SOCKET_PATH = "/tmp/arango_graph.sock"
HTTP_PORT = 8888
HTTP_HOST = "0.0.0.0"

app = FastAPI(title="Graph Socket Bridge", version="1.0.0")


class GraphRequest(BaseModel):
    """Standard graph operation request."""
    operation: str  # build, export, status, etc.
    params: Dict[str, Any] = {}


class GraphResponse(BaseModel):
    """Standard graph operation response."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


async def send_to_socket(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send request to Unix socket and get response.
    
    This is the core bridge functionality - takes HTTP request,
    forwards to socket, returns response.
    """
    socket_path = Path(UNIX_SOCKET_PATH)
    
    if not socket_path.exists():
        raise HTTPException(status_code=503, detail=f"Socket not available at {UNIX_SOCKET_PATH}")
    
    try:
        # Create async HTTP client with Unix socket transport
        async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=str(socket_path))) as client:
            # Forward request to socket
            response = await client.post(
                "http://localhost/process",  # URL doesn't matter for Unix socket
                json=request,
                timeout=300.0  # 5 minute timeout for long operations
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=response.status_code, detail=response.text)
                
    except httpx.ConnectError as e:
        logger.error(f"Failed to connect to socket: {e}")
        raise HTTPException(status_code=503, detail="Socket connection failed")
    except Exception as e:
        logger.error(f"Socket communication error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "Graph HTTP-Socket Bridge",
        "socket": str(UNIX_SOCKET_PATH),
        "socket_exists": Path(UNIX_SOCKET_PATH).exists()
    }


@app.post("/graph/build")
async def build_graph(
    background_tasks: BackgroundTasks,
    workers: int = 36,
    skip_categories: bool = False,
    threshold: float = 0.65
):
    """
    Trigger graph building through socket.
    
    Args:
        workers: Number of parallel workers
        skip_categories: Skip category edge building
        threshold: Keyword similarity threshold
    """
    request = {
        "operation": "build",
        "params": {
            "workers": workers,
            "skip_categories": skip_categories,
            "threshold": threshold
        }
    }
    
    result = await send_to_socket(request)
    return GraphResponse(success=True, data=result)


@app.post("/graph/export")
async def export_graph(
    name: str = "arxiv_graph",
    include_features: bool = True
):
    """
    Export graph through socket.
    
    Args:
        name: Export name
        include_features: Include node features in export
    """
    request = {
        "operation": "export",
        "params": {
            "name": name,
            "include_features": include_features
        }
    }
    
    result = await send_to_socket(request)
    return GraphResponse(success=True, data=result)


@app.get("/graph/status")
async def graph_status():
    """Get current graph statistics through socket."""
    request = {
        "operation": "status",
        "params": {}
    }
    
    result = await send_to_socket(request)
    return GraphResponse(success=True, data=result)


@app.post("/graph/keywords/extract")
async def extract_keywords(
    limit: Optional[int] = None,
    batch_size: int = 1000
):
    """
    Extract keywords for papers without them.
    
    Args:
        limit: Max papers to process (None for all)
        batch_size: Batch size for processing
    """
    request = {
        "operation": "extract_keywords",
        "params": {
            "limit": limit,
            "batch_size": batch_size
        }
    }
    
    result = await send_to_socket(request)
    return GraphResponse(success=True, data=result)


@app.post("/graph/edges/temporal")
async def build_temporal_edges(
    workers: int = 36,
    use_arxiv_id: bool = True
):
    """
    Build temporal proximity edges.
    
    Args:
        workers: Number of parallel workers
        use_arxiv_id: Use optimized ArXiv ID method
    """
    request = {
        "operation": "build_temporal",
        "params": {
            "workers": workers,
            "use_arxiv_id": use_arxiv_id
        }
    }
    
    result = await send_to_socket(request)
    return GraphResponse(success=True, data=result)


@app.post("/graph/edges/category")
async def build_category_edges():
    """Build same-field category edges."""
    request = {
        "operation": "build_category",
        "params": {}
    }
    
    result = await send_to_socket(request)
    return GraphResponse(success=True, data=result)


@app.post("/graph/edges/keyword")
async def build_keyword_edges(
    threshold: float = 0.65,
    top_k: int = 100
):
    """
    Build keyword similarity edges.
    
    Args:
        threshold: Similarity threshold
        top_k: Top K similar papers per paper
    """
    request = {
        "operation": "build_keywords",
        "params": {
            "threshold": threshold,
            "top_k": top_k
        }
    }
    
    result = await send_to_socket(request)
    return GraphResponse(success=True, data=result)


@app.post("/graph/analyze/interdisciplinary")
async def analyze_interdisciplinary(
    min_year_gap: int = 5,
    min_similarity: float = 0.7
):
    """
    Analyze interdisciplinary connections.
    
    Args:
        min_year_gap: Minimum year gap for temporal bridges
        min_similarity: Minimum similarity for keyword bridges
    """
    request = {
        "operation": "analyze_interdisciplinary",
        "params": {
            "min_year_gap": min_year_gap,
            "min_similarity": min_similarity
        }
    }
    
    result = await send_to_socket(request)
    return GraphResponse(success=True, data=result)


@app.post("/graph/custom")
async def custom_operation(request: GraphRequest):
    """
    Send custom operation to socket.
    
    For operations not covered by specific endpoints.
    """
    socket_request = {
        "operation": request.operation,
        "params": request.params
    }
    
    result = await send_to_socket(socket_request)
    return GraphResponse(success=True, data=result)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error"
        }
    )


def main():
    """Run the HTTP-Socket bridge server."""
    logger.info(f"Starting HTTP-Socket Bridge")
    logger.info(f"HTTP Server: http://{HTTP_HOST}:{HTTP_PORT}")
    logger.info(f"Unix Socket: {UNIX_SOCKET_PATH}")
    
    # Check if socket exists
    if not Path(UNIX_SOCKET_PATH).exists():
        logger.warning(f"Socket does not exist at {UNIX_SOCKET_PATH}")
        logger.warning("Make sure the socket server is running first!")
    
    # Run server
    uvicorn.run(
        app,
        host=HTTP_HOST,
        port=HTTP_PORT,
        log_level="info"
    )


if __name__ == "__main__":
    main()