#!/usr/bin/env python3
"""
Unix Socket Server for Graph Operations.

Receives requests via Unix socket and executes graph operations
using the existing optimized graph builder infrastructure.
"""

import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
import aiofiles
from aiohttp import web
from aiohttp.web import Request, Response
import socket

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from graph.builders.build_graph_optimized import OptimizedGraphBuilder
from graph.graph_manager import GraphManager
from graph.utils.interdisciplinary_analysis import analyze_interdisciplinary_connections

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UNIX_SOCKET_PATH = "/tmp/arango_graph.sock"


class GraphSocketServer:
    """
    Unix socket server for graph operations.
    
    Provides a simple interface to all graph operations
    through a Unix socket for maximum performance.
    """
    
    def __init__(self, socket_path: str = UNIX_SOCKET_PATH):
        """Initialize socket server."""
        self.socket_path = Path(socket_path)
        self.app = web.Application()
        self.builder = None
        self.manager = None
        
        # Setup routes
        self.setup_routes()
        
        # Ensure socket directory exists
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove old socket if exists
        if self.socket_path.exists():
            self.socket_path.unlink()
    
    def setup_routes(self):
        """Setup request routes."""
        self.app.router.add_post('/process', self.handle_request)
    
    async def handle_request(self, request: Request) -> Response:
        """
        Main request handler - routes to appropriate operation.
        
        Request format:
        {
            "operation": "build|export|status|...",
            "params": {...}
        }
        """
        try:
            data = await request.json()
            operation = data.get('operation')
            params = data.get('params', {})
            
            logger.info(f"Received operation: {operation}")
            
            # Route to appropriate handler
            if operation == 'build':
                result = await self.build_graph(**params)
            elif operation == 'export':
                result = await self.export_graph(**params)
            elif operation == 'status':
                result = await self.get_status()
            elif operation == 'extract_keywords':
                result = await self.extract_keywords(**params)
            elif operation == 'build_temporal':
                result = await self.build_temporal_edges(**params)
            elif operation == 'build_category':
                result = await self.build_category_edges(**params)
            elif operation == 'build_keywords':
                result = await self.build_keyword_edges(**params)
            elif operation == 'analyze_interdisciplinary':
                result = await self.analyze_interdisciplinary(**params)
            else:
                return web.json_response(
                    {'error': f'Unknown operation: {operation}'},
                    status=400
                )
            
            return web.json_response(result)
            
        except Exception as e:
            logger.error(f"Request handling error: {e}")
            return web.json_response(
                {'error': str(e)},
                status=500
            )
    
    async def build_graph(
        self,
        workers: int = 36,
        skip_categories: bool = False,
        threshold: float = 0.65
    ) -> Dict[str, Any]:
        """Build complete graph."""
        try:
            # Initialize builder if needed
            if not self.builder:
                self.builder = OptimizedGraphBuilder(workers=workers)
            
            # Run build in executor to not block
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._run_build,
                workers,
                skip_categories,
                threshold
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Build error: {e}")
            return {'error': str(e)}
    
    def _run_build(
        self,
        workers: int,
        skip_categories: bool,
        threshold: float
    ) -> Dict[str, Any]:
        """Run build synchronously."""
        builder = OptimizedGraphBuilder(workers=workers)
        
        # Build phases
        if not skip_categories:
            builder.build_category_edges()
        
        builder.build_temporal_edges_arxiv_id()
        builder.build_keyword_edges_faiss(threshold=threshold)
        
        stats = builder.get_stats()
        return {
            'status': 'complete',
            'stats': stats
        }
    
    async def export_graph(
        self,
        name: str = "arxiv_graph",
        include_features: bool = True
    ) -> Dict[str, Any]:
        """Export graph."""
        try:
            if not self.manager:
                self.manager = GraphManager()
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.manager.export_graph,
                name,
                include_features
            )
            
            return {
                'status': 'exported',
                'path': str(result)
            }
            
        except Exception as e:
            logger.error(f"Export error: {e}")
            return {'error': str(e)}
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current graph statistics."""
        try:
            if not self.builder:
                self.builder = OptimizedGraphBuilder()
            
            stats = self.builder.get_stats()
            return {
                'status': 'ok',
                'stats': stats
            }
            
        except Exception as e:
            logger.error(f"Status error: {e}")
            return {'error': str(e)}
    
    async def extract_keywords(
        self,
        limit: Optional[int] = None,
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """Extract keywords for papers."""
        try:
            # This would use the keyword extractor
            # For now, return placeholder
            return {
                'status': 'keywords_extracted',
                'papers_processed': limit or 'all',
                'batch_size': batch_size
            }
            
        except Exception as e:
            logger.error(f"Keyword extraction error: {e}")
            return {'error': str(e)}
    
    async def build_temporal_edges(
        self,
        workers: int = 36,
        use_arxiv_id: bool = True
    ) -> Dict[str, Any]:
        """Build temporal edges only."""
        try:
            if not self.builder:
                self.builder = OptimizedGraphBuilder(workers=workers)
            
            loop = asyncio.get_event_loop()
            
            if use_arxiv_id:
                await loop.run_in_executor(
                    None,
                    self.builder.build_temporal_edges_arxiv_id
                )
            else:
                # Fall back to date-based method if needed
                await loop.run_in_executor(
                    None,
                    self.builder.build_temporal_edges
                )
            
            stats = self.builder.get_stats()
            return {
                'status': 'temporal_edges_built',
                'stats': stats
            }
            
        except Exception as e:
            logger.error(f"Temporal edge error: {e}")
            return {'error': str(e)}
    
    async def build_category_edges(self) -> Dict[str, Any]:
        """Build category edges only."""
        try:
            if not self.builder:
                self.builder = OptimizedGraphBuilder()
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.builder.build_category_edges
            )
            
            stats = self.builder.get_stats()
            return {
                'status': 'category_edges_built',
                'stats': stats
            }
            
        except Exception as e:
            logger.error(f"Category edge error: {e}")
            return {'error': str(e)}
    
    async def build_keyword_edges(
        self,
        threshold: float = 0.65,
        top_k: int = 100
    ) -> Dict[str, Any]:
        """Build keyword edges only."""
        try:
            if not self.builder:
                self.builder = OptimizedGraphBuilder()
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.builder.build_keyword_edges_faiss,
                threshold
            )
            
            stats = self.builder.get_stats()
            return {
                'status': 'keyword_edges_built',
                'stats': stats,
                'threshold': threshold,
                'top_k': top_k
            }
            
        except Exception as e:
            logger.error(f"Keyword edge error: {e}")
            return {'error': str(e)}
    
    async def analyze_interdisciplinary(
        self,
        min_year_gap: int = 5,
        min_similarity: float = 0.7
    ) -> Dict[str, Any]:
        """Analyze interdisciplinary connections."""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                analyze_interdisciplinary_connections,
                min_year_gap,
                min_similarity
            )
            
            return {
                'status': 'analysis_complete',
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {'error': str(e)}
    
    async def start(self):
        """Start the socket server."""
        logger.info(f"Starting Unix socket server at {self.socket_path}")
        
        # Create runner
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        # Create Unix socket site
        site = web.UnixSite(runner, str(self.socket_path))
        await site.start()
        
        # Set socket permissions for access
        os.chmod(str(self.socket_path), 0o666)
        
        logger.info(f"Socket server listening at {self.socket_path}")
        logger.info("Ready to receive requests...")
        
        # Keep running
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            logger.info("Shutting down socket server...")
        finally:
            await runner.cleanup()
            if self.socket_path.exists():
                self.socket_path.unlink()


async def main():
    """Main entry point."""
    server = GraphSocketServer()
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())