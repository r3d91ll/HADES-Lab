#!/usr/bin/env python3
"""
Test client for HTTP-Socket bridge.

Simple client to test graph operations through the HTTP interface.
"""

import httpx
import asyncio
import json
from typing import Dict, Any


class GraphClient:
    """Client for graph operations via HTTP bridge."""
    
    def __init__(self, base_url: str = "http://localhost:8888"):
        """Initialize client."""
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=300.0)
    
    async def status(self) -> Dict[str, Any]:
        """Check server status."""
        response = await self.client.get(f"{self.base_url}/")
        return response.json()
    
    async def graph_status(self) -> Dict[str, Any]:
        """Get graph statistics."""
        response = await self.client.get(f"{self.base_url}/graph/status")
        return response.json()
    
    async def build_graph(
        self,
        workers: int = 36,
        skip_categories: bool = False,
        threshold: float = 0.65
    ) -> Dict[str, Any]:
        """Build complete graph."""
        response = await self.client.post(
            f"{self.base_url}/graph/build",
            params={
                "workers": workers,
                "skip_categories": skip_categories,
                "threshold": threshold
            }
        )
        return response.json()
    
    async def export_graph(
        self,
        name: str = "arxiv_graph",
        include_features: bool = True
    ) -> Dict[str, Any]:
        """Export graph."""
        response = await self.client.post(
            f"{self.base_url}/graph/export",
            params={
                "name": name,
                "include_features": include_features
            }
        )
        return response.json()
    
    async def extract_keywords(
        self,
        limit: int = None,
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """Extract keywords."""
        params = {"batch_size": batch_size}
        if limit:
            params["limit"] = limit
        
        response = await self.client.post(
            f"{self.base_url}/graph/keywords/extract",
            params=params
        )
        return response.json()
    
    async def build_temporal(self, workers: int = 36) -> Dict[str, Any]:
        """Build temporal edges."""
        response = await self.client.post(
            f"{self.base_url}/graph/edges/temporal",
            params={"workers": workers}
        )
        return response.json()
    
    async def build_categories(self) -> Dict[str, Any]:
        """Build category edges."""
        response = await self.client.post(f"{self.base_url}/graph/edges/category")
        return response.json()
    
    async def build_keywords(
        self,
        threshold: float = 0.65,
        top_k: int = 100
    ) -> Dict[str, Any]:
        """Build keyword edges."""
        response = await self.client.post(
            f"{self.base_url}/graph/edges/keyword",
            params={
                "threshold": threshold,
                "top_k": top_k
            }
        )
        return response.json()
    
    async def analyze_interdisciplinary(
        self,
        min_year_gap: int = 5,
        min_similarity: float = 0.7
    ) -> Dict[str, Any]:
        """Analyze interdisciplinary connections."""
        response = await self.client.post(
            f"{self.base_url}/graph/analyze/interdisciplinary",
            params={
                "min_year_gap": min_year_gap,
                "min_similarity": min_similarity
            }
        )
        return response.json()
    
    async def close(self):
        """Close client."""
        await self.client.aclose()


async def main():
    """Test the graph operations."""
    client = GraphClient()
    
    try:
        # Check status
        print("Checking server status...")
        status = await client.status()
        print(json.dumps(status, indent=2))
        
        # Get graph stats
        print("\nGetting graph statistics...")
        stats = await client.graph_status()
        print(json.dumps(stats, indent=2))
        
        # Example: Extract keywords for 100 papers
        print("\nExtracting keywords for 100 papers...")
        result = await client.extract_keywords(limit=100)
        print(json.dumps(result, indent=2))
        
        # Example: Build temporal edges
        print("\nBuilding temporal edges...")
        result = await client.build_temporal(workers=48)
        print(json.dumps(result, indent=2))
        
        # Example: Analyze interdisciplinary connections
        print("\nAnalyzing interdisciplinary connections...")
        result = await client.analyze_interdisciplinary()
        print(json.dumps(result, indent=2))
        
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())