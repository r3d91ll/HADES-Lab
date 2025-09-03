"""
Simplified Compute Cost Assessment for Testing
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List

import structlog

try:
    from .config import ArxivDBConfig
    from .pg import get_connection
except Exception:
    from tools.arxiv.db.config import ArxivDBConfig  # type: ignore
    from tools.arxiv.db.pg import get_connection  # type: ignore

logger = structlog.get_logger()

@dataclass
class SimpleAssessment:
    """Minimal assessment for testing"""
    total_documents: int = 0
    documents_with_pdf: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_documents': self.total_documents,
            'documents_with_pdf': self.documents_with_pdf,
            'pdf_coverage': round(self.documents_with_pdf / max(1, self.total_documents) * 100, 2)
        }

def quick_assess(
    config: ArxivDBConfig,
    where_sql: str,
    params: List[Any]
) -> SimpleAssessment:
    """Quick assessment without complex queries"""
    
    assessment = SimpleAssessment()
    
    with get_connection(config.postgres) as conn:
        cur = conn.cursor()
        
        # Just get basic counts
        count_sql = f"""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN has_pdf THEN 1 ELSE 0 END) as with_pdf
            FROM papers {where_sql}
        """
        
        logger.info(f"Running count query: {count_sql[:100]}...")
        cur.execute(count_sql, params)
        total, with_pdf = cur.fetchone()
        
        assessment.total_documents = total or 0
        assessment.documents_with_pdf = with_pdf or 0
        
        cur.close()
    
    return assessment