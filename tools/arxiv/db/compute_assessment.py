"""
Compute Cost Assessment Module for ArXiv Metadata Service

Following Actor-Network Theory, this module acts as a translation point between
human search intentions and computational resource requirements. It transforms
abstract queries into concrete resource predictions, helping orchestrators make
informed decisions about resource commitment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import structlog

# Support running as script or module
try:
    from .config import ArxivDBConfig, load_config
    from .pg import get_connection
except Exception:
    from tools.arxiv.db.config import ArxivDBConfig, load_config  # type: ignore
    from tools.arxiv.db.pg import get_connection  # type: ignore

logger = structlog.get_logger()


@dataclass
class DocumentSizeDistribution:
    """
    Represents document size distribution for resource planning.
    
    In ANT terms, this is a boundary object that translates between
    filesystem reality (bytes) and computational requirements (processing time).
    """
    small: int = 0       # < 1MB
    medium: int = 0      # 1-5MB  
    large: int = 0       # 5-20MB
    x_large: int = 0     # > 20MB
    
    total_bytes: int = 0
    avg_bytes: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'small_count': self.small,
            'medium_count': self.medium,
            'large_count': self.large,
            'x_large_count': self.x_large,
            'total_bytes': self.total_bytes,
            'avg_bytes': self.avg_bytes,
            'human_avg': self._human_size(self.avg_bytes),
            'human_total': self._human_size(self.total_bytes)
        }
    
    @staticmethod
    def _human_size(bytes: float) -> str:
        """Convert bytes to human-readable format."""
        bytes = float(bytes)  # Convert Decimal to float
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes < 1024.0:
                return f"{bytes:.2f}{unit}"
            bytes /= 1024.0
        return f"{bytes:.2f}PB"


@dataclass
class ProcessingTimeEstimate:
    """
    Estimates processing time based on empirical measurements.
    
    From an anthropological perspective, this represents the temporal
    bureaucracy of computation - how long humans must wait for machines
    to transform information.
    """
    extraction_minutes: float = 0.0
    embedding_minutes: float = 0.0
    total_minutes: float = 0.0
    
    # Empirical rates from ACID pipeline
    EXTRACTION_RATE = 36.0  # papers/minute with 8 workers
    EMBEDDING_RATE = 8.0    # papers/minute with 8 GPU workers
    
    @classmethod
    def from_document_count(cls, count: int, has_gpu: bool = True) -> ProcessingTimeEstimate:
        """Calculate time estimates based on document count."""
        extraction_time = count / cls.EXTRACTION_RATE
        embedding_time = count / cls.EMBEDDING_RATE if has_gpu else count / 1.0  # 1 paper/min CPU
        
        return cls(
            extraction_minutes=extraction_time,
            embedding_minutes=embedding_time,
            total_minutes=extraction_time + embedding_time
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'extraction_minutes': round(self.extraction_minutes, 2),
            'embedding_minutes': round(self.embedding_minutes, 2),
            'total_minutes': round(self.total_minutes, 2),
            'extraction_hours': round(self.extraction_minutes / 60, 2),
            'embedding_hours': round(self.embedding_minutes / 60, 2),
            'total_hours': round(self.total_minutes / 60, 2)
        }


@dataclass  
class ResourceRequirements:
    """
    Estimated resource requirements for processing.
    
    This class represents the material requirements of information transformation -
    the physical substrate needed to convert PDFs into actionable knowledge.
    """
    storage_gb: float = 0.0
    ram_gb: float = 0.0
    gpu_hours: float = 0.0
    cpu_hours: float = 0.0
    
    # Resource multipliers based on processing type
    STORAGE_MULTIPLIER = 3.0  # Original + staged + embedded
    RAM_PER_WORKER = 2.0      # GB per worker
    GPU_RAM_PER_WORKER = 8.0  # GB VRAM per GPU worker
    
    @classmethod
    def from_assessment(
        cls,
        doc_count: int,
        total_bytes: int,
        time_estimate: ProcessingTimeEstimate,
        num_workers: int = 8
    ) -> ResourceRequirements:
        """Calculate resource requirements from assessment data."""
        storage_gb = (float(total_bytes) * cls.STORAGE_MULTIPLIER) / (1024**3)
        ram_gb = num_workers * cls.RAM_PER_WORKER
        gpu_hours = time_estimate.embedding_minutes / 60
        cpu_hours = time_estimate.extraction_minutes / 60
        
        return cls(
            storage_gb=storage_gb,
            ram_gb=ram_gb,
            gpu_hours=gpu_hours,
            cpu_hours=cpu_hours
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'storage_gb': round(self.storage_gb, 2),
            'ram_gb': round(self.ram_gb, 2),
            'gpu_hours': round(self.gpu_hours, 2),
            'cpu_hours': round(self.cpu_hours, 2),
            'estimated_cost_usd': self._estimate_cost()
        }
    
    def _estimate_cost(self) -> float:
        """Rough cost estimate based on cloud pricing."""
        # Approximate cloud costs (AWS/GCP pricing)
        storage_cost = self.storage_gb * 0.10  # $0.10/GB/month
        gpu_cost = self.gpu_hours * 2.00       # ~$2/hour for T4 GPU
        cpu_cost = self.cpu_hours * 0.10       # ~$0.10/hour compute
        return round(storage_cost + gpu_cost + cpu_cost, 2)


@dataclass
class ComputeCostAssessment:
    """
    Complete compute cost assessment for a search query.
    
    This is the final translation product - converting a search configuration
    into a comprehensive assessment of computational requirements. It serves as
    an obligatory passage point between intention and execution.
    """
    total_documents: int = 0
    documents_with_pdf: int = 0
    documents_with_latex: int = 0
    
    size_distribution: DocumentSizeDistribution = field(default_factory=DocumentSizeDistribution)
    time_estimate: ProcessingTimeEstimate = field(default_factory=ProcessingTimeEstimate)
    resource_requirements: ResourceRequirements = field(default_factory=ResourceRequirements)
    
    # Statistical breakdowns
    by_year: Dict[int, int] = field(default_factory=dict)
    by_month: Dict[Tuple[int, int], int] = field(default_factory=dict)
    by_category: Dict[str, int] = field(default_factory=dict)
    
    # Processing complexity assessment
    complexity_distribution: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert assessment to dictionary for JSON serialization."""
        return {
            'summary': {
                'total_documents': self.total_documents,
                'documents_with_pdf': self.documents_with_pdf,
                'documents_with_latex': self.documents_with_latex,
                'pdf_coverage': round(self.documents_with_pdf / max(1, self.total_documents) * 100, 2)
            },
            'size_distribution': self.size_distribution.to_dict(),
            'time_estimate': self.time_estimate.to_dict(),
            'resource_requirements': self.resource_requirements.to_dict(),
            'temporal_distribution': {
                'by_year': {str(k): v for k, v in sorted(self.by_year.items())},
                'by_month_sample': self._sample_months(),
                'year_range': [min(self.by_year.keys()), max(self.by_year.keys())] if self.by_year else [0, 0]
            },
            'category_distribution': dict(sorted(self.by_category.items(), key=lambda x: x[1], reverse=True)[:10]),
            'complexity': self.complexity_distribution,
            'recommendations': self._generate_recommendations()
        }
    
    def _sample_months(self) -> Dict[str, int]:
        """Return sample of months for preview."""
        if not self.by_month:
            return {}
        # Show first and last 3 months
        sorted_months = sorted(self.by_month.items())
        sample = {}
        for (year, month), count in sorted_months[:3] + sorted_months[-3:]:
            sample[f"{year}-{month:02d}"] = count
        return sample
    
    def _generate_recommendations(self) -> List[str]:
        """Generate processing recommendations based on assessment."""
        recs = []
        
        # Size-based recommendations
        if self.size_distribution.x_large > self.total_documents * 0.1:
            recs.append("High proportion of large documents - consider increasing RAM allocation")
        
        # Time-based recommendations
        total_hours = self.time_estimate.total_minutes / 60
        if total_hours > 24:
            recs.append(f"Long processing time ({total_hours:.1f} hours) - consider batching")
        
        # GPU recommendations
        embedding_hours = self.time_estimate.embedding_minutes / 60
        if embedding_hours > 10:
            recs.append("Consider using multiple GPUs for parallel embedding")
        
        # Coverage recommendations
        pdf_coverage = self.documents_with_pdf / max(1, self.total_documents)
        if pdf_coverage < 0.8:
            recs.append(f"Low PDF coverage ({pdf_coverage*100:.1f}%) - consider updating artifact scan")
        
        return recs


def assess_compute_cost(
    config: ArxivDBConfig,
    where_sql: str,
    params: List[Any]
) -> ComputeCostAssessment:
    """
    Perform complete compute cost assessment for a query.
    
    This function is the primary interface for the orchestration layer,
    transforming search parameters into resource predictions.
    """
    assessment = ComputeCostAssessment()
    
    with get_connection(config.postgres) as conn:
        cur = conn.cursor()
        
        # Get total count and basic stats
        count_sql = f"""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN has_pdf THEN 1 ELSE 0 END) as with_pdf,
                SUM(CASE WHEN has_latex THEN 1 ELSE 0 END) as with_latex
            FROM papers {where_sql}
        """
        cur.execute(count_sql, params)
        total, with_pdf, with_latex = cur.fetchone()
        
        assessment.total_documents = total or 0
        assessment.documents_with_pdf = with_pdf or 0
        assessment.documents_with_latex = with_latex or 0
        
        # Get size distribution (if pdf_size_bytes is populated)
        # Need to handle the WHERE clause properly
        if where_sql:
            size_where = where_sql + " AND pdf_size_bytes IS NOT NULL"
        else:
            size_where = "WHERE pdf_size_bytes IS NOT NULL"
            
        size_sql = f"""
            SELECT 
                SUM(CASE WHEN pdf_size_bytes < 1048576 THEN 1 ELSE 0 END) as small,
                SUM(CASE WHEN pdf_size_bytes >= 1048576 AND pdf_size_bytes < 5242880 THEN 1 ELSE 0 END) as medium,
                SUM(CASE WHEN pdf_size_bytes >= 5242880 AND pdf_size_bytes < 20971520 THEN 1 ELSE 0 END) as large,
                SUM(CASE WHEN pdf_size_bytes >= 20971520 THEN 1 ELSE 0 END) as x_large,
                SUM(pdf_size_bytes) as total_bytes,
                AVG(pdf_size_bytes) as avg_bytes
            FROM papers {size_where}
        """
        cur.execute(size_sql, params)
        row = cur.fetchone()
        if row and row[4]:  # total_bytes not null
            assessment.size_distribution = DocumentSizeDistribution(
                small=row[0] or 0,
                medium=row[1] or 0,
                large=row[2] or 0,
                x_large=row[3] or 0,
                total_bytes=row[4] or 0,
                avg_bytes=row[5] or 0
            )
        else:
            # Estimate based on typical PDF sizes if not populated
            avg_size = 2 * 1024 * 1024  # 2MB average
            assessment.size_distribution.total_bytes = assessment.documents_with_pdf * avg_size
            assessment.size_distribution.avg_bytes = avg_size
        
        # Get temporal distribution
        if where_sql:
            year_where = where_sql + " AND year IS NOT NULL"
        else:
            year_where = "WHERE year IS NOT NULL"
            
        year_sql = f"""
            SELECT year, COUNT(*) 
            FROM papers {year_where}
            GROUP BY year ORDER BY year
        """
        cur.execute(year_sql, params)
        assessment.by_year = dict(cur.fetchall())
        
        # Get category distribution
        if where_sql:
            # Need to prefix all column references with p. for the papers table
            cat_where = where_sql.replace('WHERE', 'WHERE p.arxiv_id = pc.arxiv_id AND')
            # Fix any unqualified column references
            cat_where = cat_where.replace(' year ', ' p.year ')
            cat_where = cat_where.replace(' month ', ' p.month ')
            cat_where = cat_where.replace(' yymm ', ' p.yymm ')
            cat_where = cat_where.replace(' has_pdf ', ' p.has_pdf ')
            cat_where = cat_where.replace(' arxiv_id IN', ' p.arxiv_id IN')
        else:
            cat_where = ""
            
        cat_sql = f"""
            SELECT pc.category, COUNT(DISTINCT pc.arxiv_id)
            FROM paper_categories pc
            {f"JOIN papers p ON p.arxiv_id = pc.arxiv_id {cat_where}" if where_sql else ""}
            GROUP BY pc.category
            ORDER BY COUNT(DISTINCT pc.arxiv_id) DESC
            LIMIT 20
        """
        cur.execute(cat_sql, params if where_sql else [])
        assessment.by_category = dict(cur.fetchall())
        
        # Get complexity distribution (if populated)
        if where_sql:
            complexity_where = where_sql + " AND processing_complexity IS NOT NULL"
        else:
            complexity_where = "WHERE processing_complexity IS NOT NULL"
            
        complexity_sql = f"""
            SELECT 
                processing_complexity, 
                COUNT(*)
            FROM papers {complexity_where}
            GROUP BY processing_complexity
        """
        cur.execute(complexity_sql, params)
        assessment.complexity_distribution = dict(cur.fetchall())
        
        # Calculate time and resource estimates
        assessment.time_estimate = ProcessingTimeEstimate.from_document_count(
            assessment.total_documents,
            has_gpu=os.getenv('USE_GPU', 'true').lower() == 'true'
        )
        
        assessment.resource_requirements = ResourceRequirements.from_assessment(
            assessment.total_documents,
            assessment.size_distribution.total_bytes,
            assessment.time_estimate,
            num_workers=8
        )
        
        cur.close()
    
    return assessment


def print_assessment_report(assessment: ComputeCostAssessment) -> None:
    """
    Print human-readable assessment report.
    
    This function translates the computational assessment into human-readable
    format, completing the communication circuit from machine to human.
    """
    print("\n" + "=" * 80)
    print("COMPUTE COST ASSESSMENT REPORT")
    print("=" * 80)
    
    # Summary
    print(f"\nDocument Summary:")
    print(f"  Total Documents: {assessment.total_documents:,}")
    print(f"  With PDF: {assessment.documents_with_pdf:,} ({assessment.documents_with_pdf/max(1,assessment.total_documents)*100:.1f}%)")
    print(f"  With LaTeX: {assessment.documents_with_latex:,} ({assessment.documents_with_latex/max(1,assessment.total_documents)*100:.1f}%)")
    
    # Size distribution
    if assessment.size_distribution.total_bytes > 0:
        print(f"\nSize Distribution:")
        dist = assessment.size_distribution
        print(f"  Small (<1MB): {dist.small:,}")
        print(f"  Medium (1-5MB): {dist.medium:,}")
        print(f"  Large (5-20MB): {dist.large:,}")
        print(f"  X-Large (>20MB): {dist.x_large:,}")
        print(f"  Average Size: {dist._human_size(dist.avg_bytes)}")
        print(f"  Total Size: {dist._human_size(dist.total_bytes)}")
    
    # Time estimates
    print(f"\nProcessing Time Estimates:")
    time_est = assessment.time_estimate
    print(f"  Extraction: {time_est.extraction_minutes:.1f} minutes ({time_est.extraction_minutes/60:.1f} hours)")
    print(f"  Embedding: {time_est.embedding_minutes:.1f} minutes ({time_est.embedding_minutes/60:.1f} hours)")
    print(f"  Total: {time_est.total_minutes:.1f} minutes ({time_est.total_minutes/60:.1f} hours)")
    
    # Resource requirements
    print(f"\nResource Requirements:")
    resources = assessment.resource_requirements
    print(f"  Storage: {resources.storage_gb:.1f} GB")
    print(f"  RAM: {resources.ram_gb:.1f} GB")
    print(f"  GPU Hours: {resources.gpu_hours:.1f}")
    print(f"  CPU Hours: {resources.cpu_hours:.1f}")
    print(f"  Estimated Cost: ${resources._estimate_cost():.2f}")
    
    # Temporal distribution
    if assessment.by_year:
        print(f"\nTemporal Distribution:")
        print(f"  Year Range: {min(assessment.by_year.keys())}-{max(assessment.by_year.keys())}")
        print(f"  Papers by Year (top 5):")
        for year, count in sorted(assessment.by_year.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"    {year}: {count:,}")
    
    # Category distribution
    if assessment.by_category:
        print(f"\nTop Categories:")
        for cat, count in list(assessment.by_category.items())[:5]:
            print(f"  {cat}: {count:,}")
    
    # Recommendations
    recs = assessment._generate_recommendations()
    if recs:
        print(f"\nRecommendations:")
        for rec in recs:
            print(f"  • {rec}")
    
    print("\n" + "=" * 80)