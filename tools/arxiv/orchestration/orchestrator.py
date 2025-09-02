"""
ArXiv Pipeline Orchestrator

This module implements the search → preview → refine → process orchestration pattern
specified in the PRD. Following Actor-Network Theory, it acts as the central 
obligatory passage point coordinating all pipeline components.

The orchestrator is interface-agnostic, supporting CLI, MCP server, or GUI frontends
equally well through its modular backend design.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
import yaml

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.arxiv.db.compute_assessment import (
    ComputeCostAssessment,
    assess_compute_cost,
    print_assessment_report
)
from tools.arxiv.db.config import load_config as load_db_config
from tools.arxiv.db.export_ids import build_where
from tools.arxiv.utils.list_generator import ArxivListGenerator

logger = structlog.get_logger()


class OrchestrationState(Enum):
    """
    Represents the current state in the orchestration flow.
    
    In ANT terms, each state represents a different configuration of the
    actor-network, with different actors (modules) playing primary roles.
    """
    IDLE = "idle"
    SEARCHING = "searching"
    PREVIEWING = "previewing"
    REFINING = "refining"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SearchConfiguration:
    """
    Encapsulates search parameters for the pipeline.
    
    This is a boundary object that maintains coherence across different
    pipeline stages while allowing local adaptations.
    """
    start_year: int = 2010
    end_year: int = 2025
    categories: List[str] = field(default_factory=lambda: ["cs.LG", "cs.CL", "cs.CV"])
    keywords: Optional[str] = None
    target_count: Optional[int] = None
    cap_mode: str = "per-year"
    require_pdf: bool = False
    
    @classmethod
    def from_yaml(cls, path: str) -> SearchConfiguration:
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        search = data.get('search', {})
        sampling = search.get('sampling', {})
        filters = search.get('filters', {})
        
        return cls(
            start_year=search.get('start_year', 2010),
            end_year=search.get('end_year', 2025),
            categories=search.get('categories', ["cs.LG", "cs.CL", "cs.CV"]),
            keywords=search.get('keywords'),
            target_count=sampling.get('target_count'),
            cap_mode=sampling.get('cap_mode', 'per-year'),
            require_pdf=filters.get('require_pdf', False)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'start_year': self.start_year,
            'end_year': self.end_year,
            'categories': self.categories,
            'keywords': self.keywords,
            'target_count': self.target_count,
            'cap_mode': self.cap_mode,
            'require_pdf': self.require_pdf
        }


@dataclass
class OrchestrationContext:
    """
    Maintains context throughout the orchestration flow.
    
    From an anthropological perspective, this represents the collective
    memory of the pipeline - preserving decisions and assessments across
    state transitions.
    """
    search_config: SearchConfiguration
    state: OrchestrationState = OrchestrationState.IDLE
    
    # Results from each stage
    assessment: Optional[ComputeCostAssessment] = None
    paper_ids: List[str] = field(default_factory=list)
    output_files: Dict[str, str] = field(default_factory=dict)
    
    # Tracking
    iterations: int = 0
    approved: bool = False
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization."""
        return {
            'search_config': self.search_config.to_dict(),
            'state': self.state.value,
            'iterations': self.iterations,
            'approved': self.approved,
            'assessment': self.assessment.to_dict() if self.assessment else None,
            'paper_count': len(self.paper_ids),
            'output_files': self.output_files,
            'error': self.error
        }


class ArxivPipelineOrchestrator:
    """
    Central orchestrator for the ArXiv processing pipeline.
    
    This class embodies the orchestration pattern, coordinating the flow
    from search through preview, refinement, and processing. It maintains
    separation between backend logic and interface concerns.
    """
    
    def __init__(
        self,
        search_config_path: str,
        db_config_path: str = "tools/arxiv/configs/db.yaml",
        auto_approve_threshold: Optional[int] = None
    ):
        """
        Initialize the orchestrator.
        
        Args:
            search_config_path: Path to search configuration YAML
            db_config_path: Path to database configuration YAML
            auto_approve_threshold: Auto-approve if documents <= this count
        """
        self.search_config = SearchConfiguration.from_yaml(search_config_path)
        self.db_config = load_db_config(db_config_path)
        self.auto_approve_threshold = auto_approve_threshold
        
        self.context = OrchestrationContext(search_config=self.search_config)
        self.list_generator = ArxivListGenerator(search_config_path, db_config_path)
        
        logger.info(
            "Orchestrator initialized",
            search_config=search_config_path,
            auto_approve=auto_approve_threshold
        )
    
    def search(self) -> ComputeCostAssessment:
        """
        Execute search phase - query database for matching papers.
        
        Returns:
            Compute cost assessment for the search results
        """
        self.context.state = OrchestrationState.SEARCHING
        logger.info("Starting search phase", config=self.search_config.to_dict())
        
        try:
            # Build query parameters
            where_sql, params = build_where(
                start_year=self.search_config.start_year,
                end_year=self.search_config.end_year,
                months=None,
                yymm_range=None,
                categories=self.search_config.categories,
                keywords=self.search_config.keywords,
                with_pdf=self.search_config.require_pdf,
                missing_pdf=False,
                table_alias=None
            )
            
            # Perform assessment
            self.context.assessment = assess_compute_cost(
                self.db_config,
                where_sql,
                params
            )
            
            logger.info(
                "Search completed",
                total_documents=self.context.assessment.total_documents,
                with_pdf=self.context.assessment.documents_with_pdf
            )
            
            self.context.state = OrchestrationState.PREVIEWING
            return self.context.assessment
            
        except Exception as e:
            logger.error("Search failed", error=str(e))
            self.context.state = OrchestrationState.FAILED
            self.context.error = str(e)
            raise
    
    def preview(self) -> Dict[str, Any]:
        """
        Generate preview of compute costs and statistics.
        
        Returns:
            Dictionary with assessment details for review
        """
        if self.context.state != OrchestrationState.PREVIEWING:
            raise ValueError(f"Invalid state for preview: {self.context.state}")
        
        if not self.context.assessment:
            raise ValueError("No assessment available for preview")
        
        logger.info("Generating preview")
        
        # Auto-approve if below threshold
        if (self.auto_approve_threshold and 
            self.context.assessment.total_documents <= self.auto_approve_threshold):
            logger.info(
                "Auto-approving",
                documents=self.context.assessment.total_documents,
                threshold=self.auto_approve_threshold
            )
            self.context.approved = True
        
        return self.context.assessment.to_dict()
    
    def refine(self, new_config: Optional[SearchConfiguration] = None) -> ComputeCostAssessment:
        """
        Refine search parameters and re-assess.
        
        Args:
            new_config: Updated search configuration
            
        Returns:
            New compute cost assessment
        """
        self.context.state = OrchestrationState.REFINING
        self.context.iterations += 1
        
        if new_config:
            logger.info("Refining with new configuration")
            self.search_config = new_config
            self.context.search_config = new_config
        else:
            logger.info("Refining with existing configuration")
        
        # Re-run search with refined parameters
        return self.search()
    
    def approve(self) -> None:
        """Mark the current assessment as approved for processing."""
        if not self.context.assessment:
            raise ValueError("No assessment to approve")
        
        self.context.approved = True
        logger.info("Assessment approved", documents=self.context.assessment.total_documents)
    
    def process(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Execute processing phase - generate paper lists and prepare for ACID pipeline.
        
        Args:
            dry_run: If True, simulate without writing files
            
        Returns:
            Processing results including file paths
        """
        if not self.context.approved:
            raise ValueError("Assessment not approved for processing")
        
        self.context.state = OrchestrationState.PROCESSING
        logger.info("Starting processing phase", dry_run=dry_run)
        
        try:
            # Generate paper list
            result = self.list_generator.generate_list(
                override_count=self.search_config.target_count,
                dry_run=dry_run
            )
            
            self.context.paper_ids = result['paper_ids']
            self.context.output_files = result.get('output_files', {})
            
            logger.info(
                "Processing completed",
                papers_generated=len(self.context.paper_ids),
                files_written=len(self.context.output_files) if self.context.output_files else 0
            )
            
            self.context.state = OrchestrationState.COMPLETED
            
            return {
                'success': True,
                'papers_count': len(self.context.paper_ids),
                'output_files': self.context.output_files,
                'next_step': self._get_next_step_command()
            }
            
        except Exception as e:
            logger.error("Processing failed", error=str(e))
            self.context.state = OrchestrationState.FAILED
            self.context.error = str(e)
            raise
    
    def _get_next_step_command(self) -> str:
        """Generate command for running ACID pipeline on generated list."""
        if not self.context.output_files or not self.context.output_files.get('master'):
            return "No output files generated"
        
        return (
            f"python tools/arxiv/pipelines/arxiv_pipeline.py "
            f"--config tools/arxiv/configs/acid_pipeline_phased.yaml "
            f"--input-list {self.context.output_files['master']} "
            f"--arango-password $ARANGO_PASSWORD"
        )
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current orchestration status.
        
        Returns:
            Complete context information
        """
        return self.context.to_dict()
    
    def reset(self) -> None:
        """Reset orchestrator to initial state."""
        self.context = OrchestrationContext(search_config=self.search_config)
        logger.info("Orchestrator reset")


def interactive_cli_session(orchestrator: ArxivPipelineOrchestrator) -> None:
    """
    Run an interactive CLI session for the orchestrator.
    
    This demonstrates the interface-agnostic nature of the orchestrator -
    the same backend can support different interaction patterns.
    """
    print("\n" + "=" * 80)
    print("ARXIV PIPELINE ORCHESTRATOR - Interactive Mode")
    print("=" * 80)
    
    while True:
        # Search phase
        print("\n[SEARCH] Executing search with current configuration...")
        assessment = orchestrator.search()
        
        # Preview phase
        print("\n[PREVIEW] Compute cost assessment:")
        print_assessment_report(assessment)
        
        # Decision point
        while True:
            print("\nOptions:")
            print("  1. Approve and process")
            print("  2. Refine search parameters")
            print("  3. Cancel")
            
            choice = input("\nYour choice (1-3): ").strip()
            
            if choice == "1":
                orchestrator.approve()
                print("\n[PROCESS] Generating paper lists...")
                result = orchestrator.process()
                print(f"\n✓ Generated {result['papers_count']} papers")
                if result['output_files']:
                    print("\nOutput files:")
                    for key, path in result['output_files'].items():
                        print(f"  - {key}: {path}")
                print(f"\nNext step:\n  {result['next_step']}")
                return
                
            elif choice == "2":
                print("\n[REFINE] Modifying search parameters...")
                print("(In production, this would load updated config)")
                print("For now, using same config for demonstration")
                assessment = orchestrator.refine()
                print("\n[PREVIEW] Updated assessment:")
                print_assessment_report(assessment)
                
            elif choice == "3":
                print("\nCancelled by user")
                return
            else:
                print("Invalid choice, please try again")


def main():
    """Command-line interface for the orchestrator."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ArXiv Pipeline Orchestrator - Search, Preview, Refine, Process"
    )
    parser.add_argument(
        '--config',
        default='tools/arxiv/configs/arxiv_search.yaml',
        help='Search configuration file'
    )
    parser.add_argument(
        '--db-config',
        default='tools/arxiv/configs/db.yaml',
        help='Database configuration file'
    )
    parser.add_argument(
        '--mode',
        choices=['interactive', 'auto', 'preview-only'],
        default='interactive',
        help='Execution mode'
    )
    parser.add_argument(
        '--auto-approve',
        type=int,
        help='Auto-approve if document count <= this value'
    )
    parser.add_argument(
        '--output-json',
        help='Write assessment to JSON file (preview-only mode)'
    )
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = ArxivPipelineOrchestrator(
        args.config,
        args.db_config,
        args.auto_approve
    )
    
    if args.mode == 'interactive':
        interactive_cli_session(orchestrator)
        
    elif args.mode == 'auto':
        # Fully automated flow
        print("Running in auto mode...")
        assessment = orchestrator.search()
        preview = orchestrator.preview()
        
        if orchestrator.context.approved:
            result = orchestrator.process()
            print(f"✓ Auto-processed {result['papers_count']} papers")
        else:
            print("Assessment not auto-approved. Run in interactive mode to review.")
            print_assessment_report(assessment)
            
    elif args.mode == 'preview-only':
        # Just generate assessment
        assessment = orchestrator.search()
        print_assessment_report(assessment)
        
        if args.output_json:
            with open(args.output_json, 'w') as f:
                json.dump(assessment.to_dict(), f, indent=2)
            print(f"\n✓ Assessment saved to {args.output_json}")
    
    return 0


if __name__ == "__main__":
    exit(main())