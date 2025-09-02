"""
ArXiv List Generator Utility

Clean, database-driven paper list generation for ACID pipeline.
Following Actor-Network Theory, this module acts as an obligatory passage point
between search configurations and the PostgreSQL database, translating human
intentions into SQL queries that extract actionable paper lists.
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog
import yaml

# Import database utilities
from tools.arxiv.db.config import load_config as load_db_config
from tools.arxiv.db.export_ids import (
    build_query,
    build_where,
    collect_ids,
    collect_stats,
    write_outputs,
)
from tools.arxiv.db.pg import get_connection

logger = structlog.get_logger()


class ArxivListGenerator:
    """
    Generate ArXiv paper lists from PostgreSQL database.
    
    This class embodies the translation process from human-readable search
    configurations to machine-executable SQL queries, maintaining high
    conveyance (actionability) while minimizing time complexity.
    """
    
    def __init__(
        self,
        search_config_path: str,
        db_config_path: str = "tools/arxiv/configs/db.yaml"
    ):
        """
        Initialize the list generator with configuration files.
        
        Args:
            search_config_path: Path to search configuration YAML
            db_config_path: Path to database configuration YAML
        """
        self.search_config = self._load_search_config(search_config_path)
        self.db_config = load_db_config(db_config_path)
        
        # Ensure POSTGRES_USER is set if not in environment
        if not os.getenv("POSTGRES_USER") and not os.getenv("PGUSER"):
            os.environ["POSTGRES_USER"] = "postgres"
            logger.info("Set POSTGRES_USER=postgres as default")
    
    def _load_search_config(self, path: str) -> Dict[str, Any]:
        """Load and validate search configuration from YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Search config not found: {path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required fields
        required = ['search', 'output']
        for field in required:
            if field not in config:
                raise ValueError(f"Missing required config section: {field}")
        
        return config
    
    def generate_list(
        self,
        override_count: Optional[int] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Generate paper list based on search configuration.
        
        Args:
            override_count: Override target_count from config
            dry_run: If True, only return stats without writing files
            
        Returns:
            Dictionary containing:
                - total_papers: Number of papers found
                - paper_ids: List of ArXiv IDs
                - stats: Statistical breakdown
                - output_files: Paths to generated files
        """
        search = self.search_config['search']
        output = self.search_config['output']
        
        # Extract search parameters
        start_year = search.get('start_year', 2010)
        end_year = search.get('end_year', 2025)
        categories = search.get('categories', ['cs.LG', 'cs.CL', 'cs.CV'])
        keywords = search.get('keywords', '').strip()
        
        # Sampling configuration
        sampling = search.get('sampling', {})
        target_count = override_count or sampling.get('target_count')
        cap_mode = sampling.get('cap_mode', 'per-year')
        
        # Calculate caps if target_count is specified
        per_year_cap = sampling.get('per_year_cap')
        per_month_cap = sampling.get('per_month_cap')
        
        if target_count and not per_year_cap and not per_month_cap:
            years = max(1, end_year - start_year + 1)
            if cap_mode == 'per-month':
                months_count = years * 12
                per_month_cap = math.ceil(target_count / months_count)
                logger.info(f"Calculated per_month_cap={per_month_cap} for {target_count} papers")
            else:
                per_year_cap = math.ceil(target_count / years)
                logger.info(f"Calculated per_year_cap={per_year_cap} for {target_count} papers")
        
        # Filters
        filters = search.get('filters', {})
        with_pdf = filters.get('require_pdf', False)
        missing_pdf = False  # This would exclude papers WITH PDFs
        
        # Additional year constraints
        if filters.get('min_year'):
            start_year = max(start_year, filters['min_year'])
        if filters.get('max_year'):
            end_year = min(end_year, filters['max_year'])
        
        logger.info(
            "Generating ArXiv list",
            start_year=start_year,
            end_year=end_year,
            categories=categories,
            has_keywords=bool(keywords),
            target_count=target_count,
            cap_mode=cap_mode
        )
        
        # Build SQL query
        sql, params, monthly_sql = build_query(
            start_year=start_year,
            end_year=end_year,
            months=None,  # Not using month filter
            yymm_range=None,  # Not using YYMM range
            categories=categories,
            keywords=keywords if keywords else None,
            with_pdf=with_pdf,
            missing_pdf=missing_pdf,
            per_year_cap=per_year_cap,
            per_month_cap=per_month_cap,
        )
        
        # Collect IDs from database
        logger.info("Querying database for paper IDs...")
        ids = collect_ids(self.db_config, sql, params)
        logger.info(f"Found {len(ids)} papers matching criteria")
        
        # Collect statistics
        where_sql, where_params = build_where(
            start_year=start_year,
            end_year=end_year,
            months=None,
            yymm_range=None,
            categories=categories,
            keywords=keywords if keywords else None,
            with_pdf=with_pdf,
            missing_pdf=missing_pdf,
            table_alias=None,
        )
        stats = collect_stats(self.db_config, where_sql, where_params)
        
        # Collect monthly data if requested
        monthly = None
        if output.get('write_monthly_lists', True) and monthly_sql:
            logger.info("Collecting monthly breakdowns...")
            monthly = self._collect_monthly(monthly_sql, params)
        
        # Write outputs (unless dry run)
        output_files = {}
        if not dry_run:
            output_dir = Path(output['base_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp suffix
            timestamp_format = output.get('timestamp_format', '%Y%m%d_%H%M%S')
            stamp = f"{start_year}_{end_year}_{datetime.now().strftime(timestamp_format)}"
            
            output_files = write_outputs(
                out_dir=output_dir,
                prefix=output.get('prefix', 'arxiv_ids'),
                ids=ids,
                with_pdf=with_pdf,
                missing_pdf=missing_pdf,
                stats=stats,
                monthly=monthly,
                suffix=stamp,
                update_symlinks=output.get('update_symlinks', True)
            )
            
            logger.info("Output files written", files=output_files)
        
        return {
            'total_papers': len(ids),
            'paper_ids': ids,
            'stats': stats,
            'output_files': output_files,
            'search_config': self.search_config,
        }
    
    def _collect_monthly(
        self,
        monthly_sql: str,
        params: List[Any]
    ) -> Dict[Tuple[int, int], List[str]]:
        """Collect papers grouped by month."""
        monthly = {}
        
        with get_connection(self.db_config.postgres) as conn:
            cur = conn.cursor()
            try:
                cur.execute(monthly_sql, params)
                for arxiv_id, year, month, _yymm in cur.fetchall():
                    if year is None or month is None:
                        continue
                    monthly.setdefault((int(year), int(month)), []).append(arxiv_id)
            finally:
                try:
                    cur.close()
                except Exception:
                    pass
        
        return monthly
    
    def get_stats_only(self) -> Dict[str, Any]:
        """
        Get statistics without generating the full list.
        Useful for previewing what would be generated.
        """
        result = self.generate_list(dry_run=True)
        return result['stats']
    
    def validate_database(self) -> bool:
        """
        Validate that the database is accessible and has data.
        
        Returns:
            True if database is ready, False otherwise
        """
        try:
            with get_connection(self.db_config.postgres) as conn:
                cur = conn.cursor()
                
                # Check if papers table exists and has data
                cur.execute("SELECT COUNT(*) FROM papers LIMIT 1")
                count = cur.fetchone()[0]
                
                logger.info(f"Database validation successful: {count:,} papers available")
                return count > 0
                
        except Exception as e:
            logger.error(f"Database validation failed: {e}")
            return False


def main():
    """Command-line interface for list generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate ArXiv paper lists from PostgreSQL database"
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
        '--count',
        type=int,
        help='Override target_count from config'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate database connection'
    )
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Only show statistics without generating files'
    )
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = ArxivListGenerator(args.config, args.db_config)
    
    # Validate database
    if args.validate_only:
        if generator.validate_database():
            print("✅ Database validation successful")
            return 0
        else:
            print("❌ Database validation failed")
            return 1
    
    # Get stats only
    if args.stats_only:
        stats = generator.get_stats_only()
        print("\n" + "="*60)
        print("STATISTICS PREVIEW")
        print("="*60)
        print(f"Total papers matching criteria: {stats.get('total', 0):,}")
        
        by_year = stats.get('by_year', {})
        if by_year:
            print("\nBy year:")
            for year, count in sorted(by_year.items())[:10]:
                print(f"  {year}: {count:,}")
            if len(by_year) > 10:
                print(f"  ... and {len(by_year) - 10} more years")
        
        return 0
    
    # Generate full list
    result = generator.generate_list(override_count=args.count)
    
    print("\n" + "="*60)
    print("ARXIV LIST GENERATION COMPLETE")
    print("="*60)
    print(f"Total papers: {result['total_papers']:,}")
    
    if result['output_files']:
        print("\nOutput files:")
        for key, path in result['output_files'].items():
            if path:
                print(f"  - {key}: {path}")
    
    # Show sample of IDs
    if result['paper_ids']:
        print(f"\nSample IDs (first 5):")
        for arxiv_id in result['paper_ids'][:5]:
            print(f"  - {arxiv_id}")
    
    return 0


if __name__ == "__main__":
    exit(main())