"""
Monitoring module for ACID pipeline.

Provides real-time monitoring, health checks, and performance metrics
for the ArangoDB-based ACID processing pipeline.
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from arango import ArangoClient

logger = logging.getLogger(__name__)


@dataclass
class ProcessingMetrics:
    """Metrics for processing performance"""
    total_papers: int
    processed_papers: int
    failed_papers: int
    pending_papers: int
    active_locks: int
    expired_locks: int
    avg_processing_time: float
    papers_per_hour: float
    total_chunks: int
    total_equations: int
    total_tables: int
    total_images: int
    database_size_gb: float
    index_size_gb: float


@dataclass
class LockInfo:
    """Information about a lock"""
    paper_id: str
    worker_id: int
    acquired_at: str
    expires_at: str
    age_minutes: float
    is_expired: bool


class ArangoMonitor:
    """
    Monitor for ArangoDB ACID pipeline.
    
    Provides:
    - Processing status and throughput
    - Lock monitoring and cleanup
    - Collection statistics
    - Performance metrics
    - Health checks
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the monitor
        
        Note: Password must be provided in config, not read from environment.
        """
        config = config or {}
        
        # Validate password is provided
        password = config.get('password')
        if not password:
            raise ValueError("ArangoDB password must be provided in config['password']")
        
        # Connect to ArangoDB
        self.arango = ArangoClient(hosts=config.get('arango_host', ['http://192.168.1.69:8529']))
        self.db = self.arango.db(
            config.get('database', 'academy_store'),
            username=config.get('username', 'root'),
            password=password
        )
        
        logger.info("Initialized ArangoDB monitor")
    
    def get_processing_status(self) -> Dict[str, int]:
        """
        Get current processing status.
        
        Returns:
            Dictionary with counts by status
        """
        try:
            cursor = self.db.aql.execute("""
                FOR paper IN arxiv_papers
                    COLLECT status = paper.status WITH COUNT INTO count
                    RETURN { status: status, count: count }
            """)
            
            status_dict = {}
            for item in cursor:
                status_dict[item['status']] = item['count']
            
            # Cursor is automatically closed by ArangoDB client
            return status_dict
        except Exception as e:
            logger.error(f"Error getting processing status: {e}")
            return {}
    
    def get_stuck_locks(self, threshold_hours: int = 1) -> List[LockInfo]:
        """
        Find locks that have been held longer than threshold.
        
        Args:
            threshold_hours: Number of hours before considering a lock stuck
            
        Returns:
            List of stuck locks
        """
        try:
            cursor = self.db.aql.execute("""
                FOR lock IN arxiv_locks
                    LET age_minutes = DATE_DIFF(lock.acquired_at, DATE_NOW(), 'minute')
                    LET is_expired = lock.expiresAt < DATE_NOW()
                    FILTER age_minutes > @threshold_minutes
                    RETURN {
                        paper_id: lock._key,
                        worker_id: lock.worker_id,
                        acquired_at: lock.acquired_at,
                        expires_at: lock.expiresAt,
                        age_minutes: age_minutes,
                        is_expired: is_expired
                    }
            """, bind_vars={'threshold_minutes': threshold_hours * 60})
            
            locks = []
            for lock_data in cursor:
                locks.append(LockInfo(**lock_data))
            
            return locks
        except Exception as e:
            logger.error(f"Error getting stuck locks: {e}")
            return []
    
    def get_processing_throughput(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get processing throughput over specified time period.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Throughput statistics
        """
        try:
            since = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            cursor = self.db.aql.execute("""
                FOR paper IN arxiv_papers
                    FILTER paper.processing_date >= @since
                    COLLECT hour = DATE_FORMAT(paper.processing_date, '%yyyy-%mm-%dd %hh:00')
                    WITH COUNT INTO processed
                    SORT hour DESC
                    RETURN { hour: hour, processed: processed }
            """, bind_vars={'since': since})
            
            hourly_stats = list(cursor)
            
            # Calculate overall statistics
            total_processed = sum(h['processed'] for h in hourly_stats)
            avg_per_hour = total_processed / hours if hours > 0 else 0
            
            return {
                'total_processed': total_processed,
                'avg_per_hour': avg_per_hour,
                'hourly_breakdown': hourly_stats,
                'period_hours': hours
            }
        except Exception as e:
            logger.error(f"Error getting throughput: {e}")
            return {
                'total_processed': 0,
                'avg_per_hour': 0,
                'hourly_breakdown': [],
                'period_hours': hours
            }
    
    def get_collection_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all collections.
        
        Returns:
            Dictionary with collection statistics
        """
        stats = {}
        
        collections = ['arxiv_papers', 'arxiv_chunks', 'arxiv_embeddings', 'arxiv_equations', 'arxiv_tables', 'arxiv_images', 'arxiv_locks']
        
        for collection_name in collections:
            try:
                if self.db.has_collection(collection_name):
                    collection = self.db.collection(collection_name)
                    
                    # Get basic count
                    count = collection.count()
                    
                    # Get collection properties
                    properties = collection.properties()
                    
                    # Get indexes
                    indexes = collection.indexes()
                    
                    stats[collection_name] = {
                        'count': count,
                        'cache_enabled': properties.get('cacheEnabled', False),
                        'wait_for_sync': properties.get('waitForSync', False),
                        'index_count': len(indexes),
                        'indexes': [idx['type'] for idx in indexes]
                    }
                    
                    # Special handling for embeddings to get vector stats
                    if collection_name == 'arxiv_embeddings' and count > 0:
                        cursor = self.db.aql.execute("""
                            FOR e IN arxiv_embeddings
                                LIMIT 1
                                RETURN LENGTH(e.vector)
                        """)
                        vector_dims = list(cursor)
                        if vector_dims:
                            stats[collection_name]['vector_dimensions'] = vector_dims[0]
                else:
                    stats[collection_name] = {'error': 'Collection does not exist'}
            except Exception as e:
                stats[collection_name] = {'error': str(e)}
        
        return stats
    
    def get_overall_metrics(self) -> ProcessingMetrics:
        """
        Get overall processing metrics.
        
        Returns:
            ProcessingMetrics object with current statistics
        """
        try:
            # Get paper counts by status
            status = self.get_processing_status()
            
            # Get active locks
            cursor = self.db.aql.execute("""
                FOR lock IN arxiv_locks
                    COLLECT WITH COUNT INTO total
                    RETURN total
            """)
            cursor_list = list(cursor)
            active_locks = cursor_list[0] if cursor_list else 0
            
            # Get expired locks
            cursor = self.db.aql.execute("""
                FOR lock IN arxiv_locks
                    FILTER lock.expiresAt < DATE_NOW()
                    COLLECT WITH COUNT INTO total
                    RETURN total
            """)
            cursor_list = list(cursor)
            expired_locks = cursor_list[0] if cursor_list else 0
            
            # Get average processing time
            cursor = self.db.aql.execute("""
                FOR paper IN arxiv_papers
                    FILTER paper.processing_date != null
                    RETURN paper.processing_time
            """)
            processing_times = list(cursor)
            avg_processing_time = (
                sum(processing_times) / len(processing_times)
                if processing_times else 0
            )
            
            # Get throughput
            throughput = self.get_processing_throughput(1)
            papers_per_hour = throughput['avg_per_hour']
            
            # Get content counts
            cursor = self.db.aql.execute("""
                RETURN {
                    chunks: (FOR c IN arxiv_chunks COLLECT WITH COUNT INTO count RETURN count)[0],
                    equations: (FOR e IN arxiv_equations COLLECT WITH COUNT INTO count RETURN count)[0],
                    tables: (FOR t IN arxiv_tables COLLECT WITH COUNT INTO count RETURN count)[0],
                    images: (FOR i IN arxiv_images COLLECT WITH COUNT INTO count RETURN count)[0]
                }
            """)
            content_counts = list(cursor)[0] if cursor else {}
            
            # Estimate database size (more accurate)
            collection_stats = self.get_collection_statistics()
            total_docs = sum(
                s.get('count', 0) for s in collection_stats.values()
                if isinstance(s, dict) and 'count' in s
            )
            # More accurate estimate based on collection types:
            # Papers: ~5KB, Chunks: ~3KB, Embeddings: ~10KB (2048 dims * 4 bytes + metadata)
            # Equations/Tables/Images: ~2KB average
            paper_count = collection_stats.get('arxiv_papers', {}).get('count', 0)
            chunk_count = collection_stats.get('arxiv_chunks', {}).get('count', 0)
            embedding_count = collection_stats.get('arxiv_embeddings', {}).get('count', 0)
            other_count = total_docs - paper_count - chunk_count - embedding_count
            
            estimated_bytes = (
                paper_count * 5 * 1024 +  # Papers
                chunk_count * 3 * 1024 +  # Chunks
                embedding_count * 10 * 1024 +  # Embeddings
                other_count * 2 * 1024  # Others
            )
            database_size_gb = estimated_bytes / (1024 ** 3)
            
            return ProcessingMetrics(
                total_papers=sum(status.values()),
                processed_papers=status.get('PROCESSED', 0),
                failed_papers=status.get('FAILED', 0),
                pending_papers=status.get('PENDING', 0),
                active_locks=active_locks,
                expired_locks=expired_locks,
                avg_processing_time=avg_processing_time,
                papers_per_hour=papers_per_hour,
                total_chunks=content_counts.get('chunks', 0),
                total_equations=content_counts.get('equations', 0),
                total_tables=content_counts.get('tables', 0),
                total_images=content_counts.get('images', 0),
                database_size_gb=database_size_gb,
                index_size_gb=database_size_gb * 0.1  # Rough estimate
            )
        except Exception as e:
            logger.error(f"Error getting overall metrics: {e}")
            return ProcessingMetrics(
                total_papers=0,
                processed_papers=0,
                failed_papers=0,
                pending_papers=0,
                active_locks=0,
                expired_locks=0,
                avg_processing_time=0,
                papers_per_hour=0,
                total_chunks=0,
                total_equations=0,
                total_tables=0,
                total_images=0,
                database_size_gb=0,
                index_size_gb=0
            )
    
    def cleanup_expired_locks(self) -> int:
        """
        Clean up expired locks.
        
        Note: This should happen automatically via TTL index,
        but this provides manual cleanup if needed.
        
        Returns:
            Number of locks cleaned up
        """
        try:
            cursor = self.db.aql.execute("""
                FOR lock IN arxiv_locks
                    FILTER lock.expiresAt < DATE_NOW()
                    REMOVE lock IN arxiv_locks
                    RETURN OLD
            """)
            
            cleaned = len(list(cursor))
            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} expired locks")
            
            return cleaned
        except Exception as e:
            logger.error(f"Error cleaning up locks: {e}")
            return 0
    
    def check_health(self) -> Dict[str, Any]:
        """
        Perform health check on the system.
        
        Returns:
            Health status dictionary
        """
        health = {
            'status': 'healthy',
            'issues': [],
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Check database connection
            self.db.version()
        except Exception as e:
            health['status'] = 'critical'
            health['issues'].append(f"Database connection failed: {e}")
            return health
        
        # Check for stuck locks
        stuck_locks = self.get_stuck_locks(threshold_hours=2)
        if stuck_locks:
            health['status'] = 'warning'
            health['issues'].append(f"{len(stuck_locks)} stuck locks detected")
        
        # Check processing rate
        throughput = self.get_processing_throughput(1)
        if throughput['avg_per_hour'] < 0.1:  # Less than 0.1 papers/hour
            health['status'] = 'warning'
            health['issues'].append("Low processing throughput")
        
        # Check collection existence
        required_collections = ['arxiv_papers', 'arxiv_chunks', 'arxiv_embeddings', 'arxiv_locks']
        for collection in required_collections:
            if not self.db.has_collection(collection):
                health['status'] = 'critical'
                health['issues'].append(f"Missing required collection: {collection}")
        
        # Check for high failure rate
        status = self.get_processing_status()
        if status.get('FAILED', 0) > status.get('PROCESSED', 0) * 0.1:  # >10% failure
            health['status'] = 'warning'
            health['issues'].append("High failure rate detected")
        
        return health
    
    def print_summary(self):
        """Print a formatted summary of current status"""
        print("\n" + "=" * 60)
        print("ACID PIPELINE MONITORING SUMMARY")
        print("=" * 60)
        
        # Overall metrics
        metrics = self.get_overall_metrics()
        print(f"\nPAPERS:")
        print(f"  Total:     {metrics.total_papers:,}")
        print(f"  Processed: {metrics.processed_papers:,}")
        print(f"  Failed:    {metrics.failed_papers:,}")
        print(f"  Pending:   {metrics.pending_papers:,}")
        
        print(f"\nPROCESSING:")
        print(f"  Papers/hour:    {metrics.papers_per_hour:.2f}")
        print(f"  Avg time/paper: {metrics.avg_processing_time:.2f}s")
        print(f"  Active locks:   {metrics.active_locks}")
        print(f"  Expired locks:  {metrics.expired_locks}")
        
        print(f"\nCONTENT:")
        print(f"  Chunks:    {metrics.total_chunks:,}")
        print(f"  Equations: {metrics.total_equations:,}")
        print(f"  Tables:    {metrics.total_tables:,}")
        print(f"  Images:    {metrics.total_images:,}")
        
        print(f"\nSTORAGE:")
        print(f"  Database size: {metrics.database_size_gb:.2f} GB")
        print(f"  Index size:    {metrics.index_size_gb:.2f} GB")
        
        # Health check
        health = self.check_health()
        print(f"\nHEALTH: {health['status'].upper()}")
        if health['issues']:
            print("  Issues:")
            for issue in health['issues']:
                print(f"    - {issue}")
        
        print("=" * 60)


def monitor_loop(interval_seconds: int = 60, password: str = None):
    """
    Run monitoring loop that prints status periodically.
    
    Args:
        interval_seconds: How often to print status
        password: ArangoDB password (if not provided, will use environment variable)
    """
    import os
    
    # Build config with password
    config = {
        'password': password or os.environ.get('ARANGO_PASSWORD')
    }
    
    monitor = ArangoMonitor(config)
    
    print(f"Starting monitoring loop (interval: {interval_seconds}s)")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            monitor.print_summary()
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\nMonitoring stopped")