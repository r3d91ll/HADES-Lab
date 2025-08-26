#!/usr/bin/env python3
"""
Batch Processing with Transaction Isolation
============================================

Generic batch processor with per-record transaction isolation using savepoints.
Prevents one bad record from corrupting an entire batch.

Following Actor-Network Theory: Each record is treated as an independent
actant whose enrollment in the network can fail without disrupting the
enrollment of other actants in the batch.
"""

import logging
import re
from typing import List, Dict, Any, Callable, Optional
from datetime import datetime
import psycopg2
from psycopg2 import sql

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Process data in batches with fine-grained error handling.
    
    In Information Reconstructionism terms, this ensures that
    zero-propagation from one failed record doesn't affect the
    entire batch - maintaining maximum information preservation.
    """
    
    def __init__(self, 
                 db_connection: Optional[Any] = None,
                 batch_size: int = 1000,
                 use_savepoints: bool = True):
        """
        Initialize batch processor.
        
        Args:
            db_connection: Database connection (optional)
            batch_size: Size of batches to process
            use_savepoints: Use savepoints for per-record isolation
        """
        self.conn = db_connection
        self.batch_size = batch_size
        self.use_savepoints = use_savepoints
        
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'batches': 0,
            'errors': []
        }
    
    def process_items(self,
                     items: List[Any],
                     process_func: Callable[[Any], bool],
                     commit_interval: int = None) -> Dict[str, Any]:
        """
        Process items in batches.
        
        Args:
            items: List of items to process
            process_func: Function to process each item
            commit_interval: Commit after N successful items
            
        Returns:
            Processing statistics
        """
        if commit_interval is None:
            commit_interval = self.batch_size
        
        start_time = datetime.now()
        
        for batch_start in range(0, len(items), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(items))
            batch = items[batch_start:batch_end]
            
            self._process_batch(batch, process_func)
            self.stats['batches'] += 1
            
            # Commit if we have a connection and hit interval
            if self.conn and self.stats['successful'] % commit_interval == 0:
                try:
                    self.conn.commit()
                    logger.debug(f"Committed after {self.stats['successful']} successful items")
                except Exception as e:
                    logger.error(f"Commit failed: {e}")
        
        # Final commit
        if self.conn:
            try:
                self.conn.commit()
            except Exception as e:
                logger.error(f"Final commit failed: {e}")
        
        elapsed = (datetime.now() - start_time).total_seconds()
        self.stats['duration_seconds'] = elapsed
        self.stats['items_per_second'] = self.stats['total_processed'] / elapsed if elapsed > 0 else 0
        
        return self.stats
    
    def _process_batch(self, batch: List[Any], process_func: Callable[[Any], bool]):
        """
        Process a single batch with error isolation.
        
        Args:
            batch: Batch of items
            process_func: Function to process each item
        """
        for item in batch:
            self.stats['total_processed'] += 1
            cur = None
            
            try:
                # Create savepoint if using database
                if self.conn and self.use_savepoints:
                    cur = self.conn.cursor()
                    cur.execute("SAVEPOINT item_process")
                
                # Process the item
                success = process_func(item)
                
                if success:
                    # Release savepoint on success
                    if self.conn and self.use_savepoints and cur:
                        cur.execute("RELEASE SAVEPOINT item_process")
                    self.stats['successful'] += 1
                else:
                    # Rollback on failure
                    if self.conn and self.use_savepoints and cur:
                        cur.execute("ROLLBACK TO SAVEPOINT item_process")
                    self.stats['failed'] += 1
                    
            except Exception as e:
                # Rollback on exception
                if self.conn and self.use_savepoints and cur:
                    try:
                        cur.execute("ROLLBACK TO SAVEPOINT item_process")
                    except Exception as rollback_err:
                        logging.warning(f"Failed to rollback savepoint: {rollback_err}")
                        pass  # Savepoint might not exist
                
                self.stats['failed'] += 1
                error_msg = f"Item {self.stats['total_processed']}: {str(e)}"
                self.stats['errors'].append(error_msg)
                
                # Only keep last 100 errors to prevent memory issues
                if len(self.stats['errors']) > 100:
                    self.stats['errors'] = self.stats['errors'][-100:]
                
                logger.debug(f"Failed to process item: {e}")
            finally:
                # Always close cursor if it was created
                if cur:
                    try:
                        cur.close()
                    except Exception as cursor_err:
                        logging.warning(f"Error closing cursor: {cursor_err}")
                        pass  # Ignore errors during cleanup
    
    def get_summary(self) -> str:
        """Get a summary of processing results."""
        return (
            f"Processed {self.stats['total_processed']} items in {self.stats.get('duration_seconds', 0):.2f}s\n"
            f"  Successful: {self.stats['successful']}\n"
            f"  Failed: {self.stats['failed']}\n"
            f"  Rate: {self.stats.get('items_per_second', 0):.2f} items/second\n"
            f"  Batches: {self.stats['batches']}"
        )


class ParallelBatchProcessor(BatchProcessor):
    """
    Batch processor with parallel processing support.
    
    Note: Cannot use savepoints with parallel processing since
    database connections aren't thread-safe.
    """
    
    def __init__(self, 
                 batch_size: int = 1000,
                 num_workers: int = 4):
        """
        Initialize parallel batch processor.
        
        Args:
            batch_size: Size of batches
            num_workers: Number of parallel workers
        """
        super().__init__(db_connection=None, batch_size=batch_size, use_savepoints=False)
        self.num_workers = num_workers
    
    def process_items_parallel(self,
                              items: List[Any],
                              process_func: Callable[[Any], bool]) -> Dict[str, Any]:
        """
        Process items in parallel.
        
        Args:
            items: List of items to process
            process_func: Function to process each item (must be pickleable)
            
        Returns:
            Processing statistics
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from tqdm import tqdm
        
        start_time = datetime.now()
        
        # Split items into chunks for workers
        chunk_size = max(1, len(items) // (self.num_workers * 10))  # 10 chunks per worker
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all chunks
            futures = {
                executor.submit(self._process_chunk, chunk, process_func): i
                for i, chunk in enumerate(chunks)
            }
            
            # Process results as they complete
            with tqdm(total=len(items), desc="Processing") as pbar:
                for future in as_completed(futures):
                    try:
                        chunk_stats = future.result()
                        self.stats['successful'] += chunk_stats['successful']
                        self.stats['failed'] += chunk_stats['failed']
                        self.stats['errors'].extend(chunk_stats['errors'][:10])  # Keep some errors
                        pbar.update(chunk_stats['processed'])
                    except Exception as e:
                        logger.error(f"Chunk processing failed: {e}")
        
        self.stats['total_processed'] = len(items)
        elapsed = (datetime.now() - start_time).total_seconds()
        self.stats['duration_seconds'] = elapsed
        self.stats['items_per_second'] = len(items) / elapsed if elapsed > 0 else 0
        
        return self.stats
    
    @staticmethod
    def _process_chunk(chunk: List[Any], process_func: Callable[[Any], bool]) -> Dict[str, Any]:
        """
        Process a chunk of items (runs in worker process).
        
        Args:
            chunk: Items to process
            process_func: Processing function
            
        Returns:
            Chunk statistics
        """
        stats = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'errors': []
        }
        
        for item in chunk:
            stats['processed'] += 1
            try:
                if process_func(item):
                    stats['successful'] += 1
                else:
                    stats['failed'] += 1
            except Exception as e:
                stats['failed'] += 1
                stats['errors'].append(str(e))
        
        return stats


def validate_identifier(name: str) -> bool:
    """
    Validate a SQL identifier (table or column name).
    
    Args:
        name: Identifier to validate
        
    Returns:
        True if valid, False otherwise
    """
    # SQL identifier pattern: starts with letter or underscore,
    # followed by letters, digits, or underscores
    pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
    return bool(re.match(pattern, name))


def batch_insert_with_savepoints(
    conn,
    table: str,
    records: List[Dict[str, Any]],
    batch_size: int = 1000
) -> Dict[str, int]:
    """
    Batch insert records with per-record savepoint isolation.
    
    Args:
        conn: Database connection
        table: Table name
        records: List of records to insert
        batch_size: Batch size
        
    Returns:
        Statistics dictionary
    """
    # Validate table name to prevent SQL injection
    if not validate_identifier(table):
        raise ValueError(f"Invalid table name: {table}")
    
    processor = BatchProcessor(conn, batch_size=batch_size)
    cur = conn.cursor()
    
    def insert_record(record: Dict[str, Any]) -> bool:
        """Insert a single record."""
        columns = list(record.keys())
        
        # Validate all column names
        for col in columns:
            if not validate_identifier(col):
                raise ValueError(f"Invalid column name: {col}")
        
        values = list(record.values())
        
        # Use psycopg2.sql for safe identifier handling
        query = sql.SQL("""
            INSERT INTO {table} ({columns})
            VALUES ({placeholders})
            ON CONFLICT DO NOTHING
        """).format(
            table=sql.Identifier(table),
            columns=sql.SQL(', ').join(map(sql.Identifier, columns)),
            placeholders=sql.SQL(', ').join(sql.Placeholder() * len(values))
        )
        
        cur.execute(query, values)
        return cur.rowcount > 0
    
    try:
        stats = processor.process_items(records, insert_record)
    finally:
        # Ensure cursor is closed
        cur.close()
    
    return stats