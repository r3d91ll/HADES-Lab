#!/usr/bin/env python3
"""
Unit tests for ArxivSortedWorkflow's unique methods
====================================================

Focused tests for the methods that are unique to the sorted workflow:
1. _preprocess_and_sort_metadata - Sorts by character count
2. _store_processing_order - Stores sorted order with positions
3. _get_resume_position - Gets resume position from sorted order
4. _process_sorted_records - Processes in sorted order with resume
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.workflows.workflow_arxiv_sorted import ArxivSortedWorkflow
from core.tools.arxiv.arxiv_metadata_config import ArxivMetadataConfig


class TestArxivSortedUniqueMethod(unittest.TestCase):
    """Test suite for ArxivSortedWorkflow's unique methods."""

    def setUp(self):
        """
        Prepare test fixtures for ArxivSortedWorkflow unit tests.
        
        Creates:
        - self.test_metadata: a list of four metadata dicts with known abstract lengths (1, 3, 50, 100 characters).
        - a temporary JSONL metadata file at self.temp_file containing one JSON object per line (from self.test_metadata).
        - self.config: an ArxivMetadataConfig pointing to the temp file and configured with batch_size=2, num_workers=1, embedding_batch_size=2, use_gpu=False, resume_from_checkpoint=False, metadata_collection='test_arxiv_papers', embedder_model='sentence-transformers/all-MiniLM-L6-v2', and arango_database='test_db'.
        """
        # Create test metadata with known character counts
        self.test_metadata = [
            {
                "id": "2024.12345",
                "title": "Small Paper",
                "abstract": "ABC",  # 3 chars
                "authors": ["Author A"],
                "categories": ["cs.AI"]
            },
            {
                "id": "2024.67890",
                "title": "Medium Paper",
                "abstract": "A" * 50,  # 50 chars
                "authors": ["Author B"],
                "categories": ["cs.LG"]
            },
            {
                "id": "2024.11111",
                "title": "Large Paper",
                "abstract": "B" * 100,  # 100 chars
                "authors": ["Author C"],
                "categories": ["cs.CV"]
            },
            {
                "id": "2024.99999",
                "title": "Tiny Paper",
                "abstract": "X",  # 1 char
                "authors": ["Author D"],
                "categories": ["cs.NLP"]
            }
        ]

        # Create temp metadata file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        for record in self.test_metadata:
            self.temp_file.write(json.dumps(record) + '\n')
        self.temp_file.close()

        # Create minimal config
        self.config = ArxivMetadataConfig(
            metadata_file=Path(self.temp_file.name),
            batch_size=2,
            num_workers=1,
            embedding_batch_size=2,
            gpu_device='cpu',
            use_gpu=False,
            resume_from_checkpoint=False,
            metadata_collection='test_arxiv_papers',
            embedder_model='sentence-transformers/all-MiniLM-L6-v2',
            arango_database='test_db'
        )

    def tearDown(self):
        """Clean up test fixtures."""
        Path(self.temp_file.name).unlink(missing_ok=True)

    @patch('core.workflows.workflow_arxiv_sorted.DatabaseFactory')
    @patch('core.workflows.workflow_arxiv_sorted.EmbedderFactory')
    def test_preprocess_and_sort_metadata(self, mock_embedder_factory, mock_db_factory):
        """Test _preprocess_and_sort_metadata method - the core sorting logic."""
        # Setup minimal mocks
        mock_db = MagicMock()
        mock_db.has_collection.return_value = True
        mock_db_factory.get_arango.return_value = mock_db

        mock_embedder = MagicMock()
        mock_embedder_factory.create_embedder.return_value = mock_embedder

        # Create workflow
        workflow = ArxivSortedWorkflow(self.config)

        # Test the sorting method
        sorted_records = workflow._preprocess_and_sort_metadata()

        # Verify correct sorting order (1, 3, 50, 100 chars)
        expected_order = ["2024.99999", "2024.12345", "2024.67890", "2024.11111"]
        actual_order = [r['id'] for r in sorted_records]
        self.assertEqual(actual_order, expected_order)

        # Verify character counts are correct
        self.assertEqual(sorted_records[0]['abstract_length'], 1)    # X
        self.assertEqual(sorted_records[1]['abstract_length'], 3)    # ABC
        self.assertEqual(sorted_records[2]['abstract_length'], 50)   # A*50
        self.assertEqual(sorted_records[3]['abstract_length'], 100)  # B*100

        # Verify token_count field matches (for compatibility)
        for record in sorted_records:
            self.assertEqual(record['token_count'], record['abstract_length'])

        # Verify all records have required fields
        for record in sorted_records:
            self.assertIn('id', record)
            self.assertIn('record', record)
            self.assertIn('abstract_length', record)
            self.assertIn('token_count', record)

    @patch('core.workflows.workflow_arxiv_sorted.DatabaseFactory')
    @patch('core.workflows.workflow_arxiv_sorted.EmbedderFactory')
    def test_store_processing_order(self, mock_embedder_factory, mock_db_factory):
        """Test _store_processing_order method - stores sorted order with positions."""
        # Setup mocks
        mock_db = MagicMock()
        mock_order_collection = MagicMock()

        # Mock collection access
        mock_db.has_collection.return_value = False  # Force creation
        mock_db.create_collection.return_value = mock_order_collection
        mock_db.__getitem__.return_value = mock_order_collection
        mock_db_factory.get_arango.return_value = mock_db

        mock_embedder = MagicMock()
        mock_embedder_factory.create_embedder.return_value = mock_embedder

        # Create workflow and manually set db
        workflow = ArxivSortedWorkflow(self.config)
        workflow.db = mock_db  # Manually set the db since we're not calling _initialize_components

        # Create pre-sorted records
        sorted_records = [
            {'id': '2024.99999', 'abstract_length': 1, 'token_count': 1, 'record': self.test_metadata[3]},
            {'id': '2024.12345', 'abstract_length': 3, 'token_count': 3, 'record': self.test_metadata[0]},
            {'id': '2024.67890', 'abstract_length': 50, 'token_count': 50, 'record': self.test_metadata[1]},
            {'id': '2024.11111', 'abstract_length': 100, 'token_count': 100, 'record': self.test_metadata[2]},
        ]

        # Test storing the order
        workflow._store_processing_order(sorted_records)

        # Verify collection was created
        mock_db.create_collection.assert_called_once_with('arxiv_processing_order')

        # Verify truncate was called (since not resuming)
        mock_order_collection.truncate.assert_called_once()

        # Verify documents were stored with correct structure
        import_call = mock_order_collection.import_bulk.call_args_list[0]
        stored_docs = import_call[0][0]  # First argument of first call

        # Check first document structure
        self.assertEqual(stored_docs[0]['_key'], '2024_99999')
        self.assertEqual(stored_docs[0]['arxiv_id'], '2024.99999')
        self.assertEqual(stored_docs[0]['position'], 0)
        self.assertEqual(stored_docs[0]['abstract_length'], 1)

        # Check positions are sequential
        for i, doc in enumerate(stored_docs):
            self.assertEqual(doc['position'], i)

    @patch('core.workflows.workflow_arxiv_sorted.DatabaseFactory')
    @patch('core.workflows.workflow_arxiv_sorted.EmbedderFactory')
    def test_store_processing_order_with_resume(self, mock_embedder_factory, mock_db_factory):
        """Test _store_processing_order when resuming (should not truncate)."""
        # Setup mocks
        mock_db = MagicMock()
        mock_order_collection = MagicMock()

        mock_db.has_collection.return_value = True  # Collection exists
        mock_db.__getitem__.return_value = mock_order_collection
        mock_db_factory.get_arango.return_value = mock_db

        mock_embedder = MagicMock()
        mock_embedder_factory.create_embedder.return_value = mock_embedder

        # Enable resume in config
        self.config.resume_from_checkpoint = True

        # Create workflow and manually set db
        workflow = ArxivSortedWorkflow(self.config)
        workflow.db = mock_db  # Manually set the db

        # Create sorted records
        sorted_records = [
            {'id': '2024.99999', 'abstract_length': 1, 'token_count': 1, 'record': {}},
        ]

        # Test storing the order
        workflow._store_processing_order(sorted_records)

        # Verify collection was NOT created (already exists)
        mock_db.create_collection.assert_not_called()

        # Verify truncate was NOT called (resuming)
        mock_order_collection.truncate.assert_not_called()

        # Verify documents were still stored
        mock_order_collection.import_bulk.assert_called_once()

    @patch('core.workflows.workflow_arxiv_sorted.DatabaseFactory')
    @patch('core.workflows.workflow_arxiv_sorted.EmbedderFactory')
    def test_get_resume_position(self, mock_embedder_factory, mock_db_factory):
        """Test _get_resume_position method - finds last processed position."""
        # Setup mocks
        mock_db = MagicMock()

        # Mock AQL query result - last processed position is 5
        mock_cursor = iter([5])
        mock_db.aql.execute.return_value = mock_cursor
        mock_db.has_collection.return_value = True
        mock_db_factory.get_arango.return_value = mock_db

        mock_embedder = MagicMock()
        mock_embedder_factory.create_embedder.return_value = mock_embedder

        # Create workflow and manually set db
        workflow = ArxivSortedWorkflow(self.config)
        workflow.db = mock_db  # Manually set the db

        # Test getting resume position
        position = workflow._get_resume_position()

        # Should return position 6 (last + 1)
        self.assertEqual(position, 6)

        # Verify correct query was executed
        query_call = mock_db.aql.execute.call_args[0][0]
        self.assertIn('MAX(doc.position)', query_call)
        self.assertIn('arxiv_processing_order', query_call)
        self.assertIn('JOIN', query_call)  # Should join with papers collection

    @patch('core.workflows.workflow_arxiv_sorted.DatabaseFactory')
    @patch('core.workflows.workflow_arxiv_sorted.EmbedderFactory')
    def test_get_resume_position_no_progress(self, mock_embedder_factory, mock_db_factory):
        """Test _get_resume_position when no documents have been processed."""
        # Setup mocks
        mock_db = MagicMock()

        # Mock AQL query result - no documents processed
        mock_cursor = iter([None])
        mock_db.aql.execute.return_value = mock_cursor
        mock_db.has_collection.return_value = True
        mock_db_factory.get_arango.return_value = mock_db

        mock_embedder = MagicMock()
        mock_embedder_factory.create_embedder.return_value = mock_embedder

        # Create workflow and manually set db
        workflow = ArxivSortedWorkflow(self.config)
        workflow.db = mock_db  # Manually set the db

        # Test getting resume position
        position = workflow._get_resume_position()

        # Should return 0 (start from beginning)
        self.assertEqual(position, 0)

    @patch('core.workflows.workflow_arxiv_sorted.DatabaseFactory')
    @patch('core.workflows.workflow_arxiv_sorted.EmbedderFactory')
    def test_get_resume_position_with_error(self, mock_embedder_factory, mock_db_factory):
        """Test _get_resume_position handles errors gracefully."""
        # Setup mocks
        mock_db = MagicMock()

        # Mock AQL query to raise an exception
        mock_db.aql.execute.side_effect = Exception("Database error")
        mock_db.has_collection.return_value = True
        mock_db_factory.get_arango.return_value = mock_db

        mock_embedder = MagicMock()
        mock_embedder_factory.create_embedder.return_value = mock_embedder

        # Create workflow and manually set db
        workflow = ArxivSortedWorkflow(self.config)
        workflow.db = mock_db  # Manually set the db

        # Test getting resume position with error
        position = workflow._get_resume_position()

        # Should return 0 on error
        self.assertEqual(position, 0)

    @patch('core.workflows.workflow_arxiv_sorted.DatabaseFactory')
    @patch('core.workflows.workflow_arxiv_sorted.EmbedderFactory')
    def test_process_sorted_records_with_resume(self, mock_embedder_factory, mock_db_factory):
        """Test _process_sorted_records skips already processed records when resuming."""
        # Setup mocks
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_order_collection = MagicMock()

        # Mock resume position query - already processed first 2 records
        mock_cursor = iter([1])  # Last processed position is 1
        mock_db.aql.execute.return_value = mock_cursor

        mock_db.has_collection.return_value = True
        mock_db.collection.side_effect = lambda name: (
            mock_collection if name == 'test_arxiv_papers'
            else mock_order_collection if name == 'arxiv_processing_order'
            else None
        )
        mock_db_factory.get_arango.return_value = mock_db

        mock_embedder = MagicMock()
        mock_embedder.embed_batch.return_value = [[0.1] * 768]  # Mock embeddings
        mock_embedder_factory.create_embedder.return_value = mock_embedder

        # Enable resume
        self.config.resume_from_checkpoint = True
        self.config.batch_size = 1  # Process one at a time for easier testing

        # Create workflow and manually set db
        workflow = ArxivSortedWorkflow(self.config)
        workflow.db = mock_db  # Manually set the db

        # Create sorted records with token_count field
        sorted_records = [
            {'id': '2024.99999', 'abstract_length': 1, 'token_count': 1, 'record': self.test_metadata[3]},
            {'id': '2024.12345', 'abstract_length': 3, 'token_count': 3, 'record': self.test_metadata[0]},
            {'id': '2024.67890', 'abstract_length': 50, 'token_count': 50, 'record': self.test_metadata[1]},
            {'id': '2024.11111', 'abstract_length': 100, 'token_count': 100, 'record': self.test_metadata[2]},
        ]

        # Process sorted records
        workflow._process_sorted_records(sorted_records)

        # Should skip first 2 records (positions 0 and 1)
        self.assertEqual(workflow.skipped_count, 2)

        # Should only process last 2 records
        # Check that embed_batch was called with the correct abstracts
        embed_calls = mock_embedder.embed_batch.call_args_list

        # First batch should be record at position 2 (50 chars of 'A')
        first_batch_texts = embed_calls[0][0][0]  # First argument of first call
        self.assertEqual(len(first_batch_texts[0]), 50)  # 50 'A's

        # Second batch should be record at position 3 (100 chars of 'B')
        second_batch_texts = embed_calls[1][0][0]
        self.assertEqual(len(second_batch_texts[0]), 100)  # 100 'B's

    @patch('core.workflows.workflow_arxiv_sorted.DatabaseFactory')
    @patch('core.workflows.workflow_arxiv_sorted.EmbedderFactory')
    def test_process_sorted_records_batch_accumulation(self, mock_embedder_factory, mock_db_factory):
        """Test _process_sorted_records accumulates records into batches correctly."""
        # Setup mocks
        mock_db = MagicMock()
        mock_collection = MagicMock()

        mock_db.has_collection.return_value = True
        mock_db.collection.return_value = mock_collection
        mock_db_factory.get_arango.return_value = mock_db

        mock_embedder = MagicMock()
        mock_embedder.embed_batch.return_value = [[0.1] * 768] * 2  # Mock embeddings for batch of 2
        mock_embedder_factory.create_embedder.return_value = mock_embedder

        # Set batch size to 2
        self.config.batch_size = 2
        self.config.resume_from_checkpoint = False

        # Create workflow and manually set db
        workflow = ArxivSortedWorkflow(self.config)
        workflow.db = mock_db  # Manually set the db

        # Create 4 sorted records
        sorted_records = [
            {'id': f'2024.{i:05d}', 'abstract_length': i+1, 'record': {
                'id': f'2024.{i:05d}',
                'abstract': 'X' * (i+1),
                'title': f'Paper {i}',
                'authors': [],
                'categories': ['cs.AI']
            }} for i in range(4)
        ]

        # Process sorted records
        workflow._process_sorted_records(sorted_records)

        # Should have made 2 batches of 2
        self.assertEqual(mock_embedder.embed_batch.call_count, 2)
        self.assertEqual(mock_collection.insert_many.call_count, 2)

        # Verify batch sizes
        first_insert = mock_collection.insert_many.call_args_list[0][0][0]
        second_insert = mock_collection.insert_many.call_args_list[1][0][0]

        self.assertEqual(len(first_insert), 2)  # First batch has 2 records
        self.assertEqual(len(second_insert), 2)  # Second batch has 2 records


class TestSortingEdgeCases(unittest.TestCase):
    """Test edge cases in sorting functionality."""

    @patch('core.workflows.workflow_arxiv_sorted.DatabaseFactory')
    @patch('core.workflows.workflow_arxiv_sorted.EmbedderFactory')
    def test_empty_abstracts_sorted_first(self, mock_embedder_factory, mock_db_factory):
        """Test that empty abstracts are sorted first (0 length)."""
        # Create test data with empty abstract
        test_metadata = [
            {"id": "1", "abstract": "Some text", "title": "A", "authors": [], "categories": []},
            {"id": "2", "abstract": "", "title": "B", "authors": [], "categories": []},  # Empty
            {"id": "3", "abstract": "X", "title": "C", "authors": [], "categories": []},
        ]

        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        for record in test_metadata:
            temp_file.write(json.dumps(record) + '\n')
        temp_file.close()

        # Setup mocks
        mock_db = MagicMock()
        mock_db.has_collection.return_value = True
        mock_db_factory.get_arango.return_value = mock_db

        mock_embedder = MagicMock()
        mock_embedder_factory.create_embedder.return_value = mock_embedder

        # Create config
        config = ArxivMetadataConfig(
            metadata_file=Path(temp_file.name),
            batch_size=10,
            num_workers=1,
            gpu_device='cpu',
            use_gpu=False,
            arango_database='test_db'
        )

        # Create workflow and sort
        workflow = ArxivSortedWorkflow(config)
        sorted_records = workflow._preprocess_and_sort_metadata()

        # Verify empty abstract is first
        self.assertEqual(sorted_records[0]['id'], "2")
        self.assertEqual(sorted_records[0]['abstract_length'], 0)
        self.assertEqual(sorted_records[1]['id'], "3")  # Single char
        self.assertEqual(sorted_records[2]['id'], "1")  # Longest

        # Cleanup
        Path(temp_file.name).unlink()

    @patch('core.workflows.workflow_arxiv_sorted.DatabaseFactory')
    @patch('core.workflows.workflow_arxiv_sorted.EmbedderFactory')
    def test_missing_abstract_field(self, mock_embedder_factory, mock_db_factory):
        """Test handling of records with missing abstract field."""
        # Create test data with missing abstract
        test_metadata = [
            {"id": "1", "abstract": "Text", "title": "A", "authors": [], "categories": []},
            {"id": "2", "title": "B", "authors": [], "categories": []},  # No abstract field
            {"id": "3", "abstract": "X", "title": "C", "authors": [], "categories": []},
        ]

        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        for record in test_metadata:
            temp_file.write(json.dumps(record) + '\n')
        temp_file.close()

        # Setup mocks
        mock_db = MagicMock()
        mock_db.has_collection.return_value = True
        mock_db_factory.get_arango.return_value = mock_db

        mock_embedder = MagicMock()
        mock_embedder_factory.create_embedder.return_value = mock_embedder

        # Create config
        config = ArxivMetadataConfig(
            metadata_file=Path(temp_file.name),
            batch_size=10,
            num_workers=1,
            gpu_device='cpu',
            use_gpu=False,
            arango_database='test_db'
        )

        # Create workflow and sort
        workflow = ArxivSortedWorkflow(config)
        sorted_records = workflow._preprocess_and_sort_metadata()

        # Verify missing abstract is treated as empty (0 length)
        self.assertEqual(sorted_records[0]['id'], "2")
        self.assertEqual(sorted_records[0]['abstract_length'], 0)

        # Cleanup
        Path(temp_file.name).unlink()


if __name__ == '__main__':
    unittest.main(verbosity=2)