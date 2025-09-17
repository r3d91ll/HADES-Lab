#!/usr/bin/env python3
"""
Unit tests for ArxivSortedWorkflow
===================================

Tests the size-sorted processing workflow including:
- Character count-based sorting
- Processing order storage
- Checkpoint/resume functionality
- Batch processing logic
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


class TestArxivSortedWorkflow(unittest.TestCase):
    """Test suite for ArxivSortedWorkflow."""

    def setUp(self):
        """
        Prepare test fixtures for ArxivSortedWorkflow tests.
        
        Creates a small in-memory dataset of four synthetic paper records with varying abstract lengths, writes them to a temporary JSONL file, and constructs an ArxivMetadataConfig pointing to that file. The config uses batch_size=2, embedding_batch_size=2, single worker, CPU-only embedding, resume_from_checkpoint=False, and a test arango database/collection names.
        """
        # Create test metadata
        self.test_metadata = [
            {
                "id": "paper1",
                "title": "Short Paper",
                "abstract": "Short abstract.",  # 15 chars
                "authors": ["Author A"],
                "categories": ["cs.AI"]
            },
            {
                "id": "paper2",
                "title": "Medium Paper",
                "abstract": "This is a medium length abstract with more content.",  # 51 chars
                "authors": ["Author B"],
                "categories": ["cs.LG"]
            },
            {
                "id": "paper3",
                "title": "Long Paper",
                "abstract": "This is a very long abstract with lots of detailed content that goes on and on with many technical details and explanations about the research methodology and findings.",  # 168 chars
                "authors": ["Author C"],
                "categories": ["cs.CV"]
            },
            {
                "id": "paper4",
                "title": "Tiny Paper",
                "abstract": "Tiny.",  # 5 chars
                "authors": ["Author D"],
                "categories": ["cs.NLP"]
            }
        ]

        # Create temp metadata file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        for record in self.test_metadata:
            self.temp_file.write(json.dumps(record) + '\n')
        self.temp_file.close()

        # Mock configuration
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
        """
        Remove the temporary metadata file created for tests.
        
        This teardown method deletes the temporary JSONL file referenced by self.temp_file.
        If the file is already absent, deletion is ignored (no exception raised).
        """
        Path(self.temp_file.name).unlink(missing_ok=True)

    @patch('core.workflows.workflow_arxiv_sorted.DatabaseFactory')
    @patch('core.workflows.workflow_arxiv_sorted.EmbedderFactory')
    def test_sorting_by_character_count(self, mock_embedder_factory, mock_db_factory):
        """Test that abstracts are sorted by character count."""
        # Setup mocks
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_order_collection = MagicMock()

        mock_db.has_collection.side_effect = lambda name: name == 'test_arxiv_papers'
        mock_db.collection.side_effect = lambda name: (
            mock_collection if name == 'test_arxiv_papers'
            else mock_order_collection if name == 'arxiv_processing_order'
            else None
        )
        mock_db.create_collection.return_value = mock_order_collection
        mock_db_factory.get_arango.return_value = mock_db

        mock_embedder = MagicMock()
        mock_embedder.embed_batch.return_value = [[0.1] * 768] * 2  # Mock embeddings
        mock_embedder_factory.create_embedder.return_value = mock_embedder

        # Create workflow
        workflow = ArxivSortedWorkflow(self.config)

        # Call the sorting method
        sorted_records = workflow._preprocess_and_sort_metadata()

        # Verify sorting order (shortest to longest)
        expected_order = ["paper4", "paper1", "paper2", "paper3"]  # 5, 15, 52, 173 chars
        actual_order = [r['id'] for r in sorted_records]

        self.assertEqual(actual_order, expected_order)

        # Verify character counts are stored correctly
        self.assertEqual(sorted_records[0]['abstract_length'], 5)   # paper4
        self.assertEqual(sorted_records[1]['abstract_length'], 15)  # paper1
        self.assertEqual(sorted_records[2]['abstract_length'], 51)  # paper2
        self.assertEqual(sorted_records[3]['abstract_length'], 168) # paper3

    @patch('core.workflows.workflow_arxiv_sorted.DatabaseFactory')
    @patch('core.workflows.workflow_arxiv_sorted.EmbedderFactory')
    def test_processing_order_storage(self, mock_embedder_factory, mock_db_factory):
        """Test that processing order is stored in database."""
        # Setup mocks
        mock_db = MagicMock()
        mock_order_collection = MagicMock()

        mock_db.has_collection.return_value = False
        mock_db.create_collection.return_value = mock_order_collection
        mock_db.collection.return_value = mock_order_collection
        mock_db_factory.get_arango.return_value = mock_db

        mock_embedder = MagicMock()
        mock_embedder_factory.create_embedder.return_value = mock_embedder

        # Create workflow
        workflow = ArxivSortedWorkflow(self.config)

        # Create sorted records
        sorted_records = [
            {'id': 'paper4', 'abstract_length': 5, 'record': self.test_metadata[3]},
            {'id': 'paper1', 'abstract_length': 15, 'record': self.test_metadata[0]},
        ]

        # Store processing order
        workflow._store_processing_order(sorted_records)

        # Verify collection was created
        mock_db.create_collection.assert_called_once_with('arxiv_processing_order')

        # Verify documents were inserted with correct positions
        expected_calls = [
            call({
                '_key': 'paper4',
                'paper_id': 'paper4',
                'position': 0,
                'abstract_length': 5
            }),
            call({
                '_key': 'paper1',
                'paper_id': 'paper1',
                'position': 1,
                'abstract_length': 15
            })
        ]

        # Check that insert was called correctly
        self.assertEqual(mock_order_collection.insert.call_count, 2)
        actual_calls = mock_order_collection.insert.call_args_list
        for expected, actual in zip(expected_calls, actual_calls):
            self.assertEqual(actual, expected)

    @patch('core.workflows.workflow_arxiv_sorted.DatabaseFactory')
    @patch('core.workflows.workflow_arxiv_sorted.EmbedderFactory')
    def test_resume_from_checkpoint(self, mock_embedder_factory, mock_db_factory):
        """Test resume functionality from checkpoint."""
        # Setup mocks
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_order_collection = MagicMock()

        # Mock AQL query for resume position
        mock_cursor = iter([2])  # Resume from position 2
        mock_db.aql.execute.return_value = mock_cursor

        mock_db.has_collection.return_value = True
        mock_db.collection.side_effect = lambda name: (
            mock_collection if name == 'test_arxiv_papers'
            else mock_order_collection if name == 'arxiv_processing_order'
            else None
        )
        mock_db_factory.get_arango.return_value = mock_db

        mock_embedder = MagicMock()
        mock_embedder_factory.create_embedder.return_value = mock_embedder

        # Enable resume in config
        self.config.resume_from_checkpoint = True

        # Create workflow
        workflow = ArxivSortedWorkflow(self.config)

        # Get resume position
        position = workflow._get_resume_position()

        # Should resume from position 3 (last completed + 1)
        self.assertEqual(position, 3)

        # Verify query was executed
        mock_db.aql.execute.assert_called_once()
        query_call = mock_db.aql.execute.call_args[0][0]
        self.assertIn('MAX(doc.position)', query_call)
        self.assertIn('arxiv_processing_order', query_call)

    @patch('core.workflows.workflow_arxiv_sorted.DatabaseFactory')
    @patch('core.workflows.workflow_arxiv_sorted.EmbedderFactory')
    def test_batch_processing(self, mock_embedder_factory, mock_db_factory):
        """Test that records are processed in correct batches."""
        # Setup mocks
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_order_collection = MagicMock()

        mock_db.has_collection.return_value = True
        mock_db.collection.side_effect = lambda name: (
            mock_collection if name == 'test_arxiv_papers'
            else mock_order_collection if name == 'arxiv_processing_order'
            else None
        )
        mock_db_factory.get_arango.return_value = mock_db

        mock_embedder = MagicMock()
        mock_embedder.embed_batch.return_value = [[0.1] * 768] * 2
        mock_embedder_class.return_value = mock_embedder

        # Track batch inserts
        batch_inserts = []
        mock_collection.insert_many.side_effect = lambda docs: batch_inserts.append(len(docs))

        # Create workflow with batch size 2
        workflow = ArxivSortedWorkflow(self.config)

        # Process sorted records
        sorted_records = workflow._preprocess_and_sort_metadata()
        workflow._process_sorted_records(sorted_records)

        # Should have processed 2 batches (4 records with batch size 2)
        self.assertEqual(len(batch_inserts), 2)
        self.assertEqual(batch_inserts[0], 2)  # First batch
        self.assertEqual(batch_inserts[1], 2)  # Second batch

        # Verify embedder was called with correct batch sizes
        self.assertEqual(mock_embedder.embed_batch.call_count, 2)

    @patch('core.workflows.workflow_arxiv_sorted.DatabaseFactory')
    @patch('core.workflows.workflow_arxiv_sorted.EmbedderFactory')
    def test_character_count_not_token_estimation(self, mock_embedder_factory, mock_db_factory):
        """Verify using character count, not token estimation."""
        # Setup mocks
        mock_db = MagicMock()
        mock_db.has_collection.return_value = True
        mock_db_factory.get_arango.return_value = mock_db

        mock_embedder = MagicMock()
        mock_embedder_factory.create_embedder.return_value = mock_embedder

        # Create workflow
        workflow = ArxivSortedWorkflow(self.config)

        # Process and sort
        sorted_records = workflow._preprocess_and_sort_metadata()

        # Verify character counts match actual string lengths
        for record in sorted_records:
            original = next(m for m in self.test_metadata if m['id'] == record['id'])
            actual_length = len(original['abstract'])

            # Should be exact character count, not divided by 4
            self.assertEqual(record['abstract_length'], actual_length)
            self.assertEqual(record['token_count'], actual_length)  # Field kept for compatibility

            # Ensure we're not doing token estimation
            self.assertNotEqual(record['abstract_length'], actual_length // 4)

    @patch('core.workflows.workflow_arxiv_sorted.DatabaseFactory')
    @patch('core.workflows.workflow_arxiv_sorted.EmbedderFactory')
    def test_empty_abstract_handling(self, mock_embedder_factory, mock_db_factory):
        """Test handling of papers with empty abstracts."""
        # Add paper with empty abstract
        self.test_metadata.append({
            "id": "paper5",
            "title": "No Abstract Paper",
            "abstract": "",  # Empty abstract
            "authors": ["Author E"],
            "categories": ["cs.AI"]
        })

        # Rewrite temp file
        Path(self.temp_file.name).unlink()
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        for record in self.test_metadata:
            self.temp_file.write(json.dumps(record) + '\n')
        self.temp_file.close()
        self.config.metadata_file = Path(self.temp_file.name)

        # Setup mocks
        mock_db = MagicMock()
        mock_db.has_collection.return_value = True
        mock_db_factory.get_arango.return_value = mock_db

        mock_embedder = MagicMock()
        mock_embedder_factory.create_embedder.return_value = mock_embedder

        # Create workflow
        workflow = ArxivSortedWorkflow(self.config)

        # Process and sort
        sorted_records = workflow._preprocess_and_sort_metadata()

        # Empty abstract should be first (0 chars)
        self.assertEqual(sorted_records[0]['id'], 'paper5')
        self.assertEqual(sorted_records[0]['abstract_length'], 0)

    @patch('core.workflows.workflow_arxiv_sorted.DatabaseFactory')
    @patch('core.workflows.workflow_arxiv_sorted.EmbedderFactory')
    def test_logging_metrics(self, mock_embedder_factory, mock_db_factory):
        """Test that workflow logs appropriate metrics."""
        # Setup mocks with logging capture
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_order_collection = MagicMock()

        mock_db.has_collection.return_value = True
        mock_db.collection.side_effect = lambda name: (
            mock_collection if name == 'test_arxiv_papers'
            else mock_order_collection if name == 'arxiv_processing_order'
            else None
        )
        mock_db_factory.get_arango.return_value = mock_db

        mock_embedder = MagicMock()
        mock_embedder.embed_batch.return_value = [[0.1] * 768] * 2
        mock_embedder_class.return_value = mock_embedder

        # Create workflow
        workflow = ArxivSortedWorkflow(self.config)

        # Capture log calls
        with patch.object(workflow.logger, 'info') as mock_log_info:
            sorted_records = workflow._preprocess_and_sort_metadata()

            # Check for sorting metrics log
            sorting_log_calls = [
                call for call in mock_log_info.call_args_list
                if 'sorting_metrics' in str(call)
            ]

            self.assertTrue(len(sorting_log_calls) > 0, "Should log sorting metrics")

            # Verify metrics content
            if sorting_log_calls:
                metrics_call = sorting_log_calls[0]
                # Check that min, max, avg are logged
                self.assertIn('min_size', str(metrics_call))
                self.assertIn('max_size', str(metrics_call))
                self.assertIn('avg_size', str(metrics_call))


class TestIntegrationWorkflow(unittest.TestCase):
    """Integration tests for the complete workflow."""

    @patch('core.workflows.workflow_arxiv_sorted.DatabaseFactory')
    @patch('core.workflows.workflow_arxiv_sorted.EmbedderFactory')
    def test_end_to_end_processing(self, mock_embedder_factory, mock_db_factory):
        """Test complete workflow from file to database."""
        # Create larger test dataset
        test_data = []
        for i in range(10):
            test_data.append({
                "id": f"paper_{i}",
                "title": f"Paper {i}",
                "abstract": "x" * (i * 20 + 10),  # Varying lengths
                "authors": [f"Author {i}"],
                "categories": ["cs.AI"]
            })

        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        for record in test_data:
            temp_file.write(json.dumps(record) + '\n')
        temp_file.close()

        # Setup mocks
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_order_collection = MagicMock()

        mock_db.has_collection.return_value = False
        mock_db.create_collection.return_value = mock_order_collection
        mock_db.collection.side_effect = lambda name: (
            mock_collection if name == 'test_papers'
            else mock_order_collection if name == 'arxiv_processing_order'
            else None
        )
        mock_db_factory.get_arango.return_value = mock_db

        mock_embedder = MagicMock()
        mock_embedder.embed_batch.return_value = [[0.1] * 768] * 5
        mock_embedder_factory.create_embedder.return_value = mock_embedder

        # Configure workflow
        config = ArxivMetadataConfig(
            metadata_file=Path(temp_file.name),
            batch_size=5,
            num_workers=1,
            embedding_batch_size=5,
            gpu_device='cpu',
            use_gpu=False,
            resume_from_checkpoint=False,
            metadata_collection='test_papers',
            embedder_model='sentence-transformers/all-MiniLM-L6-v2',
            arango_database='test_db'
        )

        # Run workflow
        workflow = ArxivSortedWorkflow(config)
        workflow.run()

        # Verify processing order collection was created
        mock_db.create_collection.assert_called_with('arxiv_processing_order')

        # Verify all papers were processed
        self.assertEqual(mock_collection.insert_many.call_count, 2)  # 10 papers / batch_size 5

        # Verify papers were processed in size order
        order_inserts = mock_order_collection.insert.call_args_list
        positions = [call[0][0]['position'] for call in order_inserts]
        self.assertEqual(positions, list(range(10)))  # Should be 0-9 in order

        # Clean up
        Path(temp_file.name).unlink(missing_ok=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)