#!/usr/bin/env python3
"""Download a single ArXiv paper and generate an LLM-ready artifact via Docling.

The workflow keeps the late chunking guarantee mandated by the Conveyance
Framework, ensuring the produced payload retains full-document context while
avoiding repeated extraction/embedding work.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

from core.workflows.workflow_base import WorkflowBase, WorkflowConfig, WorkflowResult
from core.processors.document_processor import (
    DocumentProcessor,
    ProcessingConfig,
    ProcessingResult,
)

_ARXIV_IMPORT_ERROR: Optional[Exception] = None
try:  # pragma: no cover - exercised indirectly
    from core.tools.arxiv.arxiv_api_client import ArXivAPIClient, DownloadResult, ArXivMetadata
except ImportError as exc:  # pragma: no cover - optional dependency path
    ArXivAPIClient = None  # type: ignore[assignment]
    DownloadResult = None  # type: ignore[assignment]
    ArXivMetadata = None  # type: ignore[assignment]
    _ARXIV_IMPORT_ERROR = exc

logger = logging.getLogger(__name__)


class ArxivSinglePDFWorkflow(WorkflowBase):
    """Execute the single-paper ArXiv pipeline with Docling extraction and late chunking.

    The workflow accepts either an ArXiv identifier or canonical PDF URL, downloads the
    document through the HTTP client, and routes it through :class:`DocumentProcessor`
    (Docling + embedder) so the resulting payload preserves full context for LLM
    consumption. Outputs are serialized to JSON together with the captured metadata so
    downstream agents can reload the full conveyance bundle without re-running the
    expensive steps.
    """

    def __init__(
        self,
        config: Optional[WorkflowConfig] = None,
        processing_config: Optional[ProcessingConfig] = None,
        api_client: Optional[ArXivAPIClient] = None,
        document_processor: Optional[DocumentProcessor] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        wf_config = config or WorkflowConfig(
            name="arxiv_single_pdf",
            batch_size=1,
            num_workers=1,
            checkpoint_enabled=False,
        )
        super().__init__(wf_config)

        self.processing_config = processing_config or ProcessingConfig()
        if self.processing_config.chunking_strategy != "late":
            logger.info("Overriding chunking strategy to 'late' for ArXiv workflow")
            self.processing_config.chunking_strategy = "late"

        self.document_processor = document_processor or DocumentProcessor(self.processing_config)

        if api_client is not None:
            self.api_client = api_client
        else:
            if ArXivAPIClient is None:
                raise ImportError(
                    "ArXivAPIClient dependency is unavailable. Ensure optional dependencies are installed."
                ) from _ARXIV_IMPORT_ERROR
            self.api_client = ArXivAPIClient()

        self.download_root = self.config.staging_path / "arxiv_single_pdf" / "downloads"
        self.download_root.mkdir(parents=True, exist_ok=True)

        self.output_root = output_dir or (self.config.staging_path / "arxiv_single_pdf" / "outputs")
        self.output_root.mkdir(parents=True, exist_ok=True)

    def validate_inputs(
        self,
        *,
        arxiv_id: Optional[str] = None,
        arxiv_url: Optional[str] = None,
        **_: Any,
    ) -> bool:
        identifier = arxiv_id or arxiv_url
        if not identifier:
            logger.error("No ArXiv identifier provided")
            return False

        normalized = self._normalize_arxiv_identifier(arxiv_id, arxiv_url)
        if not normalized:
            return False

        if not self.api_client.validate_arxiv_id(normalized):
            logger.error("ArXiv ID failed validation: %s", normalized)
            return False

        return True

    def execute(
        self,
        *,
        arxiv_id: Optional[str] = None,
        arxiv_url: Optional[str] = None,
        force_download: bool = False,
    ) -> WorkflowResult:
        start_time = datetime.now(timezone.utc)
        normalized_id = self._normalize_arxiv_identifier(arxiv_id, arxiv_url)

        if not normalized_id:
            return self._result_for_failure(
                start_time,
                error_message="Invalid ArXiv identifier",
            )

        if not self.api_client.validate_arxiv_id(normalized_id):
            return self._result_for_failure(
                start_time,
                arxiv_id=normalized_id,
                error_message="ArXiv ID validation failed",
            )

        download_result = self.api_client.download_paper(
            normalized_id,
            pdf_dir=self.download_root / "pdf",
            latex_dir=None,
            force=force_download,
        )

        if not download_result.success or not download_result.pdf_path:
            return self._result_for_failure(
                start_time,
                arxiv_id=normalized_id,
                error_message=download_result.error_message or "Download failed",
            )

        processing_result = self._run_processor(download_result, normalized_id)

        output_payload = self._build_output_payload(processing_result, download_result)
        output_path = self._persist_payload(normalized_id, output_payload)

        end_time = datetime.now(timezone.utc)
        success = processing_result.success

        metadata: Dict[str, Any] = {
            "arxiv_id": normalized_id,
            "output_path": str(output_path),
            "pdf_path": str(download_result.pdf_path),
            "processing_success": success,
        }

        if download_result.metadata:
            metadata.update(self._serialize_metadata(download_result.metadata))

        errors = processing_result.errors if not success else []

        return WorkflowResult(
            workflow_name=self.name,
            success=success,
            items_processed=1 if success else 0,
            items_failed=0 if success else 1,
            start_time=start_time,
            end_time=end_time,
            metadata=metadata,
            errors=errors,
        )

    def _run_processor(self, download_result: DownloadResult, doc_id: str) -> ProcessingResult:
        return self.document_processor.process_document(
            pdf_path=download_result.pdf_path,
            latex_path=download_result.latex_path,
            document_id=doc_id,
        )

    def _build_output_payload(
        self,
        processing_result: ProcessingResult,
        download_result: DownloadResult,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "arxiv_id": download_result.arxiv_id,
            "source_pdf": str(download_result.pdf_path) if download_result.pdf_path else None,
            "document": processing_result.to_dict(),
        }

        if download_result.metadata:
            payload["metadata"] = self._serialize_metadata(download_result.metadata)

        return payload

    def _persist_payload(self, normalized_id: str, payload: Dict[str, Any]) -> Path:
        safe_id = normalized_id.replace('/', '_')
        output_path = self.output_root / f"{safe_id}.json"
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return output_path

    def _normalize_arxiv_identifier(
        self,
        arxiv_id: Optional[str],
        arxiv_url: Optional[str],
    ) -> Optional[str]:
        if arxiv_id:
            return arxiv_id.strip()

        if not arxiv_url:
            return None

        url = arxiv_url.strip()
        if not url:
            return None

        path = url.split("?", 1)[0]
        path = path.rstrip("/")
        if path.endswith(".pdf"):
            path = path[:-4]

        identifier = path.split("/")[-1]
        return identifier or None

    def _serialize_metadata(self, metadata: Any) -> Dict[str, Any]:
        if hasattr(metadata, "__dataclass_fields__"):
            serialized = asdict(metadata)
        elif hasattr(metadata, "__dict__"):
            serialized = dict(metadata.__dict__)
        else:
            serialized = {}

        published = serialized.get("published")
        if hasattr(published, "isoformat"):
            serialized["published"] = published.isoformat()

        updated = serialized.get("updated")
        if hasattr(updated, "isoformat"):
            serialized["updated"] = updated.isoformat()

        return serialized

    def _result_for_failure(
        self,
        start_time: datetime,
        *,
        arxiv_id: Optional[str] = None,
        error_message: str,
    ) -> WorkflowResult:
        end_time = datetime.now(timezone.utc)
        metadata: Dict[str, Any] = {}
        if arxiv_id:
            metadata["arxiv_id"] = arxiv_id

        return WorkflowResult(
            workflow_name=self.name,
            success=False,
            items_processed=0,
            items_failed=1,
            start_time=start_time,
            end_time=end_time,
            metadata=metadata,
            errors=[error_message],
        )


__all__ = ["ArxivSinglePDFWorkflow"]
