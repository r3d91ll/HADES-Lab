"""
Progress Tracking Utilities
===========================

Theory Connection - Information Reconstructionism:
Progress tracking implements temporal measurement of information transformation
across the Conveyance Framework dimensions. It quantifies the WHERE → WHAT
transformation process, enabling real-time assessment of Context amplification
and system efficiency optimization.

From Actor-Network Theory: Progress trackers act as "immutable mobiles" that
carry information about transformation states across distributed components,
maintaining coherent understanding of processing advancement throughout
the network.

The tracker optimizes C = (W·R·H/T)·Ctx^α by providing visibility into:
- WHERE: Processing location and file path progression
- WHAT: Content transformation quality and completeness
- WHO: Worker efficiency and resource utilization patterns
- TIME: Processing velocity and remaining duration estimates
- Context: System coherence and error tracking across phases
"""

from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import time
import logging
import json
from pathlib import Path
from collections import deque
import math

logger = logging.getLogger(__name__)


class ProgressState(Enum):
    """Progress tracking states."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressStep:
    """
    Individual progress step.

    Theory Connection: Represents discrete transformation in WHERE → WHAT
    progression, with metadata tracking Context coherence factors.
    """
    id: str
    name: str
    total_items: int
    completed_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    state: ProgressState = ProgressState.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_messages: List[str] = field(default_factory=list)

    @property
    def duration(self) -> Optional[timedelta]:
        """Get step duration."""
        if self.start_time is None:
            return None
        end_time = self.end_time or datetime.utcnow()
        return end_time - self.start_time

    @property
    def completion_percent(self) -> float:
        """Calculate completion percentage."""
        if self.total_items == 0:
            return 100.0
        return (self.completed_items / self.total_items) * 100

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        processed = self.completed_items + self.failed_items
        if processed == 0:
            return 100.0
        return (self.completed_items / processed) * 100

    @property
    def processing_rate(self) -> Optional[float]:
        """Calculate processing rate (items per second)."""
        if self.duration is None or self.duration.total_seconds() == 0:
            return None
        processed = self.completed_items + self.failed_items + self.skipped_items
        return processed / self.duration.total_seconds()

    @property
    def estimated_completion(self) -> Optional[datetime]:
        """Estimate completion time based on current rate."""
        if self.state != ProgressState.RUNNING or self.processing_rate is None:
            return None

        remaining_items = self.total_items - self.completed_items - self.failed_items - self.skipped_items
        if remaining_items <= 0:
            return datetime.utcnow()

        remaining_seconds = remaining_items / self.processing_rate
        return datetime.utcnow() + timedelta(seconds=remaining_seconds)

    def start(self) -> None:
        """Start the progress step."""
        if self.state != ProgressState.PENDING:
            logger.warning(f"Step {self.id} is not pending (current: {self.state})")
            return

        self.start_time = datetime.utcnow()
        self.state = ProgressState.RUNNING
        logger.debug(f"Started progress step: {self.id}")

    def pause(self) -> None:
        """Pause the progress step."""
        if self.state != ProgressState.RUNNING:
            logger.warning(f"Step {self.id} is not running (current: {self.state})")
            return

        self.state = ProgressState.PAUSED
        logger.debug(f"Paused progress step: {self.id}")

    def resume(self) -> None:
        """Resume the progress step."""
        if self.state != ProgressState.PAUSED:
            logger.warning(f"Step {self.id} is not paused (current: {self.state})")
            return

        self.state = ProgressState.RUNNING
        logger.debug(f"Resumed progress step: {self.id}")

    def complete(self) -> None:
        """Complete the progress step."""
        if self.state not in [ProgressState.RUNNING, ProgressState.PAUSED]:
            logger.warning(f"Step {self.id} is not active (current: {self.state})")
            return

        self.end_time = datetime.utcnow()
        self.state = ProgressState.COMPLETED
        logger.debug(f"Completed progress step: {self.id}")

    def fail(self, error_message: Optional[str] = None) -> None:
        """Mark the progress step as failed."""
        self.end_time = datetime.utcnow()
        self.state = ProgressState.FAILED

        if error_message:
            self.error_messages.append(error_message)

        logger.debug(f"Failed progress step: {self.id}")

    def cancel(self) -> None:
        """Cancel the progress step."""
        self.end_time = datetime.utcnow()
        self.state = ProgressState.CANCELLED
        logger.debug(f"Cancelled progress step: {self.id}")

    def update_progress(self, completed: Optional[int] = None,
                       failed: Optional[int] = None,
                       skipped: Optional[int] = None) -> None:
        """
        Update progress counters.

        Args:
            completed: Number of completed items
            failed: Number of failed items
            skipped: Number of skipped items
        """
        if completed is not None:
            self.completed_items = completed
        if failed is not None:
            self.failed_items = failed
        if skipped is not None:
            self.skipped_items = skipped

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'id': self.id,
            'name': self.name,
            'total_items': self.total_items,
            'completed_items': self.completed_items,
            'failed_items': self.failed_items,
            'skipped_items': self.skipped_items,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'state': self.state.value,
            'completion_percent': self.completion_percent,
            'success_rate': self.success_rate,
            'processing_rate': self.processing_rate,
            'duration_seconds': self.duration.total_seconds() if self.duration else None,
            'estimated_completion': self.estimated_completion.isoformat() if self.estimated_completion else None,
            'metadata': self.metadata,
            'error_messages': self.error_messages
        }


class ProgressTracker:
    """
    Multi-step progress tracking system.

    Theory Connection - Conveyance Framework Implementation:
    Tracks progression across all dimensions of C = (W·R·H/T)·Ctx^α:

    1. WHERE (R): File paths, processing locations, component transitions
    2. WHAT (W): Content transformation quality and completeness metrics
    3. WHO (H): Worker efficiency, resource utilization, capacity tracking
    4. TIME (T): Processing velocity, ETA calculation, duration measurement
    5. Context (Ctx): System coherence, error tracking, validation success

    The tracker enables Context amplification (Ctx^α) by providing visibility
    into processing coherence and enabling real-time optimization decisions.
    """

    def __init__(self, name: str, description: str = ""):
        """
        Initialize progress tracker.

        Args:
            name: Tracker name
            description: Human-readable description
        """
        self.name = name
        self.description = description
        self.created_at = datetime.utcnow()

        # Step management
        self._steps: Dict[str, ProgressStep] = {}
        self._step_order: List[str] = []
        self._current_step_id: Optional[str] = None

        # Progress history
        self._progress_history: deque = deque(maxlen=10000)  # Keep last 10k updates

        # Callbacks
        self._step_callbacks: Dict[ProgressState, List[Callable]] = {
            state: [] for state in ProgressState
        }
        self._update_callbacks: List[Callable] = []

        # Thread safety
        self._lock = threading.RLock()

        # Overall tracking
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.overall_state = ProgressState.PENDING

        logger.info(f"Created progress tracker: {name}")

    def add_step(self, step_id: str, name: str, total_items: int,
                metadata: Optional[Dict[str, Any]] = None) -> ProgressStep:
        """
        Add a new progress step.

        Theory Connection: Establishes WHERE positioning in processing
        pipeline and defines WHAT transformation boundaries.

        Args:
            step_id: Unique step identifier
            name: Human-readable step name
            total_items: Total number of items to process
            metadata: Additional step metadata

        Returns:
            Created progress step

        Raises:
            ValueError: If step_id already exists
        """
        with self._lock:
            if step_id in self._steps:
                raise ValueError(f"Step {step_id} already exists")

            step = ProgressStep(
                id=step_id,
                name=name,
                total_items=total_items,
                metadata=metadata or {}
            )

            self._steps[step_id] = step
            self._step_order.append(step_id)

            logger.debug(f"Added progress step: {step_id} ({total_items} items)")
            return step

    def get_step(self, step_id: str) -> Optional[ProgressStep]:
        """
        Get progress step by ID.

        Args:
            step_id: Step identifier

        Returns:
            Progress step or None if not found
        """
        with self._lock:
            return self._steps.get(step_id)

    def start_step(self, step_id: str) -> bool:
        """
        Start a progress step.

        Args:
            step_id: Step identifier

        Returns:
            True if step was started successfully
        """
        with self._lock:
            if step_id not in self._steps:
                logger.error(f"Unknown step: {step_id}")
                return False

            step = self._steps[step_id]
            step.start()

            self._current_step_id = step_id

            # Start overall tracking if this is the first step
            if self.overall_state == ProgressState.PENDING:
                self.start_time = datetime.utcnow()
                self.overall_state = ProgressState.RUNNING

            # Trigger callbacks
            self._trigger_step_callbacks(ProgressState.RUNNING, step)
            self._record_progress_update("step_started", step)

            return True

    def update_step(self, step_id: str, completed: Optional[int] = None,
                   failed: Optional[int] = None, skipped: Optional[int] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update step progress.

        Theory Connection: Records incremental WHERE → WHAT transformation
        progress, updating Context coherence measurements.

        Args:
            step_id: Step identifier
            completed: Number of completed items
            failed: Number of failed items
            skipped: Number of skipped items
            metadata: Updated metadata

        Returns:
            True if step was updated successfully
        """
        with self._lock:
            if step_id not in self._steps:
                logger.error(f"Unknown step: {step_id}")
                return False

            step = self._steps[step_id]
            step.update_progress(completed, failed, skipped)

            if metadata:
                step.metadata.update(metadata)

            # Record progress update
            self._record_progress_update("step_updated", step)

            # Trigger update callbacks
            for callback in self._update_callbacks:
                try:
                    callback(self, step)
                except Exception as e:
                    logger.error(f"Update callback failed: {e}")

            return True

    def complete_step(self, step_id: str) -> bool:
        """
        Complete a progress step.

        Args:
            step_id: Step identifier

        Returns:
            True if step was completed successfully
        """
        with self._lock:
            if step_id not in self._steps:
                logger.error(f"Unknown step: {step_id}")
                return False

            step = self._steps[step_id]
            step.complete()

            # Check if this was the current step
            if self._current_step_id == step_id:
                self._current_step_id = None

            # Trigger callbacks
            self._trigger_step_callbacks(ProgressState.COMPLETED, step)
            self._record_progress_update("step_completed", step)

            # Check if all steps are complete
            self._check_overall_completion()

            return True

    def fail_step(self, step_id: str, error_message: Optional[str] = None) -> bool:
        """
        Mark step as failed.

        Args:
            step_id: Step identifier
            error_message: Error description

        Returns:
            True if step was marked as failed
        """
        with self._lock:
            if step_id not in self._steps:
                logger.error(f"Unknown step: {step_id}")
                return False

            step = self._steps[step_id]
            step.fail(error_message)

            # Check if this was the current step
            if self._current_step_id == step_id:
                self._current_step_id = None

            # Trigger callbacks
            self._trigger_step_callbacks(ProgressState.FAILED, step)
            self._record_progress_update("step_failed", step)

            return True

    def get_overall_progress(self) -> Dict[str, Any]:
        """
        Get overall progress across all steps.

        Theory Connection: Provides aggregate view of Context coherence
        and Conveyance optimization across the entire processing pipeline.

        Returns:
            Overall progress dictionary
        """
        with self._lock:
            total_items = sum(step.total_items for step in self._steps.values())
            completed_items = sum(step.completed_items for step in self._steps.values())
            failed_items = sum(step.failed_items for step in self._steps.values())
            skipped_items = sum(step.skipped_items for step in self._steps.values())

            completion_percent = (completed_items / total_items * 100) if total_items > 0 else 0.0
            success_rate = (completed_items / max(1, completed_items + failed_items)) * 100

            # Calculate overall processing rate
            processing_rate = None
            if self.start_time:
                duration = datetime.utcnow() - self.start_time
                if duration.total_seconds() > 0:
                    processed = completed_items + failed_items + skipped_items
                    processing_rate = processed / duration.total_seconds()

            # Estimate completion time
            estimated_completion = None
            if processing_rate and processing_rate > 0:
                remaining_items = total_items - completed_items - failed_items - skipped_items
                if remaining_items > 0:
                    remaining_seconds = remaining_items / processing_rate
                    estimated_completion = datetime.utcnow() + timedelta(seconds=remaining_seconds)

            return {
                'name': self.name,
                'description': self.description,
                'state': self.overall_state.value,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'duration_seconds': (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else None,
                'total_steps': len(self._steps),
                'completed_steps': len([s for s in self._steps.values() if s.state == ProgressState.COMPLETED]),
                'failed_steps': len([s for s in self._steps.values() if s.state == ProgressState.FAILED]),
                'current_step_id': self._current_step_id,
                'total_items': total_items,
                'completed_items': completed_items,
                'failed_items': failed_items,
                'skipped_items': skipped_items,
                'completion_percent': completion_percent,
                'success_rate': success_rate,
                'processing_rate': processing_rate,
                'estimated_completion': estimated_completion.isoformat() if estimated_completion else None
            }

    def get_step_summary(self) -> List[Dict[str, Any]]:
        """
        Get summary of all steps.

        Returns:
            List of step summaries
        """
        with self._lock:
            return [step.to_dict() for step_id in self._step_order
                   for step in [self._steps[step_id]]]

    def calculate_conveyance_metrics(self) -> Dict[str, float]:
        """
        Calculate Conveyance Framework metrics from progress data.

        Theory Connection: Quantifies C = (W·R·H/T)·Ctx^α using
        empirical measurements from processing progression.

        Returns:
            Conveyance metrics dictionary
        """
        with self._lock:
            if not self._steps:
                return {}

            # WHERE (R): Processing location consistency
            # Measured as spatial coherence across steps
            where_score = 1.0  # Base score, could be enhanced with file path analysis

            # WHAT (W): Content transformation quality
            # Measured as success rate across all processing
            total_processed = sum(s.completed_items + s.failed_items for s in self._steps.values())
            total_successful = sum(s.completed_items for s in self._steps.values())
            what_score = total_successful / max(1, total_processed)

            # WHO (H): System capability utilization
            # Measured as processing efficiency
            total_capacity = sum(s.total_items for s in self._steps.values())
            actual_processed = sum(s.completed_items + s.failed_items + s.skipped_items
                                 for s in self._steps.values())
            who_score = actual_processed / max(1, total_capacity)

            # TIME (T): Processing efficiency
            # Measured as inverse of normalized processing time
            if self.start_time:
                elapsed = (datetime.utcnow() - self.start_time).total_seconds()
                # Normalize time score (lower is better, so use inverse)
                time_score = min(1.0, 3600 / max(1, elapsed))  # 1 hour = perfect score
            else:
                time_score = 0.0

            # Context (Ctx): Overall coherence
            # Equal weights: L=I=A=G=0.25
            local_coherence = 1.0 - (len([s for s in self._steps.values() if s.error_messages]) /
                                   max(1, len(self._steps)))
            instruction_fit = sum(1.0 for s in self._steps.values() if s.state != ProgressState.PENDING) / \
                            max(1, len(self._steps))
            actionability = sum(1.0 for s in self._steps.values()
                              if s.state in [ProgressState.RUNNING, ProgressState.COMPLETED]) / \
                          max(1, len(self._steps))
            grounding = 1.0 if self.start_time else 0.0

            context_score = 0.25 * (local_coherence + instruction_fit + actionability + grounding)

            # Alpha (super-linear amplification)
            alpha = 1.8

            # Conveyance calculation (efficiency view)
            if time_score > 0:
                conveyance_score = (where_score * what_score * who_score / time_score) * \
                                 (context_score ** alpha)
            else:
                conveyance_score = 0.0

            return {
                'where': where_score,
                'what': what_score,
                'who': who_score,
                'time': time_score,
                'context': context_score,
                'alpha': alpha,
                'conveyance': conveyance_score,
                'local_coherence': local_coherence,
                'instruction_fit': instruction_fit,
                'actionability': actionability,
                'grounding': grounding
            }

    def add_step_callback(self, state: ProgressState, callback: Callable) -> None:
        """
        Add callback for step state changes.

        Args:
            state: Progress state to listen for
            callback: Callback function (tracker, step) -> None
        """
        self._step_callbacks[state].append(callback)

    def add_update_callback(self, callback: Callable) -> None:
        """
        Add callback for progress updates.

        Args:
            callback: Callback function (tracker, step) -> None
        """
        self._update_callbacks.append(callback)

    def _trigger_step_callbacks(self, state: ProgressState, step: ProgressStep) -> None:
        """Trigger callbacks for step state change."""
        for callback in self._step_callbacks[state]:
            try:
                callback(self, step)
            except Exception as e:
                logger.error(f"Step callback failed: {e}")

    def _record_progress_update(self, event_type: str, step: ProgressStep) -> None:
        """Record progress update in history."""
        update_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'step_id': step.id,
            'step_state': step.state.value,
            'completion_percent': step.completion_percent,
            'processing_rate': step.processing_rate
        }

        self._progress_history.append(update_record)

    def _check_overall_completion(self) -> None:
        """Check if all steps are complete and update overall state."""
        completed_states = {ProgressState.COMPLETED, ProgressState.FAILED, ProgressState.CANCELLED}
        all_steps_done = all(step.state in completed_states for step in self._steps.values())

        if all_steps_done:
            self.end_time = datetime.utcnow()

            # Determine overall state
            if any(step.state == ProgressState.FAILED for step in self._steps.values()):
                self.overall_state = ProgressState.FAILED
            elif any(step.state == ProgressState.CANCELLED for step in self._steps.values()):
                self.overall_state = ProgressState.CANCELLED
            else:
                self.overall_state = ProgressState.COMPLETED

            logger.info(f"Progress tracker {self.name} finished with state: {self.overall_state.value}")

    def save_progress(self, file_path: Path) -> None:
        """
        Save progress state to file.

        Args:
            file_path: Output file path
        """
        with self._lock:
            progress_data = {
                'tracker': self.get_overall_progress(),
                'steps': self.get_step_summary(),
                'conveyance_metrics': self.calculate_conveyance_metrics(),
                'history': list(self._progress_history)[-100:]  # Last 100 updates
            }

            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(progress_data, f, indent=2, default=str)

                logger.debug(f"Progress saved to {file_path}")

            except Exception as e:
                logger.error(f"Failed to save progress: {e}")

    @classmethod
    def load_progress(cls, file_path: Path) -> Optional['ProgressTracker']:
        """
        Load progress state from file.

        Args:
            file_path: Input file path

        Returns:
            Loaded progress tracker or None if failed
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)

            tracker_data = progress_data.get('tracker', {})
            tracker = cls(
                name=tracker_data.get('name', 'loaded_tracker'),
                description=tracker_data.get('description', '')
            )

            # Restore steps
            for step_data in progress_data.get('steps', []):
                step = ProgressStep(
                    id=step_data['id'],
                    name=step_data['name'],
                    total_items=step_data['total_items'],
                    completed_items=step_data['completed_items'],
                    failed_items=step_data['failed_items'],
                    skipped_items=step_data['skipped_items'],
                    state=ProgressState(step_data['state']),
                    metadata=step_data.get('metadata', {}),
                    error_messages=step_data.get('error_messages', [])
                )

                if step_data.get('start_time'):
                    step.start_time = datetime.fromisoformat(step_data['start_time'])
                if step_data.get('end_time'):
                    step.end_time = datetime.fromisoformat(step_data['end_time'])

                tracker._steps[step.id] = step
                tracker._step_order.append(step.id)

            # Restore tracker state
            if tracker_data.get('start_time'):
                tracker.start_time = datetime.fromisoformat(tracker_data['start_time'])
            if tracker_data.get('end_time'):
                tracker.end_time = datetime.fromisoformat(tracker_data['end_time'])

            tracker.overall_state = ProgressState(tracker_data.get('state', 'pending'))
            tracker._current_step_id = tracker_data.get('current_step_id')

            logger.info(f"Loaded progress tracker from {file_path}")
            return tracker

        except Exception as e:
            logger.error(f"Failed to load progress tracker: {e}")
            return None
