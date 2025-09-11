"""Asynchronous processing module for video frame analysis.

This module provides a threaded processing system that runs video analysis tasks
on a separate thread from the UI to prevent interface freezing during heavy
processing operations.
"""

import time
import threading
import queue
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from PyQt5.QtCore import QObject, pyqtSignal, QThread

from ..config.settings import get_setting
from .inference import run_inference
from .tracking import run_tracking, reset_tracker
from .field_segmentation import run_field_segmentation
from .player_id import run_player_id_on_tracks


class ProcessingTaskType(Enum):
    """Types of processing tasks that can be performed."""
    INFERENCE = "inference"
    TRACKING = "tracking"
    FIELD_SEGMENTATION = "field_segmentation"
    PLAYER_ID = "player_id"
    FULL_ANALYSIS = "full_analysis"  # Run all enabled processing


@dataclass
class ProcessingTask:
    """Represents a processing task to be executed."""
    task_id: str
    task_type: ProcessingTaskType
    frame: np.ndarray
    frame_index: int
    timestamp: float
    enabled_options: Dict[str, bool]
    previous_results: Optional[Dict[str, Any]] = None
    priority: int = 1  # Higher number = higher priority
    
    def __lt__(self, other):
        """Less than comparison for priority queue ordering."""
        if not isinstance(other, ProcessingTask):
            return NotImplemented
        # Primary sort by priority (higher priority first)
        if self.priority != other.priority:
            return self.priority > other.priority
        # Secondary sort by timestamp (newer first)
        return self.timestamp > other.timestamp


@dataclass
class ProcessingResult:
    """Results from a completed processing task."""
    task_id: str
    frame_index: int
    timestamp: float
    processing_time_ms: float
    success: bool
    error_message: Optional[str] = None
    
    # Processing results
    detections: Optional[List[Dict]] = None
    tracks: Optional[List[Any]] = None
    field_results: Optional[List[Any]] = None
    player_ids: Optional[Dict[int, Tuple[str, Any]]] = None
    
    # Performance metrics
    performance_metrics: Optional[Dict[str, float]] = None


class AsyncProcessorWorker(QThread):
    """Worker thread for processing video frames asynchronously."""
    
    # Signals to communicate with UI thread
    processing_complete = pyqtSignal(object)  # ProcessingResult
    processing_error = pyqtSignal(str, str)  # task_id, error_message
    queue_status_changed = pyqtSignal(int)  # queue_size
    
    def __init__(self):
        super().__init__()
        self.task_queue = queue.PriorityQueue()
        self.is_running = True
        self.current_task_id = None
        
        # Performance tracking
        self.total_tasks_processed = 0
        self.average_processing_time = 0.0
        
        # Thread-safe shutdown event
        self.shutdown_event = threading.Event()
        
    def run(self):
        """Main processing loop running in separate thread."""
        print("[ASYNC_PROCESSOR] Worker thread started")
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Get next task with timeout to allow for shutdown
                try:
                    task = self.task_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                if task is None:  # Shutdown signal
                    break
                    
                self.current_task_id = task.task_id
                self.queue_status_changed.emit(self.task_queue.qsize())
                
                # Process the task
                result = self._process_task(task)
                
                # Emit result
                self.processing_complete.emit(result)
                
                # Update performance metrics
                self._update_performance_metrics(result.processing_time_ms)
                
                self.task_queue.task_done()
                self.current_task_id = None
                
            except Exception as e:
                error_msg = f"Processing error: {str(e)}"
                print(f"[ASYNC_PROCESSOR] {error_msg}")
                if hasattr(task, 'task_id'):
                    self.processing_error.emit(task.task_id, error_msg)
                
        print("[ASYNC_PROCESSOR] Worker thread stopped")
    
    def _process_task(self, task: ProcessingTask) -> ProcessingResult:
        """Process a single task and return results."""
        start_time = time.time()
        performance_metrics = {}
        
        try:
            print(f"[ASYNC_PROCESSOR] Processing task {task.task_id} "
                  f"(type: {task.task_type.value}, frame: {task.frame_index})")
            
            # Initialize results
            detections = []
            tracks = []
            field_results = []
            player_ids = {}
            
            # Run inference if enabled
            if task.enabled_options.get('inference', False):
                inference_start = time.time()
                detections = run_inference(task.frame)
                performance_metrics['Inference'] = (time.time() - inference_start) * 1000
            
            # Run tracking if enabled and we have detections
            if task.enabled_options.get('tracking', False) and detections:
                tracking_start = time.time()
                tracks = run_tracking(task.frame, detections)
                performance_metrics['Tracking'] = (time.time() - tracking_start) * 1000
            
            # Run field segmentation if enabled
            if task.enabled_options.get('field_segmentation', False):
                field_start = time.time()
                field_results = run_field_segmentation(task.frame)
                performance_metrics['Field Segmentation'] = (time.time() - field_start) * 1000
            
            # Run player ID if enabled and we have tracks
            if task.enabled_options.get('player_id', False) and tracks:
                player_id_start = time.time()
                player_ids = run_player_id_on_tracks(task.frame, tracks)
                performance_metrics['Player ID'] = (time.time() - player_id_start) * 1000
            
            processing_time_ms = (time.time() - start_time) * 1000
            performance_metrics['Total Runtime'] = processing_time_ms
            
            return ProcessingResult(
                task_id=task.task_id,
                frame_index=task.frame_index,
                timestamp=task.timestamp,
                processing_time_ms=processing_time_ms,
                success=True,
                detections=detections,
                tracks=tracks,
                field_results=field_results,
                player_ids=player_ids,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            print(f"[ASYNC_PROCESSOR] Error processing task {task.task_id}: {e}")
            import traceback
            traceback.print_exc()
            
            return ProcessingResult(
                task_id=task.task_id,
                frame_index=task.frame_index,
                timestamp=task.timestamp,
                processing_time_ms=processing_time_ms,
                success=False,
                error_message=str(e)
            )
    
    def add_task(self, task: ProcessingTask):
        """Add a task to the processing queue."""
        # Use task directly in priority queue (now that ProcessingTask is comparable)
        self.task_queue.put(task)
        self.queue_status_changed.emit(self.task_queue.qsize())
    
    def clear_queue(self):
        """Clear all pending tasks from the queue."""
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
                self.task_queue.task_done()
            except queue.Empty:
                break
        self.queue_status_changed.emit(0)
        print("[ASYNC_PROCESSOR] Queue cleared")
    
    def stop(self):
        """Stop the worker thread gracefully."""
        print("[ASYNC_PROCESSOR] Stopping worker thread...")
        self.is_running = False
        self.shutdown_event.set()
        
        # Add shutdown signal to queue
        self.task_queue.put(None)
        
        # Wait for thread to finish
        if self.isRunning():
            self.wait(5000)  # Wait up to 5 seconds
        
        print("[ASYNC_PROCESSOR] Worker thread stopped")
    
    def _update_performance_metrics(self, processing_time_ms: float):
        """Update running performance metrics."""
        self.total_tasks_processed += 1
        if self.total_tasks_processed == 1:
            self.average_processing_time = processing_time_ms
        else:
            # Running average
            alpha = 0.1  # Smoothing factor
            self.average_processing_time = (
                alpha * processing_time_ms + 
                (1 - alpha) * self.average_processing_time
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return {
            'total_tasks_processed': self.total_tasks_processed,
            'average_processing_time_ms': self.average_processing_time,
            'queue_size': self.task_queue.qsize(),
            'current_task_id': self.current_task_id,
            'is_running': self.is_running
        }


class AsyncVideoProcessor(QObject):
    """Main interface for asynchronous video processing."""
    
    # Signals for UI communication
    processing_complete = pyqtSignal(object)  # ProcessingResult
    processing_error = pyqtSignal(str, str)  # task_id, error_message
    queue_status_changed = pyqtSignal(int)  # queue_size
    
    def __init__(self):
        super().__init__()
        
        # Create worker thread
        self.worker = AsyncProcessorWorker()
        
        # Connect worker signals
        self.worker.processing_complete.connect(self.processing_complete.emit)
        self.worker.processing_error.connect(self.processing_error.emit)
        self.worker.queue_status_changed.connect(self.queue_status_changed.emit)
        
        # Configuration
        self.max_queue_size = get_setting("processing.async.max_queue_size", 10)
        self.auto_clear_old_tasks = get_setting("processing.async.auto_clear_old_tasks", True)
        
        # Task ID counter
        self.task_counter = 0
        
        # Start worker thread
        self.worker.start()
        
        print("[ASYNC_PROCESSOR] Async video processor initialized")
    
    def process_frame_async(self, 
                           frame: np.ndarray,
                           frame_index: int,
                           enabled_options: Dict[str, bool],
                           priority: int = 1) -> str:
        """Submit a frame for asynchronous processing.
        
        Args:
            frame: Video frame to process
            frame_index: Index of the frame in the video
            enabled_options: Dictionary of enabled processing options
            priority: Task priority (higher = more important)
            
        Returns:
            Task ID for tracking the processing task
        """
        # Generate unique task ID
        self.task_counter += 1
        task_id = f"frame_{frame_index}_{self.task_counter}"
        
        # Clear old tasks if queue is getting full
        if (self.auto_clear_old_tasks and 
            self.worker.task_queue.qsize() >= self.max_queue_size):
            self._clear_old_tasks()
        
        # Create processing task
        task = ProcessingTask(
            task_id=task_id,
            task_type=ProcessingTaskType.FULL_ANALYSIS,
            frame=frame.copy(),  # Create copy to avoid threading issues
            frame_index=frame_index,
            timestamp=time.time(),
            enabled_options=enabled_options.copy(),
            priority=priority
        )
        
        # Submit task
        self.worker.add_task(task)
        
        print(f"[ASYNC_PROCESSOR] Submitted task {task_id} for frame {frame_index}")
        return task_id
    
    def _clear_old_tasks(self):
        """Clear older tasks from the queue to prevent backlog."""
        # Keep only the most recent tasks
        keep_count = self.max_queue_size // 2
        
        # Get all tasks from queue
        tasks = []
        while not self.worker.task_queue.empty():
            try:
                task = self.worker.task_queue.get_nowait()
                if task is not None:  # Skip shutdown signal
                    tasks.append(task)
            except queue.Empty:
                break
        
        # Sort by timestamp (most recent first) and keep only recent ones
        tasks.sort(key=lambda x: x.timestamp, reverse=True)
        tasks_to_keep = tasks[:keep_count]
        
        # Put kept tasks back in queue
        for task in tasks_to_keep:
            self.worker.task_queue.put(task)
        
        cleared_count = len(tasks) - len(tasks_to_keep)
        if cleared_count > 0:
            print(f"[ASYNC_PROCESSOR] Cleared {cleared_count} old tasks from queue")
    
    def clear_queue(self):
        """Clear all pending processing tasks."""
        self.worker.clear_queue()
    
    def get_queue_size(self) -> int:
        """Get current number of tasks in processing queue."""
        return self.worker.task_queue.qsize()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return self.worker.get_performance_stats()
    
    def is_processing(self) -> bool:
        """Check if processor is currently working on a task."""
        return self.worker.current_task_id is not None
    
    def shutdown(self):
        """Shutdown the async processor and cleanup resources."""
        print("[ASYNC_PROCESSOR] Shutting down async processor...")
        
        # Clear queue first
        self.clear_queue()
        
        # Stop worker thread
        self.worker.stop()
        
        print("[ASYNC_PROCESSOR] Async processor shutdown complete")
