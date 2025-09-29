import cv2
import time
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple

from detector import BlinkDetector, MouthOpenDetector


class PipelineState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ActionStep:
    """Represents a single action step in the pipeline"""
    
    def __init__(self, detector, options: Dict[str, Any], description: str = ""):
        self.detector = detector
        self.options = options
        self.description = description
        self.start_time = None
        self.completed = False
        self.success = False
    
    def reset(self):
        """Reset step state"""
        if hasattr(self.detector, 'action_count'):
            self.detector.action_count = 0
        if hasattr(self.detector, 'start_time'):
            self.detector.start_time = time.time()
        self.start_time = time.time()
        self.completed = False
        self.success = False


class ActionPipeline:
    def __init__(self, sequence: List[Dict[str, Any]], timeout_per_step: float = 30.0):
        """
        Initialize ActionPipeline

        Args:
            sequence: List of step dictionaries with 'detector', 'options', and optional 'description'
            timeout_per_step: Maximum time allowed per step in seconds
        """
        # Initialize camera
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            raise RuntimeError("Error: Could not open camera")
        
        # Convert sequence to ActionStep objects
        self._steps = []
        for i, step_dict in enumerate(sequence):
            description = step_dict.get('description', f"Step {i + 1}")
            action_step = ActionStep(
                detector=step_dict['detector'],
                options=step_dict['options'],
                description=description
            )
            self._steps.append(action_step)
        
        self._current_step_index = 0
        self._success_count = 0
        self._state = PipelineState.IDLE
        self._timeout_per_step = timeout_per_step
        self._pipeline_start_time = None
        
        # UI settings
        self._window_name = "Action Pipeline"
        self._fps_counter = 0
        self._fps_start_time = time.time()
        self._current_fps = 0
    
    @property
    def current_step(self) -> Optional[ActionStep]:
        """Get current step being processed"""
        if 0 <= self._current_step_index < len(self._steps):
            return self._steps[self._current_step_index]
        return None
    
    @property
    def progress(self) -> Tuple[int, int]:
        """Get current progress (completed, total)"""
        return self._success_count, len(self._steps)
    
    @property
    def is_complete(self) -> bool:
        """Check if pipeline is complete"""
        return self._success_count >= len(self._steps)
    
    def _calculate_fps(self) -> None:
        """Calculate and update FPS"""
        self._fps_counter += 1
        if self._fps_counter % 30 == 0:
            fps_end_time = time.time()
            self._current_fps = 30 / (fps_end_time - self._fps_start_time)
            self._fps_start_time = fps_end_time
    
    def _draw_ui_overlay(self, frame, step: ActionStep, action_count: bool) -> None:
        """Draw UI overlay on frame"""
        height, width = frame.shape[:2]
        
        # Background for status
        cv2.rectangle(frame, (0, 0), (width, 200), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (width, 200), (255, 255, 255), 2)
        
        # Pipeline progress
        progress_text = f"Pipeline Progress: {self._success_count} / {len(self._steps)}"
        cv2.putText(frame, progress_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Current step info
        step_text = f"Current Step: {step.description}"
        cv2.putText(frame, step_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Action instruction
        if hasattr(step.detector, 'statement'):
            cv2.putText(frame, step.detector.statement, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Action count and target
        action_name = getattr(step.detector, 'CTA', 'Action')
        current_count = getattr(step.detector, 'action_count', 0)
        target_count = step.options.get('exp_count', 1)
        
        count_text = f"{action_name}: {current_count} / {target_count}"
        cv2.putText(frame, count_text, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Timeout countdown
        if step.start_time:
            elapsed = time.time() - step.start_time
            remaining = max(0, self._timeout_per_step - elapsed)
            timeout_text = f"Time Remaining: {remaining:.1f}s"
            color = (0, 255, 0) if remaining > 10 else (0, 165, 255) if remaining > 5 else (0, 0, 255)
            cv2.putText(frame, timeout_text, (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {self._current_fps:.1f}", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Success/Action detection alert
        if action_count:
            alert_height = 60
            cv2.rectangle(frame, (0, height - alert_height), (width, height), (0, 255, 0), -1)
            cv2.putText(frame, f"{action_name.upper()} DETECTED!",
                        (width // 2 - 120, height - alert_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Controls help
        help_y = height - 120
        cv2.rectangle(frame, (width - 200, help_y), (width, height), (50, 50, 50), -1)
        cv2.putText(frame, "Controls:", (width - 190, help_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Q - Quit", (width - 190, help_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "R - Reset Step", (width - 190, help_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "S - Skip Step", (width - 190, help_y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "N - Next Pipeline", (width - 190, help_y + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _show_completion_screen(self, success: bool) -> None:
        """Show completion screen"""
        # Create a completion screen
        completion_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        if success:
            color = (0, 255, 0)
            message = "PIPELINE COMPLETED!"
            sub_message = f"All {len(self._steps)} steps completed successfully"
        else:
            color = (0, 0, 255)
            message = "PIPELINE FAILED"
            sub_message = f"Completed {self._success_count} of {len(self._steps)} steps"
        
        cv2.putText(completion_frame, message, (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(completion_frame, sub_message, (50, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        total_time = time.time() - self._pipeline_start_time
        time_text = f"Total Time: {total_time:.1f} seconds"
        cv2.putText(completion_frame, time_text, (50, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(completion_frame, "Press any key to exit...", (50, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow(self._window_name, completion_frame)
        cv2.waitKey(0)
    
    def _process_step(self, step: ActionStep) -> bool:
        """
        Process a single step

        Returns:
            bool: True if step completed successfully, False otherwise
        """
        step.reset()
        print(f"\n--- Starting: {step.description} ---")
        print("Controls: Q=Quit, R=Reset, S=Skip, N=Next")
        
        while not step.completed:
            ret, frame = self._cap.read()
            if not ret:
                print("Error: Could not read frame")
                return False
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect actions using the detector
            try:
                if hasattr(step.detector, 'detect'):
                    detection_result = step.detector.detect(frame)
                    if isinstance(detection_result, tuple) and len(detection_result) >= 2:
                        processed_frame, detected = detection_result[:2]
                    else:
                        processed_frame, detected = frame, detection_result
                else:
                    # Fallback for different detector interfaces
                    processed_frame = frame
                    detected = False
                    print("Warning: Detector doesn't have 'detect' method")
            except Exception as e:
                print(f"Error in detector: {e}")
                processed_frame = frame
                detected = False
            
            # Calculate FPS
            self._calculate_fps()
            
            # Draw UI overlay
            self._draw_ui_overlay(processed_frame, step, detected)
            
            # Check for step completion
            if detected:
                print('here')
                current_count = getattr(step.detector, 'action_count', 0)
                target_count = step.options.get('exp_count', 1)
                
                if current_count >= target_count:
                    step.success = True
                    step.completed = True
                    print(f"✓ Step completed: {step.description}")
                    # Brief success display
                    time.sleep(0.5)
                    break
            
            # Check timeout
            if step.start_time and time.time() - step.start_time > self._timeout_per_step:
                print(f"✗ Step timeout: {step.description}")
                step.completed = True
                break
            
            # Display frame
            cv2.imshow(self._window_name, processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self._state = PipelineState.CANCELLED
                return False
            elif key == ord('r'):
                step.reset()
                print(f"Step reset: {step.description}")
            elif key == ord('s'):
                print(f"Step skipped: {step.description}")
                step.success = True
                step.completed = True
                break
            elif key == ord('n'):
                print(f"Pipeline cancelled by user")
                self._state = PipelineState.CANCELLED
                return False
        
        return step.success
    
    def perform_action(self) -> bool:
        """
        Execute the action pipeline

        Returns:
            bool: True if all steps completed successfully
        """
        if not self._steps:
            print("No steps defined in pipeline")
            return False
        
        self._state = PipelineState.RUNNING
        self._pipeline_start_time = time.time()
        self._success_count = 0
        self._current_step_index = 0
        
        print(f"🚀 Starting Action Pipeline with {len(self._steps)} steps")
        print("=" * 50)
        
        try:
            for i, step in enumerate(self._steps):
                self._current_step_index = i
                
                if self._process_step(step):
                    self._success_count += 1
                else:
                    if self._state == PipelineState.CANCELLED:
                        print("\n❌ Pipeline cancelled by user")
                        return False
                    else:
                        print(f"\n❌ Pipeline failed at step: {step.description}")
                        self._state = PipelineState.FAILED
                        self._show_completion_screen(False)
                        return False
            
            # All steps completed successfully
            print(f"\n🎉 Pipeline completed successfully!")
            print(f"Time taken: {time.time() - self._pipeline_start_time:.1f} seconds")
            self._state = PipelineState.SUCCESS
            self._show_completion_screen(True)
            return True
        
        except KeyboardInterrupt:
            print("\n⚠️  Pipeline interrupted by user")
            self._state = PipelineState.CANCELLED
            return False
        except Exception as e:
            print(f"\n💥 Pipeline error: {e}")
            self._state = PipelineState.FAILED
            return False
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up resources"""
        if self._cap and self._cap.isOpened():
            self._cap.release()
        cv2.destroyAllWindows()
        print("\n🧹 Cleanup completed")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline execution statistics"""
        total_time = time.time() - self._pipeline_start_time if self._pipeline_start_time else 0
        
        return {
            'total_steps': len(self._steps),
            'completed_steps': self._success_count,
            'success_rate': self._success_count / len(self._steps) * 100 if self._steps else 0,
            'total_time': total_time,
            'state': self._state.value,
            'current_step': self._current_step_index
        }
    
    def reset(self):
        """Reset pipeline to initial state"""
        self._current_step_index = 0
        self._success_count = 0
        self._state = PipelineState.IDLE
        self._pipeline_start_time = None
        
        for step in self._steps:
            step.reset()
        
        print("Pipeline reset to initial state")


# Example usage and utility functions
def create_blink_sequence(blink_detector, target_blinks: int = 3):
    """Helper function to create a blink sequence step"""
    return {
        'detector': blink_detector,
        'options': {'exp_count': target_blinks},
        'description': f'Blink {target_blinks} times'
    }


def create_mouth_sequence(mouth_detector, target_opens: int = 2):
    """Helper function to create a mouth opening sequence step"""
    return {
        'detector': mouth_detector,
        'options': {'exp_count': target_opens},
        'description': f'Open mouth {target_opens} times'
    }


# Example of how to use the refined pipeline
if __name__ == "__main__":
    import numpy as np
    
    
    # Create mock sequence
    sequence = [
        create_mouth_sequence(MouthOpenDetector(statement="Please open mouth 2 times"), 3),
        create_blink_sequence(BlinkDetector(statement="Please blink 3 times"), 3),
    ]
    
    try:
        pipeline = ActionPipeline(sequence, timeout_per_step=15.0)
        success = pipeline.perform_action()
        
        # Print final statistics
        stats = pipeline.get_statistics()
        print(f"\nFinal Statistics: {stats}")
    
    except RuntimeError as e:
        print(f"Pipeline initialization failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    
    
    
    
    
    
    
    