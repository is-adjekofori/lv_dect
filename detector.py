import cv2
import mediapipe as mp
import numpy as np
import time


class BlinkDetector:
	CTA = "Blink"
	def __init__(self, ear_threshold=0.25, consec_frames=1, statement="Pls Blink"):
		"""
		Initialize blink detector

		Args:
			ear_threshold (float): Eye Aspect Ratio threshold for blink detection
			consec_frames (int): Number of consecutive frames below threshold to count as blink
		"""
		self.ear_threshold = ear_threshold
		self.consec_frames = consec_frames
		self.statement = statement
		
		# Initialize MediaPipe Face Mesh
		self.mp_face_mesh = mp.solutions.face_mesh
		self.face_mesh = self.mp_face_mesh.FaceMesh(
			max_num_faces=1,
			refine_landmarks=True,
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5,
		)
		
		self.mp_drawing = mp.solutions.drawing_utils
		
		# Eye landmark indices for MediaPipe Face Mesh
		self.LEFT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
		self.RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
		
		# More precise eye contour landmarks
		self.LEFT_EYE_POINTS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
		self.RIGHT_EYE_POINTS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
		
		# Simplified eye landmarks for EAR calculation
		self.LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]  # outer, top, bottom corners
		self.RIGHT_EYE_EAR = [362, 387, 385, 263, 373, 380]
		
		# Blink detection variables
		self.blink_counter = 0
		self.frame_counter = 0
		self.action_count = 0
		self.start_time = time.time()
	
	@staticmethod
	def __calculate_ear(eye_landmarks):
		"""
		Calculate Eye Aspect Ratio (EAR)
		EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
		"""
		# Vertical eye landmarks
		A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])  # p2-p6
		B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])  # p3-p5
		
		# Horizontal eye landmark
		C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])  # p1-p4
		
		# Calculate EAR
		if C > 0:
			ear = (A + B) / (2.0 * C)
		else:
			ear = 0
		
		return ear
	
	@staticmethod
	def __extract_eye_landmarks(landmarks, eye_indices, img_width, img_height):
		"""Extract eye landmark coordinates"""
		eye_points = []
		for idx in eye_indices:
			x = int(landmarks.landmark[idx].x * img_width)
			y = int(landmarks.landmark[idx].y * img_height)
			eye_points.append([x, y])
		return np.array(eye_points)
	
	@staticmethod
	def __draw_eye_landmarks(img, eye_points, color=(0, 255, 0)):
		"""Draw eye landmarks on the image"""
		for point in eye_points:
			cv2.circle(img, tuple(point), 2, color, -1)
	
	def detect(self, frame):
		"""
		Process frame and detect blinks

		Args:
			frame: Input video frame

		Returns:
			tuple: (processed_frame, blink_detected, ear_left, ear_right)
		"""
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		results = self.face_mesh.process(rgb_frame)
		
		blink_detected = False
		ear_left = 0
		ear_right = 0
		
		if results.multi_face_landmarks:
			img_height, img_width = frame.shape[:2]
			for face_landmarks in results.multi_face_landmarks:
				
				# Extract eye landmarks for EAR calculation
				left_eye = self.__extract_eye_landmarks(face_landmarks, self.LEFT_EYE_EAR, img_width, img_height)
				right_eye = self.__extract_eye_landmarks(face_landmarks, self.RIGHT_EYE_EAR, img_width, img_height)
				
				# Calculate EAR for both eyes
				ear_left = self.__calculate_ear(left_eye)
				ear_right = self.__calculate_ear(right_eye)
				
				# Average EAR
				avg_ear = (ear_left + ear_right) / 2.0
				
				# Draw eye landmarks
				self.__draw_eye_landmarks(frame, left_eye, (0, 255, 0))
				self.__draw_eye_landmarks(frame, right_eye, (0, 255, 0))
				
				# Check for blink
				if avg_ear < self.ear_threshold:
					self.frame_counter += 1
				else:
					if self.frame_counter >= self.consec_frames:
						self.action_count += 1
						blink_detected = True
					self.frame_counter = 0
				
				# Draw EAR values
				cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, 30),
				            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
				cv2.putText(frame, f"Left: {ear_left:.3f}", (10, 60),
				            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
				cv2.putText(frame, f"Right: {ear_right:.3f}", (10, 85),
				            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
				
				# Draw blink status
				if self.frame_counter > 0:
					cv2.putText(frame, "BLINKING", (10, 120),
					            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		
		return frame, blink_detected

class MouthOpenDetector:
	
	CTA = "Open Mouth"
	def __init__(self, mar_threshold=0.5, consec_frames=3, statement="Pls Blink"):
		"""
		Initialize mouth opening detector

		Args:
			mar_threshold (float): Mouth Aspect Ratio threshold for mouth opening detection
			consec_frames (int): Number of consecutive frames above threshold to count as mouth opening
		"""
		self.mar_threshold = mar_threshold
		self.consec_frames = consec_frames
		self.statement = statement
		
		# Initialize MediaPipe Face Mesh
		self.mp_face_mesh = mp.solutions.face_mesh
		self.face_mesh = self.mp_face_mesh.FaceMesh(
			max_num_faces=1,
			refine_landmarks=True,
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5
		)
		
		# Mouth landmark indices for MediaPipe Face Mesh
		# Outer lip landmarks
		self.MOUTH_LANDMARKS = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
		
		# Inner lip landmarks for more precise detection
		self.INNER_MOUTH_LANDMARKS = [78, 81, 13, 311, 308, 324, 318, 402, 317, 14, 87, 178]
		
		# Key points for MAR calculation (top, bottom, left, right)
		self.MOUTH_MAR_POINTS = [13, 14, 78, 308, 81, 178, 87, 317]  # top center, bottom center, corners
		
		# Mouth opening detection variables
		self.mouth_counter = 0
		self.frame_counter = 0
		self.action_count = 0
		self.is_mouth_open = False
		self.start_time = time.time()
	
	def calculate_mar(self, mouth_landmarks):
		"""
		Calculate Mouth Aspect Ratio (MAR)
		MAR = (vertical_dist1 + vertical_dist2) / (2 * horizontal_dist)
		"""
		# Vertical distances (top to bottom lip)
		# Upper lip to lower lip (center points)
		A = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[1])  # top center to bottom center
		
		# Additional vertical measurements for better accuracy
		B = np.linalg.norm(mouth_landmarks[4] - mouth_landmarks[5])  # left inner vertical
		C = np.linalg.norm(mouth_landmarks[6] - mouth_landmarks[7])  # right inner vertical
		
		# Horizontal distance (left to right corner)
		D = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[3])  # left corner to right corner
		
		# Calculate MAR
		if D > 0:
			mar = (A + B + C) / (3.0 * D)
		else:
			mar = 0
		
		return mar
	
	def __extract_mouth_landmarks(self, landmarks, img_width, img_height):
		"""Extract mouth landmark coordinates"""
		mouth_points = []
		for idx in self.MOUTH_MAR_POINTS:
			x = int(landmarks.landmark[idx].x * img_width)
			y = int(landmarks.landmark[idx].y * img_height)
			mouth_points.append([x, y])
		return np.array(mouth_points)
	
	def __draw_mouth_landmarks(self, img, mouth_points, color=(0, 255, 255)):
		"""Draw mouth landmarks on the image"""
		for point in mouth_points:
			cv2.circle(img, tuple(point), 2, color, -1)
		
		# # Draw mouth contour
		# if len(mouth_points) >= 4:
		# 	pts = mouth_points.reshape((-1, 1, 2))
		# 	cv2.polylines(img, [pts], True, color, 1)
	
	def detect(self, frame):
		"""
		Process frame and detect mouth opening

		Args:
			frame: Input video frame

		Returns:
			tuple: (processed_frame, mouth_opened, mar_value)
		"""
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		results = self.face_mesh.process(rgb_frame)
		
		mouth_opened = False
		mar_value = 0
		
		if results.multi_face_landmarks:
			for face_landmarks in results.multi_face_landmarks:
				img_height, img_width = frame.shape[:2]
				
				# Extract mouth landmarks for MAR calculation
				mouth_points = self.__extract_mouth_landmarks(face_landmarks, img_width, img_height)
				
				# Extract outer mouth landmarks for visualization
				outer_mouth = self.__extract_mouth_landmarks(face_landmarks, img_width, img_height)
				
				# Calculate MAR
				mar_value = self.calculate_mar(mouth_points)
				
				# Draw mouth landmarks
				self.__draw_mouth_landmarks(frame, outer_mouth)
				
				# Check for mouth opening
				if mar_value > self.mar_threshold:
					self.frame_counter += 1
					if self.frame_counter >= self.consec_frames and not self.is_mouth_open:
						self.action_count += 1
						self.is_mouth_open = True
						mouth_opened = True
				else:
					if self.frame_counter > 0:
						self.is_mouth_open = False
					self.frame_counter = 0
				
				# Reset mouth open state when closed
				if mar_value <= self.mar_threshold * 0.7:  # Hysteresis to prevent flickering
					self.is_mouth_open = False
				
				# Draw MAR value
				cv2.putText(frame, f"MAR: {mar_value:.3f}", (300, 30),
				            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
				
				# Draw mouth status
				if self.is_mouth_open:
					cv2.putText(frame, "MOUTH OPEN", (300, 60),
					            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
				else:
					cv2.putText(frame, "MOUTH CLOSED", (300, 60),
					            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
		
		return frame, mouth_opened


