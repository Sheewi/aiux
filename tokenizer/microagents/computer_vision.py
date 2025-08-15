"""
Computer Vision Agent using OpenCV + ONNX Runtime

This microagent specializes in image processing and computer vision tasks
with optimized model inference.
"""

import cv2
import numpy as np
import onnxruntime as ort
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import base64
import io
from PIL import Image
import requests

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result of an object detection operation."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    

@dataclass
class VisionResult:
    """Result of a computer vision operation."""
    success: bool
    operation: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    error: Optional[str] = None


class ONNXModelManager:
    """Manager for ONNX model loading and inference."""
    
    def __init__(self):
        self.models = {}
        self.sessions = {}
        
    def load_model(self, model_name: str, model_path: str, 
                  class_names: List[str] = None) -> bool:
        """
        Load an ONNX model for inference.
        
        Args:
            model_name: Identifier for the model
            model_path: Path to ONNX model file
            class_names: List of class names for the model
            
        Returns:
            True if model loaded successfully
        """
        try:
            # Create ONNX Runtime session
            session = ort.InferenceSession(model_path)
            
            self.sessions[model_name] = session
            self.models[model_name] = {
                "path": model_path,
                "class_names": class_names or [],
                "input_shape": session.get_inputs()[0].shape,
                "input_name": session.get_inputs()[0].name,
                "output_names": [output.name for output in session.get_outputs()]
            }
            
            logger.info(f"Loaded ONNX model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model {model_name}: {e}")
            return False
            
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a loaded model."""
        return self.models.get(model_name)
        
    def predict(self, model_name: str, input_data: np.ndarray) -> Optional[List[np.ndarray]]:
        """
        Run inference on a loaded model.
        
        Args:
            model_name: Name of the model to use
            input_data: Preprocessed input data
            
        Returns:
            List of output arrays
        """
        if model_name not in self.sessions:
            logger.error(f"Model {model_name} not loaded")
            return None
            
        try:
            session = self.sessions[model_name]
            model_info = self.models[model_name]
            
            input_name = model_info["input_name"]
            outputs = session.run(None, {input_name: input_data})
            
            return outputs
            
        except Exception as e:
            logger.error(f"Inference failed for model {model_name}: {e}")
            return None


class ComputerVisionAgent:
    """
    Computer vision agent with OpenCV and ONNX Runtime integration.
    
    Features:
    - Image preprocessing and enhancement
    - Object detection and classification
    - Image segmentation
    - Feature extraction and matching
    - Real-time video processing
    - Custom model inference
    """
    
    def __init__(self):
        self.model_manager = ONNXModelManager()
        
    def load_image(self, source: Union[str, np.ndarray, bytes]) -> Optional[np.ndarray]:
        """
        Load image from various sources.
        
        Args:
            source: File path, URL, numpy array, or bytes
            
        Returns:
            Image as numpy array (BGR format)
        """
        try:
            if isinstance(source, np.ndarray):
                return source
            elif isinstance(source, bytes):
                # Convert bytes to image
                nparr = np.frombuffer(source, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return image
            elif isinstance(source, str):
                if source.startswith(('http://', 'https://')):
                    # Download from URL
                    response = requests.get(source)
                    response.raise_for_status()
                    nparr = np.frombuffer(response.content, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    return image
                else:
                    # Load from file path
                    image = cv2.imread(source)
                    return image
            else:
                logger.error(f"Unsupported image source type: {type(source)}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return None
            
    def preprocess_image(self, image: np.ndarray, 
                        target_size: Tuple[int, int] = (640, 640),
                        normalize: bool = True,
                        to_rgb: bool = False) -> np.ndarray:
        """
        Preprocess image for model inference.
        
        Args:
            image: Input image
            target_size: Target dimensions (width, height)
            normalize: Whether to normalize pixel values
            to_rgb: Whether to convert BGR to RGB
            
        Returns:
            Preprocessed image
        """
        try:
            # Resize image
            processed = cv2.resize(image, target_size)
            
            # Convert color space if needed
            if to_rgb:
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                
            # Normalize pixel values
            if normalize:
                processed = processed.astype(np.float32) / 255.0
                
            return processed
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return image
            
    def detect_objects_yolo(self, image: np.ndarray, 
                          model_name: str = "yolo",
                          confidence_threshold: float = 0.5,
                          nms_threshold: float = 0.4) -> VisionResult:
        """
        Detect objects using YOLO model.
        
        Args:
            image: Input image
            model_name: Name of loaded YOLO model
            confidence_threshold: Minimum confidence for detections
            nms_threshold: NMS threshold for duplicate removal
            
        Returns:
            VisionResult with detection results
        """
        try:
            model_info = self.model_manager.get_model_info(model_name)
            if not model_info:
                return VisionResult(
                    success=False,
                    operation="object_detection",
                    data={},
                    metadata={},
                    error=f"Model {model_name} not loaded"
                )
                
            # Preprocess image
            input_shape = model_info["input_shape"]
            if len(input_shape) == 4:  # Batch dimension
                target_size = (input_shape[3], input_shape[2])
            else:
                target_size = (640, 640)
                
            processed = self.preprocess_image(image, target_size, normalize=True, to_rgb=True)
            
            # Prepare input for model
            if len(processed.shape) == 3:
                processed = np.expand_dims(processed, axis=0)  # Add batch dimension
            processed = np.transpose(processed, (0, 3, 1, 2))  # NCHW format
            
            # Run inference
            outputs = self.model_manager.predict(model_name, processed)
            if not outputs:
                return VisionResult(
                    success=False,
                    operation="object_detection",
                    data={},
                    metadata={},
                    error="Inference failed"
                )
                
            # Post-process results
            detections = self._postprocess_yolo_output(
                outputs[0], image.shape[:2], target_size,
                confidence_threshold, nms_threshold,
                model_info.get("class_names", [])
            )
            
            return VisionResult(
                success=True,
                operation="object_detection",
                data={"detections": detections},
                metadata={
                    "model": model_name,
                    "confidence_threshold": confidence_threshold,
                    "num_detections": len(detections)
                }
            )
            
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return VisionResult(
                success=False,
                operation="object_detection",
                data={},
                metadata={},
                error=str(e)
            )
            
    def _postprocess_yolo_output(self, output: np.ndarray, 
                                original_shape: Tuple[int, int],
                                input_shape: Tuple[int, int],
                                confidence_threshold: float,
                                nms_threshold: float,
                                class_names: List[str]) -> List[DetectionResult]:
        """Post-process YOLO model output."""
        detections = []
        
        try:
            # Extract boxes, scores, and class IDs
            boxes = []
            confidences = []
            class_ids = []
            
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > confidence_threshold:
                    # Scale bounding box back to original image size
                    center_x = int(detection[0] * original_shape[1] / input_shape[0])
                    center_y = int(detection[1] * original_shape[0] / input_shape[1])
                    width = int(detection[2] * original_shape[1] / input_shape[0])
                    height = int(detection[3] * original_shape[0] / input_shape[1])
                    
                    # Convert to top-left corner format
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)
                    
                    boxes.append([x, y, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    
            # Apply Non-Maximum Suppression
            if boxes:
                indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
                
                if len(indices) > 0:
                    for i in indices.flatten():
                        x, y, w, h = boxes[i]
                        class_name = class_names[class_ids[i]] if class_ids[i] < len(class_names) else f"class_{class_ids[i]}"
                        
                        detections.append(DetectionResult(
                            class_id=class_ids[i],
                            class_name=class_name,
                            confidence=confidences[i],
                            bbox=(x, y, w, h)
                        ))
                        
        except Exception as e:
            logger.error(f"YOLO post-processing failed: {e}")
            
        return detections
        
    def extract_features(self, image: np.ndarray, 
                        method: str = "orb") -> VisionResult:
        """
        Extract features from image.
        
        Args:
            image: Input image
            method: Feature extraction method ('orb', 'sift', 'surf')
            
        Returns:
            VisionResult with extracted features
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            if method.lower() == "orb":
                detector = cv2.ORB_create()
            elif method.lower() == "sift":
                detector = cv2.SIFT_create()
            elif method.lower() == "surf":
                detector = cv2.xfeatures2d.SURF_create()
            else:
                return VisionResult(
                    success=False,
                    operation="feature_extraction",
                    data={},
                    metadata={},
                    error=f"Unsupported feature extraction method: {method}"
                )
                
            keypoints, descriptors = detector.detectAndCompute(gray, None)
            
            # Convert keypoints to serializable format
            kp_data = []
            for kp in keypoints:
                kp_data.append({
                    "x": float(kp.pt[0]),
                    "y": float(kp.pt[1]),
                    "size": float(kp.size),
                    "angle": float(kp.angle),
                    "response": float(kp.response)
                })
                
            return VisionResult(
                success=True,
                operation="feature_extraction",
                data={
                    "keypoints": kp_data,
                    "descriptors": descriptors.tolist() if descriptors is not None else [],
                    "num_features": len(keypoints)
                },
                metadata={
                    "method": method,
                    "image_shape": image.shape
                }
            )
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return VisionResult(
                success=False,
                operation="feature_extraction",
                data={},
                metadata={},
                error=str(e)
            )
            
    def enhance_image(self, image: np.ndarray, 
                     operations: List[str] = None) -> VisionResult:
        """
        Enhance image using various techniques.
        
        Args:
            image: Input image
            operations: List of enhancement operations
            
        Returns:
            VisionResult with enhanced image
        """
        try:
            enhanced = image.copy()
            applied_ops = []
            
            operations = operations or ["denoise", "sharpen", "contrast"]
            
            for op in operations:
                if op == "denoise":
                    enhanced = cv2.fastNlMeansDenoisingColored(enhanced)
                    applied_ops.append("denoise")
                elif op == "sharpen":
                    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                    enhanced = cv2.filter2D(enhanced, -1, kernel)
                    applied_ops.append("sharpen")
                elif op == "contrast":
                    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=10)
                    applied_ops.append("contrast")
                elif op == "gamma":
                    gamma = 1.2
                    inv_gamma = 1.0 / gamma
                    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                                    for i in np.arange(0, 256)]).astype("uint8")
                    enhanced = cv2.LUT(enhanced, table)
                    applied_ops.append("gamma")
                    
            # Convert to base64 for output
            _, buffer = cv2.imencode('.jpg', enhanced)
            enhanced_b64 = base64.b64encode(buffer).decode()
            
            return VisionResult(
                success=True,
                operation="image_enhancement",
                data={
                    "enhanced_image_b64": enhanced_b64,
                    "applied_operations": applied_ops
                },
                metadata={
                    "original_shape": image.shape,
                    "enhanced_shape": enhanced.shape
                }
            )
            
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            return VisionResult(
                success=False,
                operation="image_enhancement",
                data={},
                metadata={},
                error=str(e)
            )
            
    def analyze_image(self, image: np.ndarray) -> VisionResult:
        """
        Perform comprehensive image analysis.
        
        Args:
            image: Input image
            
        Returns:
            VisionResult with analysis results
        """
        try:
            analysis = {}
            
            # Basic properties
            height, width = image.shape[:2]
            channels = image.shape[2] if len(image.shape) > 2 else 1
            
            analysis["dimensions"] = {
                "width": width,
                "height": height,
                "channels": channels,
                "total_pixels": width * height
            }
            
            # Color analysis
            if channels == 3:
                # Color statistics
                mean_color = np.mean(image, axis=(0, 1))
                std_color = np.std(image, axis=(0, 1))
                
                analysis["color_stats"] = {
                    "mean_bgr": mean_color.tolist(),
                    "std_bgr": std_color.tolist(),
                    "brightness": float(np.mean(mean_color)),
                    "contrast": float(np.mean(std_color))
                }
                
                # Dominant colors using K-means
                data = image.reshape((-1, 3))
                data = np.float32(data)
                
                k = 5  # Number of dominant colors
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
                _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                
                analysis["dominant_colors"] = centers.tolist()
                
            # Edge detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if channels == 3 else image
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (width * height)
            
            analysis["edges"] = {
                "edge_density": float(edge_density),
                "total_edge_pixels": int(np.sum(edges > 0))
            }
            
            # Blur detection using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            analysis["blur_score"] = float(laplacian_var)
            analysis["is_blurry"] = laplacian_var < 100
            
            # Histogram analysis
            hist_data = {}
            if channels == 3:
                for i, color in enumerate(['blue', 'green', 'red']):
                    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                    hist_data[color] = hist.flatten().tolist()
            else:
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                hist_data['gray'] = hist.flatten().tolist()
                
            analysis["histograms"] = hist_data
            
            return VisionResult(
                success=True,
                operation="image_analysis",
                data=analysis,
                metadata={
                    "analysis_timestamp": "now",
                    "opencv_version": cv2.__version__
                }
            )
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return VisionResult(
                success=False,
                operation="image_analysis",
                data={},
                metadata={},
                error=str(e)
            )


# Convenience functions for quick usage
def quick_object_detection(image_path: str, model_path: str = None) -> List[DetectionResult]:
    """Quick object detection on an image."""
    agent = ComputerVisionAgent()
    
    # Load a default model if available
    if model_path and Path(model_path).exists():
        agent.model_manager.load_model("default", model_path)
        
    image = agent.load_image(image_path)
    if image is None:
        return []
        
    result = agent.detect_objects_yolo(image, "default")
    if result.success:
        return result.data.get("detections", [])
    return []


def quick_image_analysis(image_path: str) -> Dict[str, Any]:
    """Quick comprehensive image analysis."""
    agent = ComputerVisionAgent()
    image = agent.load_image(image_path)
    
    if image is None:
        return {"success": False, "error": "Could not load image"}
        
    result = agent.analyze_image(image)
    return result.data if result.success else {"success": False, "error": result.error}


if __name__ == "__main__":
    # Example usage
    agent = ComputerVisionAgent()
    
    # Load an example image
    # image = agent.load_image("path/to/image.jpg")
    # 
    # if image is not None:
    #     # Analyze the image
    #     analysis = agent.analyze_image(image)
    #     print(f"Analysis success: {analysis.success}")
    #     
    #     # Extract features
    #     features = agent.extract_features(image)
    #     print(f"Extracted {features.data.get('num_features', 0)} features")
    #     
    #     # Enhance the image
    #     enhanced = agent.enhance_image(image)
    #     print(f"Enhancement success: {enhanced.success}")
    
    print("Computer Vision Agent initialized successfully!")
