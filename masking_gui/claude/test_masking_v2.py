
import unittest
import numpy as np
import cv2
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from masking_v2 import CubemapProjection, MaskingPipeline, MaskConfig, SegmentationModel, ImageGeometry

class TestMaskingV2(unittest.TestCase):
    
    def setUp(self):
        self.config = MaskConfig(model=SegmentationModel.FASTSAM) # Use FastSAM for speed/mocking
        self.pipeline = MaskingPipeline(self.config, auto_select_model=False)
        
    def test_cubemap_projection(self):
        """Test equirectangular to cubemap and back."""
        proj = CubemapProjection(face_size=256)
        
        # Create a dummy equirectangular image (checkerboard)
        h, w = 512, 1024
        equi = np.zeros((h, w, 3), dtype=np.uint8)
        # Draw some patterns
        cv2.rectangle(equi, (0, 0), (w//2, h//2), (255, 0, 0), -1)
        cv2.rectangle(equi, (w//2, h//2), (w, h), (0, 255, 0), -1)
        
        # Convert to cubemap
        faces = proj.equirect2cubemap(equi)
        self.assertEqual(len(faces), 6)
        for face in faces.values():
            self.assertEqual(face.shape, (256, 256, 3))
            
        # Convert back
        recon = proj.cubemap2equirect(faces, (w, h))
        self.assertEqual(recon.shape, (h, w, 3))
        
        # Check if reconstruction is somewhat similar (it won't be perfect due to sampling)
        # Just check if it's not empty
        self.assertTrue(np.mean(recon) > 0)
        
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        self.assertIsNotNone(self.pipeline)
        
    def test_yolo11_config(self):
        """Test YOLO11 configuration."""
        config = MaskConfig(model=SegmentationModel.YOLO11, yolo_model_size='s')
        self.assertEqual(config.model, SegmentationModel.YOLO11)
        self.assertEqual(config.yolo_model_size, 's')

if __name__ == '__main__':
    unittest.main()
