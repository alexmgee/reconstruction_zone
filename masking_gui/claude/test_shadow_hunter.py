
import unittest
import numpy as np
import sys
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from masking_v2 import MaskingPipeline, MaskConfig, SegmentationModel, MaskResult, MaskQuality

class TestShadowHunter(unittest.TestCase):
    
    def setUp(self):
        self.config = MaskConfig(
            model=SegmentationModel.FASTSAM, # Use FastSAM to avoid loading big models
            enable_shadow_hunter=True
        )
        
    @patch('masking_v2.ShadowHunter')
    @patch('masking_v2.FastSAMSegmenter')
    def test_shadow_hunter_integration(self, MockFastSAM, MockShadowHunter):
        # Setup mocks
        mock_shadow_hunter = MockShadowHunter.return_value
        mock_shadow_hunter.detect_shadow.return_value = [10, 10, 50, 50] # Dummy shadow box
        
        mock_segmenter = MockFastSAM.return_value
        # Mock segment_image to return one object with a box
        mock_segmenter.segment_image.return_value = [
            MaskResult(
                mask=np.zeros((100, 100), dtype=np.uint8),
                confidence=0.9,
                quality=MaskQuality.EXCELLENT,
                metadata={'box': [20, 20, 40, 40], 'class': 'person'}
            )
        ]
        
        # Initialize pipeline
        pipeline = MaskingPipeline(self.config, auto_select_model=False)
        
        # Inject mocks (pipeline creates them internally, so we need to patch class instantiation)
        # But since we patched the class in the decorator, pipeline.shadow_hunter should be our mock instance
        # Let's verify
        self.assertIsNotNone(pipeline.shadow_hunter)
        
        # Process dummy image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = pipeline.process_image(image)
        
        # Verify detect_shadow was called
        mock_shadow_hunter.detect_shadow.assert_called_once()
        
        # Verify we got a shadow result
        # The pipeline combines masks, so we check if the combined mask is not empty
        # Or we can check if the shadow result was added to the list internally
        # But process_image returns a combined mask.
        # Since our mock shadow box is [10, 10, 50, 50], the mask should have pixels there.
        
        # Actually, let's check if the pipeline logic extended the results list
        # We can't easily inspect local variables.
        # But we can check if `detect_shadow` was called with the correct box.
        args, _ = mock_shadow_hunter.detect_shadow.call_args
        self.assertEqual(args[1], [20, 20, 40, 40])

if __name__ == '__main__':
    unittest.main()
