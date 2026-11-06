"""Check temporal sampling on an encoded video and explicit token-grid rejection."""
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from common import read_window
from explanations import token_rollout


class DataContractTests(unittest.TestCase):
    def test_decode_window_excludes_later_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'timed.avi'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10.,(64,48))
            self.assertTrue(writer.isOpened())
            for frame_number in range(100):
                writer.write(np.full((48,64,3),frame_number*2,dtype=np.uint8))
            writer.release()
            frames,times=read_window(path,2.,2.,16,10.)
            intensities=np.array([frame.mean() for frame in frames])
            self.assertEqual(len(frames),16)
            self.assertGreaterEqual(intensities.min(),38)
            self.assertLessEqual(intensities.max(),80)
            self.assertTrue(np.all(np.diff(intensities)>=0))
            self.assertAlmostEqual(times[0],2.)
            self.assertAlmostEqual(times[-1],3.9)

    def test_mean_pool_grid_preserves_every_token_and_rejects_wrong_layout(self):
        attention=torch.eye(8).reshape(1,1,8,8)
        result=token_rollout([attention],(2,2,2),'mean')
        self.assertEqual(tuple(result.shape),(1,2,2,2))
        self.assertTrue(torch.allclose(result,torch.full_like(result,1/8)))
        with self.assertRaises(ValueError):
            token_rollout([attention],(2,2,2),'cls')
        with self.assertRaises(ValueError):
            token_rollout([torch.eye(7)[None,None]],(2,2,2),'mean')


if __name__=='__main__':unittest.main()
