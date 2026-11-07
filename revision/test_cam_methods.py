"""Numerical contracts for the model-specific CAM comparison."""
import unittest

import torch

from cam_methods import activation_volume, divided_rollout, gradient_cams, joint_rollout, restore_divided_grid
from explanations import token_rollout


class CamContracts(unittest.TestCase):
    def test_explicit_token_layouts(self):
        temporal = 4
        for name, length in [('vivit', 5), ('videomae', 4)]:
            value = torch.arange(length).reshape(1, length, 1).float()
            actual = activation_volume(value, name, temporal, 1, 2)
            expected = torch.tensor([[[[[1., 2.]], [[3., 4.]]]]]) if name == 'vivit' else torch.tensor([[[[[0., 1.]], [[2., 3.]]]]])
            torch.testing.assert_close(actual, expected)
        value = torch.arange(12).reshape(4, 3, 1).float()
        actual = activation_volume(value, 'timesformer', 4, 1, 2)
        torch.testing.assert_close(actual[0, 0, :, 0], torch.tensor([[1., 2.], [4., 5.], [7., 8.], [10., 11.]]))
        with self.assertRaises(ValueError):
            activation_volume(torch.zeros(1, 7, 2), 'videomae', 4, 1, 2)

    def test_zero_gradient_and_signed_layercam(self):
        activation = torch.tensor([1., -2., 3., 4.]).reshape(1, 2, 1, 1, 2)
        gradient = torch.tensor([2., 2., -1., 1.]).reshape_as(activation)
        result = gradient_cams(activation, gradient)
        torch.testing.assert_close(result['layercam'], torch.tensor([2., 0.]).reshape(1, 1, 1, 2))
        for value in gradient_cams(activation, torch.zeros_like(gradient)).values():
            self.assertEqual(float(value.sum()), 0.)

    def test_legacy_configured_frame_grouping_is_inverted(self):
        # Four actual frames, but two configured frames: each physical patch
        # has been split into two virtual spatial positions by the old model.
        value = torch.tensor([[[-1.], [0.], [2.], [4.], [6.]],
                              [[-1.], [1.], [3.], [5.], [7.]]])
        actual = activation_volume(value, 'timesformer', 4, 1, 2, attention_frames=2)
        expected = torch.tensor([[0., 4.], [1., 5.], [2., 6.], [3., 7.]])
        torch.testing.assert_close(actual[0, 0, :, 0], expected)
        grid = value[:, 1:, 0].reshape(1, 2, 2, 2)
        torch.testing.assert_close(restore_divided_grid(grid, 4, 1, 2)[0, :, 0], expected)

    def test_joint_row_readout_equals_full_rollout(self):
        torch.manual_seed(5)
        for cls in [True, False]:
            size = 6 + int(cls)
            attentions = [torch.softmax(torch.randn(1, 2, size, size), -1) for _ in range(3)]
            expected = token_rollout(attentions, (3, 1, 2), 'cls' if cls else 'mean')
            torch.testing.assert_close(joint_rollout(attentions, (3, 1, 2), cls), expected)

    def test_divided_flow_equals_explicit_dense_transitions(self):
        torch.manual_seed(7)
        frames, patches, layers = 3, 2, 3
        spatial = [torch.softmax(torch.randn(frames, patches + 1, patches + 1), -1) for _ in range(layers)]
        temporal = [torch.softmax(torch.randn(patches, frames, frames), -1) for _ in range(layers)]
        size = 1 + patches * frames
        rollout = torch.eye(size)
        for space, time in zip(spatial, temporal):
            u, v = torch.zeros(size, size), torch.zeros(size, size)
            u[0, 0] = 1
            v[0, 0] = .5 + .5 * space[:, 0, 0].mean()
            for n in range(patches):
                for t in range(frames):
                    position = 1 + n * frames + t
                    v[0, position] = .5 / frames * space[t, 0, n + 1]
                    v[position, 0] = .5 * space[t, n + 1, 0]
                    for s in range(frames):
                        u[position, 1 + n * frames + s] = .5 * time[n, t, s] + .5 * (t == s)
                    for m in range(patches):
                        v[position, 1 + m * frames + t] = .5 * space[t, n + 1, m + 1] + .5 * (n == m)
            torch.testing.assert_close(u.sum(1), torch.ones(size))
            torch.testing.assert_close(v.sum(1), torch.ones(size))
            rollout = v @ u @ rollout
        expected = rollout[0, 1:].reshape(patches, frames).T.reshape(1, frames, 1, patches)
        torch.testing.assert_close(divided_rollout(spatial, temporal, frames, 1, patches), expected)
        with self.assertRaises(ValueError):
            divided_rollout(spatial, temporal[:-1], frames, 1, patches)


if __name__ == '__main__':
    unittest.main(verbosity=2)
