import importlib
import os
import sys
import types
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Minimal numpy stub

def _sum(x):
    if isinstance(x, list):
        return sum(_sum(i) for i in x)
    return x

def _mean(x):
    vals = list(x)
    return sum(vals) / len(vals) if vals else 0

np_stub = types.SimpleNamespace(
    array=lambda x: x,
    mean=_mean,
    sum=_sum,
    isscalar=lambda x: not isinstance(x, (list, tuple, dict))
)
sys.modules['numpy'] = np_stub

import metrics
importlib.reload(metrics)
from metrics import PixelwiseMetrics


class FakeTensor:
    def __init__(self, data):
        self.data = data

    def _apply(self, other, op):
        if isinstance(other, FakeTensor):
            other = other.data
        return FakeTensor(_apply_recursive(self.data, other, op))

    def __eq__(self, other):
        return self._apply(other, lambda a, b: 1 if a == b else 0)

    def __mul__(self, other):
        return self._apply(other, lambda a, b: a * b)

    def sum(self):
        return FakeTensor(_sum(self.data))

    def cpu(self):
        return self

    def detach(self):
        return self

    def numpy(self):
        return self.data


def _apply_recursive(a, b, op):
    if isinstance(a, list) and isinstance(b, list):
        return [_apply_recursive(x, y, op) for x, y in zip(a, b)]
    if isinstance(a, list):
        return [_apply_recursive(x, b, op) for x in a]
    if isinstance(b, list):
        return [_apply_recursive(a, y, op) for y in b]
    return op(a, b)


def test_pixelwise_metrics_correct_accuracy():
    metrics_obj = PixelwiseMetrics(num_classes=2)
    y = FakeTensor([[0, 1], [1, 0]])
    y_hat = FakeTensor([[0, 1], [0, 1]])
    metrics_obj.add_batch(y, y_hat)
    cw = metrics_obj.get_classwise_accuracy()
    assert cw['pixelclass_0'] == pytest.approx(0.5)
    assert cw['pixelclass_1'] == pytest.approx(0.5)
    assert metrics_obj.get_average_accuracy() == pytest.approx(0.5)


def test_pixelwise_metrics_handles_missing_class():
    metrics_obj = PixelwiseMetrics(num_classes=2)
    y = FakeTensor([[0, 0], [0, 0]])
    y_hat = FakeTensor([[0, 0], [0, 0]])
    metrics_obj.add_batch(y, y_hat)
    cw = metrics_obj.get_classwise_accuracy()
    assert cw['pixelclass_0'] == pytest.approx(1.0)
    assert cw['pixelclass_1'] == pytest.approx(0.0)

