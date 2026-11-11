import sys
import os
from pathlib import Path
from unittest.mock import MagicMock
from importlib.machinery import ModuleSpec

class AutoMockFinder:
    def __init__(self, prefixes):
        self.prefixes = prefixes
    def find_spec(self, fullname, path, target=None):
        if any(fullname == p or fullname.startswith(p + '.') for p in self.prefixes):
            spec = ModuleSpec(fullname, None)
            spec.loader = self
            return spec
        return None
    def create_module(self, spec):
        mod = MagicMock()
        mod.__name__ = spec.name
        mod.__file__ = '<mock>'
        mod.__loader__ = self
        mod.__package__ = spec.name.rpartition('.')[0]
        mod.__path__ = []
        return mod
    def exec_module(self, module):
        pass

sys.meta_path.insert(0, AutoMockFinder(['official', 'tensorflow_io', 'tensorflow.compat.v1.estimator', 'seqeval', 'tensorboard']))

import tf_keras
sys.modules['keras'] = tf_keras
import tensorflow as tf
tf.keras = tf_keras

from tensorflow.python.ops import control_flow_ops
control_flow_ops.case = tf.case
control_flow_ops.cond = tf.cond
control_flow_ops.while_loop = tf.while_loop

BASE_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = BASE_DIR / 'models' / 'research'
SLIM_DIR = RESEARCH_DIR / 'slim'
for p in [str(BASE_DIR / 'models'), str(RESEARCH_DIR), str(SLIM_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from object_detection import exporter_main_v2
from absl import app

if __name__ == '__main__':
    app.run(exporter_main_v2.main)
