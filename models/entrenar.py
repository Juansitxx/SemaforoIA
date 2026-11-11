import sys
import os
from pathlib import Path
from unittest.mock import MagicMock
from importlib.machinery import ModuleSpec

# 1. Interceptor universal para módulos secundarios no requeridos
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

# 2. Inyectar tf_keras como motor principal de Keras (Keras 2 nativo para TF Object Detection)
import tf_keras
sys.modules['keras'] = tf_keras
import tensorflow as tf
tf.keras = tf_keras

# Evitar TBNotInstalledError en entornos sin TensorBoard completo
for _fn in ('image', 'scalar', 'histogram', 'text', 'audio', 'write', 'flush'):
    setattr(tf.summary, _fn, lambda *args, **kwargs: True)
    setattr(tf.compat.v2.summary, _fn, lambda *args, **kwargs: True)

# 3. Compatibilidad para control_flow_ops en tf_slim (decodificación de imágenes)
from tensorflow.python.ops import control_flow_ops
control_flow_ops.case = tf.case
control_flow_ops.cond = tf.cond
control_flow_ops.while_loop = tf.while_loop

# 4. Inyectar rutas locales de TensorFlow Object Detection
BASE_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = BASE_DIR / 'models' / 'research'
SLIM_DIR = RESEARCH_DIR / 'slim'
for p in [str(BASE_DIR / 'models'), str(RESEARCH_DIR), str(SLIM_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from object_detection import model_lib_v2
from object_detection import model_main_tf2
from absl import app

model_lib_v2.LOG_EVERY = 10
model_lib_v2.NUM_STEPS_PER_ITERATION = 10
_orig_train_loop = model_lib_v2.train_loop

def _patched_train_loop(*args, **kwargs):
    train_steps = kwargs.get('train_steps') or 100
    step_iter = min(10, max(1, int(train_steps)))
    kwargs['num_steps_per_iteration'] = step_iter
    kwargs['checkpoint_every_n'] = min(50, step_iter)
    return _orig_train_loop(*args, **kwargs)

model_lib_v2.train_loop = _patched_train_loop

if __name__ == '__main__':
    app.run(model_main_tf2.main)
