"""
TensorFlow Compatibility Layer

This module provides compatibility between TensorFlow 1.x and 2.x APIs
for the SoundWatch server. Import this module at the top of any file 
that uses TensorFlow to ensure compatibility with TensorFlow 2.x.
"""

import tensorflow as tf
import numpy as np
from functools import wraps
import warnings
import logging

logger = logging.getLogger(__name__)

# Check TensorFlow version
TF_VERSION = tf.__version__
IS_TF2 = TF_VERSION.startswith('2.')

if IS_TF2:
    import tensorflow.compat.v1 as tf1
    tf1.disable_eager_execution()
    tf1.disable_v2_behavior()
    logger.info(f"TensorFlow {TF_VERSION} detected. Enabling compatibility mode.")
else:
    tf1 = tf
    logger.info(f"TensorFlow {TF_VERSION} detected. Using native APIs.")

# Create compatibility functions
def get_default_graph():
    """Compatibility function for tf.get_default_graph()"""
    if IS_TF2:
        return tf1.get_default_graph()
    else:
        return tf.get_default_graph()

def load_model_compat(model_path):
    """Compatibility function for loading Keras models"""
    try:
        if IS_TF2:
            # Try to load with tf.keras first
            model = tf.keras.models.load_model(model_path, compile=False)
            logger.info("Model loaded with tf.keras")
            return model
        else:
            # Use standard keras in TF1
            from keras.models import load_model
            return load_model(model_path)
    except Exception as e:
        logger.error(f"Error loading model with primary method: {e}")
        try:
            # Fallback to direct h5py loading if available
            warnings.warn("Falling back to h5py model loading. This may affect model functionality.")
            import h5py
            from keras.models import model_from_json
            
            # Load model architecture
            with h5py.File(model_path, 'r') as f:
                model_json = f.attrs.get('model_config')
                if model_json is None:
                    model_json = f.attrs.get('model_config').decode('utf-8')
                model = model_from_json(model_json)
                
                # Load weights
                model.load_weights(model_path)
                return model
        except Exception as inner_e:
            logger.error(f"Error with fallback loading: {inner_e}")
            raise RuntimeError(f"Failed to load model: {e}\nAnd fallback failed: {inner_e}")

def numpy_compat_fromstring(string, dtype=None, sep=''):
    """Compatibility function for np.fromstring() which is deprecated"""
    try:
        if dtype is None:
            return np.frombuffer(string.encode() if isinstance(string, str) else string)
        else:
            if isinstance(string, str):
                if sep:
                    # Handle string with separators
                    return np.array(string.split(sep), dtype=dtype)
                else:
                    # Handle raw string
                    return np.frombuffer(string.encode(), dtype=dtype)
            else:
                # Handle bytes or other non-string
                return np.frombuffer(string, dtype=dtype)
    except Exception as e:
        logger.error(f"Error in numpy_compat_fromstring: {e}, falling back to fromstring")
        # Fallback to fromstring with warning
        warnings.warn("Using deprecated np.fromstring(), consider updating code to use np.frombuffer()")
        if sep and isinstance(string, str):
            return np.array(string.split(sep), dtype=dtype)
        return np.fromstring(string, dtype=dtype, sep=sep)

# Define a decorator for graph context
def with_graph_context(func):
    """Decorator to ensure function runs with the right graph context in TF1.x or TF2.x"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if IS_TF2:
            with tf1.get_default_graph().as_default():
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return wrapper

# Export these references for compatibility
graph = get_default_graph() 