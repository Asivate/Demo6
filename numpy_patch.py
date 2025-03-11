#!/usr/bin/env python3
"""
NumPy compatibility patch for TensorFlow

This script patches NumPy to add back the deprecated 'object' attribute
that was removed in NumPy 1.20+ but is still used by older TensorFlow versions.

Usage:
    Import this module before importing TensorFlow:
    
    import numpy_patch  # Apply patch
    import tensorflow as tf  # Now TensorFlow should work
"""

import numpy as np
import sys

# Check if numpy.object is missing
if not hasattr(np, 'object'):
    # Add the missing attribute
    np.object = object
    print("NumPy patch applied: Added np.object compatibility for TensorFlow") 