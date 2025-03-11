# NumPy Compatibility Fix for TensorFlow

## Problem

When running the SoundWatch server with newer versions of NumPy (1.20+) and older versions of TensorFlow, you may encounter this error:

```
AttributeError: module 'numpy' has no attribute 'object'.
`np.object` was a deprecated alias for the builtin `object`. To avoid this error in existing code, use `object` by itself.
```

This happens because:
1. Newer versions of NumPy (1.20+) removed the deprecated `np.object` attribute
2. Older versions of TensorFlow still use `np.object` internally

## Solution

This repository includes a patch that adds the missing attribute back to NumPy. The patch has been applied to:

- `main.py`
- `test_model.py`

The patch simply adds the following code before importing TensorFlow:

```python
import numpy as np
# Apply NumPy patch for TensorFlow compatibility
if not hasattr(np, 'object'):
    np.object = object
```

## Alternative Solutions

If the patch doesn't work for you, you can try these alternatives:

1. Downgrade NumPy to a compatible version:
   ```
   pip install numpy==1.19.5
   ```

2. Upgrade TensorFlow to a newer version that doesn't use the deprecated attribute:
   ```
   pip install tensorflow>=2.4.0
   ```

## Requirements

For reference, here are the compatible versions:

- For TensorFlow 1.x: Use NumPy < 1.20
- For TensorFlow 2.0-2.3: Use NumPy < 1.20
- For TensorFlow 2.4+: NumPy 1.20+ should work

The `requirements_updated.txt` file specifies compatible versions. 