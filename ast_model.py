import os
import torch
import numpy as np
import soundfile as sf
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import warnings
import time
import traceback
from tensorflow.keras import layers, models

# Suppress transformers warnings
warnings.filterwarnings("ignore", category=UserWarning)

def check_float16_support():
    """Check if the system supports float16 operations in a backward-compatible way."""
    try:
        # For server stability, we'll be conservative and recommend not using float16
        # unless we're 100% certain it's compatible with the system
        
        # Try the new way (newer PyTorch versions)
        if hasattr(torch.backends, 'cpu') and hasattr(torch.backends.cpu, 'supports_float16'):
            return False  # Returning False for now as we've seen compatibility issues
        
        # Since we're having issues with float16, return False to use float32 instead
        return False
    except Exception as e:
        print(f"Error checking float16 support: {e}")
        return False

def load_ast_model(model_name="MIT/ast-finetuned-audioset-10-10-0.4593", **kwargs):
    """
    Load the Audio Spectrogram Transformer model and feature extractor from Hugging Face
    
    Args:
        model_name (str): The name of the model on Hugging Face
        **kwargs: Additional arguments to pass to the from_pretrained method
            - attn_implementation: "eager" or "sdpa" for Scaled Dot Product Attention
            - torch_dtype: torch.float32, torch.float16 or torch.bfloat16
        
    Returns:
        tuple: (model, feature_extractor)
    """
    print(f"Loading AST model: {model_name}")
    try:
        # Load feature extractor
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        
        # Check for optimization parameters
        attn_implementation = kwargs.get("attn_implementation", None)
        
        # Use standard precision for compatibility
        print("Using standard precision (float32) for maximum compatibility")
        if "torch_dtype" in kwargs:
            del kwargs["torch_dtype"]  # Remove any torch_dtype setting
        
        # Print optimization settings
        if attn_implementation:
            print(f"Using attention implementation: {attn_implementation}")
        
        # Load model - explicitly use float32 regardless of what was passed in kwargs
        model = AutoModelForAudioClassification.from_pretrained(model_name, torch_dtype=torch.float32, **kwargs)
        
        # Explicitly convert any half-precision parameters to float32
        for param in model.parameters():
            if param.dtype != torch.float32:
                param.data = param.data.to(torch.float32)
        
        # Set all model parameters to float32 to avoid mixed precision issues
        model = model.to(torch.float32)
        
        print("Verified all model parameters are using float32 precision")
        
        # Put model in evaluation mode and move to CPU (or GPU if available)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model = model.to(device)
        
        return model, feature_extractor
    except Exception as e:
        print(f"Error loading AST model: {e}")
        raise e

# Map AST model labels to homesounds labels
# This mapping is based on semantic similarity between the labels from both systems
def map_ast_labels_to_homesounds(predictions, threshold=0.05):
    """
    Map AST model predictions to homesounds labels
    
    Args:
        predictions: List of prediction dictionaries from AST model
        threshold: Confidence threshold for predictions (default: 0.05)
        
    Returns:
        list: Mapped predictions with homesounds labels
    """
    # Comprehensive mapping from AST/AudioSet labels to homesounds labels
    # These mappings are based on semantic similarity
    ast_to_homesounds = {
        # Test sounds (for development and testing)
        "Sine wave": "hazard-alarm",  # Map sine wave to hazard-alarm for testing
        "Beep, bleep": "hazard-alarm",
        "Chirp tone": "hazard-alarm",
        "Dial tone": "phone-ring",
        "Busy signal": "phone-ring",
        "White noise": "water-running",
        "Static": "water-running",
        "Noise": "water-running",
        
        # Dog sounds
        "Dog": "dog-bark",
        "Bark": "dog-bark",
        "Howl": "dog-bark",
        
        # Tool sounds
        "Drill": "drill",
        "Drilling": "drill",
        "Power tool": "drill",
        "Electric drill": "drill",
        "Hammer": "hammer",
        "Hammering": "hammer",
        "Saw": "saw",
        "Chainsaw": "saw",
        "Vacuum cleaner": "vacuum",
        "Vacuum": "vacuum",
        
        # Alarms
        "Smoke alarm": "hazard-alarm",
        "Fire alarm": "hazard-alarm",
        "Alarm": "hazard-alarm",
        "Carbon monoxide detector": "hazard-alarm",
        "Siren": "hazard-alarm",
        "Emergency vehicle": "hazard-alarm",
        "Smoke detector": "hazard-alarm",
        "Alarm clock": "alarm-clock",
        "Beep": "hazard-alarm",
        
        # Phones
        "Telephone": "phone-ring",
        "Telephone bell ringing": "phone-ring",
        "Ringtone": "phone-ring",
        "Ringer": "phone-ring",
        "Cell phone": "phone-ring",
        "Telephone ringing": "phone-ring",
        
        # Human sounds
        "Speech": "speech",
        "Male speech": "male-conversation",
        "Female speech": "female-conversation",
        "Conversation": "conversation",
        "Narration": "conversation",
        "Talk": "conversation",
        "Child speech": "conversation",
        "Crying, sobbing": "baby-cry",
        "Baby cry": "baby-cry",
        "Baby crying": "baby-cry",
        "Infant cry": "baby-cry",
        "Cough": "cough",
        "Coughing": "cough",
        "Snore": "snore",
        "Snoring": "snore",
        "Typing": "typing",
        "Keyboard": "typing",
        "Computer keyboard": "typing",
        "Typing on keyboard": "typing",
        "Keypress": "typing",
        "Laugh": "laugh",
        "Laughter": "laugh",
        "Chuckle": "laugh",
        "Giggle": "laugh",
        
        # Kitchen appliances and activities
        "Blender": "blender",
        "Food processor": "blender",
        "Mixer": "blender",
        "Microwave oven": "microwave",
        "Microwave": "microwave",
        "Chopping (food)": "chopping",
        "Knife": "chopping",
        "Cutting": "chopping",
        "Slicing": "chopping",
        "Dishwasher": "dishwasher",
        "Frying (food)": "cooking",
        "Sizzle": "cooking",
        "Frying pan": "cooking",
        "Cooking": "cooking",
        "Cutlery": "cooking",
        "Silverware": "cooking",
        "Food": "cooking",
        
        # Door sounds
        "Door": "door",
        "Slam": "door",
        "Knock": "knock",
        "Knocking": "knock",
        "Doorbell": "doorbell",
        "Ding-dong": "doorbell",
        "Bell": "doorbell",
        
        # Bathroom sounds
        "Water": "water-running",
        "Water tap": "water-running",
        "Running water": "water-running",
        "Sink (filling or washing)": "water-running",
        "Bathtub (filling or washing)": "water-running",
        "Faucet": "water-running",
        "Toilet flush": "flush",
        "Toilet": "flush",
        "Flush toilet": "flush",
        "Electric toothbrush": "toothbrush",
        "Toothbrush": "toothbrush",
        "Electric razor": "shaver",
        "Shaver": "shaver",
        "Electric shaver": "shaver",
        "Hair dryer": "hair-dryer",
        "Blow-dryer": "hair-dryer",
        
        # Vehicle sounds
        "Car": "engine",
        "Vehicle": "engine",
        "Engine": "engine",
        "Motor vehicle (road)": "engine",
        "Engine starting": "engine",
        "Car passing by": "engine",
        "Traffic noise": "engine",
        "Road noise": "engine",
        "Car horn": "car-horn",
        "Honk": "car-horn",
        "Beeping": "car-horn",
        
        # Animal sounds
        "Cat": "cat-meow",
        "Meow": "cat-meow",
        "Cat meowing": "cat-meow",
        
        # NEW: Hand sounds - special focus on finger snapping
        "Finger snapping": "finger-snap",
        "Snap": "finger-snap",
        "Clap": "hand-clap",
        "Clapping": "hand-clap",
        "Hands": "hand-sounds",
        "Slap": "hand-sounds",
        "Tap": "hand-sounds",
        "Percussion": "hand-sounds",
        "Click": "hand-sounds",
        "Applause": "applause",
        
        # NEW: Ambient and silence related
        "Silence": "silence",
        "Quiet": "silence",
        "Room tone": "silence",
        "Background noise": "background",
        "Ambient noise": "background",
        "Inside, small room": "background",
        "Inside, large room": "background",
        "Rustling": "background",
        
        # NEW: Music and instruments
        "Music": "music",
        "Musical instrument": "music",
        "Singing": "music",
        "Guitar": "music",
        "Piano": "music",
        "Drum": "music",
        "Percussion": "music",
        "Wind instrument": "music",
        "String instrument": "music",
        "Plucked string instrument": "music",
        "Brass instrument": "music",
        "Synthesizer": "music",
        "Sound effect": "sound-effect",
        
        # NEW: Electronic sounds
        "Electronic sound": "electronic-sound",
        "Computer": "electronic-sound",
        "Electronic device": "electronic-sound",
        "Notification": "notification",
        "Alert": "notification",
        "Chime": "notification",
    }
    
    # Map predictions to homesounds categories
    mapped_predictions = []
    
    # Special handling for finger snapping with lower threshold
    finger_snap_prediction = None
    finger_snap_confidence = 0
    
    for pred in predictions:
        # Extract label and confidence
        ast_label = pred["label"]
        confidence = pred["confidence"]
        
        # Special case for finger snapping
        if ast_label == "Finger snapping" or ast_label == "Snap":
            if confidence > finger_snap_confidence:
                finger_snap_prediction = {
                    "original_label": ast_label,
                    "label": "finger-snap",
                    "confidence": confidence
                }
                finger_snap_confidence = confidence
        
        # Check if this label is mapped
        if ast_label in ast_to_homesounds:
            homesound_label = ast_to_homesounds[ast_label]
            mapped_predictions.append({
                "original_label": ast_label,
                "label": homesound_label,
                "confidence": confidence
            })
    
    # Add finger snapping if detected, even with lower confidence
    if finger_snap_prediction and finger_snap_confidence > 0.03:  # Lower threshold just for finger snapping
        # If already in mapped_predictions, remove it (to avoid duplicates)
        mapped_predictions = [p for p in mapped_predictions if p["label"] != "finger-snap"]
        mapped_predictions.append(finger_snap_prediction)
    
    # Filter by threshold (except for the special case handled above)
    mapped_predictions = [p for p in mapped_predictions if p["confidence"] >= threshold or p["label"] == "finger-snap"]
    
    # Sort by confidence
    mapped_predictions = sorted(mapped_predictions, key=lambda x: x["confidence"], reverse=True)
    
    return mapped_predictions

def preprocess_audio_for_ast(audio_data, sample_rate, feature_extractor):
    """
    Preprocess audio data specifically for AST model
    
    Args:
        audio_data (np.ndarray): Raw audio samples
        sample_rate (int): Sample rate of the audio
        feature_extractor: The feature extractor for AST
        
    Returns:
        torch.Tensor: Processed input for AST model
    """
    # Ensure the audio data is the right shape and type
    if len(audio_data.shape) > 1:
        # If multi-channel, convert to mono by averaging
        audio_data = np.mean(audio_data, axis=1)
    
    # Normalize audio data
    if np.abs(audio_data).max() > 1.0:
        audio_data = audio_data / np.abs(audio_data).max()
    
    # Convert to float32 before feature extraction to ensure consistent precision
    audio_data = audio_data.astype(np.float32)
    
    # Extract features using the feature extractor
    inputs = feature_extractor(
        audio_data, 
        sampling_rate=sample_rate, 
        return_tensors="pt",
        padding=True
    )
    
    # This is critical: ensure inputs are float32 for compatibility
    inputs = {k: v.to(torch.float32) for k, v in inputs.items()}
    
    return inputs

def predict_sound(audio_data, sample_rate, model, feature_extractor, threshold=0.05, top_k=5):
    """
    Process audio input and return predictions
    
    Args:
        audio_data (np.ndarray): Audio data as a numpy array
        sample_rate (int): Sample rate of the audio data
        model: The AST model
        feature_extractor: The feature extractor for the AST model
        threshold (float): Confidence threshold for predictions
        top_k (int): Number of top predictions to return
    
    Returns:
        dict: Dictionary with raw and mapped predictions
    """
    try:
        start_time = time.time()
        
        # Ensure the audio data is the right shape and type
        if len(audio_data.shape) > 1:
            # If multi-channel, convert to mono by averaging
            audio_data = np.mean(audio_data, axis=1)
        
        # Normalize audio data
        if np.abs(audio_data).max() > 1.0:
            audio_data = audio_data / np.abs(audio_data).max()
        
        # Force float32 precision for audio data
        audio_data = audio_data.astype(np.float32)
        
        # Calculate RMS and dB for silence detection
        rms = np.sqrt(np.mean(audio_data**2))
        db_level = 20 * np.log10(rms) if rms > 0 else -100
        
        print(f"Audio data shape: {audio_data.shape}, sample rate: {sample_rate}")
        
        # Check for silence first (very low volume)
        if db_level < -65:
            print(f"Silence detected (db level: {db_level:.2f})")
            return {
                "top_predictions": [{"label": "Silence", "confidence": 0.95}],
                "mapped_predictions": [{"original_label": "Silence", "label": "silence", "confidence": 0.95}],
                "raw_predictions": None  # No raw predictions for silence
            }
        
        # Debug information about model parameter dtypes
        param_dtypes = {}
        for name, param in model.named_parameters():
            if param.dtype not in param_dtypes:
                param_dtypes[param.dtype] = 0
            param_dtypes[param.dtype] += 1
        
        print(f"Model is using dtype: {next(model.parameters()).dtype}")
        
        try:
            # Use the preprocess function to prepare audio input
            inputs = preprocess_audio_for_ast(audio_data, sample_rate, feature_extractor)
            
            # Get the model's device
            device = next(model.parameters()).device
            
            # Ensure all tensors are float32 and on the correct device
            if isinstance(inputs, dict):
                for key in inputs:
                    # First check the dtype and device before conversion to avoid unnecessary operations
                    if inputs[key].dtype != torch.float32:
                        try:
                            print(f"Converting input tensor {key} from {inputs[key].dtype} to float32")
                            inputs[key] = inputs[key].to(dtype=torch.float32)
                        except Exception as e:
                            print(f"Error converting {key} to float32: {str(e)}")
                            # Fallback to safe conversion
                            inputs[key] = inputs[key].float()
                    
                    # Move to the model's device if needed
                    if inputs[key].device != device:
                        try:
                            print(f"Moving input tensor {key} from {inputs[key].device} to {device}")
                            inputs[key] = inputs[key].to(device=device)
                        except Exception as e:
                            print(f"Error moving {key} to device {device}: {str(e)}")
                            # Try with cpu first then move
                            inputs[key] = inputs[key].to("cpu").to(device)
            
            print(f"Feature extraction completed. Running inference...")
            
            # Run inference with no gradient tracking for efficiency
            with torch.no_grad():
                try:
                    # Verify model parameters are float32
                    for name, param in model.named_parameters():
                        if param.dtype != torch.float32:
                            print(f"Warning: Parameter {name} has dtype {param.dtype}, converting to float32")
                            param.data = param.data.to(torch.float32)
                    
                    # Run the model with the prepared inputs, explicit cast to float32
                    outputs = model(**inputs)
                    logits = outputs.logits.float()  # Ensure logits are float32
                    
                    # Apply softmax to get probabilities
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    
                    # Get the model's id2label mapping
                    id2label = model.config.id2label
                    
                    # Convert to numpy for easier processing
                    probs_np = probs[0].cpu().numpy()
                    
                    # Store the raw predictions for aggregation
                    raw_predictions = probs_np
                    
                    # Process predictions to get the top results with label information
                    predictions = process_predictions(probs_np, id2label, threshold, top_k)
                    
                    # Also map AST predictions to the homesounds categories if possible
                    mapped_predictions = map_ast_labels_to_homesounds(predictions["top_predictions"], threshold)
                    
                    # Include the raw probability vector for aggregation
                    predictions["raw_predictions"] = raw_predictions
                    predictions["mapped_predictions"] = mapped_predictions
                    
                    # Record and log the prediction time
                    end_time = time.time()
                    elapsed = end_time - start_time
                    print(f"AST prediction completed in {elapsed:.2f} seconds")
                    
                    return predictions
                    
                except RuntimeError as e:
                    if "Expected all tensors to be on the same device" in str(e):
                        print(f"Device error in model inference: {str(e)}")
                        print("Attempting recovery with CPU...")
                        # Move model to CPU as a fallback
                        model = model.to("cpu")
                        
                        # Move all inputs to CPU
                        for key in inputs:
                            inputs[key] = inputs[key].to("cpu")
                        
                        # Retry with everything on CPU
                        outputs = model(**inputs)
                        logits = outputs.logits
                        probs = torch.nn.functional.softmax(logits, dim=-1)
                        id2label = model.config.id2label
                        probs_np = probs[0].cpu().numpy()
                        raw_predictions = probs_np
                        predictions = process_predictions(probs_np, id2label, threshold, top_k)
                        mapped_predictions = map_ast_labels_to_homesounds(predictions["top_predictions"], threshold)
                        predictions["raw_predictions"] = raw_predictions
                        predictions["mapped_predictions"] = mapped_predictions
                        
                        end_time = time.time()
                        elapsed = end_time - start_time
                        print(f"AST prediction completed in {elapsed:.2f} seconds (recovery path)")
                        
                        return predictions
                    else:
                        # Rethrow if it's not a device error
                        raise
                        
        except Exception as e:
            print(f"Error in feature extraction or model inference: {str(e)}")
            traceback.print_exc()
            # Return a fallback result
            return {
                "top_predictions": [{"label": "Error", "confidence": 0.0}],
                "mapped_predictions": [],
                "raw_predictions": None
            }
    
    except Exception as e:
        print(f"Critical error in AST prediction: {str(e)}")
        traceback.print_exc()
        return {
            "top_predictions": [{"label": "Error", "confidence": 0.0}],
            "mapped_predictions": [],
            "raw_predictions": None
        }

def process_predictions(probs_np, id2label, threshold=0.05, top_k=5):
    """
    Process raw model predictions to get top-k predictions with labels
    
    Args:
        probs_np (np.ndarray): Raw probabilities from the model
        id2label (dict): Mapping from prediction indices to label names
        threshold (float): Confidence threshold for predictions
        top_k (int): Number of top predictions to return
        
    Returns:
        dict: Dictionary with processed predictions
    """
    # Get indices of top-k predictions
    top_indices = np.argsort(probs_np)[::-1][:top_k]
    
    # Get corresponding probabilities/confidences
    top_probs = probs_np[top_indices]
    
    # Get corresponding label names
    top_labels = [id2label[idx] for idx in top_indices]
    
    # Create list of top predictions
    top_predictions = []
    for i in range(len(top_indices)):
        # Skip predictions with confidence below threshold
        if top_probs[i] < threshold:
            continue
            
        top_predictions.append({
            "label": top_labels[i],
            "confidence": float(top_probs[i])  # Convert from numpy to python float
        })
    
    # Print top predictions for debugging
    print("===== AST MODEL RAW PREDICTIONS =====")
    for i in range(min(5, len(top_predictions))):
        if i < len(top_predictions):
            print(f"  {top_predictions[i]['label']}: {top_predictions[i]['confidence']:.6f}")
    
    # Map AST labels to homesounds categories
    mapped_predictions = map_ast_labels_to_homesounds(top_predictions, threshold)
    
    return {
        "top_predictions": top_predictions,
        "mapped_predictions": mapped_predictions,
        "has_predictions": len(top_predictions) > 0
    }

# Make class labels available at module level for aggregation
class_labels = []

def initialize_class_labels(model):
    """Initialize the class labels from the model"""
    global class_labels
    if model and hasattr(model, 'config') and hasattr(model.config, 'id2label'):
        class_labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    return class_labels

# Testing
if __name__ == "__main__":
    # Generate a test sound (1 second of 440Hz sine wave)
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # Load the model
    model, feature_extractor = load_ast_model()
    
    # Run prediction
    predictions = predict_sound(audio_data, sample_rate, model, feature_extractor)
    
    # Print results
    print("\nTop predictions from AST model:")
    for pred in predictions["top_predictions"]:
        print(f"  {pred['label']}: {pred['confidence']:.6f}")
    
    print("\nMapped predictions for SoundWatch:")
    for pred in predictions["mapped_predictions"]:
        print(f"  {pred['label']} (from {pred['original_label']}): {pred['confidence']:.6f}")

# Update model architecture to match TensorFlow best practices
def create_improved_model(input_shape, num_classes):
    norm_layer = layers.Normalization()
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Resizing(32, 32),  # Downsample for efficiency
        norm_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Conv2D(128, 3, activation='relu'),  # Additional layer
        layers.GlobalAveragePooling2D(),  # Better than Flatten for audio
        layers.Dense(256, activation='relu', kernel_regularizer='l2'),  # Regularization
        layers.Dropout(0.5),
        layers.Dense(num_classes),
    ])
    
    # Add proper normalization adaptation
    norm_layer.adapt(data=train_spectrogram_ds.map(lambda spec, label: spec))
    return model 