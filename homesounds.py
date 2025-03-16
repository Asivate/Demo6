"""
Homesounds module for SoundWatch server.
Contains labels and mappings for sound classification.
"""

# HomeSounds Label Definition
labels = {
    'dog-bark':0,
    'drill':1,
    'hazard-alarm':2,
    'phone-ring':3,
    'speech':4,
    'vacuum':5,
    'baby-cry':6,
    'chopping':7,
    'cough':8,
    'door':9,
    'water-running':10,
    'knock':11,
    'microwave':12,
    'shaver':13,
    'toothbrush':14,
    'blender':15,
    'dishwasher':16,
    'doorbell':17,
    'flush':18,
    'hair-dryer':19,
    'laugh':20,
    'snore':21,
    'typing':22,
    'hammer':23,
    'car-horn':24,
    'engine':25,
    'saw':26,
    'cat-meow':27,
    'alarm-clock':28,
    'cooking':29,
    # New sound labels
    'finger-snap':30,
    'hand-clap':31,
    'hand-sounds':32,
    'applause':33,
    'silence':34,
    'background':35,
    'music':36,
    'sound-effect':37,
    'electronic-sound':38,
    'notification':39,
    'male-conversation':40,
    'female-conversation':41,
    'conversation':42,
}

bathroom = ['water-running','flush','toothbrush','shaver','hair-dryer']
kitchen = ['water-running','chopping','cooking','microwave','blender','hazard-alarm','dishwasher','speech']
bedroom = ['alarm-clock','snore','cough','baby-cry','speech', 'music']
office = ['knock','typing','phone-ring','door','cough','speech', 'finger-snap', 'notification']
entrance = ['knock','door','doorbell','speech','laugh', 'finger-snap', 'hand-clap']
living = ['knock','door','doorbell','speech','laugh', 'music', 'applause', 'finger-snap']
workshop = ['hammer','saw','drill','vacuum','hazard-alarm','speech']
outdoor = ['dog-bark','cat-meow','engine','car-horn','speech','hazard-alarm', 'music']

everything = [
    'dog-bark', 'drill', 'hazard-alarm', 'phone-ring', 'speech', 
    'vacuum', 'baby-cry', 'chopping', 'cough', 'door', 
    'water-running', 'knock', 'microwave', 'shaver', 'toothbrush', 
    'blender', 'dishwasher', 'doorbell', 'flush', 'hair-dryer', 
    'laugh', 'snore', 'typing', 'hammer', 'car-horn', 
    'engine', 'saw', 'cat-meow', 'alarm-clock', 'cooking',
    # Add new sounds to everything context
    'finger-snap', 'hand-clap', 'hand-sounds', 'applause', 'silence',
    'background', 'music', 'sound-effect', 'electronic-sound', 'notification',
    'male-conversation', 'female-conversation', 'conversation'
]

context_mapping = {
    'kitchen': kitchen, 
    'bathroom': bathroom, 
    'bedroom': bedroom, 
    'office': office, 
    'entrance': entrance, 
    'workshop':workshop, 
    'outdoor':outdoor, 
    'everything': everything
}

# Sound class labels (index to label mapping)
to_human_labels = {
    0: "Alarm",
    1: "Baby crying",
    2: "Car horn",
    3: "Cat meowing",
    4: "Dishes",
    5: "Dog barking",
    6: "Door knock",
    7: "Doorbell",
    8: "Footsteps",
    9: "Glass breaking",
    10: "Keyboard typing",
    11: "Microwave",
    12: "Phone ringing",
    13: "Shower",
    14: "Sink water running",
    15: "Snoring",
    16: "Speech",
    17: "Toilet flush",
    18: "Vacuum cleaner",
    19: "Water boiling"
}

# Label to index mapping
labels = {
    "alarm": 0,
    "baby_crying": 1,
    "car_horn": 2,
    "cat_meowing": 3,
    "dishes": 4,
    "dog_barking": 5,
    "door_knock": 6,
    "doorbell": 7,
    "footsteps": 8,
    "glass_breaking": 9,
    "keyboard_typing": 10,
    "microwave": 11,
    "phone_ringing": 12,
    "shower": 13,
    "sink_water_running": 14,
    "snoring": 15,
    "speech": 16,
    "toilet_flush": 17,
    "vacuum_cleaner": 18,
    "water_boiling": 19
}

# Group sounds by category
sound_categories = {
    "Alerts": ["Alarm", "Doorbell", "Phone ringing", "Door knock", "Car horn"],
    "Human": ["Speech", "Baby crying", "Footsteps", "Snoring", "Keyboard typing"],
    "Animal": ["Dog barking", "Cat meowing"],
    "Water": ["Sink water running", "Shower", "Toilet flush", "Water boiling"],
    "Appliance": ["Microwave", "Vacuum cleaner"],
    "Household": ["Dishes", "Glass breaking"]
}

# Priority levels for sounds (higher number = higher priority)
sound_priorities = {
    "Alarm": 10,
    "Baby crying": 9,
    "Door knock": 8,
    "Doorbell": 8,
    "Glass breaking": 8,
    "Phone ringing": 7,
    "Car horn": 7,
    "Dog barking": 6,
    "Cat meowing": 5,
    "Speech": 5,
    "Footsteps": 4,
    "Toilet flush": 3,
    "Shower": 3,
    "Sink water running": 3,
    "Water boiling": 3,
    "Microwave": 2,
    "Vacuum cleaner": 2,
    "Dishes": 2,
    "Keyboard typing": 1,
    "Snoring": 1
}

# Enhanced threshold configuration with temporal smoothing
SOUND_THRESHOLDS = {
    # Base thresholds (db > 60)
    'speech': {'base': 0.65, 'min': 0.4, 'priority': 1},
    'hazard-alarm': {'base': 0.4, 'min': 0.3, 'priority': 3},
    'door': {'base': 0.5, 'min': 0.35, 'priority': 2},
    'water-running': {'base': 0.55, 'min': 0.4, 'priority': 2},
    'baby-cry': {'base': 0.45, 'min': 0.3, 'priority': 3},
    # ... other classes
}

CONTEXT_WEIGHTS = {
    'kitchen': {
        'weights': {'water-running': 1.5, 'hazard-alarm': 2.0},
        'suppress': ['speech', 'music']
    },
    'bedroom': {
        'weights': {'baby-cry': 2.0, 'snore': 1.8},
        'suppress': ['drill', 'hammer']
    },
    # ... other contexts
}

# Stateful prediction history for temporal smoothing 
PREDICTION_HISTORY = {}

def process_predictions(preds, context, db_level, history_length=5):
    """Stateful prediction processing with temporal analysis"""
    global PREDICTION_HISTORY
    
    # Dynamic threshold adjustment
    thresholds = {
        k: max(v['min'], v['base'] * (db_level/70)) 
        for k, v in SOUND_THRESHOLDS.items()
    }
    
    # Context weighting
    weighted = {
        label: prob * CONTEXT_WEIGHTS[context]['weights'].get(label, 1)
        for label, prob in preds.items()
    }
    
    # Temporal smoothing
    current_time = time.time()
    PREDICTION_HISTORY.setdefault(context, []).append({
        'time': current_time,
        'predictions': weighted,
        'db': db_level
    })
    
    # Remove old entries
    PREDICTION_HISTORY[context] = [
        entry for entry in PREDICTION_HISTORY[context]
        if current_time - entry['time'] < history_length
    ]
    
    # Calculate moving averages
    temporal_predictions = defaultdict(list)
    for entry in PREDICTION_HISTORY[context]:
        for label, prob in entry['predictions'].items():
            temporal_predictions[label].append(prob)
    
    averaged = {
        label: np.mean(probs) * (1 + 0.2*np.std(probs))
        for label, probs in temporal_predictions.items()
    }
    
    # Apply final thresholds
    valid = [
        (label, prob) 
        for label, prob in averaged.items() 
        if prob >= thresholds.get(label, 0.5)
    ]
    
    return sorted(valid, key=lambda x: x[1], reverse=True)

# Update data preparation pipeline
def create_optimized_dataset(dataset):
    """Enhanced data pipeline with augmentation"""
    augmentation = tf.keras.Sequential([
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.1),
        layers.GaussianNoise(0.001)
    ])
    
    return dataset \
        .map(lambda x, y: (augmentation(x, training=True), y), 
             num_parallel_calls=tf.data.AUTOTUNE) \
        .cache() \
        .shuffle(10000, reshuffle_each_iteration=True) \
        .batch(32) \
        .prefetch(tf.data.AUTOTUNE) \
        .map(lambda x, y: (tf.image.per_image_standardization(x), y),
             num_parallel_calls=tf.data.AUTOTUNE)

def apply_speech_correction(preds, db_level):
    speech_prob = preds.get('speech', 0)
    # Dynamic correction based on dB levels and presence of other sounds
    # Reduced correction factor from 0.3 to 0.25
    correction = 0.25 * (1 - np.exp(-db_level/60))
    corrected_prob = max(speech_prob - correction, 0)
    return {**preds, 'speech': corrected_prob}
