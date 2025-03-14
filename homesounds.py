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

to_human_labels = {
    'dog-bark': "Dog Barking",
    'drill': "Drill In-Use",
    'hazard-alarm': "Fire/Smoke Alarm",
    'phone-ring': "Phone Ringing",
    'speech': "Speech",
    'vacuum': "Vacuum In-Use",
    'baby-cry': "Baby Crying",
    'chopping': "Chopping",
    'cough': "Coughing",
    'door': "Door In-Use",
    'water-running': "Water Running",
    'knock': "Knocking",
    'microwave': "Microwave",
    'shaver': "Shaver In-Use",
    'toothbrush': "Toothbrushing",
    'blender': "Blender In-Use",
    'dishwasher': "Dishwasher",
    'doorbell': "Doorbell In-Use",
    'flush': "Toilet Flushing",
    'hair-dryer': "Hair Dryer In-Use",
    'laugh': "Laughing",
    'snore': "Snoring",
    'typing': "Typing",
    'hammer': "Hammering",
    'car-horn': "Car Honking",
    'engine': "Vehicle Running",
    'saw': "Saw In-Use",
    'cat-meow': "Cat Meowing",
    'alarm-clock': "Alarm Clock",
    'cooking': "Utensils and Cutlery",
    # New sound labels
    'finger-snap': "Finger Snap",
    'hand-clap': "Hand Clap",
    'hand-sounds': "Hand Sounds",
    'applause': "Applause",
    'silence': "Silence",
    'background': "Background Sounds",
    'music': "Music",
    'sound-effect': "Sound Effect",
    'electronic-sound': "Electronic Sound",
    'notification': "Notification Sound",
    'male-conversation': "Male Speech",
    'female-conversation': "Female Speech",
    'conversation': "Conversation",
}

# Add dynamic threshold configuration
SOUND_THRESHOLDS = {
    'speech': 0.65,
    'hazard-alarm': 0.4,
    'door': 0.5,
    'water-running': 0.55,
    # Add other class-specific thresholds
}

CONTEXT_WEIGHTS = {
    'kitchen': {'water-running': 1.5, 'hazard-alarm': 2.0},
    'bedroom': {'baby-cry': 2.0, 'snore': 1.8},
    # Add other context boosts
}

def process_predictions(predictions, context='general', db_level=70):
    # Apply context-aware weighting
    weighted_preds = apply_context_weights(predictions, context)
    
    # Dynamic threshold adjustment based on dB level
    base_threshold = 0.5
    db_adjustment = np.clip((db_level - 60) / 20, -0.2, 0.2)
    dynamic_threshold = base_threshold + db_adjustment
    
    # Multi-label approach
    valid_predictions = []
    for label, prob in weighted_preds.items():
        threshold = SOUND_THRESHOLDS.get(label, dynamic_threshold)
        if prob >= threshold:
            valid_predictions.append((label, prob))
    
    # Temporal smoothing (requires state management)
    if not valid_predictions:
        return handle_ambient_sounds(db_level)
    
    return sorted(valid_predictions, key=lambda x: x[1], reverse=True)

# Update data preparation pipeline
def create_optimized_dataset(dataset):
    return dataset.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE).batch(32)

def apply_speech_correction(preds, db_level):
    speech_prob = preds.get('speech', 0)
    # Dynamic correction based on dB levels and presence of other sounds
    correction = 0.3 * (1 - np.exp(-db_level/60))
    corrected_prob = max(speech_prob - correction, 0)
    return {**preds, 'speech': corrected_prob}
