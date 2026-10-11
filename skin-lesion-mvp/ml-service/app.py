import os
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as torchvision_models  # Renamed to avoid conflict
from collections import Counter
import json
import warnings
import argparse
warnings.filterwarnings("ignore", category=UserWarning)

# Define class names and info
CLASS_NAMES = {
    0: 'akiec',  # Actinic Keratoses and Intraepithelial Carcinoma
    1: 'bcc',    # Basal Cell Carcinoma
    2: 'bkl',    # Benign Keratosis-like Lesions
    3: 'df',     # Dermatofibroma
    4: 'mel',    # Melanoma
    5: 'nv',     # Melanocytic Nevi
    6: 'vasc'    # Vascular Lesions
}

CLASS_INFO = {
    'akiec': {
        'display': 'Actinic Keratosis / Intraepithelial Carcinoma',
        'risk': 'High',
        'description': 'A pre-cancerous growth or early form of skin cancer.',
        'recommendation': 'Consult a dermatologist promptly for evaluation.'
    },
    'bcc': {
        'display': 'Basal Cell Carcinoma',
        'risk': 'High',
        'description': 'The most common type of skin cancer, usually slow-growing.',
        'recommendation': 'Consult a dermatologist for proper treatment options.'
    },
    'bkl': {
        'display': 'Benign Keratosis',
        'risk': 'Low',
        'description': 'A non-cancerous skin growth that appears as a waxy, scaly patch.',
        'recommendation': 'Generally no treatment needed, but monitor for changes.'
    },
    'df': {
        'display': 'Dermatofibroma',
        'risk': 'Low',
        'description': 'A common benign skin growth or nodule that is usually harmless.',
        'recommendation': 'No treatment needed unless causing discomfort.'
    },
    'mel': {
        'display': 'Melanoma',
        'risk': 'Very High',
        'description': 'A serious form of skin cancer that can spread if not treated early.',
        'recommendation': 'Seek immediate medical attention.'
    },
    'nv': {
        'display': 'Melanocytic Nevus (Mole)',
        'risk': 'Low',
        'description': 'A benign growth of melanocytes, usually harmless but should be monitored.',
        'recommendation': 'Monitor for changes in size, shape, or color.'
    },
    'vasc': {
        'display': 'Vascular Lesion',
        'risk': 'Low',
        'description': 'Abnormalities of blood vessels that appear on the skin surface.',
        'recommendation': 'Typically harmless but consult a doctor if concerned.'
    }
}

def load_models(models_dir="./models"):
    """Load all models from the models directory"""
    loaded_models = {}  # Renamed from 'models' to avoid conflict with the imported module
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if models directory exists
    if not os.path.exists(models_dir):
        print(f"Models directory {models_dir} not found. Searching for models...")
        # Try to find models in parent directories
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for root, dirs, files in os.walk(os.path.dirname(script_dir)):
            if "models" in dirs:
                models_dir = os.path.join(root, "models")
                print(f"Found models directory at: {models_dir}")
                break
    
    # Possible model filenames (with variations)
    possible_models = {
        "mobilenetv3": ["final_mobilenetv3_model.pt", "mobilenetv3.pt", "final_mobilenetv3.pt"],
        "densenet121": ["final_densenet121_model.pt", "densenet121.pt", "final_densenet121.pt"],
        "resnet50": ["final_resnet50_model.pt", "resnet50.pt", "final_resnet50.pt"]
    }
    
    # Find and load each model
    for model_name, filenames in possible_models.items():
        model_found = False
        for filename in filenames:
            model_path = os.path.join(models_dir, filename)
            if os.path.exists(model_path):
                print(f"Loading {model_name} model from {model_path}")
                try:
                    # Initialize the base model architecture first
                    if 'resnet' in model_name:
                        model = torchvision_models.resnet50(weights=None)
                        model.fc = torch.nn.Linear(model.fc.in_features, 7)
                    elif 'densenet' in model_name:
                        model = torchvision_models.densenet121(weights=None)
                        model.classifier = torch.nn.Linear(model.classifier.in_features, 7)
                    elif 'mobilenet' in model_name:
                        model = torchvision_models.mobilenet_v3_large(weights=None)
                        # the fine-tuned checkpoint replaced the whole head: 960 -> 256 -> 7
                        model.classifier = torch.nn.Sequential(
                            torch.nn.Linear(960, 256), torch.nn.Hardswish(), torch.nn.Dropout(p=0.2), torch.nn.Linear(256, 7))
                    else:
                        raise ValueError(f"Unknown model architecture: {model_name}")
                    
                    # Load model to the appropriate device
                    model_data = torch.load(model_path, map_location=device)
                    
                    # Handle different saved formats
                    if isinstance(model_data, dict) and 'model' in model_data:
                        # If the file contains a dict with 'model' key
                        model.load_state_dict(model_data['model'])
                    elif isinstance(model_data, dict) and 'state_dict' in model_data:
                        # If the file contains a state_dict
                        model.load_state_dict(model_data['state_dict'])
                    elif isinstance(model_data, dict) and len(model_data) > 0:
                        # If the file contains just a state dict (OrderedDict)
                        # Clean state_dict if needed (remove module. prefix)
                        cleaned_state_dict = {k.replace('module.', ''): v for k, v in model_data.items()}
                        model.load_state_dict(cleaned_state_dict, strict=False)
                    elif hasattr(model_data, 'state_dict'):
                        # If the file contains the model object directly
                        model.load_state_dict(model_data.state_dict())
                    
                    model.eval()  # Set model to evaluation mode
                    model = model.to(device)  # Ensure model is on the correct device
                    loaded_models[model_name] = model
                    print(f"Successfully loaded {model_name}")
                    model_found = True
                    break
                except Exception as e:
                    print(f"Error loading {model_name} from {model_path}: {e}")
        
        if not model_found:
            print(f"Warning: Could not find or load any {model_name} model files!")
    
    # Check if any models were loaded
    if not loaded_models:
        print("No models could be loaded! Unable to make predictions.")
        return None, device
    else:
        print(f"Loaded {len(loaded_models)} models successfully")
        
    return loaded_models, device

def preprocess_image(image_path, device):
    """Process image for model input"""
    try:
        # Define image transformations
        image_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Open image from file
        image = Image.open(image_path).convert('RGB')
        
        # Apply transformations and prepare tensor
        tensor = image_transforms(image).unsqueeze(0).to(device)
        return tensor
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict(image_path, loaded_models, device):
    """Predict skin lesion from image file"""
    if not loaded_models:
        print("Error: No models loaded for prediction")
        return None
        
    # Preprocess image
    image_tensor = preprocess_image(image_path, device)
    if image_tensor is None:
        return None
    
    # Get predictions from each model
    predictions = {}
    overall_probs = {i: 0.0 for i in range(7)}
    
    with torch.no_grad():  # No need to track gradients
        for model_name, model in loaded_models.items():
            # Get model output
            output = model(image_tensor)
            probs = F.softmax(output, dim=1)[0].cpu().numpy()
            
            # Get predicted class
            pred_class = int(probs.argmax())
            pred_confidence = float(probs[pred_class])
            
            # Store prediction
            predictions[model_name] = {
                "class_id": pred_class,
                "class_name": CLASS_NAMES[pred_class],
                "confidence": pred_confidence
            }
            
            # Add to overall probabilities
            for i, prob in enumerate(probs):
                overall_probs[i] += float(prob) / len(loaded_models)
    
    # Ensemble prediction using majority voting
    pred_classes = [pred["class_id"] for pred in predictions.values()]
    counter = Counter(pred_classes)
    ensemble_class = counter.most_common(1)[0][0]
    ensemble_class_name = CLASS_NAMES[ensemble_class]
    
    # Format result
    result = {
        "prediction": {
            "class_id": ensemble_class,
            "class_name": ensemble_class_name,
            "display_name": CLASS_INFO[ensemble_class_name]["display"],
            "risk": CLASS_INFO[ensemble_class_name]["risk"],
            "description": CLASS_INFO[ensemble_class_name]["description"],
            "recommendation": CLASS_INFO[ensemble_class_name]["recommendation"],
            "confidence": counter[ensemble_class] / len(loaded_models)  # Percentage of models that voted for this class
        },
        "model_predictions": predictions,
        "class_probabilities": {CLASS_NAMES[i]: float(prob) for i, prob in overall_probs.items()}
    }
    
    return result

def main():
    """Main function to run the script"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Please Input the Name of your File: ')
    parser.add_argument('-f', '--file', help='Path to input file')
    parser.add_argument('--models', type=str, help='Path to the models directory (default: auto-detect)')
    args = parser.parse_args()
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define models directory path
    models_dir = args.models if args.models else os.environ.get("MODELS_DIR", os.path.join(script_dir, "models"))
    
    # Process the file argument
    if args.file:
        print(f'Processing file: {args.file}')
        image_path = args.file
    else:
        print('File not specified')
        # Default image path if file not specified
        image_path = os.path.join(script_dir, "melanoma.jpeg")
        
    print(f"Processing image: {image_path}")
    print(f"Script directory: {script_dir}")
    print(f"Looking for models in: {models_dir}")
    
    # Check if image exists
    if not os.path.exists(image_path):
        # Try to find the image in the parent directory structure
        print("Image not found at the expected location. Searching...")
        for root, dirs, files in os.walk(os.path.dirname(script_dir)):
            image_filename = os.path.basename(image_path)
            if image_filename in files:
                image_path = os.path.join(root, image_filename)
                print(f"Found image at: {image_path}")
                break
        
        # If still not found
        if not os.path.exists(image_path):
            print(f"Error: Could not find image {image_path} anywhere!")
            return
    
    # Load models and make prediction
    loaded_models, device = load_models(models_dir)
    if not loaded_models:
        print("Error: Could not load any models. Exiting.")
        return
        
    result = predict(image_path, loaded_models, device)
    if not result:
        print("Error: Could not make prediction. Exiting.")
        return
    
    # Print results
    print("\n=== PREDICTION RESULTS ===")
    print(f"Classification: {result['prediction']['display_name']}")
    print(f"Risk Level: {result['prediction']['risk']}")
    print(f"Description: {result['prediction']['description']}")
    print(f"Recommendation: {result['prediction']['recommendation']}")
    print(f"Confidence: {result['prediction']['confidence'] * 100:.2f}%\n")
    
    print("Individual Model Predictions:")
    for model_name, pred in result['model_predictions'].items():
        print(f"  - {model_name}: {CLASS_INFO[pred['class_name']]['display']} (Confidence: {pred['confidence'] * 100:.2f}%)")
    
    print("\nClass Probabilities:")
    for class_name, prob in result['class_probabilities'].items():
        print(f"  - {CLASS_INFO[class_name]['display']}: {prob * 100:.2f}%")
    
    # Save results to JSON file
    output_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_prediction.json"
    with open(output_filename, "w") as f:
        json.dump(result, f, indent=4)
    print(f"\nResults also saved to {output_filename}")

if __name__ == "__main__":
    main()