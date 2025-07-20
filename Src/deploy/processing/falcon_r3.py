import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from ultralytics import YOLO
import yaml

def load_confidence_threshold():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('confidence_threshold', 0.5)

class ClassificationModel:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence = load_confidence_threshold()
        try:
            # Load model
            self.model = YOLO(model_path)
            self.is_yolo = True
            print("Loaded YOLO model successfully")
        except Exception as e:
            print(f"Not a YOLO model ({str(e)}), trying PyTorch model...")
            try:
                self.model = torch.load(model_path, map_location=self.device)
                if hasattr(self.model, 'eval'):
                    self.model.eval()
                self.is_yolo = False
                print("Loaded PyTorch model successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")

    def predict(self, image_path):
        """Predict class for an image"""
        try:
            if self.is_yolo:
                results = self.model(image_path, conf=self.confidence)
                if len(results) > 0 and hasattr(results[0], 'probs'):
                    probs = results[0].probs
                    predicted_class = int(probs.top1)
                    class_name = self.model.names[predicted_class]
                    return class_name
                else:
                    raise RuntimeError("No valid predictions from YOLO model")
            else:
                image = Image.open(image_path).convert('RGB')
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
                ])
                image_tensor = transform(image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(image_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    predicted_class = torch.argmax(probabilities).item()
                    
                    if hasattr(self.model, 'class_names'):
                        return self.model.class_names[predicted_class]
                    else:
                        # Try to load class names from file
                        class_names_path = os.path.join(os.path.dirname('models/falcon_r3.pt'), 'class_names.txt')
                        if os.path.exists(class_names_path):
                            with open(class_names_path, 'r') as f:
                                class_names = [line.strip() for line in f]
                            return class_names[predicted_class]
                        return f"class_{predicted_class}"
                    
        except Exception as e:
            raise RuntimeError(f"Error predicting {image_path}: {str(e)}")

def process_r3(input_data):
    """Stage 3: Classify images"""
    if not input_data:
        print("Warning: No input data received for classification")
        return []
        
    print(f"R3 Processing: Received {len(input_data)} items for classification")
    
    model_path = 'models/falcon_r3.pt'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    try:
        classifier = ClassificationModel(model_path)
        results = []
        classification_counts = {}
        
        for i, item in enumerate(input_data):
            if not isinstance(item, dict) or 'path' not in item:
                print(f"Warning: Invalid item format at index {i}: {item}")
                continue
                
            if not os.path.exists(item['path']):
                print(f"Warning: Image not found: {item['path']}")
                continue
                
            try:
                print(f"  Classifying {i+1}/{len(input_data)}: {os.path.basename(item['path'])}")
                class_name = classifier.predict(item['path'])
                item['classified_class'] = class_name
                results.append(item)
                
                # Count classifications
                classification_counts[class_name] = classification_counts.get(class_name, 0) + 1
                print(f"    → Classified as: {class_name}")
                
            except Exception as e:
                print(f"Warning: Failed to classify {item['path']}: {str(e)}")
                # Still add the item but with unknown classification
                item['classified_class'] = 'unknown'
                results.append(item)
                continue
        
        print(f"R3 Classification Summary:")
        for class_name, count in classification_counts.items():
            print(f"  {class_name}: {count} items")
        print(f"Total classified items: {len(results)}")
        
        return results
        
    except Exception as e:
        raise RuntimeError(f"Classification error: {str(e)}")
