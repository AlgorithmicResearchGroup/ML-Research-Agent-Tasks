import torch
import torch.nn.functional as F
from torchvision import transforms
from datasets import load_dataset
import timm
from PIL import Image

# Load the dataset (we'll use the test split for inference)
dataset = load_dataset("AlgorithmicResearchGroup/budget_model_inference")
test_dataset = dataset['test']

# Define the same transforms used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Set up the model
num_classes = len(set(dataset['train']['class']))
model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)

# Load the trained weights
model.load_state_dict(torch.load('efficientnet_classifier.pth'))
model.eval()

# Move model to appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Function to get class label from index
def get_class_label(index):
    # Replace this with actual class labels from your dataset
    return test_dataset[index]['class']

# Inference function
def predict(image):
    img = Image.fromarray(image).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence

# Perform inference on the test set
correct = 0
total = 0

for i in range(len(test_dataset)):
    image = test_dataset[i]['image']
    true_label = test_dataset[i]['class']
    
    predicted_class, confidence = predict(image)
    predicted_label = get_class_label(predicted_class)
    
    if predicted_label == true_label:
        correct += 1
    total += 1
    
    if i % 100 == 0:  # Print every 100 images
        print(f"Image {i+1}:")
        print(f"True label: {true_label}")
        print(f"Predicted label: {predicted_label}")
        print(f"Confidence: {confidence:.2f}")
        print("--------------------")

# Calculate and print the overall accuracy
accuracy = correct / total
print(f"\nOverall accuracy on test set: {accuracy:.2f}")