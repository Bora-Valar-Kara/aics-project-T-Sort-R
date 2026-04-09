import torch  # pytorch 
import torch.nn as nn  # neural network
import torch.optim as optim  # optimization algorithms
from torch.utils.data import Dataset, DataLoader  # some tools for handling and loading data in batches
from torchvision import transforms, models  # pre-built image processing tools and models
from PIL import Image  # the library for opening, manipulating and saving image filess
import numpy as np  # numpy
import matplotlib.pyplot as plt  # for plotting
from datasets import load_dataset  # to load out dataset from Hugging Face
import mss  # a library for taking screenshots from monitors
import cv2  # computer vision library for image processing
import time  # for time related functions like sleep and timing
import argparse  # to customize bash command arguments, to use the same script for testing, training etc. 
from pathlib import Path  # to handle file system paths
from sklearn.metrics import confusion_matrix, classification_report  # evaluating the model
import seaborn as sns  # data visualization
from tqdm import tqdm  # progress bars to show loop process
import serial  # this is the library for serial communication with Arduino
import serial.tools.list_ports  # to handle serial ports

class Config:  # here we define a Config class to store all our configuration settings in one place for ease of use
    BATCH_SIZE = 32  # to process x images at a time
    NUM_EPOCHS = 15  # x times we loop the training dataset
    LEARNING_RATE = 0.001  # they say 1e-3 is good
    IMG_SIZE = 224  # all images will be resized to 224x224 pixels which is standard size for ResNet that we will use as architecture
    MODEL_PATH = 'trashnet_model.pth'  # the trained model will be saved because we will test it multiple times after trainging
    CATEGORIES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']  # our list of trash types
    CAPTURE_INTERVAL = 2.0  # screen captures will happen every 2 seconds
    MONITOR_NUMBER = 1  # to tag the primary monitor because I use multiple at home
    ARDUINO_PORT = None  # serial port for arduino (None is auto detection)
    ARDUINO_BAUDRATE = 9600  # the communication speed with arduino, 9600 bits per second is standard
    ARDUINO_TIMEOUT = 2  # to wait for arduino response
    SERVO_ANGLES = {  # a dictionary mapping each trash category to a servo motor angle 180 is max my servo can do
        'cardboard': 0,
        'glass': 36,
        'metal': 72,
        'paper': 108,
        'plastic': 144,
        'trash': 180
    }
    COLORS = {  # RGB for visualizing categories
        'cardboard': (139, 69, 19),  # brown
        'glass': (65, 105, 225),  # blue
        'metal': (112, 128, 144),  # gray
        'paper': (245, 222, 179),  # bej
        'plastic': (255, 99, 71),  # red
        'trash': (47, 79, 79)  # dark gray
    }
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU if available, trained on RTX 3060

class ArduinoController:  # we define ArduinoController class to manage all arduino communication as a blueprint
    def __init__(self, port=None, baudrate=9600, timeout=2):  # arguments are port, because ports are different for different computers, baudrate is the speed of the communication that can differ, timeout can differ as well
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        #instance variables
        self.serial_connection = None  # we initialize serial_connection as None which means we are not connected to arduino yet
        self.is_connected = False  # we track whether we are connected to arduino or not
    
    def find_arduino_port(self):  # to automatically search for arduino on available ports, to not having to search for which port is the arduino on everytime on several computers
        print("Searching Arduino port.")
        ports = serial.tools.list_ports.comports()  # all available serial ports on the computer
        arduino_keywords = ['Arduino', 'ttyUSB', 'ttyACM', 'CH340', 'USB Serial']  # this list of keywords are usually Arduino ports
        for port in ports:  # loop through each port found
            port_info = f"{port.device} - {port.description}"  # we create a string with port information
            print(f"  Found: {port_info}")  # we print information about this port
            for keyword in arduino_keywords:  # check each arduino keyword
                if keyword.lower() in port.description.lower() or keyword in port.device:  # if it matches the keyword, we found an Arduino port
                    print(f"Arduino on {port.device} port.")
                    return port.device  # return the device path
        print("We failed to find Arduino automatically")  # if we exit the loop without finding Arduino
        return None  # return None then
    
    def connect(self):  # method to make connection with Arduino
        if self.port is None:  # if no port was specified, try to find it automatically
            self.port = self.find_arduino_port()  # we call our find_arduino_port method
            if self.port is None:  # if still None, auto-detection failed, show help message to tell user how to fix the issue
                print("\n Could not auto detect Arduino port.")
                print("You can specify port manually with --port argument")  # then they need to find the port manually
                print("Examples:")  # example commands
                print("on Linux: --port /dev/ttyUSB0 or --port /dev/ttyACM0 usually works")
                print("on Windows: --port COM3 or --port COM4")
                return False  # return False to indicate that the connection failed
        try:  # try & except block catches errors that might occur during connection
            print(f"\nConnecting to Arduino on {self.port}.")  # the port we are attempting to connect to
            self.serial_connection = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=self.timeout)  # we create serial connection object with our parameters; this opens the serial port for communication
            print("Waiting for Arduino to start...")  # verbose outputs: waiting message
            time.sleep(2)  # we wait 2 seconds - arduino resets when serial connection opens
            self.serial_connection.reset_input_buffer()  # we clear any garbage data that might be in the input buffer
            self.is_connected = True  # to mark that weare now connected
            print("Arduino connected.")
            self.move_servo(90)  # we move servo to center position (90 degrees) as initialization
            return True  # we return True to indicate successful connection
        except serial.SerialException as e:  # to catch serial communication errors specifically
            print(f"Failed to connect to Arduino: {e}")  # we print error message with the specific error (e)
            print("")
            print("Troubleshooting:")  # troubleshooting tips that are usually the problem
            print("1. Check USB cable is connected")
            print("2. Make sure Arduino code is uploaded")
            print("3. Close Arduino Serial Monitor if it is open")
            print("4. Try different USB port")
            print("5. Check port permissions")
            return False  # we return False to indicate connection failed
    
    def move_servo(self, angle):  # method to move the servo motor to a specific angle -- angle parameter is target angle in degrees (0-180)
        if not self.is_connected:  # we check if we are connected before trying to send commands -- 'not' should flip the boolean -- if is_connected is False, this fires:
            print("Arduino not connected")
            return False  # we return False to indicate command failed
        try:  # try--except block to handle communication errors
            if not (0 <= angle <= 180):  # to validate that angle is within servo's valid range (0-180 degrees) -- I tried more than 180 and less than 0, it doesn't accept those values:
                print(f"Invalid angle: {angle} (must be between 0-180)")
                return False  # we return False to indicate invalid angle
            command = f"{angle}\n"  # we create command string with angle value and newline character -- f-string allows us to insert the angle variable; \n is the newline character that Arduino looks for
            self.serial_connection.write(command.encode())  # we convert string to bytes and send to Arduino -- .encode() converts the string to bytes that can be sent over serial
            time.sleep(0.1)  # adding a delay for Arduino to process the command, sometimes problematic if no delay
            if self.serial_connection.in_waiting > 0:  # we check if Arduino sent back any response -- in_waiting tells us how many bytes are waiting to be read
                response = self.serial_connection.readline().decode().strip()  # we read one line of response from Arduino -- .decode() converts bytes back to a string -- .strip() removes whitespace and newline characters from ends
                if response and ('ERROR' in response or 'Invalid' in response):  # we check if there's a response and if it contains an error message; 'and' means both conditions must be true
                    print(f"Arduino error: {response}")
            return True
        except serial.SerialException as e:  # same catch block
            print(f"Error communicating with Arduino: {e}") 
            self.is_connected = False
            return False
    
    def move_to_category(self, category):  # method to move servo to the angle associated with a trash category -- category parameter: string like 'plastic' or 'glass'
        if category not in Config.SERVO_ANGLES:  # we check if the category exists in our SERVO_ANGLES dictionary; 'not in' checks if the key is not in the dictionary
            return False
        angle = Config.SERVO_ANGLES[category]  # to look up the angle for this category from our dictionary -- dictionary lookup: dict[key] returns the value associated with that key
        return self.move_servo(angle)  # we call move_servo with the looked up angle and return its result
    
    def disconnect(self):  # this method is to safely close the Arduino connection
        if self.serial_connection and self.is_connected:  # we check if we have a connection object & we are marked as connected -- 'and' for both conditions must be true
            print("\nDisconnecting from Arduino...")
            self.serial_connection.close()  # we close the serial connection
            self.is_connected = False  # we update our connection status
            print("Arduino disconnected")
    
    def __del__(self):  # it is a special dunder method -- it is called when the object is about to be destroyed -- it ensures we always disconnect when program ends
        self.disconnect()  # it calls our disconnect method to clean up

class TrashNetDataset(Dataset):  # we define a custom Dataset class for loading TrashNet data -- inheriting from Dataset should mean we get all the functionality PyTorch expects
    def __init__(self, hf_dataset, transform=None):  # constructor takes a HuggingFace dataset and optional transforms: image preprocessing steps such as resizing, normalization...
        self.dataset = hf_dataset  # to store the dataset for later use
        self.transform = transform  # and the transform function
    
    def __len__(self):  # dunder that returns the dataset size -- len(dataset) will call this method
        return len(self.dataset)  # return the number of items in the dataset
    
    def __getitem__(self, idx):  # this dunder is called when we access dataset[idx] -- index of the item we want (like dataset[0] for first item)
        item = self.dataset[idx]  # get the item at this index from our HuggingFace dataset
        image = item['image'].convert('RGB')  # we extract the image from the item and ensure its in RGB format
        label = item['label']  # wextract the label (which is category number) from the item
        if self.transform:  # if we have transform functions-- apply them
            image = self.transform(image)
        return image, label  # return both the processed image and its label as a tuple

def create_model(num_classes=6, pretrained=True):  # making a function to create and configure a ResNet18 model -- num_classes: number of categories to classify (default 6) -- pretrained: whether to use ImageNet weights
    model = models.resnet18(pretrained=pretrained)  # loading a ResNet18 model architecture from torchvision -- pretrained=True means we start with weights from ImageNet (thus; "transfer learning")
    for param in model.parameters():  # now here we freeze all the pre-trained layers so we don't retrain them -- this loop goes through every parameter in the model
        param.requires_grad = False  # means that this parameters won't be updated during training
    num_features = model.fc.in_features  # to get the number of input features to the final fully connected layer -- it needed to know how to replace the final layer
    model.fc = nn.Sequential(  # Replace the final fully connected layer with our custom classifier; nn.Sequential creates a sequence of layers that data flows through
        nn.Dropout(0.5),  # Dropout randomly turns off 50% of neurons during training (prevents overfitting)
        nn.Linear(num_features, 256),  # Linear layer: transforms num_features inputs to 256 outputs; this is a fully connected layer where each input connects to each output
        nn.ReLU(),  # ReLU activation: max(0, x) - adds non-linearity to help learn complex patterns
        nn.Dropout(0.3),  # Another dropout layer with 30% probability (less aggressive)
        nn.Linear(256, num_classes)  # Final linear layer: 256 inputs to num_classes outputs (one per category)
    )
    return model  # Return the modified model

def train_epoch(model, loader, criterion, optimizer, device):  # Function to train the model for one epoch; model: neural network; loader: provides batches; criterion: loss function; optimizer: updates weights; device: CPU/GPU
    model.train()  # Set model to training mode (enables dropout, batch normalization training behavior)
    running_loss = 0.0  # Initialize running loss to accumulate loss across all batches
    correct = 0  # Track number of correct predictions
    total = 0  # Track total number of samples processed
    pbar = tqdm(loader, desc='Training')  # Create a progress bar using tqdm, wrapping our data loader; desc: description text shown with the progress bar
    for images, labels in pbar:  # Loop through batches of data from the loader; each iteration gives us one batch of images and labels
        images, labels = images.to(device), labels.to(device)  # Move images and labels to the device (GPU or CPU); .to(device) copies the data to GPU memory if available
        optimizer.zero_grad()  # Zero out gradients from previous iteration; gradients accumulate by default, so we need to clear them
        outputs = model(images)  # Forward pass: feed images through the model to get predictions; outputs is a tensor with shape [batch_size, num_classes]
        loss = criterion(outputs, labels)  # Calculate loss (how wrong the predictions are); compares model outputs to true labels
        loss.backward()  # Backward pass: calculate gradients (how to change weights to reduce loss); computes the derivative of loss with respect to each parameter
        optimizer.step()  # Update model weights using the computed gradients; this is where the model actually learns
        running_loss += loss.item()  # Add this batch's loss to our running total; .item() extracts the Python number from a tensor with one element
        _, predicted = outputs.max(1)  # Get the predicted class for each image; .max(1) finds the maximum value along dimension 1 (across classes); returns (max_values, indices_of_max_values)
        total += labels.size(0)  # Add batch size to total count; .size(0) returns the first dimension size (batch size)
        correct += predicted.eq(labels).sum().item()  # Count how many predictions matched the true labels; .eq() creates boolean tensor; .sum() counts True values; .item() converts to Python number
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})  # Update the progress bar with current loss and accuracy; set_postfix updates the text shown after the progress bar
    return running_loss / len(loader), 100. * correct / total  # Return average loss (total loss / number of batches) and accuracy percentage

def evaluate(model, loader, criterion, device):  # Function to evaluate model performance on validation/test data; similar to train_epoch but without updating weights
    model.eval()  # Set model to evaluation mode (disables dropout, changes batch norm behavior)
    running_loss = 0.0  # Initialize tracking variables
    correct = 0  # Count correct predictions
    total = 0  # Count total samples
    all_preds = []  # Lists to store all predictions for detailed analysis
    all_labels = []  # Store all true labels
    with torch.no_grad():  # torch.no_grad() disables gradient calculation (saves memory, speeds up); we don't need gradients during evaluation since we're not training
        for images, labels in tqdm(loader, desc='Evaluating'):  # Loop through validation batches
            images, labels = images.to(device), labels.to(device)  # Move data to device
            outputs = model(images)  # Forward pass to get predictions
            loss = criterion(outputs, labels)  # Calculate loss
            running_loss += loss.item()  # Accumulate loss
            _, predicted = outputs.max(1)  # Get predicted classes
            total += labels.size(0)  # Update counts
            correct += predicted.eq(labels).sum().item()  # Count correct predictions
            all_preds.extend(predicted.cpu().numpy())  # Store predictions; .cpu() moves tensor to CPU; .numpy() converts to NumPy array; .extend() adds all items to our list
            all_labels.extend(labels.cpu().numpy())  # Store true labels
    return running_loss / len(loader), 100. * correct / total, all_preds, all_labels  # Return average loss, accuracy, and all predictions/labels for analysis

def train_model():  # Main training function that orchestrates the entire training process
    print("=" * 70)  # Print decorative header; "=" * 70 creates a string of 70 equal signs
    print("TrashNet Model Training")  # Print title
    print("=" * 70)  # Print footer
    print(f"Device: {Config.DEVICE}")  # Print configuration information
    print(f"Batch size: {Config.BATCH_SIZE}")  # Print batch size
    print(f"Epochs: {Config.NUM_EPOCHS}\n")  # Print epochs; \n creates a blank line
    
    print("Loading TrashNet dataset...")  # Print loading message
    dataset = load_dataset("garythung/trashnet")  # Load the TrashNet dataset from Hugging Face; downloads and caches the dataset
    print(f"Total samples: {len(dataset['train'])}")  # Print total number of samples
    print("Splitting into train/test sets (80/20)...")  # Print splitting message
    dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)  # Split data into training and testing sets (80% train, 20% test); train_test_split creates two subsets; seed=42 ensures reproducibility
    print(f"Train samples: {len(dataset['train'])}")  # Print train set size
    print(f"Test samples: {len(dataset['test'])}\n")  # Print test set size
    
    train_transform = transforms.Compose([  # Define data augmentation and preprocessing for training data; transforms.Compose chains multiple transformations together
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),  # Resize all images to 224x224 (standard input size for ResNet)
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally (50% chance) - data augmentation
        transforms.RandomRotation(15),  # Randomly rotate images up to 15 degrees - helps model learn rotation invariance
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Randomly adjust brightness, contrast, and saturation - makes model more robust
        transforms.ToTensor(),  # Convert PIL Image to PyTorch tensor (changes HWC to CHW format, scales to 0-1)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet mean and std (helps model converge faster); these are the standard values used when ResNet was trained
    ])
    
    test_transform = transforms.Compose([  # Define preprocessing for test data (no augmentation, just resize and normalize)
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),  # Resize to standard size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with same values as training
    ])
    
    train_dataset = TrashNetDataset(dataset['train'], transform=train_transform)  # Create Dataset objects that apply our transforms
    test_dataset = TrashNetDataset(dataset['test'], transform=test_transform)  # Create test dataset
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)  # Create DataLoaders that handle batching and shuffling; shuffle=True randomizes order; num_workers=4 uses 4 parallel processes; pin_memory=True speeds up GPU transfer
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)  # Create test loader; don't shuffle test data (order doesn't matter)
    
    print("Creating model...")  # Print model creation message
    model = create_model(num_classes=len(Config.CATEGORIES))  # Create our modified ResNet18 model
    model = model.to(Config.DEVICE)  # Move model to GPU or CPU
    
    criterion = nn.CrossEntropyLoss()  # Define loss function (CrossEntropyLoss is standard for classification); combines softmax activation and negative log likelihood loss
    optimizer = optim.Adam(model.fc.parameters(), lr=Config.LEARNING_RATE)  # Define optimizer (Adam is an advanced version of gradient descent); only optimize the final fully connected layer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)  # Learning rate scheduler reduces LR when validation loss plateaus; mode='min': reduce when metric stops decreasing; patience=3: wait 3 epochs; factor=0.5: multiply LR by 0.5
    
    best_acc = 0.0  # Track best accuracy to save best model
    train_losses, train_accs = [], []  # Lists to store training history for plotting
    val_losses, val_accs = [], []  # Lists for validation history
    
    print("\nStarting training...\n")  # Print start message
    
    for epoch in range(Config.NUM_EPOCHS):  # Main training loop - iterate through epochs; range(NUM_EPOCHS) creates sequence [0, 1, 2, ..., NUM_EPOCHS-1]
        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")  # Print current epoch (add 1 for human-readable numbering)
        print("-" * 70)  # Print separator line
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)  # Train for one epoch and get metrics
        train_losses.append(train_loss)  # Store training metrics
        train_accs.append(train_acc)  # Store training accuracy
        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, Config.DEVICE)  # Evaluate on validation set; _ ignores predictions and labels
        val_losses.append(val_loss)  # Store validation metrics
        val_accs.append(val_acc)  # Store validation accuracy
        scheduler.step(val_loss)  # Update learning rate based on validation loss
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")  # Print epoch results
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")  # Print validation results
        if val_acc > best_acc:  # Check if this is the best model so far
            best_acc = val_acc  # Update best accuracy
            torch.save({'model_state_dict': model.state_dict(), 'categories': Config.CATEGORIES, 'epoch': epoch, 'accuracy': val_acc}, Config.MODEL_PATH)  # Save model checkpoint; torch.save saves a dictionary containing model state and metadata
            print(f"Best model saved... Accuracy: {best_acc:.2f}%")  # Print success message
        print()  # Print blank line for readability
    
    print("\n" + "=" * 70)  # After training, perform final evaluation; print header
    print("Final Evaluation")  # Print title
    print("=" * 70)  # Print footer
    checkpoint = torch.load(Config.MODEL_PATH)  # Load the best model checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])  # Restore model weights
    _, _, all_preds, all_labels = evaluate(model, test_loader, criterion, Config.DEVICE)  # Evaluate to get predictions for confusion matrix
    
    cm = confusion_matrix(all_labels, all_preds)  # Create confusion matrix showing how predictions compare to true labels
    plt.figure(figsize=(10, 8))  # Create a figure for plotting
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=Config.CATEGORIES, yticklabels=Config.CATEGORIES)  # Draw heatmap of confusion matrix; annot=True: show numbers; fmt='d': format as integers; cmap='Blues': use blue color scheme
    plt.title('Confusion Matrix')  # Set plot title
    plt.ylabel('True Label')  # Set y-axis label
    plt.xlabel('Predicted Label')  # Set x-axis label
    plt.xticks(rotation=45)  # Rotate x-axis labels 45 degrees for readability
    plt.yticks(rotation=45)  # Rotate y-axis labels
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    plt.savefig('confusion_matrix.png')  # Save figure to file
    print("Confusion matrix saved as 'confusion_matrix.png'")  # Print save message
    
    print("\nClassification Report:")  # Print detailed classification metrics header
    print(classification_report(all_labels, all_preds, target_names=Config.CATEGORIES))  # Print detailed metrics (precision, recall, f1-score)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))  # Create training history plots; create figure with 2 subplots side by side
    ax1.plot(train_losses, label='Train Loss', marker='o')  # First subplot: loss over epochs; plot training loss with circle markers
    ax1.plot(val_losses, label='Val Loss', marker='s')  # Plot validation loss with square markers
    ax1.set_xlabel('Epoch')  # Set x-axis label
    ax1.set_ylabel('Loss')  # Set y-axis label
    ax1.set_title('Loss')  # Set subplot title
    ax1.legend()  # Show legend
    ax1.grid(True)  # Show grid
    ax2.plot(train_accs, label='Train Acc', marker='o')  # Second subplot: accuracy over epochs
    ax2.plot(val_accs, label='Val Acc', marker='s')  # Plot validation accuracy
    ax2.set_xlabel('Epoch')  # Set x-axis label
    ax2.set_ylabel('Accuracy (%)')  # Set y-axis label
    ax2.set_title('Accuracy')  # Set subplot title
    ax2.legend()  # Show legend
    ax2.grid(True)  # Show grid
    plt.tight_layout()  # Adjust spacing between subplots
    plt.savefig('training_history.png')  # Save the figure
    print("Training history saved as 'training_history.png'")  # Print save message
    print(f"\nTraining complete... Best accuracy: {best_acc:.2f}%")  # Print final summary
    print(f"Model saved as '{Config.MODEL_PATH}'")  # Print model path

def load_model():  # Function to load a trained model from disk
    if not Path(Config.MODEL_PATH).exists():  # Check if model file exists; Path().exists() returns True if file is found
        raise FileNotFoundError(f"Model not found at '{Config.MODEL_PATH}'. Please train the model first: python trashnet_arduino.py --train")  # Raise an error if model file doesn't exist; stops execution and shows helpful error message
    model = create_model(num_classes=len(Config.CATEGORIES))  # Create model architecture (same structure as training)
    checkpoint = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)  # Load checkpoint from file; map_location ensures it loads on correct device (CPU/GPU)
    model.load_state_dict(checkpoint['model_state_dict'])  # Restore saved weights into model
    model = model.to(Config.DEVICE)  # Move model to computation device
    model.eval()  # Set to evaluation mode
    print(f"Model loaded from '{Config.MODEL_PATH}'")  # Print success message
    print(f"Accuracy: {checkpoint['accuracy']:.2f}%")  # Print model accuracy
    return model  # Return the loaded model

def classify_image(model, image_path_or_array, transform):  # Function to classify a single image; image_path_or_array: either a file path string or NumPy array; transform: preprocessing transformations
    if isinstance(image_path_or_array, str):  # Check if input is a file path (string); isinstance checks if object is of a specific type
        img = Image.open(image_path_or_array).convert('RGB')  # Open image from file path and convert to RGB
        original_img = cv2.imread(image_path_or_array)  # Load image with OpenCV for visualization
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)  # Convert from OpenCV's BGR format to RGB
    else:  # Input is already an array
        img = Image.fromarray(image_path_or_array)  # Convert to PIL Image
        original_img = image_path_or_array  # Use the array directly as original image
    img_tensor = transform(img)  # Apply preprocessing transforms
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension (model expects batches); unsqueeze(0) adds a dimension at position 0: [C,H,W] -> [1,C,H,W]
    img_tensor = img_tensor.to(Config.DEVICE)  # Move to computation device
    with torch.no_grad():  # Disable gradient computation (we're not training)
        outputs = model(img_tensor)  # Get model predictions
        probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Apply softmax to convert logits to probabilities; dim=1 means apply softmax across the class dimension
        confidence, predicted = probabilities.max(1)  # Get the highest probability and its index; .max(1) returns (max_values, indices) along dimension 1
    predicted_class = Config.CATEGORIES[predicted.item()]  # Convert predicted index to category name
    confidence_value = confidence.item() * 100  # Convert confidence to percentage; .item() extracts the Python number from a 1-element tensor
    all_probs = probabilities[0].cpu().numpy()  # Get all probabilities as a NumPy array; [0] removes batch dimension; .cpu() moves to CPU; .numpy() converts to array
    return predicted_class, confidence_value, all_probs, original_img  # Return prediction results

def test_with_screen(use_arduino=False, arduino_port=None):  # Function to continuously capture and classify screen content; use_arduino: whether to control servo; arduino_port: which port Arduino is on
    print("=" * 70)  # Print header
    print("TrashNet Screen Testing")  # Print title
    if use_arduino:  # If Arduino is enabled
        print("with Arduino Servo Control")  # Print Arduino status
    print("=" * 70)  # Print footer
    
    model = load_model()  # Load the trained model
    
    arduino = None  # Initialize Arduino controller as None
    if use_arduino:  # If Arduino control is enabled
        print("\n" + "=" * 70)  # Print Arduino setup section
        print("Arduino Setup")  # Print title
        print("=" * 70)  # Print footer
        arduino = ArduinoController(port=arduino_port, baudrate=Config.ARDUINO_BAUDRATE)  # Create Arduino controller object
        if not arduino.connect():  # Try to connect
            print("\nContinuing without Arduino...")  # If connection fails, continue without Arduino
            arduino = None  # Set to None
        else:  # Connection succeeded
            print("\nServo angle mapping:")  # Print servo angle mapping header
            for category, angle in Config.SERVO_ANGLES.items():  # Loop through each category and its angle; .items() returns (key, value) pairs
                print(f"  {category:10s} → {angle:3d}°")  # Print formatted mapping; :10s = left-align string in 10 chars; :3d = right-align integer in 3 chars
    
    transform = transforms.Compose([  # Define image preprocessing pipeline
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),  # Resize to standard size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet values
    ])
    
    sct = mss.mss()  # Create screenshot capture object
    monitors = sct.monitors  # Get list of available monitors; monitors[0] is combined virtual monitor, monitors[1:] are physical monitors
    print(f"\nAvailable monitors: {len(monitors)-1}")  # Print monitor information
    for i, monitor in enumerate(monitors[1:], 1):  # Loop through monitors (skip index 0); enumerate(list, start) creates (index, item) pairs starting from 'start'
        print(f" Monitor {i}: {monitor['width']}x{monitor['height']}")  # Print monitor resolution
    
    monitor = monitors[Config.MONITOR_NUMBER]  # Select the monitor to capture from
    print(f"\nCapturing from Monitor {Config.MONITOR_NUMBER}")  # Print capture settings
    print(f"Capture interval: {Config.CAPTURE_INTERVAL}s")  # Print interval
    print("\nPress Ctrl+C to stop\n")  # Print stop instruction
    
    try:  # try-except-finally block for clean shutdown
        while True:  # Infinite loop for continuous capture; True is always true, so this loops forever until interrupted
            screenshot = sct.grab(monitor)  # Capture screenshot from selected monitor
            img = np.array(screenshot)  # Convert screenshot to NumPy array
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)  # Convert from BGRA (Blue, Green, Red, Alpha) to RGB format
            predicted_class, confidence, all_probs, _ = classify_image(model, img, transform)  # Classify the captured image; _ ignores original_img
            print(f"{predicted_class.upper()} - {confidence:.2f}%")  # Print prediction result; .upper() converts string to uppercase
            if arduino and arduino.is_connected:  # If Arduino is connected and working
                arduino.move_to_category(predicted_class)  # Move servo to corresponding category angle
            time.sleep(Config.CAPTURE_INTERVAL)  # Wait before next capture
    except KeyboardInterrupt:  # Catch KeyboardInterrupt (Ctrl+C)
        print("\n\nTesting stopped by user")  # Print stopping message
    finally:  # finally block always executes, even if there's an error
        if arduino:  # If Arduino is connected
            arduino.disconnect()  # Disconnect it

def test_with_image(image_path, use_arduino=False, arduino_port=None):  # Function to classify a single image file; image_path: path to the image file
    print("=" * 70)  # Print header
    print("TrashNet Image Testing")  # Print title
    if use_arduino:  # If Arduino is enabled
        print("with Arduino Servo Control")  # Print Arduino status
    print("=" * 70)  # Print footer
    
    model = load_model()  # Load model
    
    arduino = None  # Initialize Arduino if requested
    if use_arduino:  # If Arduino control is enabled
        print("\n" + "=" * 70)  # Print Arduino setup section
        print("Arduino Setup")  # Print title
        print("=" * 70)  # Print footer
        arduino = ArduinoController(port=arduino_port, baudrate=Config.ARDUINO_BAUDRATE)  # Create controller
        if not arduino.connect():  # Try to connect
            print("\nContinuing without Arduino...")  # If connection fails
            arduino = None  # Set to None
    
    transform = transforms.Compose([  # Define preprocessing
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),  # Resize
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
    ])
    
    print(f"\nClassifying: {image_path}")  # Print which image we're classifying
    predicted_class, confidence, all_probs, original_img = classify_image(model, image_path, transform)  # Classify the image
    
    print(f"\nPrediction: {predicted_class.upper()}")  # Print prediction
    print(f"Confidence: {confidence:.2f}%")  # Print confidence
    print("\nAll probabilities:")  # Print probabilities header
    for i, category in enumerate(Config.CATEGORIES):  # Print probabilities for all categories; enumerate gives us (index, item) pairs
        print(f"  {category:10s}: {all_probs[i]*100:5.2f}%")  # Print each category with its probability; :10s = left-align in 10 chars; :5.2f = float with 5 total digits, 2 after decimal
    
    if arduino and arduino.is_connected:  # Move servo if Arduino is connected
        print()  # Print blank line for spacing
        arduino.move_to_category(predicted_class)  # Move to category
    
    plt.figure(figsize=(10, 6))  # Create visualization; create figure with 2 subplots (1 row, 2 columns)
    plt.subplot(1, 2, 1)  # First subplot: show original image with prediction
    plt.imshow(original_img)  # Display image
    plt.title(f'Prediction: {predicted_class}\nConfidence: {confidence:.2f}%')  # Set title with prediction and confidence
    plt.axis('off')  # Hide axis ticks and labels
    plt.subplot(1, 2, 2)  # Second subplot: bar chart of probabilities
    colors = [Config.COLORS[cat] for cat in Config.CATEGORIES]  # Get colors for each category from our config; list comprehension: [expression for item in list]
    colors = [(r/255, g/255, b/255) for r, g, b in colors]  # Normalize RGB values from 0-255 to 0-1 for matplotlib
    plt.barh(Config.CATEGORIES, all_probs * 100, color=colors)  # Create horizontal bar chart; all_probs * 100 converts probabilities to percentages
    plt.xlabel('Confidence (%)')  # Set x-axis label
    plt.title('Category Probabilities')  # Set title
    plt.xlim(0, 100)  # Set x-axis range from 0 to 100%
    plt.tight_layout()  # Adjust spacing
    plt.savefig('classification_result.png')  # Save visualization
    print("\nVisualization saved as 'classification_result.png'")  # Print save message
    plt.show()  # Display the plot window
    
    if arduino:  # Wait for user to press Enter before disconnecting Arduino
        input("\nPress Enter to disconnect Arduino...")  # Wait for user input
        arduino.disconnect()  # Disconnect

def main():  # Main function - entry point of the program
    parser = argparse.ArgumentParser(description='TrashNet Model Testing with Arduino Servo Control', formatter_class=argparse.RawDescriptionHelpFormatter, epilog="""  # Create argument parser for command-line interface; description: help text shown at top; formatter_class: how to format help; epilog: additional text at bottom
Examples:
  Train model:
    python trashnet_arduino.py --train
  
  Test with screen capture (no Arduino):
    python trashnet_arduino.py --test
  
  Test with screen capture + Arduino (auto-detect port):
    python trashnet_arduino.py --test --arduino
  
  Test with Arduino on specific port:
    python trashnet_arduino.py --test --arduino --port /dev/ttyUSB0
    python trashnet_arduino.py --test --arduino --port COM3
  
  Test with image file:
    python trashnet_arduino.py --test-image trash.jpg --arduino
        """)
    parser.add_argument('--train', action='store_true', help='Train the model')  # Add command-line arguments; action='store_true': flag is False by default, becomes True if present
    parser.add_argument('--test', action='store_true', help='Test with screen capture')  # Add test flag
    parser.add_argument('--test-image', type=str, help='Test with image file')  # Add test-image argument; type=str means this argument takes a string value
    parser.add_argument('--arduino', action='store_true', help='Enable Arduino servo control')  # Add Arduino flag
    parser.add_argument('--port', type=str, help='Arduino serial port (auto-detect if not specified)')  # Add port argument
    parser.add_argument('--interval', type=float, default=2.0, help='Capture interval in seconds (default: 2.0)')  # Add interval argument; type=float, default=2.0: takes decimal number, defaults to 2.0
    parser.add_argument('--monitor', type=int, default=1, help='Monitor number to capture (default: 1)')  # Add monitor argument; type=int, default=1: takes integer, defaults to 1
    args = parser.parse_args()  # Parse the command-line arguments; creates an object (args) with attributes for each argument
    if args.interval:  # Update Config based on provided arguments; if user specified interval
        Config.CAPTURE_INTERVAL = args.interval  # Update the config
    if args.monitor:  # If user specified monitor number
        Config.MONITOR_NUMBER = args.monitor  # Update the config
    if args.port:  # If user specified port
        Config.ARDUINO_PORT = args.port  # Update the config
    if args.train:  # Determine which mode to run based on arguments; if --train flag was provided
        train_model()  # Run training
    elif args.test:  # elif = "else if" - only checked if previous if was False; if --test flag was provided
        test_with_screen(use_arduino=args.arduino, arduino_port=args.port)  # Run screen testing
    elif args.test_image:  # If --test-image was provided with a filename
        test_with_image(args.test_image, use_arduino=args.arduino, arduino_port=args.port)  # Run image testing
    else:  # If no mode was specified
        parser.print_help()  # Print the help message showing available options

if __name__ == '__main__':  # Standard Python idiom for script entry point; __name__ is a special variable that equals '__main__' when script is run directly (not when imported as a module)
    main()  # Call the main function to start the program
