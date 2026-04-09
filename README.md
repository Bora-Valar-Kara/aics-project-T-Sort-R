# Embodied AI for Waste Classification

**Author:** Bora Kara
**Course:** LT2318 H25 Artificial Intelligence: Cognitive Systems
**University of Gothenburg**

## Project Overview
A prototype embodied AI system that classifies waste using a ResNet18 model (transfer learning on TrashNet) and controls an Arduino servo motor for physical sorting. Demonstrates the perception → cognition → action loop.

## Repository Structure
- `code/` — Python classification script and Arduino servo code
- `data/` — Demo videos and real-world test images of Swedish waste
- `paper/` — Report (.tex, .bib, .pdf) and figures
- `notes/` — Lab log and real-world testing results
- `library/` — Supporting files

## How to Run

### Requirements
```
pip install torch torchvision pillow opencv-python mss numpy matplotlib datasets pyserial scikit-learn seaborn tqdm
```

### Training
```
python code/trashtest.py --train
```

### Testing with screen capture
```
python code/trashtest.py --test
```

### Testing with screen capture + Arduino
```
python code/trashtest.py --test --arduino
```

### Testing a single image
```
python code/trashtest.py --test-image path/to/image.jpg
```

### Arduino
Upload `code/trashnet_servo.ino` to an Arduino Uno via the Arduino IDE.

## Data on MLTGPU Server
Large files (trained model, dataset cache) are at:
```
/srv/data/guskarabo/
```

## Key Results
- **Benchmark accuracy:** 82.10% on TrashNet test set
- **Real-world accuracy:** 65% on 17 Swedish waste items
- **Domain gap:** Cardboard/paper transferred well; plastic/glass did not

## Resources
- TrashNet dataset: https://github.com/garythung/trashnet
- HuggingFace: https://huggingface.co/datasets/garythung/trashnet
- Kaggle demo dataset: https://www.kaggle.com/datasets/feyzazkefe/waste-classification-data
