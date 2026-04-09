# Real-World Testing Results: Swedish Waste Classification

## Testing Methodology
- Gothenburg
- 17 real trash items
- Wooden surface
- With different zoom levels and angles

## Test Results Summary

| Category | Total Tests | Correct | Incorrect | Accuracy |
|----------|-------------|---------|-----------|----------|
| Cardboard | 3 | 3 | 0 | 100% |
| Paper | 1 | 1 | 0 | 100% |
| Plastic | 4 | 1 | 3 | 25% |
| Metal | 7 | 6 | 1 | 86% |
| Glass | 2 | 0 | 2 | 0% |
| **Overall** | **17** | **11** | **6** | **65%** |

## Detailed Results

### Cardboard (3/3 correct - 100%)
1.  Cardboard - Correctly classified
2.  Cardboard - Correctly classified
3.  Cardboard - Correctly classified

### Paper (1/1 correct - 100%)
1.  Paper - Correctly classified

### Plastic (1/4 correct - 25%)
1.  Plastic - Misclassified
2.  Plastic - Misclassified
3.  Plastic - Correctly classified
4.  Plastic - Misclassified

### Metal (6/7 correct - 86%)
1. - Metal - Correctly classified
2. - Metal - Correctly classified
3. - Metal (covered with plastic) → Misclassified as cardboard
4. - Metal - Correctly classified
5. - Metal - Correctly classified
6. - Metal - Correctly classified
7. - Metal - Correctly classified

### Glass (0/2 correct - 0%)
1. - Glass → Misclassified
2. - Glass → Misclassified

## Observations

### 1. Strong Performance
- **Cardboard/Paper**: 100% accuracy (4/4)
  - In Sweden, I believe cardboard and paper are recycled together, so distinguishing between them is not critical
  - Model performs excellently on fibrous materials

### 2. Moderate Performance
- **Metal**: 86% accuracy (6/7)
  - Good recognition of clean metal surfaces
  - Failed when metal was covered with plastic wrapper (classified as cardboard)
  - Colored metals performed worse than straight metals

### 3. Poor Performance
- **Plastic**: 25% accuracy (1/4)
  - Worst performing category in real world tests
  - Significant domain gap between TrashNet and Swedish waste
  - Colored plastics particularly problematic
  
- **Glass**: 0% accuracy (0/2)
  - Complete failure on glass bottles/containers
  - Possible reasons: transparency, reflections, different glass types than TrashNet

## Environmental Factors

### Background Effect
- **Wooden surface background** may have influenced classifications
- Wood texture similar to cardboard - possible confusion
- I could test with neutral backgrounds (white/gray)

### Zoom Level Effect
- **Varying zoom helped** in some cases
- Close-up shots: Better detail but lose context
- Wide shots: More context but less detail
- Optimal zoom: Object fills around 70% of frame

### Lighting Conditions
- Indoor lighting with wooden background
- No controlled studio lighting

## Domain Gap Analysis

### TrashNet vs Swedish Waste
1. **Different packaging styles**
   - TrashNet: US-style packaging
   - Swedish test: European/Nordic packaging designs
   - Color schemes differ significantly

2. **Material variations**
   - Plastic types vary by region
   - Swedish plastic containers may use different polymers
   - Glass bottles have different shapes/colors

3. **Background context**
   - TrashNet: Clean, uniform backgrounds
   - Real-world: Wooden surface, varied lighting
   - Domain adaptation needed for deployment

## Swedish Recycling Context

### Practical Implications
- **Cardboard/Paper distinction not needed**: Sweden recycles these together (Pappersförpackningar)
- **Model's strength aligns with needs**: Excellent at cardboard/paper
- **Critical weaknesses**: Plastic and glass classification needs improvement for Swedish trash

### Swedish Recycling Categories
1. **Pappersförpackningar** (Paper packaging) - Excellent
2. **Plastförpackningar** (Plastic packaging) - Poor
3. **Metallförpackningar** (Metal packaging) - Moderate
4. **Glasförpackningar** (Glass packaging) - Very Poor
5. **Restavfall** (Residual waste) - Not tested

## Conclusions

### Strengths
1. Excellent performance on fibrous materials (cardboard/paper)
2. Reasonably good metal detection (when unobstructed)
3. Fast inference time suitable for real-time use

### Weaknesses
1. Poor plastic recognition with Swedish packaging
2. Failed completely on glass items
3. Could be sensitive to background and lighting
4. Struggles with colored/wrapped materials

### For Improvement
1. **Fine-tune on Swedish waste data**
   - Collect dataset of Swedish packaging
   - Include variety of backgrounds
   - Capture different lighting conditions (see ideal conditions for the robot in Readme.md, this is a prototype)

2. **Data augmentation strategies**
   - Add background variation during training (or ideal conditions for robot)
   - Add transparency/reflection augmentation for glass

3. **Model improvements**
   - Maybe add attention mechanisms for material texture


## Overall Assessment
The model shows **promising but insufficient performance** for real Swedish waste sorting:
- Training accuracy: 82.10% (TrashNet test set)
- Real-world accuracy: 65% (Swedish waste)

**Conclusion**: Requires fine-tuning on Swedish waste data before production deployment.
