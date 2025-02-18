# Dataset Construction for Abstract Art Generation

## Dataset Composition and Rationale

The dataset was constructed to train a VQ-VAE model for abstract art generation, with careful consideration given to both the quantity and quality of training images. The final dataset comprises 11,056 images, strategically sourced from multiple artistic movements and collections to ensure comprehensive coverage of abstract artistic styles.

### Core Components

1. Pure Abstract Art Foundation (2,700 images)
   - Sourced from a curated collection of contemporary abstract art
   - Split between two galleries (Abstract_gallery and Abstract_gallery_2)
   - Represents modern abstract artistic practices

2. Core Abstract Movements (~5,390 images)
   - Abstract Expressionism: 2,500 images
   - Action Painting: 90 images
   - Color Field Painting: 1,500 images
   - Minimalism: 1,300 images
   - Selected for their direct relevance to abstract art generation

3. Cubist Influences (~1,050 images)
   - Analytical Cubism: 80 images
   - Synthetic Cubism: 170 images
   - General Cubism: 800 images
   - Chosen for their role in the development of abstract art

4. Supporting Modern Movements (~1,800 images)
   - Art Nouveau Modern: 600 images
   - Expressionism: 800 images
   - Fauvism: 400 images
   - Selected to provide additional stylistic elements and compositional variety

## Methodology

### Selection Criteria
1. Core abstract works were included with minimal filtering to preserve the full range of abstract expression
2. Cubist and supporting movements were filtered more strictly to exclude representational works
3. Images were automatically filtered to remove pieces with explicitly representational descriptions (e.g., "portrait", "landscape", "still-life")

### Data Organization
1. Maintained an 80/20 train/test split across all categories
2. Preserved original movement classifications in filenames
3. Implemented robust file handling to manage special characters and ensure consistent naming

### Quality Control
1. Automated filtering for representational content
2. Maintained movement-specific selection ratios to ensure balanced representation
3. Achieved a 99.82% successful processing rate (11,056 successful out of 11,076 total attempts)

## Technical Implementation

The dataset construction was automated using a Python-based pipeline that handled:
1. File management and organization
2. Special character normalization in filenames
3. Train/test splitting
4. Movement-based categorization
5. Automated filtering of representational works
6. Statistical tracking and verification

This methodological approach ensures a dataset that is:
1. Sufficiently large for deep learning applications
2. Well-balanced across artistic movements
3. Focused on abstract styles while maintaining stylistic diversity
4. Properly organized for machine learning tasks
5. Well-documented and reproducible

The resulting dataset provides a robust foundation for training generative models, with particular emphasis on abstract art generation while maintaining sufficient stylistic diversity to prevent overfitting or mode collapse.
