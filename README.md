# Network Intrusion Detection using ACGAN - 91.5% Accuracy

## Problem
NSL-KDD dataset has <1% U2R/R2L attacks. Normal models get 72% accuracy.

## My Solution
Built Auxiliary Classifier GAN (ACGAN) to generate synthetic attack samples.
Generator creates fake attack traffic, Discriminator learns to detect attacks.

## Results
- Baseline (No GAN): 72.3% accuracy
- With ACGAN: 91.5% accuracy
- Tech: Python, TensorFlow, Keras, NSL-KDD

## How to Run
pip install tensorflow pandas scikit-learn
python main.py
