# baseline-text-classifier-logreg
Simple baseline text classifier using TF-IDF features and logistic regression.

# Overview
- Data: AG News
- Task: 4-class topic classification
- Model: Logisitic Regression (scikit-learn)
- Features: TF-IDF
- Goal: Establish a clean, reproducible baseline

# Project Structure
baseline-text-classifier-logreg/ 
├── README.md
├── src/ 
│ ├── predict.py # prediction script
│ ├── train.py # main training script 
│ └── utils.py # preprocessing helpers 
└── requirements.txt

# How to Run
pip install -r requirements.txt 
python src/train.py

# Results (Output)
Accuracy: 0.9056578947368421
F1 Score: 0.9054772607107149
Confusion Matrix: 
[[1706 59 86 49] 
[ 27 1849 16 8] 
[ 67 18 1650 165] 
[ 59 21 142 1678]]

# Discussion
This model did well with sports, with an almost perfect classification. However, this model struggled with differentiating Business and Sci/Tech.