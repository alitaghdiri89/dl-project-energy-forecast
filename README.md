# ⚡ Project Deep Learning: Electricity Price Forecasting

This repository contains experiments with deep learning models for **forecasting electricity prices** using historical London energy data. The workflow covers dataset preparation, visualization, model training, and evaluation.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/Project-Deep-Learning.git
cd Project-Deep-Learning
```
### 2. Create a virtual enviornment
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage
### Datasets
To get the datasets either run the `01-Dataset_Preparation.ipynb` notebook or copy the contents of `\data\processed.zip` in the `\data\processed\`directory.
### Data analysis and visualization
Notebook `02-Plot_data.ipynb` contains various visualizations for analysis.
### Train the model
Run the main training script:
```bash
python 03-Main.py
```
### 📊 Results
The evaluation notebook (`04-Evaluation.ipynb`) contains metrics and visualizations of the trained model’s accuracy and forecast capability.

## ✅ Requirements
This project requires Python 3.8+ and the following libraries:
- numpy
- pandas
- matplotlib
- scikit-learn
- torch (PyTorch, used as the deep learning framework)
- jupyter

## 👥 Contributors

- [Morteza Feizbakhsh](https://github.com/mortezaflb)
- [Aliakbar Taghdiri](https://github.com/alitaghdiri89)


## 📚 References

1. Nugaliyadde A., Somaratne U., Wong K. W. (2019). *Predicting Electricity Consumption using Deep Recurrent Neural Networks*. arXiv:1909.08182v1, https://doi.org/10.48550/arXiv.1909.08182
2. Smart meter data from London area Available at: https://www.kaggle.com/jeanmidev/smart-meters-in-london
