# 🧭 *Tourism Analysis & Recommendation System* 
A data-driven platform that analyzes tourism data and provides personalized recommendations using machine learning, with an interactive interface for end users.

---

## 🌍 Project Overview  
This project builds a comprehensive **Tourism Analysis and Recommendation System** using Python, machine learning models, and a Streamlit-based UI. It processes and analyzes tourism data from multiple dimensions, predicts user behavior (visit mode and rating), and recommends travel attractions tailored to individual preferences.  

The system is modular, scalable, and designed for both analytical exploration and practical user interaction.

---

## ✨ Features  

### 🛫 Visit Mode Prediction  
Predicts the likely **mode of visit** (e.g., Family, Solo, Couple, Group) for a user visiting a destination.  
- Trained using a **Random Forest Classifier**  
- Uses user demographics, travel history, attraction type, seasonality, and more  
- Provides intelligent suggestions on how a user might travel  

### ⭐ Rating Prediction  
Predicts the **rating a user might assign** to an attraction.  
- Uses a regression model (Random Forest Regressor)  
- Considers user preferences, past ratings, location characteristics, and item features  
- Enables early feedback and satisfaction forecasting

### 🎯 Personalized Recommendation System  
Generates travel recommendations based on a hybrid approach.  
- Combines **collaborative filtering** and **content-based filtering**  
- Uses user-item interaction history and attraction metadata  
- Delivers personalized and relevant destination suggestions  

### 📊 Data Analysis Dashboard  
Includes a module for **exploratory data analysis** with visualizations:  
- Regional & seasonal tourism trends  
- Attraction popularity by type and location  
- Demographic insights into user behavior  
- Rating distributions and travel patterns  

### 🖥️ Interactive Streamlit Web Application  
A web app built with **Streamlit** provides an intuitive interface for real-time interaction:  
- Predict visit mode and ratings via form inputs  
- Explore personalized recommendations  
- Visualize data trends with charts and summaries  

---

## ⚙️ Technical Implementation  

The project is structured in a modular fashion with clearly separated concerns:

### 📁 Data Layer  
- **Input:** Excel datasets (`City`, `Country`, `User`, `Transaction`, etc.)  
- **Process:** Data merging, missing value handling, standardization, and feature engineering  
- **Output:** `Final Dataset.xlsx` for modeling and app usage  

### 🤖 Modeling Layer  
- **Visit Mode Prediction Model:** Classification using Random Forest  
- **Rating Prediction Model:** Regression using Random Forest  
- **Recommendation Engine:** Hybrid recommender (user-item similarity + content metadata)  
- Models are trained, evaluated, and saved as `.pkl` files  

### 💻 Application Layer  
- **Streamlit App (`Tourism_Recommender.py`)**  
  - Loads models and processed dataset  
  - Provides UI for predictions and recommendations  
  - Visualizes analytical insights interactively  

---

## Project Structure
```
├── requirements.txt
├── Datasets/
│ ├── City.xlsx
│ ├── Continent.xlsx
│ ├── Country.xlsx
│ ├── Final Dataset.xlsx
│ ├── Item.xlsx
│ ├── Mode.xlsx
│ ├── Region.xlsx
│ ├── Transaction.xlsx
│ ├── Type.xlsx
│ └── User.xlsx
├── Interactive_App/
│ └── Tourism_Recommender.py
├── Models/
│ ├── Rating Predictor.pkl
│ ├── Recommendation Model.pkl
│ └── Visit Mode Predictor.pkl
├── Notebooks/
│ └── Data Analysis.ipynb
└── Source/
├── Data Preparation.ipynb
├── Rating Prediction Model.ipynb
├── Recommendation System.ipynb
└── Visit Mode Prediction Model.ipynb
```

---

## 📦 Dependencies  
All required Python packages are listed in `requirements.txt`.

---

## 🛠️ Installation  

### 1. Clone the Repository  
```
git clone https://github.com/sree-9523/Tourism-Analysis
```
### 2. Install required packages: 
```
pip install -r requirements.txt
```
### 3. Ensure Datasets & Models Are Present
  - Place all .xlsx files in the Datasets/ folder
  - If .pkl models are missing, run the notebooks in Source/ to train and generate them

---

## 🚀 Running the Application

### Step 1: Data Preparation
Open and Run:
```
Source/Data Preparation.ipynb
```

### Step 2: Model Training
Train each model by running:
```
* Visit Mode Prediction Model.ipynb
* Rating Prediction Model.ipynb
* Recommendation System.ipynb
```

### Step 3: Launching the Streamlit App
Run
```
streamlit run Interactive_App/Tourism_Recommender.py
```
The app will launch in your browser with navigation tabs for predictions, recommendations, and insights.

---

## 🧪 Example Use Cases
- “How would a solo traveler rate this attraction in Europe during winter?”
- “Recommend 5 destinations for a family traveling from India.”
- “Show me trends in attraction ratings by region and season.”

---

## 🔮 Future Improvements
- Integrate real-time APIs for dynamic attraction data
- Include sentiment analysis from travel reviews
- Add multilingual support for global accessibility
- Implement deep learning-based ranking for better recommendations
- Deploy as a hosted web app for broader use

---

## 📙 License
MIT License

Copyright (c) 2025 sree-9523
