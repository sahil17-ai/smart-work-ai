# smart-work-ai
# Smart Work Assistant AI

Hybrid AI system that predicts task completion risk using:
- Deep Learning (LSTM) for text understanding
- Machine Learning (Random Forest) for behavior analysis

## Features
- Detects unclear commitments from text
- Predicts performance risk from past behavior
- Combines both for final decision
- Interactive UI using Streamlit

## Tech Stack
Python, TensorFlow, Scikit-learn, Streamlit

## How to Run
pip install -r requirements.txt
streamlit run app.py

## Example
Input: "I will try to finish"
Output: ⚠️ Unclear commitment

Input: "I will finish by 5pm"
Output: ✅ Likely to complete
