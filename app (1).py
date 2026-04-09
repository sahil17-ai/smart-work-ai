
import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model in .keras format
model = load_model("dl_model.keras")
clf = pickle.load(open("ml_model.pkl", "rb"))
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

st.title("🤖 Smart Work Assistant AI")
st.subheader("📊 Task Risk Analyzer")

user_text = st.text_input("Task Statement")

tasks = st.number_input("Tasks Completed", 0, 20)
hours = st.number_input("Hours Worked", 0, 12)
delays = st.number_input("Delays", 0, 10)

if st.button("Analyze"):

    if user_text.strip() == "":
        st.warning("Enter text")
        st.stop()

    seq = tokenizer.texts_to_sequences([user_text])
    pad = pad_sequences(seq, maxlen=6)
    text_pred = model.predict(pad)[0][0]

    behavior_pred = clf.predict([[tasks, hours, delays]])[0]

    st.markdown("### 🔍 Result")

    if text_pred < 0.5 and behavior_pred == 0:
        st.error("🚨 High Risk")
    elif text_pred < 0.5:
        st.warning("⚠️ Unclear commitment")
    elif behavior_pred == 0:
        st.warning("⚠️ Poor performance")
    else:
        st.success("✅ Likely to complete")

    st.progress(float(text_pred))
    st.write(f"Confidence Score: {text_pred:.2f}")
