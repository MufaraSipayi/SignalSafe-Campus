import streamlit as st
import joblib

# Page settings
st.set_page_config(
    page_title="SignalSafe Campus",
    page_icon="🛡️",
    layout="centered"
)

# Load the saved model and vectorizer
@st.cache_resource
def load_model():
    bundle = joblib.load("signalsafe_model.pkl")
    return bundle["model"], bundle["vectorizer"]

model, vectorizer = load_model()

# Website heading
st.title("🛡️ SignalSafe Campus")
st.subheader("Smart Text Detection System")

st.write(
    "Enter a message below to check the prediction "
    "made by the trained machine-learning model."
)

# Message input
message = st.text_area(
    "Enter your message:",
    placeholder="Type or paste a message here..."
)

# Prediction button
if st.button("Check Message"):
    if message.strip():
        # Convert text into numerical features
        message_vector = vectorizer.transform([message])

        # Make prediction
        prediction = model.predict(message_vector)[0]

        st.subheader("Prediction Result")
        st.success(f"Model prediction: {prediction}")

        # Show confidence if supported by the model
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(message_vector)[0]
            confidence = max(probabilities) * 100
            st.write(f"Model confidence: {confidence:.2f}%")
        else:
            st.info("This model does not provide prediction confidence.")

    else:
        st.warning("Please enter a message first.")

st.divider()

st.caption(
    "SignalSafe Campus is an educational demonstration. "
    "Predictions may be incorrect and should not be used "
    "for real-world safety decisions."
)
