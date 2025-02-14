import streamlit as st
from model import load_or_train_model

# Configure the page
st.set_page_config(
    page_title="Spam Detection App",
    page_icon="✉️",
    layout="centered",
    initial_sidebar_state="expanded"
)

def main():
    st.title("Spam Detection App")
    st.write("Enter a SMS message below and click 'Predict' to determine if it's spam or not.")

    # Load or train the model (cached)
    model = load_or_train_model()

    # initialize counter in session state 
    if "reset_counter" not in st.session_state:
        st.session_state.reset_counter = 0

    # Text input widget with a session state key
    user_input = st.text_area("Your Message:", key=f"user_input_{st.session_state.reset_counter}")

    # Create two columns for the Predict and Clear buttons side by side
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Predict"):
            if user_input:
                prediction = model.predict([user_input])[0]
                result = "Spam" if prediction == 1 else "Not Spam"
                st.success(f"Prediction: {result}")
            else:
                st.warning("Please enter a message to analyze.")

    with col2:
        if st.button("Clear"):
            st.session_state.reset_counter += 1
            st.rerun()

if __name__ == "__main__":
    main()