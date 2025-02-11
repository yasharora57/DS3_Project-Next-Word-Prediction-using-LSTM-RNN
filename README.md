# Next Word Prediction Using LSTM and GRU Models

This project focuses on building a **Next Word Prediction** model using **Recurrent Neural Networks (RNNs)** with **LSTM (Long Short-Term Memory)** and **GRU (Gated Recurrent Unit)** architectures. The model is trained on Shakespeare's *Hamlet* text dataset, which is preprocessed, tokenized, and used to predict the next word in a given sequence of words. The project also includes a **Streamlit web application** for interactive next-word prediction.

---

### How to Use the Project

1. **Clone the Repository**:
   - Clone the GitHub repository to your local machine.

2. **Install Dependencies**:
   - Install the required libraries using `pip install -r requirements.txt`.

3. **Run the Streamlit App**:
   - Navigate to the project directory and run `streamlit run app.py`.
   - Open the provided URL in your browser to access the web app.

4. **Predict Next Word**:
   - Enter a sequence of words in the input box and click "Predict Next Word" to see the prediction.

### Key Components and Workflow

1. **Data Collection**:
   - The dataset used is Shakespeare's *Hamlet*, downloaded using the **NLTK Gutenberg corpus**.
   - The text is saved into a file (`hamlet.txt`) for further processing.

2. **Data Preprocessing**:
   - The text is converted to lowercase to ensure uniformity.
   - **Tokenization** is performed using TensorFlow's `Tokenizer`, which assigns unique indices to each word in the text.
   - Input sequences are created by generating n-grams (sequences of words) from the text.
   - Sequences are padded to ensure uniform length using `pad_sequences`.

3. **Model Architecture**:
   - Two RNN models are implemented:
     - **LSTM Model**: A sequential model with an embedding layer, two LSTM layers, dropout for regularization, and a dense output layer with softmax activation.
     - **GRU Model**: Similar to the LSTM model but uses GRU layers instead of LSTM.
   - Both models use **categorical cross-entropy** as the loss function and **Adam** as the optimizer.
   - Early stopping is implemented to prevent overfitting, monitoring validation loss with a patience of 3 epochs.

4. **Training**:
   - The dataset is split into training and testing sets (80-20 split).
   - The models are trained for 50 epochs, with early stopping to ensure optimal performance.

5. **Next Word Prediction**:
   - A function (`predict_next_word`) is implemented to predict the next word in a given sequence using the trained model.
   - The function tokenizes the input text, pads it to the required length, and uses the model to predict the most likely next word.

6. **Model and Tokenizer Saving**:
   - The trained LSTM model is saved as `next_word_lstm.h5`.
   - The tokenizer is saved using `pickle` for future use.

7. **Streamlit Web Application**:
   - A simple web app is built using **Streamlit** to provide an interactive interface for next-word prediction.
   - Users can input a sequence of words, and the app predicts the next word using the saved LSTM model and tokenizer.

---

### Key Features and Techniques

- **RNN Architectures**: Utilizes both LSTM and GRU, which are well-suited for sequential data like text.
- **Early Stopping**: Prevents overfitting by stopping training when validation loss stops improving.
- **Tokenization and Padding**: Ensures the text data is in a format suitable for neural networks.
- **Interactive Web App**: Provides a user-friendly interface for testing the model.


---

This project demonstrates the power of RNNs in natural language processing tasks and provides a foundation for building more advanced text prediction systems. The inclusion of a Streamlit app makes it accessible and interactive for users.
