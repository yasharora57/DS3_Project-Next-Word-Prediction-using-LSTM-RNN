# Next Word Prediction Using LSTM/GRU RNN

## Project Summary

This project focuses on building a predictive model that performs next word prediction from text, using deep learning techniques with Recurrent Neural Networks (RNNs). The project is implemented in Python using TensorFlow/Keras and is designed to showcase techniques such as data preprocessing, sequence generation, and model deployment via a web application. The dataset used in this project is derived from Shakespeare's *Hamlet*, which is obtained using the NLTK Gutenberg corpus.

### Data Collection

- **Source:**  
  The dataset is sourced from NLTK's Gutenberg corpus. Specifically, the raw text of *Hamlet* is downloaded and saved to a file named `hamlet.txt`.
  
- **Libraries Used:**  
  - `nltk` (for accessing the Gutenberg corpus)
  - `pandas` (for data manipulation)

### Data Preprocessing

- **Text Loading and Cleaning:**  
  The text is loaded from the `hamlet.txt` file and converted to lowercase to maintain uniformity during tokenization.
  
- **Tokenization:**  
  Using TensorFlow’s `Tokenizer`, the text is converted into sequences of integers, where each unique word is assigned a unique index. The total number of words is computed to define the vocabulary size.
  
- **Sequence Generation:**  
  The text is split into lines, and for each line, n-gram sequences are generated. For example, from the sentence “To be or not to be”, multiple sequences such as `[To, be]`, `[To, be, or]`, etc., are created.
  
- **Padding:**  
  All sequences are padded using TensorFlow’s `pad_sequences` to ensure that they are of equal length, which is essential for batch processing in neural networks.

- **Creating Predictors and Labels:**  
  The padded sequences are then split into predictors (`x`) and labels (`y`), where `x` contains all tokens except the last one and `y` contains the last token of each sequence. The labels are one-hot encoded using TensorFlow utilities.

### Model Development

Two models were explored in this project:

1. **LSTM-Based Model:**  
   - **Architecture:**  
     - An Embedding layer to convert word indices to dense vectors.
     - Two LSTM layers with dropout in between to learn sequential patterns.
     - A Dense output layer with softmax activation to predict the probability distribution over the vocabulary.
   - **Compilation:**  
     The model is compiled using the categorical crossentropy loss function and the Adam optimizer.
   
2. **GRU-Based Model:**  
   - **Architecture:**  
     - Similar to the LSTM model, but with GRU layers replacing LSTM layers.
     - This alternative model is evaluated for performance and efficiency in capturing sequence information.
   - **Compilation:**  
     Also compiled with categorical crossentropy and the Adam optimizer.

- **Training:**  
  The data is split into training and test sets using scikit-learn’s `train_test_split`. An early stopping mechanism is implemented to prevent overfitting by monitoring the validation loss. The models are trained for multiple epochs until the validation loss stops improving.

### Prediction Function

A custom function is defined to predict the next word given an input sequence. The function:
- Converts the input text to sequences using the tokenizer.
- Pads the sequence to match the required input length.
- Uses the model to predict the next word.
- Finds and returns the word corresponding to the highest predicted probability.

### Model Deployment

- **Model Saving:**  
  After training, the best performing model is saved in HDF5 format (`next_word_lstm.h5`).
  
- **Tokenizer Saving:**  
  The tokenizer object is saved using Python’s `pickle` module to ensure the same vocabulary is used during prediction.

- **Streamlit Web App:**  
  A simple Streamlit application is built to provide an interactive interface. Users can input a sequence of words, and the app displays the next word prediction using the loaded model and tokenizer.

---

This project demonstrates the complete pipeline for building a next word prediction model using deep learning, from data collection and preprocessing to model training and interactive deployment. It showcases the practical application of LSTM/GRU networks in natural language processing and provides an end-to-end solution that can be further extended or fine-tuned for various language modeling tasks.
