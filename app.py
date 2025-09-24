import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
import time
import io

# Define the GDregressor class
class GDregressor:
    def __init__(self, learning_rate, epochs):
        self.history = []  # Store the history of m and b values
        self.m = 100  # Initial slope
        self.b = -120  # Initial intercept
        self.lr = learning_rate  # Learning rate
        self.epochs = epochs  # Number of epochs
        
    def fit(self, X, y):
        for i in range(self.epochs):
            # Calculate gradients
            loss_b = -2 * np.sum(y - self.m * X.ravel() - self.b)
            loss_m = -2 * np.sum((y - self.m * X.ravel() - self.b) * X.ravel())
            
            # Update the parameters
            self.b = self.b - (self.lr * loss_b)
            self.m = self.m - (self.lr * loss_m)
            
            # Store the history of parameters (slope and intercept)
            self.history.append((self.m, self.b))

# Streamlit app interface
st.title("Gradient Descent Regression Visualization")

# Sidebar for user inputs
st.sidebar.header("Configure the dataset")
n_samples = st.sidebar.slider("Number of samples", 50, 300, 100)
n_features = st.sidebar.slider("Number of features", 1, 5, 1)
noise_level = st.sidebar.slider("Noise level", 0, 50, 20)

# Generate the regression data based on the user inputs
X, y = make_regression(n_samples=n_samples, n_targets=1, n_informative=1, 
                       n_features=n_features, noise=noise_level, random_state=13)

# Create a regression model
learning_rate = st.sidebar.slider("Learning rate", 0.0001, 0.1, 0.001)
epochs = st.sidebar.slider("Epochs", 1, 200, 35)
gd = GDregressor(learning_rate=learning_rate, epochs=epochs)

# Add a start button to trigger the gradient descent

# Full screen CSS styling for maximized layout
st.markdown("""
    <style>
        .streamlit-expanderHeader {
            font-size: 22px;
            font-weight: bold;
        }
        .stButton>button {
            background-color: #FF5733;
            color: white;
            font-size: 20px;
            padding: 15px;
            border-radius: 10px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #FF5733;
            color: black;
        }
        /* Fullscreen style for the animation container */
        .full-screen-container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 80vh;  /* 80% of the screen height */
            width: 100%;
            margin-top: 5%;
        }
        /* Upward Arrow styling */
        .arrow {
            display: block;
            width: 0;
            height: 0;
            border-left: 10px solid transparent;
            border-right: 10px solid transparent;
            border-bottom: 20px solid black;  /* Upward arrow */
            margin: auto;
        }
        /* Position the arrow directly below the button */
        .arrow-container {
            text-align: center;
            margin-top: 20px;  /* Adjust the position to be right below the button */
            cursor: pointer;
        }
    </style>
    """, unsafe_allow_html=True)

# Add a start button to trigger the gradient descent
start_button = st.button("Start Gradient Descent")

# Hide the button after click
if start_button:
    # Remove the button from the interface
    start_button = st.empty()  # This clears the button

    # Display the instruction text and the upward arrow
    st.markdown("""
        <div class="arrow-container">
            <p><strong>Click on the arrow below to start the animation</strong></p>
            <div class="arrow"></div>
        </div>
    """, unsafe_allow_html=True)

    # After the user clicks the arrow, the animation will start
    if st.button("Start Animation"):
        # Proceed with animation (insert the animation code here)
        st.write("Animation started!")

