import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
import time

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
start_button = st.button("Start Gradient Descent")

# Full screen CSS styling
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
        .full-screen-container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 90vh;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True)

if start_button:
    # Fit the model to the data
    gd.fit(X, y)

    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the data points
    ax.scatter(X, y, color='blue', label='Data points')

    # Line object for the regression line (initially empty)
    line, = ax.plot([], [], color='red', label='SGD Regression Line')

    # Set up the axis limits
    ax.set_xlim(np.min(X) - 1, np.max(X) + 1)
    ax.set_ylim(np.min(y) - 10, np.max(y) + 10)
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend()

    # Streamlit container for the animation
    placeholder = st.empty()

    # Function to update the line during animation
    def update(epoch):
        # Get the current values of m and b from the history
        m, b = gd.history[epoch]
        
        # Update the regression line
        line.set_data(X, m * X + b)
        
        # Update the title to show the current epoch
        ax.set_title(f'Epoch {epoch + 1}')
        return line,

    # Create the FuncAnimation object
    ani = FuncAnimation(fig, update, frames=len(gd.history), interval=500)

    # Save the animation to a temporary file
    buf = io.BytesIO()
    ani.save(buf, writer='ffmpeg', fps=24)
    buf.seek(0)

    # Display the animation inside the Streamlit app
    with placeholder.container():
        st.markdown('<div class="full-screen-container">', unsafe_allow_html=True)
        st.video(buf)
        st.markdown('</div>', unsafe_allow_html=True)

    # After the animation ends, display the MSE loss and gradient
    st.subheader("Gradient Descent Loss Function and Gradient (MSE)")
    losses = [np.mean((y - (m * X + b))**2) for m, b in gd.history]
    gradients = [(-2 * np.sum(y - m * X.ravel() - b), -2 * np.sum((y - m * X.ravel() - b) * X.ravel())) for m, b in gd.history]

    # Plot the loss function and gradient
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    # Plot the MSE loss
    ax2.plot(losses, label='MSE Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_title("Loss Function (MSE) over Epochs")
    ax2.legend()

    # Plot the gradient (for both m and b)
    fig3, ax3 = plt.subplots(figsize=(8, 6))

    gradient_m = [g[0] for g in gradients]
    gradient_b = [g[1] for g in gradients]
    ax3.plot(gradient_m, label='Gradient for m')
    ax3.plot(gradient_b, label='Gradient for b')

    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Gradient')
    ax3.set_title("Gradients with respect to Loss Function (MSE)")
    ax3.legend()

    # Display loss and gradient plots
    st.pyplot(fig2)
    st.pyplot(fig3)

