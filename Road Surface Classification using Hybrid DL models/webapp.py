import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet import preprocess_input

# Function to load the selected model
def load_selected_model(selected_model):
    if selected_model == 'HybridModel':
        return load_model('model_hybrid_1.keras')
    elif selected_model == 'ResNet':
        return load_model('ResNet_balanced.keras')
    elif selected_model == 'MobileNet':
        return load_model('MobileNet_balanced.keras')

# Define class labels
class_labels = ['Wet Asphalt Smooth', 'Wet Concrete Smooth', 'Wet Gravel']

# Dictionary containing information about each model
model_info = {
    'HybridModel': 'The Hybrid Model combines features from ResNet and MobileNet to achieve better performance in road surface image classification.',
    'ResNet': 'ResNet is a deep convolutional neural network architecture known for its residual connections, which help alleviate the vanishing gradient problem.',
    'MobileNet': 'MobileNet is a lightweight neural network architecture optimized for mobile and embedded devices, leveraging depthwise separable convolutions for efficiency without sacrificing accuracy.'
}

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to match the input size
    image = image.resize((224, 224))
    # Convert PIL image to numpy array
    img_array = np.array(image)
    # Preprocess the image according to the model requirements
    #img_array = preprocess_input(img_array)
    return img_array

# Function to make predictions
def predict(image, model):
    processed_image = preprocess_image(image)
    # Assuming your model outputs probabilities for each class
    prediction = model.predict(np.expand_dims(processed_image, axis=0))
    return prediction

# Streamlit web application
st.title('Road Surface Image Classification')

# Sidebar for model selection
st.sidebar.title('Model Selection')
selected_model = st.sidebar.selectbox('Choose a Model', ['HybridModel', 'ResNet', 'MobileNet'])
with st.spinner('Working on it...'):
    # Load the selected model
    model = load_selected_model(selected_model)

    # Display model information in the right sidebar
    st.sidebar.title('Model Information')
    st.sidebar.write(model_info[selected_model])

    # Upload new image

    uploaded_file = st.file_uploader("Upload an image of road surface...", type=["jpg", "jpeg", "png"])


# Display uploaded image
if uploaded_file is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make prediction on the uploaded image
        with st.spinner('Classifying the image...'):
            prediction = predict(image, model)[0]  # Extracting the prediction from the array
            max_class_index = np.argmax(prediction)
            max_class = class_labels[max_class_index]
            max_probability = prediction[max_class_index]

        st.write(f'The Predicted Class is: {max_class} with the score of {max_probability*100:.2f}%')

    except Exception as e:
        st.error("An error occurred: {}".format(str(e)))

# Display hyperlinks to download Python notebooks
st.sidebar.title('Developed by')
st.sidebar.markdown('Sharaj Jagadeesan 23100668')
st.sidebar.markdown('Estella Roberts 23102324')
st.sidebar.markdown('Jason Martin 23101224')
st.sidebar.markdown('Merlin Mariam Moncy 23103030')
st.sidebar.markdown('Sreelakshmi Anjilimarayil Shyjumon 23103188')
#st.sidebar.markdown("Download the Python notebooks used for model training and development:")
#st.sidebar.markdown("[HybridModel Python Notebook](https://github.com/Sharaj17/Road-Surface-Classification/blob/master/CT5103_Assignment1.pdf)")
#st.sidebar.markdown("[Sample Images for Testing](https://github.com/Sharaj17/Road-Surface-Classification/tree/master/Test%20Images)")

