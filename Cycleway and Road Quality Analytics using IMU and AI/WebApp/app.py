from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import pandas as pd
import numpy as np
import io
import base64
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import folium
from scipy.signal import savgol_filter
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio

app = Flask(__name__)

image_folder = r"D:\Semester 2\Cycleway_Quality_Analysis\static\images"

# Load the models
models = {
    'rf_model': joblib.load(r"D:\Semester 2\Cycleway_Quality_Analysis\models\random_forest_model.pkl"),
    'svm_model': joblib.load(r"D:\Semester 2\Cycleway_Quality_Analysis\models\svm_model.pkl"),
    'knn_model': joblib.load(r"D:\Semester 2\Cycleway_Quality_Analysis\models\knn_model.pkl"),
    'logreg_model': joblib.load(r"D:\Semester 2\Cycleway_Quality_Analysis\models\logistic_regression_model.pkl"),
    'nn_model': tf.keras.models.load_model(r"D:\Semester 2\Cycleway_Quality_Analysis\models\neural_network_model.h5")
}

# Preprocessing and smoothing data
def preprocess_and_smooth_data(data):
    col_to_select = ['seconds_elapsed', 'accelerometer_z', 'accelerometer_x', 'accelerometer_y', 'location_latitude', 'location_longitude']
    data = data[col_to_select]
    data = data.dropna().reset_index(drop=True)

    smoothed_columns = ['accelerometer_z', 'accelerometer_x', 'accelerometer_y']

    for col in smoothed_columns:
        data[col + '_smooth'] = savgol_filter(data[col], window_length=5, polyorder=3)

    return data

def process_new_data(data):
    new_data = preprocess_and_smooth_data(data)
    relevant_columns = ['accelerometer_x', 'accelerometer_y', 'accelerometer_z', 'accelerometer_y_smooth', 'accelerometer_x_smooth', 'accelerometer_z_smooth']
    new_data = new_data[relevant_columns]
    return new_data

def create_map_with_path_and_markers(sample_data, images_folder):
    if sample_data.empty:
        raise ValueError("The sample_data DataFrame is empty. Cannot create a map.")

    map_display = folium.Map(location=[sample_data['location_latitude'].mean(), sample_data['location_longitude'].mean()], zoom_start=15)
    path_coords = []

    for idx, row in sample_data.iterrows():
        path_coords.append([row['location_latitude'], row['location_longitude']])
        
        if row['label'] == 0:
            image_url = url_for('serve_image', filename=f"{int(row['seconds_elapsed'])}.png", _external=True)
            # Assuming the images are square and of fixed size, you can adjust width and height as needed
            html = f"""<html><body><img src="{image_url}" style="width: 300px; height: auto;" /></body></html>"""
            iframe = folium.IFrame(html, width=320, height=320)  # Adjust width and height if needed
            popup = folium.Popup(iframe, max_width=320)  # Ensure max_width matches iframe width

            folium.CircleMarker(
                location=[row['location_latitude'], row['location_longitude']],
                radius=5,
                color='red',
                fill=True,
                fill_color='red',
                popup=popup
            ).add_to(map_display)

    folium.PolyLine(path_coords, color="blue", weight=2.5, opacity=1).add_to(map_display)

    return map_display


def plot_graphs(sample_data):
    # Create a line chart for accelerometer data using Plotly
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=sample_data['seconds_elapsed'], y=sample_data['accelerometer_x_smooth'], mode='lines', name='Accelerometer X', line=dict(color='royalblue')))
    fig1.add_trace(go.Scatter(x=sample_data['seconds_elapsed'], y=sample_data['accelerometer_y_smooth'], mode='lines', name='Accelerometer Y', line=dict(color='orange')))
    fig1.add_trace(go.Scatter(x=sample_data['seconds_elapsed'], y=sample_data['accelerometer_z_smooth'], mode='lines', name='Accelerometer Z', line=dict(color='green')))
    fig1.update_layout(title='Smoothed Accelerometer Data Over Time', xaxis_title='Time (seconds)', yaxis_title='Acceleration')

    line_chart_url = pio.to_json(fig1)

    # Create a heatmap to show correlations between accelerometer axes
    corr = sample_data[['accelerometer_x', 'accelerometer_y', 'accelerometer_z']].corr()
    fig2 = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='Viridis'))
    fig2.update_layout(title='Correlation between Accelerometer Axes')

    heatmap_url = pio.to_json(fig2)

    # Create a boxplot to show the distribution of accelerometer data
    fig3 = go.Figure()
    fig3.add_trace(go.Box(y=sample_data['accelerometer_x_smooth'], name='Accelerometer X', boxmean='sd'))
    fig3.add_trace(go.Box(y=sample_data['accelerometer_y_smooth'], name='Accelerometer Y', boxmean='sd'))
    fig3.add_trace(go.Box(y=sample_data['accelerometer_z_smooth'], name='Accelerometer Z', boxmean='sd'))
    fig3.update_layout(title='Boxplot of Smoothed Accelerometer Data', xaxis_title='Axis', yaxis_title='Smoothed Acceleration')

    boxplot_url = pio.to_json(fig3)

    # Create a pie chart for label distribution
    label_counts = sample_data['label'].value_counts().sort_index()
    
    # Ensure there are at least two labels for a valid pie chart
    if len(label_counts) < 2:
        label_counts = label_counts._append(pd.Series([0], index=[1], name=''))
        labels = [f'Label {i}' for i in range(len(label_counts))]
    else:
        labels = [f'Label {i}' for i in range(len(label_counts))]
    
    fig4 = px.pie(values=label_counts.values, names=labels, title='Distribution of Labels')
    pie_chart_url = pio.to_json(fig4)

    return line_chart_url, heatmap_url, boxplot_url, pie_chart_url


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files or 'model' not in request.form:
            return redirect(request.url)
        file = request.files['file']
        model_name = request.form['model']
        
        if file.filename == '':
            return redirect(request.url)

        if model_name not in models:
            return redirect(request.url)
        
        if file:
            data = pd.read_csv(file)
            sample_data = preprocess_and_smooth_data(data)
            new_data = process_new_data(sample_data)
            
            scaler = StandardScaler()
            new_data_scaled = scaler.fit_transform(new_data)
            
            selected_model = models[model_name]
            
            if model_name == 'nn_model':
                predictions = np.argmax(selected_model.predict(new_data_scaled), axis=1)
            else:
                predictions = selected_model.predict(new_data_scaled)
            
            sample_data['label'] = predictions

            # Generate the map
            map_display = create_map_with_path_and_markers(sample_data, image_folder)
            map_html = map_display._repr_html_()  # Convert the map to HTML

            # Generate graphs
            line_chart_url, heatmap_url, boxplot_url, pie_chart_url = plot_graphs(sample_data)

            return render_template(
                'results.html',
                line_chart_url=line_chart_url,
                heatmap_url=heatmap_url,
                boxplot_url=boxplot_url,
                pie_chart_url=pie_chart_url,
                map_html=map_html,
                time_labels=sample_data['seconds_elapsed'].tolist(),
                accelerometer_z=sample_data['accelerometer_z'].tolist(),
                distribution_data=sample_data['label'].value_counts().sort_index().tolist()
            )

    return render_template('index.html')

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(image_folder, filename)

if __name__ == "__main__":
    app.run(debug=True)
