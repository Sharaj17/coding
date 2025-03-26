# Packages

import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ipywidgets as widgets
from IPython.display import display
from flask import Flask, render_template, request, jsonify


app = Flask(__name__)

# Loading the datasets

movies = pd.read_csv(r"D:\Data Analytics Materials\DataQuest\ml-25m\ml-25m\movies.csv")
ratings = pd.read_csv(r"D:\Data Analytics Materials\DataQuest\ml-25m\ml-25m\ratings.csv")
ratings

# Preprocessing step

def cleaning_title(title):
    return re.sub("[^a-zA-Z0-9 ]","",title)
movies["clean_title"] = movies["title"].apply(cleaning_title)
movies.head()

# Creating TFIDF matrix

vectorizer = TfidfVectorizer(ngram_range = (1,2))
tfidf = vectorizer.fit_transform(movies["clean_title"])

# Computing cosine similarity
def search_title(title):
    title = cleaning_title(title)
    query_vector = vectorizer.transform([title])
    similarity = cosine_similarity(query_vector, tfidf).flatten()
    indices = np.argsort(similarity)[::-1][:5]
    results = movies.iloc[indices]
    return results

# Creating an interactive widget

movie_input = widgets.Text(value = "", description = "Movie Title:", disabled = False)

movie_list = widgets.Output()

def on_type(data):
    with movie_list:
        movie_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            display(search_title(title))
movie_input.observe(on_type, names="value")
display(movie_input, movie_list)

# Creating a function to find similar movies

def find_similar_movies(movie_id):
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
    similar_user_rec = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
    
    similar_user_rec = similar_user_rec.value_counts() / len(similar_users)
    similar_user_rec = similar_user_rec[similar_user_rec > 0.1]
    
    all_users = ratings[(ratings["movieId"].isin(similar_user_rec.index)) & (ratings["rating"] > 4)]
    all_users_rec = all_users["movieId"].value_counts() / len(all_users["userId"].unique())

    rec_percentages = pd.concat([similar_user_rec, all_users_rec], axis = 1)
    rec_percentages.columns = ["similar", "all"]
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending = False)
    return rec_percentages.head(10).merge(movies, left_index = True, right_on = "movieId")[["score","title","genres"]]

# Display the Recommendations

movie_input_name = widgets.Text(value = "", description = "Movie Title: ", disabled = False)
recommendation_list = widgets.Output()

def on_type_rec(data):
    with recommendation_list:
        recommendation_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            result = search_title(title)
            movie_id = result.iloc[0]["movieId"]
            display(find_similar_movies(movie_id))
            
movie_input_name.observe(on_type_rec, names="value")
display(movie_input_name, recommendation_list)

@app.route('/', methods=['GET', 'POST'])
def index():
    search_results = []
    recommendations = []

    if request.method == 'POST':
        title = request.form['title']
        results = search_title(title)
        search_results = results.to_dict(orient='records')

        if len(search_results) > 0:
            movie_id = search_results[0]["movieId"]
            recommendations = find_similar_movies(movie_id).to_dict(orient='records')

    return render_template('index.html', search_results=search_results, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)