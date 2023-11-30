import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler
import pickle

def recommend_hospitals(data, disease_name, filter_by):
    disease_data = data[data['Disease'] == disease_name]
    
    if disease_data.empty:
        return "No hospitals found for the given disease."

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(disease_data['Disease'].fillna(''))

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    results = pd.DataFrame()  # Define results DataFrame outside of conditions

    if not disease_data.empty:
        idx = disease_data.index[0]
        sim_scores = list(enumerate(cosine_sim[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1] > 0, reverse=True)

        hospital_indices = [i[0] for i in sim_scores]

        if filter_by == 'Rating':
            # Filter based on rating above 4.5
            results = disease_data.iloc[hospital_indices].loc[disease_data['Rating'] > 4.5]
            results = results.sort_values('Rating', ascending=False)
        elif filter_by == 'Cost':
            # Sort based on price in ascending order to get lowest cost first
            results = disease_data.iloc[hospital_indices].sort_values('Price')
        
    return results.head(5)

# Save the function to a pickle file
with open('Cost_rating.pkl', 'wb') as f:
    pickle.dump(recommend_hospitals, f)
