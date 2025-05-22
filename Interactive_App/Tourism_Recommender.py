import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

class TourismRecommender:
    def __init__(self):
        self.user_item_matrix = None
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        self.attraction_features = None
        self.content_similarity_matrix = None
        self.user_profiles = None
        self.attraction_profiles = None

    def preprocess_data(self, df):
        """
        Preprocess the tourism dataset for recommendation system
        """     
        # Create user-item matrix (users as rows, attractions as columns, ratings as values)
        self.user_item_matrix = df.pivot_table(
            index='UserId', 
            columns='AttractionId', 
            values='Rating',
            fill_value=0
        )
        
        # Create attraction feature dataframe
        attraction_data = df.drop_duplicates('AttractionId')[
            ['AttractionId', 'AttractionType', 'Continent', 'Region', 'Country']
        ]
        
        # Process attraction features for content-based filtering
        self.attraction_features = pd.get_dummies(
            attraction_data, 
            columns=['AttractionType', 'Continent', 'Region', 'Country'],
            drop_first=False
        )
        self.attraction_features.set_index('AttractionId', inplace=True)
        
        # Calculate visit counts and average ratings for each attraction
        attraction_stats = df.groupby('AttractionId').agg({
            'UserId': 'count',
            'Rating': 'mean'
        }).rename(columns={'UserId': 'visit_count'})
        
        # Merge stats into features
        self.attraction_features = self.attraction_features.join(attraction_stats)
        
        # Fill missing values
        self.attraction_features.fillna(0, inplace=True)
    
        return self
    
    def build_collaborative_model(self):
        """
        Build a collaborative filtering model based on user-user similarity
        """
            
        # Calculate user similarity matrix
        self.user_similarity_matrix = cosine_similarity(self.user_item_matrix)
        
        # Convert to DataFrame for easier indexing
        self.user_similarity_matrix = pd.DataFrame(
            self.user_similarity_matrix,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        
        # Calculate item similarity matrix
        self.item_similarity_matrix = cosine_similarity(self.user_item_matrix.T)
        
        # Convert to DataFrame for easier indexing
        self.item_similarity_matrix = pd.DataFrame(
            self.item_similarity_matrix,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )
        
        return self
    
    def build_content_model(self):
        """
        Build a content-based filtering model based on attraction features
        """
        
        # Scale numerical features
        scaler = MinMaxScaler()
        numerical_cols = ['visit_count', 'Rating']
        self.attraction_features[numerical_cols] = scaler.fit_transform(self.attraction_features[numerical_cols])
        
        # Calculate content similarity matrix
        self.content_similarity_matrix = cosine_similarity(self.attraction_features)
        
        # Convert to DataFrame for easier indexing
        self.content_similarity_matrix = pd.DataFrame(
            self.content_similarity_matrix,
            index=self.attraction_features.index,
            columns=self.attraction_features.index
        )
        
        return self
    
    def build_user_profiles(self, df):
        """
        Build user profiles based on their rating history and demographics
        """
                
        # Group by user to get their demographic information
        user_data = df.groupby('UserId').first()
        
        # Get user preferences for attraction types
        user_preferences = df.groupby(['UserId', 'AttractionType'])['Rating'].mean().unstack(fill_value=0)
        
        # Get user preferences for regions
        user_regions = df.groupby(['UserId', 'Region'])['Rating'].mean().unstack(fill_value=0)
        
        # Combine into user profiles
        self.user_profiles = user_data[['Continent', 'Country', 'user_avg_rating_before']]
        self.user_profiles = self.user_profiles.join(user_preferences, how='left')
        self.user_profiles = self.user_profiles.join(user_regions, how='left')
        
        # Fill missing values
        self.user_profiles.fillna(0, inplace=True)
       
        return self
    
    def recommend_collaborative(self, user_id, n_recommendations=5, n_neighbors=10):
        """
        Generate recommendations using user-based collaborative filtering
        """
        # # Check if user exists
        # if user_id not in self.user_similarity_matrix.index:
        #     print(f"User {user_id} not found in training data")
        #     return pd.DataFrame()
        
        # Get user's rated items
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index.tolist()
        
        # Find similar users
        similar_users = self.user_similarity_matrix[user_id].sort_values(ascending=False)
        similar_users = similar_users.drop(user_id)  # Remove the user itself
        similar_users = similar_users.head(n_neighbors)
        
        # Get recommendations from similar users
        recommendations = pd.DataFrame(columns=['pred_rating', 'count'])  # Initialize with columns
        
        for similar_user, similarity in similar_users.items():
            # Get items rated by the similar user
            similar_user_ratings = self.user_item_matrix.loc[similar_user]
            similar_user_rated = similar_user_ratings[similar_user_ratings > 0]
            
            # Filter out items already rated by the target user
            new_attractions = similar_user_rated.drop(rated_items, errors='ignore')
            
            # Weight ratings by similarity
            weighted_ratings = new_attractions * similarity
            
            # Add to recommendations
            for item, rating in weighted_ratings.items():
                if item not in recommendations.index:
                    recommendations.at[item, 'pred_rating'] = rating
                    recommendations.at[item, 'count'] = 1
                else:
                    recommendations.at[item, 'pred_rating'] += rating
                    recommendations.at[item, 'count'] += 1
        
        # Check if empty before calculating final_rating
        if recommendations.empty:
            return pd.DataFrame()
        
        # Calculate average predicted rating
        recommendations['final_rating'] = recommendations['pred_rating'] / recommendations['count']
        
        # Sort and return top N recommendations
        return recommendations.sort_values('final_rating', ascending=False).head(n_recommendations)
    
    def recommend_content_based(self, user_id, n_recommendations=5):
        """
        Generate recommendations using content-based filtering
        """
        # Check if user exists
        # if user_id not in self.user_item_matrix.index:
        #     print(f"User {user_id} not found in training data")
        #     return pd.DataFrame()
        
        # Get user's rated items
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index.tolist()
        
        # Calculate weighted average content similarity
        recommendations = pd.DataFrame()
        
        for item_id in self.attraction_features.index:
            # Skip items already rated by the user
            if item_id in rated_items:
                continue
            
            weighted_sum = 0
            similarity_sum = 0
            
            for rated_item in rated_items:
                rating = user_ratings[rated_item]
                if rated_item in self.content_similarity_matrix.index:
                    similarity = self.content_similarity_matrix.loc[item_id, rated_item]
                    weighted_sum += similarity * (rating - 3)
                    similarity_sum += abs(similarity)
            
            if similarity_sum > 0:
                # Calculate predicted rating (adjust back to 1-5 scale)
                recommendations.at[item_id, 'pred_rating'] = 3 + (weighted_sum / similarity_sum)
        
        # Sort and return top N recommendations
        return recommendations.sort_values('pred_rating', ascending=False).head(n_recommendations)
    
    def recommend_hybrid(self, user_id, n_recommendations=5, collab_weight=0.7):
        """
        Generate recommendations using a hybrid approach (collaborative + content)
        """
        # Get collaborative filtering recommendations
        cf_recs = self.recommend_collaborative(user_id, n_recommendations=10)
        
        # Get content-based recommendations
        cb_recs = self.recommend_content_based(user_id, n_recommendations=10)
        
        # Combine recommendations with weights
        if cf_recs.empty and cb_recs.empty:
            return pd.DataFrame(columns=['hybrid_rating'])
        elif cf_recs.empty:
            cb_recs = cb_recs.copy()
            cb_recs['hybrid_rating'] = cb_recs['pred_rating']
            return cb_recs[['hybrid_rating']].head(n_recommendations)
        elif cb_recs.empty:
            cf_recs = cf_recs.copy()
            cf_recs['hybrid_rating'] = cf_recs['final_rating']
            return cf_recs[['hybrid_rating']].head(n_recommendations)
        
        # Normalize ratings to 0-1 scale for fair combination
        cf_min = cf_recs['final_rating'].min()
        cf_max = cf_recs['final_rating'].max()
        if cf_max > cf_min:
            cf_recs['norm_rating'] = (cf_recs['final_rating'] - cf_min) / (cf_max - cf_min)
        else:
            cf_recs['norm_rating'] = 0.5
        
        cb_min = cb_recs['pred_rating'].min()
        cb_max = cb_recs['pred_rating'].max()
        if cb_max > cb_min:
            cb_recs['norm_rating'] = (cb_recs['pred_rating'] - cb_min) / (cb_max - cb_min)
        else:
            cb_recs['norm_rating'] = 0.5
        
        # Combine recommendations
        hybrid_recs = pd.DataFrame()
        
        # Add collaborative filtering recommendations
        for item_id, row in cf_recs.iterrows():
            hybrid_recs.at[item_id, 'cf_rating'] = row['norm_rating']
            hybrid_recs.at[item_id, 'cf_weight'] = collab_weight
        
        # Add content-based recommendations
        for item_id, row in cb_recs.iterrows():
            if item_id in hybrid_recs.index:
                hybrid_recs.at[item_id, 'cb_rating'] = row['norm_rating']
                hybrid_recs.at[item_id, 'cb_weight'] = 1 - collab_weight
            else:
                hybrid_recs.at[item_id, 'cf_rating'] = 0
                hybrid_recs.at[item_id, 'cf_weight'] = collab_weight
                hybrid_recs.at[item_id, 'cb_rating'] = row['norm_rating']
                hybrid_recs.at[item_id, 'cb_weight'] = 1 - collab_weight
        
        # Fill missing values
        hybrid_recs.fillna(0, inplace=True)
        
        # Calculate weighted ratings
        hybrid_recs['hybrid_rating'] = (
            (hybrid_recs['cf_rating'] * hybrid_recs['cf_weight']) + 
            (hybrid_recs['cb_rating'] * hybrid_recs['cb_weight'])
        )
        
        # Sort and return top N recommendations
        return hybrid_recs.sort_values('hybrid_rating', ascending=False).head(n_recommendations)
    
    def get_attraction_details(self, attraction_ids, attraction_data):
        """
        Get details for recommended attractions
        """
        # Filter attraction data for the recommended IDs
        attraction_details = attraction_data[attraction_data['AttractionId'].isin(attraction_ids)]
        
        # Return relevant columns
        return attraction_details[['AttractionId', 'Attraction', 'AttractionType', 'Country', 'Region', 'Rating']]
    
    def save_model(self, filepath):
        """
        Save the trained recommendation model
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        # print(f"Model saved to {filepath}")

    def load_model(cls, filepath):
        """
        Load a trained recommendation model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        # print(f"Model loaded from {filepath}")
        return model

# Set page configuration
st.set_page_config(
    page_title="Tourism Recommendation System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models
@st.cache_resource
def load_models():
    try:
        # Use the correct file paths
        rating_model_path = "C:/Users/Saima Modak/Capstone Projects/Tourism Analysis/Models/Rating Predictor.pkl"
        visit_mode_model_path = "C:/Users/Saima Modak/Capstone Projects/Tourism Analysis/Models/Visit Mode Predictor.pkl"
        recommendation_model_path = "C:/Users/Saima Modak/Capstone Projects/Tourism Analysis/Models/Recommendation Model.pkl"
        
        with open(rating_model_path, "rb") as f:
            rating_model = pickle.load(f)
        
        with open(visit_mode_model_path, "rb") as f:
            visit_mode_model = pickle.load(f)
        
        with open(recommendation_model_path, "rb") as f:
            recommendation_model = pickle.load(f)
        
        # Load dataset
        df = pd.read_excel("C:/Users/Saima Modak/Capstone Projects/Tourism Analysis/Datasets/Final Dataset.xlsx")
        
        return rating_model, visit_mode_model, recommendation_model, df
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

# Load models and data
rating_model, visit_mode_model, recommendation_model, df = load_models()

# Extract probability dictionaries from visit_mode_model
month_mode_probs = visit_mode_model['month_mode_probs']
continent_mode_probs = visit_mode_model['continent_mode_probs']

# Function to predict visit mode
def predict_visit_mode(input_data, model_data):
    model = model_data['model']
    scaler = model_data['scaler']
    encoders = model_data['encoders']
    features = model_data['features']
    target_encoder = model_data['target_encoder']
    features_final = model_data.get('features_final', model.feature_names_in_ if hasattr(model, 'feature_names_in_') else features)

    # Prepare DataFrame for prediction
    X = pd.DataFrame([input_data])

    # Only keep columns that the model expects
    X = X[[f for f in features if f in X.columns]]

    # Identify categorical features (those present in encoders)
    categorical_features = [f for f in X.columns if f in encoders]
    numerical_features = [f for f in X.columns if f not in categorical_features]

    # Encode categorical features
    for feature in categorical_features:
        encoder = encoders[feature]
        X[feature] = encoder.transform(X[feature])

    # Scale numerical features
    if numerical_features and scaler:
        X[numerical_features] = scaler.transform(X[numerical_features])

    X = X.reindex(columns=features_final, fill_value=0)

    # Make prediction
    prediction_encoded = model.predict(X)
    probabilities = model.predict_proba(X)[0]

    # Decode prediction
    if target_encoder:
        prediction = target_encoder.inverse_transform([[prediction_encoded[0]]])[0][0]
    else:
        prediction = prediction_encoded[0]

    # Get class names and probabilities
    classes = model.classes_
    prob_dict = dict(zip(classes, probabilities))

    return prediction, prob_dict

# Function to predict rating

def predict_rating(input_data, model_data):
    model = model_data['model']
    scaler = model_data['scaler']
    encoders = model_data['encoders']
    features = model_data['features']
    X = pd.DataFrame([input_data])

    # Encode categorical features using saved LabelEncoders
    for feature in encoders:
        if feature in X.columns:
            try:
                encoder = encoders[feature]
                X[feature] = encoder.transform(X[feature].astype(str))
            except Exception as e:
                st.error(f"Error encoding feature '{feature}': {e}")
                st.stop()

    # Scale numerical features
    numerical_features = [f for f in features if f not in encoders]
    if numerical_features and scaler:
        try:
            X[numerical_features] = scaler.transform(X[numerical_features])
        except Exception as e:
            st.error(f"Error during scaling: {e}")
            st.stop()
            
    # Force columns to match model's feature_names_in_ if available
    if hasattr(model, "feature_names_in_"):
        X = X.reindex(columns=list(model.feature_names_in_), fill_value=0)
    else:
        X = X.reindex(columns=features, fill_value=0)
    X = X.astype(np.float32)
    X = X.reset_index(drop=True)

    try:
        prediction = model.predict(X)[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.stop()

    return prediction

# Main app
def main():
    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page", ["Home", "Visit Mode Prediction", "Rating Prediction", "Recommendations", "Data Insights"])
    
    # Home page
    if page == "Home":
        st.title("Tourism Recommendation System")
        st.write("Welcome to the Tourism Recommendation System. This application provides:")
        st.write("1. Visit Mode Prediction - Predict how users will visit attractions")
        st.write("2. Rating Prediction - Predict ratings for attractions")
        st.write("3. Personalized Recommendations - Get attraction recommendations")
        st.write("4. Data Insights - Explore tourism patterns")
    
    # Visit Mode Prediction
    elif page == "Visit Mode Prediction":
        st.title("Visit Mode Prediction")
        st.write("Predict how a user is likely to visit an attraction")
        
        # Create form for input using the actual features
        with st.form("visit_mode_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Categorical features
                encoders = visit_mode_model['encoders']
                continent = st.selectbox("Continent", encoders['Continent'].classes_)
                region = st.selectbox("Region", encoders['Region'].classes_)
                country = st.selectbox("Country", encoders['Country'].classes_)
                city = st.selectbox("City Name", encoders['CityName'].classes_)
                attraction_type = st.selectbox("Attraction Type", encoders['AttractionType'].classes_)
                prev_visit_mode = st.selectbox("Previous Visit Mode", encoders['prev_visit_mode'].classes_)
                            
            with col2:
                # Numerical features
                visit_month = st.slider("Visit Month", 1, 12, 6)
                visit_quarter = (visit_month - 1) // 3 + 1  # Calculate quarter
                visit_year = st.slider("Visit Year", 2022, 2025, 2023)
                
                # Sinusoidal transformations for cyclical month feature
                sin_month = np.sin(2 * np.pi * visit_month / 12)
                cos_month = np.cos(2 * np.pi * visit_month / 12)
                
                # User metrics
                user_pct_couples = st.slider("% of Couples Visits", 0.0, 1.0, 0.25)
                user_pct_family = st.slider("% of Family Visits", 0.0, 1.0, 0.25)
                user_pct_friends = st.slider("% of Friends Visits", 0.0, 1.0, 0.25)
                user_pct_business = st.slider("% of Business Visits", 0.0, 1.0, 0.25)
                user_travel_diversity = st.slider("User Travel Diversity", 0.0, 1.0, 0.5)
                
                # Attraction and user metrics
                attraction_avg_rating = st.slider("Attraction Avg Rating", 1.0, 5.0, 4.0)
                user_previous_visits = st.slider("User Previous Visits", 0, 20, 5)
                city_popularity = st.slider("City Popularity", 0.0, 1.0, 0.7)
                user_avg_rating = st.slider("User Avg Rating", 1.0, 5.0, 4.0)
                user_attraction_compatibility = st.slider("User-Attraction Compatibility", 0.0, 1.0, 0.7)
            
            submit_button = st.form_submit_button("Predict Visit Mode")
        
        if submit_button and visit_mode_model is not None:
            # Prepare input data based on model features
            input_data = {
                'VisitMonth': visit_month,
                'VisitQuarter': visit_quarter,
                'VisitYear': visit_year,
                'Continent': continent,
                'Region': region,
                'Country': country,
                'CityName': city,
                'AttractionType': attraction_type,
                'prev_visit_mode': prev_visit_mode,
                'user_pct_Couples': user_pct_couples,
                'user_pct_Family': user_pct_family,
                'user_pct_Friends': user_pct_friends,
                'user_pct_Business': user_pct_business,
                'user_travel_diversity': user_travel_diversity,
                'attraction_avg_rating_before': attraction_avg_rating,
                'user_previous_visits': user_previous_visits,
                'city_popularity': city_popularity,
                'user_avg_rating_before': user_avg_rating,
                'user_attraction_compatibility': user_attraction_compatibility,
                'sin_month': sin_month,
                'cos_month': cos_month,
                'month_mode_prob_Business': month_mode_probs['Business'].get(visit_month, 0),
                'month_mode_prob_Family': month_mode_probs['Family'].get(visit_month, 0),
                'month_mode_prob_Friends': month_mode_probs['Friends'].get(visit_month, 0),
                'month_mode_prob_Couples': month_mode_probs['Couples'].get(visit_month, 0),
                'continent_mode_prob_Business': continent_mode_probs['Business'].get(continent, 0),
                'continent_mode_prob_Couples': continent_mode_probs['Couples'].get(continent, 0)
            }
            
            try:
                # Make prediction
                prediction, probabilities = predict_visit_mode(input_data, visit_mode_model)
                
                # Display results
                mode_map = {'F': 'Family', 'Fr': 'Friends', 'B': 'Business', 'C': 'Couples', 'S' : 'Solo'}
                prediction_full = mode_map.get(prediction, prediction)
                st.success(f"Predicted Visit Mode: **{prediction_full}**")
                
                # Show probability chart
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Sort by probability for better visualization
                sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                modes = [item[0] for item in sorted_items]
                probs = [item[1] for item in sorted_items]
                
                ax.bar(modes, probs, color='skyblue')
                ax.set_ylabel("Probability")
                ax.set_title("Visit Mode Probabilities")
                ax.set_ylim(0, 1)
                
                # Add probability values on top of bars
                for i, v in enumerate(probs):
                    ax.text(i, v + 0.02, f'{v:.2f}', ha='center')
                    
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.write("Please check if all required features are available and properly formatted.")
        elif submit_button:
            st.error("Model not loaded. Please check the file paths.")
    
    # Rating Prediction
    elif page == "Rating Prediction":
        st.title("Rating Prediction")
        st.write("Predict the rating a user might give to an attraction")
        
        # Create form for input using the actual features from model
        with st.form("rating_form"):
            col1, col2 = st.columns(2)

            with col1:
                # User features
                user_rating_trend = st.slider("User Rating Trend", -1.0, 1.0, 0.0, 
                                            help="Trend in user's recent ratings (-1: decreasing, 0: stable, 1: increasing)")
                user_avg_rating_before = st.slider("User's Average Rating", 1.0, 5.0, 4.0)
                user_previous_visits = st.slider("User's Previous Visits", 0, 20, 5)

                # Visit details
                visit_month = st.slider("Visit Month", 1, 12, datetime.now().month)
                visit_quarter = (visit_month - 1) // 3 + 1  # Calculate quarter
                country = st.selectbox("Country", df['Country'].unique() if df is not None else ["United States", "France", "Japan"])
                visit_mode = st.selectbox("Visit Mode", df['VisitMode'].unique() if df is not None else ["Family", "Business", "Couples", "Friends"])
                visit_season = st.selectbox("Visit Season", df['VisitSeason'].unique() if df is not None else ["Summer", "Winter", "Spring", "Autumn"])
                continent = st.selectbox("Continent", df['Continent'].unique() if df is not None else ["Asia", "Europe", "Africa", "America"])
                region = st.selectbox("Region", df['Region'].unique() if df is not None else ["East", "West", "North", "South"])

            with col2:
                # Attraction features
                attraction_avg_rating_before = st.slider("Attraction's Average Rating", 1.0, 5.0, 4.0)
                attraction_previous_visits = st.slider("Attraction's Previous Visits", 0, 1000, 500)
                attraction_previous_visitors = st.slider("Attraction's Previous Visitors", 0, 10000, 500)
                city_popularity = st.slider("City Popularity", 0.0, 1.0, 0.7)
                attraction_type = st.selectbox("Attraction Type", df['AttractionType'].unique() if df is not None else ["Beach", "Museum", "Historical"])

            submit_button = st.form_submit_button("Predict Rating")
        
        if submit_button and rating_model is not None:
            # Build input_data in the exact order as model expects
            feature_values = {
                'user_rating_trend': user_rating_trend,
                'user_avg_rating_before': user_avg_rating_before,
                'attraction_avg_rating_before': attraction_avg_rating_before,
                'attraction_previous_visits': attraction_previous_visits,
                'user_previous_visits': user_previous_visits,
                'city_popularity': city_popularity,
                'VisitMonth': visit_month,
                'Country': country,
                'VisitMode': visit_mode,
                'AttractionType': attraction_type,
                'Region': region,
                'attraction_previous_visitors': attraction_previous_visitors,
                'VisitSeason': visit_season,
                'Continent': continent,
                'VisitQuarter': visit_quarter
            }

            # Ensure input_data order matches model's features
            input_data = {k: feature_values[k] for k in rating_model['features']}

            try:
                
                # Make prediction
                predicted_rating = predict_rating(input_data, rating_model)
                
                # Display results
                st.success(f"Predicted Rating: **{predicted_rating:.2f}** / 5.0")
                
                # Simple visualization of the predicted rating
                fig, ax = plt.subplots(figsize=(10, 2))
                ax.set_xlim(1, 5)
                ax.set_ylim(0, 1)
                ax.axvline(x=predicted_rating, color='red', linestyle='-')
                ax.set_title(f"Predicted Rating: {predicted_rating:.2f}")
                ax.set_xlabel("Rating Scale")
                for i in range(1, 6):
                    ax.text(i, 0.5, str(i), ha='center')
                st.pyplot(fig)
                
                # Interpretation
                if predicted_rating >= 4.5:
                    st.write("This is an excellent rating! The user is likely to be very satisfied.")
                elif predicted_rating >= 4.0:
                    st.write("This is a very good rating. The user is likely to be satisfied.")
                elif predicted_rating >= 3.5:
                    st.write("This is a good rating. The user will likely find this acceptable.")
                elif predicted_rating >= 3.0:
                    st.write("This is an average rating. The user may have mixed feelings.")
                else:
                    st.write("This is a below-average rating. The user may be unsatisfied.")
                    
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.write("Please check if all required features are available and properly formatted.")
        elif submit_button:
            st.error("Model not loaded. Please check the file paths.")
    
    # Recommendations
    elif page == "Recommendations":
        st.title("Attraction Recommendations")
        st.write("Get personalized recommendations for attractions")
        
        user_id = st.selectbox("Select User ID", options=df['UserId'].unique())
        num_recommendations = st.slider("Number of recommendations", 1, 10, 5)
        
        if st.button("Get Recommendations", key="get_recommendations_main") and recommendation_model is not None:
            if user_id:
                with st.spinner("Generating recommendations..."):
                    try:
                        # Use the hybrid recommendation method
                        recommendations = recommendation_model.recommend_hybrid(user_id, n_recommendations=num_recommendations)
                        
                        if recommendations.empty or 'hybrid_rating' not in recommendations.columns:
                            st.warning(f"No recommendations found for user {user_id}.")
                        
                        elif not recommendations.empty:
                            # Get attraction details
                            attraction_ids = recommendations.index.tolist()
                            attraction_details = recommendation_model.get_attraction_details(attraction_ids, df)
                            
                            # Merge with recommendations
                            result = pd.merge(
                                recommendations.reset_index().rename(columns={'index': 'AttractionId'}),
                                attraction_details,
                                on='AttractionId'
                            )
                            
                            # Format the display table
                            display_df = result[['Attraction', 'AttractionType', 'Country', 'Region', 'Rating', 'hybrid_rating']]
                            display_df.columns = ['Attraction', 'Type', 'Country', 'Region', 'Avg Rating', 'Score']

                            # Only show the number requested
                            display_df = display_df.head(num_recommendations)
                                               
                            # Show as table
                            st.markdown("#### Top Recommendations:")
                            st.dataframe(display_df)

                            # Display results
                            st.success(f"Found {len(display_df)} recommendations for user {user_id}")

                            # Visualization
                            fig, ax = plt.subplots(figsize=(10, 6))
                            bars = ax.barh(display_df['Attraction'], display_df['Score'], color='skyblue')
                            ax.set_xlabel('Recommendation Score')
                            ax.set_title('Top Recommendations')

                            # Add score labels
                            for bar in bars:
                                width = bar.get_width()
                                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}', va='center')

                            st.pyplot(fig)
                            
                        else:
                            st.warning(f"No recommendations found for user {user_id}.")
                    except Exception as e:
                        st.error(f"Error generating recommendations: {e}")
                        st.write("Please check if the user exists in the dataset.")
            else:
                st.warning("Please enter a User ID")
        else:
            if recommendation_model is None:
                st.error("Recommendation model not loaded. Please check the file paths.")

    # Data Insights
    elif page == "Data Insights":
        st.title("Tourism Data Insights")
        st.write("Explore patterns in tourism data")
        
        # Simple statistics
        if df is not None:
            st.subheader("Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Unique Users", df['UserId'].nunique())
            with col3:
                st.metric("Unique Attractions", df['AttractionId'].nunique())
            with col4:
                st.metric("Average Rating", f"{df['Rating'].mean():.2f}/5.0")
            
            # Sample of data
            st.subheader("Sample Data")
            st.dataframe(df.sample(5))
            
            # Visualizations
            st.subheader("Visualizations")
            chart_type = st.selectbox("Select chart", ["Rating Distribution", "Visit Modes", "Top Attractions"])
            
            if chart_type == "Rating Distribution":
                # Bar plot of average rating by visit mode, with value labels
                avg_ratings = df.groupby('VisitMode')['Rating'].mean().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(8, 5))
                bars = ax.bar(avg_ratings.index, avg_ratings.values, color=sns.color_palette("Set2", n_colors=len(avg_ratings)))
                ax.set_title('Average Rating by Visit Mode', fontsize=16, color='#222')
                ax.set_xlabel('Visit Mode', fontsize=12)
                ax.set_ylabel('Average Rating', fontsize=12)
                ax.set_ylim(0, 5)
                for i, v in enumerate(avg_ratings.values):
                    ax.text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=12, fontweight='bold')
                ax.grid(axis='y', linestyle='--', alpha=0.4)
                plt.tight_layout()
                st.pyplot(fig)
            
            elif chart_type == "Visit Modes":
                # Visit mode distribution
                visit_mode_counts = df['VisitMode'].value_counts()
                
                fig, ax = plt.subplots(figsize=(10, 6))
                pastel_colors = sns.color_palette("Accent", n_colors=len(visit_mode_counts))
                visit_mode_counts.plot(
                    kind='pie',
                    autopct='%1.1f%%',
                    ax=ax,
                    colors=pastel_colors,
                    textprops={'fontsize': 12}
                )
                ax.set_title('Visit Mode Distribution')
                ax.set_ylabel('')  # Hide 'None' ylabel
                ax.set_aspect('equal')
                st.pyplot(fig)
            
            elif chart_type == "Top Attractions":
                # Top attractions by visit count
                top_attractions = df.groupby(['AttractionId', 'Attraction']).size().reset_index(name='VisitCount')
                top_attractions = top_attractions.sort_values('VisitCount', ascending=False).head(10)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                palette = sns.color_palette("rocket", n_colors=len(top_attractions))
                bars = ax.barh(top_attractions['Attraction'], top_attractions['VisitCount'], color=palette)
                ax.set_title('Top 10 Most Visited Attractions')
                ax.set_xlabel('Number of Visits')

                # Add count labels
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 5, bar.get_y() + bar.get_height()/2, f'{int(width)}', va='center')

                st.pyplot(fig)
        else:
            st.error("Dataset not loaded. Please check the file path.")

# Run the app
if __name__ == "__main__":
    main()