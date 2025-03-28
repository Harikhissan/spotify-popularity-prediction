import streamlit as st
import pandas as pd
import pickle
import random
import time

# Load the model, mappings, and features
model = pickle.load(open("random_forest_model.pkl", "rb"))
mappings = pickle.load(open("mappings.pkl", "rb"))
training_features = pickle.load(open("training_features.pkl", "rb"))
original_df = pd.read_csv("original_data.csv")  # Load the original dataset
selected_songs = pd.read_csv("selected_songs.csv")  # Load selected songs

# UI Header
st.title("Spotify Music Recommendation Bot")
st.subheader("Select a song or get a random one!")

# --- Song Selection UI ---
song_option = st.radio("Choose a song selection method:", ["Select a song", "Get a random song"])

if song_option == "Select a song":
    selected_song = st.selectbox(
        "Select a song:",
        selected_songs["track_name"].unique(),  # Display track_name
        key="song_select",
    )
    # Get selected song data
    selected_song_data = selected_songs[selected_songs["track_name"] == selected_song].iloc[0]
    
elif song_option == "Get a random song":
    random_song_data = original_df.sample(n=1).iloc[0]  # Pick a random song from original data
    selected_song_data = random_song_data
    selected_song = selected_song_data["track_name"]

# Display song and artist name
st.markdown(f"**Song:** {selected_song}  \n**Artist:** {selected_song_data['track_artist']}")

# --- Data Preprocessing ---
def preprocess_input_for_streamlit(input_data):
    input_data = pd.DataFrame([input_data])  # Convert to DataFrame

    # 1. One-Hot Encoding for 'playlist_genre'
    input_data = pd.get_dummies(input_data, columns=["playlist_genre"], prefix=["genre"])

    # 2. Target Encoding for 'track_artist', 'track_album_name', 'track_name'
    for col in ["track_artist", "track_album_name", "track_name"]:
        input_data[col] = input_data[col].map(mappings.get(f"{col}_mapping", {})).fillna(0)

    # 3. Frequency Encoding for 'track_album_release_date'
    fe = original_df.groupby("track_album_release_date").size() / len(original_df)
    input_data["track_album_release_date_freq"] = input_data["track_album_release_date"].map(fe).fillna(0)

    # Convert 'track_album_release_date' to numerical features
    input_data["track_album_release_date"] = pd.to_datetime(
        input_data["track_album_release_date"], errors="coerce"
    )
    input_data["release_year"] = input_data["track_album_release_date"].dt.year
    input_data["release_month"] = input_data["track_album_release_date"].dt.month
    input_data["release_day"] = input_data["track_album_release_date"].dt.day
    input_data = input_data.drop(columns=["track_album_release_date"])

    # 4. Label Encoding for 'playlist_subgenre'
    input_data["playlist_subgenre"] = input_data["playlist_subgenre"].map(mappings.get("playlist_subgenre_mapping", {})).fillna(0)

    # 5. One-Hot Encoding for 'playlist_name'
    input_data = pd.get_dummies(input_data, columns=["playlist_name"], prefix=["playlist"])

    # Reorder or select columns in input_data to match training data
    input_data = input_data.reindex(columns=training_features, fill_value=0)

    return input_data

# Prediction Button
if st.button("🎵 Predict Song Recommendation"):
    with st.spinner("🔍 Analyzing song features..."):
        time.sleep(2)  # Simulate loading time

        # Preprocess the song data
        preprocessed_song_data = preprocess_input_for_streamlit(selected_song_data)

        # Make a prediction
        prediction = model.predict(preprocessed_song_data)[0]

    # Interpret Prediction
    popularity = "**Popular**" if prediction == 1 else "**Unpopular**"

    if popularity == "**Popular**":
        st.success(f"**{selected_song}** is popular!")
    else:
        st.error(f"**{selected_song}** is unpopular!")
