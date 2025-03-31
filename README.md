# spotify-popularity-prediction
How to Run the Code
Prerequisites

Python 3.8 or later
pip package manager
Spotify Developer account (for API access)

Setup

Clone the repository
bashCopygit clone https://github.com/yourusername/spotify-popularity-prediction.git
cd spotify-popularity-prediction

Create and activate a virtual environment (optional but recommended)
bashCopypython -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

Install dependencies
bashCopypip install -r requirements.txt

Configure Spotify API credentials

Create a .env file in the project root with your Spotify API credentials:
CopySPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here




Data Collection

Fetch data from Spotify API
bashCopypython src/data_collection.py
This will save raw data to data/raw/spotify_tracks.csv

Data Processing

Process and prepare the data
bashCopypython src/data_processing.py
This will create preprocessed data in data/processed/processed_tracks.csv

Model Training

Train the Random Forest model
bashCopypython src/train_model.py
This will save the trained model to models/random_forest_model.pkl

Evaluation

Evaluate model performance
bashCopypython src/evaluate_model.py
This will generate performance metrics and save them to results/metrics.json

Prediction

Make predictions with new data
bashCopypython src/predict.py --input path/to/your/songs.csv
This will output predictions to results/predictions.csv

Visualization

Generate visualizations
bashCopypython src/visualize.py
This will create visualizations in the visualization/ directory

Web Application (Optional)

Run the web application
bashCopypython app.py
This will start a Flask web application at http://localhost:5000 where you can upload songs and get popularity predictions in real-time.

Troubleshooting

If you encounter issues with the Spotify API rate limits, add a delay between requests in src/data_collection.py
For memory issues during model training, adjust the n_jobs parameter in src/train_model.py
