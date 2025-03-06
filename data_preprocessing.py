import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def load_and_preprocess_data(filepath):
    # Load the cleaned dataset
    df = pd.read_csv(filepath)

    # Drop rows where 'winner' is missing
    df_cleaned = df.dropna(subset=['winner']).copy()

    # Replace old team names for consistency
    old_to_new_teams = {
        "Delhi Daredevils": "Delhi Capitals",
        "Kings XI Punjab": "Punjab Kings",
        "Rising Pune Supergiants": "Rising Pune Supergiant",
        "Deccan Chargers": "Sunrisers Hyderabad",
        "Gujarat Lions": "Gujarat Titans",
        "Pune Warriors India": "Rising Pune Supergiant"
    }
    df_cleaned[['team1', 'team2', 'winner']] = df_cleaned[['team1', 'team2', 'winner']].replace(old_to_new_teams)

    # Ensure data is sorted by match date if the column exists
    if 'match_date' in df_cleaned.columns:
        df_cleaned = df_cleaned.sort_values(by='match_date')

    # Encode the 'winner' column to numeric values for calculating recent form
    winner_encoder = LabelEncoder()
    df_cleaned['winner_encoded'] = winner_encoder.fit_transform(df_cleaned['winner'])

    # Function to calculate last 5 match win rate
    def calculate_recent_form(df, team_col, match_result_col):
        team_form = df.groupby(team_col)[match_result_col].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)
        return team_form

    df_cleaned['team1_recent_form'] = calculate_recent_form(df_cleaned, 'team1', 'winner_encoded')
    df_cleaned['team2_recent_form'] = calculate_recent_form(df_cleaned, 'team2', 'winner_encoded')

    # Example: List of key players for each team
    key_players = {
        "Chennai Super Kings": ["Ruturaj Gaikwad", "Ravindra Jadeja", "MS Dhoni", "Sam Curran", "Devon Conway"],
        "Delhi Capitals": ["KL Rahul", "Axar Patel", "Kuldeep Yadav", "Mitchell Starc", "Harry Brook"],
        "Gujarat Titans": ["Shubman Gill", "Jos Buttler", "Rashid Khan", "Kagiso Rabada", "Mohammed Siraj"],
        "Kolkata Knight Riders": ["Rinku Singh", "Varun Chakravarthy", "Sunil Narine", "Andre Russell", "Ramandeep Singh"],
        "Lucknow Super Giants": ["Rishabh Pant", "Nicholas Pooran", "Ravi Bishnoi", "Mohsin Khan", "Ayush Badoni"],
        "Mumbai Indians": ["Rohit Sharma", "Suryakumar Yadav", "Jasprit Bumrah", "Hardik Pandya", "Tilak Varma"],
        "Punjab Kings": ["Shreyas Iyer", "Yuzvendra Chahal", "Arshdeep Singh", "Prabhsimran Singh", "Shashank Singh"],
        "Rajasthan Royals": ["Sanju Samson", "Yashasvi Jaiswal", "Shimron Hetmyer", "Riyan Parag", "Sandeep Sharma"],
        "Royal Challengers Bengaluru": ["Rajat Patidar", "Virat Kohli", "Yash Dayal", "Liam Livingstone", "Phil Salt"],
        "Sunrisers Hyderabad": ["Pat Cummins", "Heinrich Klaasen", "Abhishek Sharma", "Travis Head", "Nitish Kumar Reddy"]
    }

    # Function to check if key player is missing
    def key_player_absent(row):
        team1_key = key_players.get(row['team1'])
        team2_key = key_players.get(row['team2'])
    
        if team1_key and any(player in row['team1_players'] for player in team1_key):
            team1_status = 1
        else:
            team1_status = 0
    
        if team2_key and any(player in row['team2_players'] for player in team2_key):
            team2_status = 1
        else:
            team2_status = 0
    
        return team1_status, team2_status

    df_cleaned[['team1_key_player_available', 'team2_key_player_available']] = df_cleaned.apply(key_player_absent, axis=1, result_type='expand')

    # Define Features (X) and Target (y)
    X = df_cleaned[['team1', 'team2', 'team1_win_percentage', 'team2_win_percentage',
                    'head_to_head', 'home_advantage', 'toss_winner', 'toss_decision',
                    'batter_form', 'bowler_form', 'venue', 'venue_win_percentage', 'day_night',
                    'team1_recent_form', 'team2_recent_form', 'team1_key_player_available', 'team2_key_player_available', 'winner']].copy()
    y = df_cleaned['winner'].copy()

    # Encode categorical features
    label_encoders = {}
    categorical_cols = ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision','winner']

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))  # Ensure transformation is applied correctly
        label_encoders[col] = le  # Store for later decoding

    # Encode target variable
    label_encoders['winner'] = LabelEncoder()
    y = label_encoders['winner'].fit_transform(y.astype(str))

    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['team1_win_percentage', 'team2_win_percentage', 'head_to_head',
                      'batter_form', 'bowler_form', 'venue_win_percentage', 'team1_recent_form', 'team2_recent_form']
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols].astype(float))

    # Convert boolean columns explicitly to integer
    X['home_advantage'] = X['home_advantage'].astype(int)
    X['day_night'] = X['day_night'].astype(int)
    X['team1_key_player_available'] = X['team1_key_player_available'].astype(int)
    X['team2_key_player_available'] = X['team2_key_player_available'].astype(int)

    # Ensure all columns are of the correct type (float or int)
    X = X.astype(float)

    # Save preprocessed data
    joblib.dump((X, y, label_encoders, scaler, categorical_cols, numerical_cols), 'data/preprocessed_data.pkl')

    return X, y, label_encoders, scaler, categorical_cols, numerical_cols

# Run the preprocessing function
load_and_preprocess_data(r'C:\Users\Chandan Bhat\OneDrive\Desktop\IPL_miniproject\dataset\final_model\cleaned_combined_ipl_data.csv')