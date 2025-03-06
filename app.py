import numpy as np
import pandas as pd
import joblib
import streamlit as st
import ollama
X, y, label_encoders, scaler, categorical_cols, numerical_cols = joblib.load('preprocessed_data.pkl')
def predict_match_outcome(team1, team2, toss_winner, toss_decision, venue, day_night, team1_players, team2_players):
    global team1_proba, team2_proba
    # Load preprocessed data, trained model, and X_test
    best_model = joblib.load('trained_model.pkl')
    X_test = joblib.load('X_test.pkl')

    # Validate user inputs
    def validate_input(input_value, encoder, input_name):
        if input_value not in encoder.classes_:
            raise ValueError(f"{input_name} contains previously unseen labels: '{input_value}'")
        return encoder.transform([input_value])[0]

    try:
        team1_encoded = validate_input(team1, label_encoders['team1'], "Team 1")
        team2_encoded = validate_input(team2, label_encoders['team2'], "Team 2")
        toss_winner_encoded = validate_input(toss_winner, label_encoders['toss_winner'], "Toss Winner")
        toss_decision_encoded = validate_input(toss_decision, label_encoders['toss_decision'], "Toss Decision")
        venue_encoded = validate_input(venue, label_encoders['venue'], "Venue")
    except ValueError as e:
        return str(e)

    day_night_encoded = 1 if day_night.lower() == 'night' else 0

    # Filter the dataset for matches between team1 and team2
    matches = X_test[((X_test['team1'] == team1_encoded) & (X_test['team2'] == team2_encoded)) |
                     ((X_test['team1'] == team2_encoded) & (X_test['team2'] == team1_encoded))]

    if matches.empty:
        return "No match data found for the specified teams."

    # Get the latest match data
    latest_match = matches.iloc[-1]

    # Extract relevant features
    team1_win_percentage = latest_match['team1_win_percentage']
    team2_win_percentage = latest_match['team2_win_percentage']
    head_to_head = latest_match['head_to_head']
    home_advantage = latest_match['home_advantage']
    batter_form = latest_match['batter_form']
    bowler_form = latest_match['bowler_form']
    venue_win_percentage = latest_match['venue_win_percentage']
    team1_recent_form = latest_match['team1_recent_form']
    team2_recent_form = latest_match['team2_recent_form']
    team1_key_player_available = latest_match['team1_key_player_available']
    team2_key_player_available = latest_match['team2_key_player_available']

    # Check if 'winner' column exists in the latest match and is valid
    if 'winner' in latest_match and latest_match['winner'] in label_encoders['winner'].classes_:
        Winner = label_encoders['winner'].inverse_transform([int(latest_match['winner'])])[0]
    else:
        Winner = None

    # Create the custom test case DataFrame
    custom_test_case = pd.DataFrame({
        'team1': [team1_encoded],
        'team2': [team2_encoded],
        'team1_win_percentage': [team1_win_percentage],
        'team2_win_percentage': [team2_win_percentage],
        'head_to_head': [head_to_head],
        'home_advantage': [home_advantage],
        'toss_winner': [toss_winner_encoded],
        'toss_decision': [toss_decision_encoded],
        'batter_form': [batter_form],
        'bowler_form': [bowler_form],
        'venue': [venue_encoded],
        'venue_win_percentage': [venue_win_percentage],
        'day_night': [day_night_encoded],
        'team1_recent_form': [team1_recent_form],
        'team2_recent_form': [team2_recent_form],
        'team1_key_player_available': [team1_key_player_available],
        'team2_key_player_available': [team2_key_player_available],
        'winner': [Winner]
    })

    # Ensure 'winner' is included in categorical_cols
    if 'winner' not in categorical_cols:
        categorical_cols.append('winner')

    # Encode the custom test case
    for col in categorical_cols:
        custom_test_case[col] = custom_test_case[col].apply(
            lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1
        )

    # Scale numerical features
    custom_test_case[numerical_cols] = scaler.transform(custom_test_case[numerical_cols].astype(float))

    # Convert categorical boolean/int features to int
    custom_test_case['home_advantage'] = custom_test_case['home_advantage'].astype(int)
    custom_test_case['day_night'] = custom_test_case['day_night'].astype(int)
    custom_test_case['team1_key_player_available'] = custom_test_case['team1_key_player_available'].astype(int)
    custom_test_case['team2_key_player_available'] = custom_test_case['team2_key_player_available'].astype(int)

    # Ensure all columns are of the correct dtype (float)
    custom_test_case = custom_test_case.astype(float)

    # Make a prediction
    prediction_proba = best_model.predict_proba(custom_test_case)

    # Get the indices of the teams in the match
    team1_index = label_encoders['winner'].transform([team1])[0]
    team2_index = label_encoders['winner'].transform([team2])[0]

    # Get the probabilities for the teams in the match
    team1_proba = prediction_proba[0][team1_index]
    team2_proba = prediction_proba[0][team2_index]

    # Determine the predicted winner based on the highest probability
    if team1_proba > team2_proba:
        predicted_winner = team1
    else:
        predicted_winner = team2

    # Decode the custom test case for printing
    decoded_custom_test_case = custom_test_case.copy()
    for col in categorical_cols:
        # Skip decoding if the column contains -1
        if -1 in custom_test_case[col].values:
            decoded_custom_test_case[col] = custom_test_case[col]
        else:
            decoded_custom_test_case[col] = label_encoders[col].inverse_transform(custom_test_case[col].astype(int))

    # Prepare the result
    result = {
        "team1": decoded_custom_test_case['team1'].values[0],
        "team2": decoded_custom_test_case['team2'].values[0],
        "toss_winner": decoded_custom_test_case['toss_winner'].values[0],
        "toss_decision": decoded_custom_test_case['toss_decision'].values[0],
        "venue": decoded_custom_test_case['venue'].values[0],
        "predicted_winner": predicted_winner,
        "team1_proba": team1_proba,
        "team2_proba": team2_proba,
        "Winner": Winner,
        "prediction_correct": Winner == predicted_winner
    }

    return result

def generate_explanation(result):
    response = ollama.generate(
        model="llama3.2:1b",
        prompt=f"""Who is likely to win the match between {team1} and {team2}? The model predicts:

                {team1} win probability: {team1_proba}
                {team2} win probability: {team2_proba}
                Give the winner and explain in simple terms why, considering factors like:

                Recent team form
                Head-to-head record
                Key player availability
                Toss impact
                Venue conditions
                Keep it straightforward and useful for betting decisions."""
    )
    explanation = response["response"].strip()
    return explanation

# Streamlit UI
st.title("IPL Match Outcome Predictor")

team1 = st.text_input("Enter Team 1:").strip()
team2 = st.text_input("Enter Team 2:").strip()
toss_winner = st.text_input("Enter Toss Winner:").strip()
toss_decision = st.selectbox("Enter Toss Decision:", ["bat", "field"]).strip()
venue = st.text_input("Enter Venue:").strip()
day_night = st.selectbox("Enter Playing Time:", ["day", "night"]).strip()
team1_players = st.text_area("Enter Playing 11 for Team 1 (comma-separated):").strip().split(',')
team2_players = st.text_area("Enter Playing 11 for Team 2 (comma-separated):").strip().split(',')

if st.button("Predict Outcome"):
    result = predict_match_outcome(team1, team2, toss_winner, toss_decision, venue, day_night, team1_players, team2_players)
    
    if isinstance(result, dict):  # Check if result is a dictionary before accessing keys
        st.write("Predicted Winner:", result['predicted_winner'])
        
        explanation = generate_explanation(result)
        st.write("Explanation for the prediction:")
        st.write(explanation)
    else:
        st.error(result)
