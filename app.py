import pandas as pd
import streamlit as st
import joblib

teams = ['Royal Challengers Bengaluru',
 'Mumbai Indians',
 'Chennai Super Kings',
 'Delhi Capitals',
 'Gujarat Titans',
 'Lucknow Super Giants',
 'Sunrisers Hyderabad',
 'Rajasthan Royals',
 'Punjab Kings',
 'Kolkata Knight Riders']

cities = ['Delhi', 'Navi Mumbai', 'Mumbai', 'Kolkata', 'Chandigarh',
       'Nagpur', 'Abu Dhabi', 'Chennai', 'Hyderabad', 'Dubai',
       'Johannesburg', 'Indore', 'Cuttack', 'Bengaluru', 'Visakhapatnam',
       'Lucknow', 'Rajkot', 'Kimberley', 'Durban', 'Cape Town',
       'Port Elizabeth', 'Sharjah', 'Ahmedabad', 'Jaipur', 'Pune',
       'Guwahati', 'Dharamsala', 'Centurion', 'Ranchi', 'Raipur',
       'Bloemfontein', 'Kanpur', 'Mohali', 'East London']

model = joblib.load('pipe.joblib')

st.title("IPL Match Win Predictor")

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Select the batting team", sorted(teams))

with col2:
    # Filter out the selected batting team from the fielding team dropdown
    fielding_team_options = [team for team in teams if team != batting_team]
    bowling_team = st.selectbox("Select the fielding team", sorted(fielding_team_options))

selected_city = st.selectbox("Select the host city", sorted(cities))

target_to_chase = st.number_input("Target for " + batting_team)

col3, col4, col5 = st.columns(3)

with col3:
    current_score = st.number_input("Current score")

with col4:
    overs = st.number_input("Overs completed")

with col5:
    wickets = st.number_input("Wickets down")

if st.button("Predict Probability"):
    # Perform calculations
    runs_left = target_to_chase - current_score
    balls_left = 120 - 6 * overs
    wickets_left = 10 - wickets
    crr = current_score / overs
    rrr = (6 * runs_left) / balls_left

    # Input DataFrame to model
    input_df = pd.DataFrame({"batting_team": [batting_team], "bowling_team": [bowling_team], "city": [selected_city], "runs_left": [runs_left], "balls_left": [balls_left], "wickets_left": [wickets_left], "target_to_chase": [target_to_chase], "crr": [crr], "rrr": [rrr]})
    # st.table(input_df)

    prediction = model.predict_proba(input_df)

    lose_prob = prediction[0][0] * 100
    win_prob = prediction[0][1] * 100
    
    st.subheader(f"{batting_team} - {win_prob:.2f} %")
    st.subheader(f"{bowling_team} - {lose_prob:.2f} %")
