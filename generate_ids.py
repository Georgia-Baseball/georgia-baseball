import pandas as pd

pitches_df = pd.read_csv('all_pitches.csv')

def get_batter_df(pitches_df):
    if pitches_df['UTCDateTime'].dtype == 'object':
        pitches_df['UTCDateTime'] = pd.to_datetime(pitches_df['UTCDateTime'])

    sorted_df = pitches_df.sort_values(by=['BatterId', 'Batter', 'UTCDateTime'], ascending=[True, True, False])

    batter_df = sorted_df.drop_duplicates(subset=['BatterId', 'Batter'], keep='first')[['BatterId', 'Batter', 'BatterTeam']]

    batter_df = batter_df.reset_index(drop=True)
    
    return batter_df

def get_pitcher_df(pitches_df):
    if pitches_df['UTCDateTime'].dtype == 'object':
        pitches_df['UTCDateTime'] = pd.to_datetime(pitches_df['UTCDateTime'])

    sorted_df = pitches_df.sort_values(by=['PitcherId', 'Pitcher', 'UTCDateTime'], ascending=[True, True, False])

    pitcher_df = sorted_df.drop_duplicates(subset=['PitcherId', 'Pitcher'], keep='first')[['PitcherId', 'Pitcher', 'PitcherTeam']]

    pitcher_df = pitcher_df.reset_index(drop=True)
    
    return pitcher_df


def generate_id_csv(pitches_df, output_path='player_ids.csv'):
    batter_df = get_batter_df(pitches_df)
    pitcher_df = get_pitcher_df(pitches_df)

    batter_df['Type'] = 'Batter'
    pitcher_df['Type'] = 'Pitcher'

    batter_df = batter_df.rename(columns={'BatterId': 'Id', 'Batter': 'Name', 'BatterTeam': 'Team'})
    pitcher_df = pitcher_df.rename(columns={'PitcherId': 'Id', 'Pitcher': 'Name', 'PitcherTeam': 'Team'})

    combined_df = pd.concat([batter_df, pitcher_df], ignore_index=True)
    combined_df = combined_df.sort_values(by='Name', ascending=True)

    combined_df.to_csv(output_path, index=False)

    print(f"CSV saved to {output_path}")

generate_id_csv(pitches_df)