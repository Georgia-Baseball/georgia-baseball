import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from itertools import product
from scipy.ndimage import gaussian_filter
from itertools import product
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
from fpdf import FPDF
import joblib
from datetime import datetime
import helper_functions as hf
from constants import(
    platoon_state_mapping,
    side_buckets,
    height_buckets,
    count_values,
    num_clusters,
    numerical_features,
    pseudo_sample_size,
    median_features
)
import warnings
warnings.filterwarnings('ignore')

rv_model = joblib.load('rv_model.pkl')
gmm_models = hf.load_gmm_models()

pitches_df = pd.read_csv('all_pitches.csv')
global_means = pd.read_csv('global_means.csv')

pitches_df = hf.prepare_data(pitches_df, game_only=False)

def calculate_batter_metrics(synthetic_df, pitches_df, pitcher_id, batter_id):
    platoon_state_encoded = synthetic_df['PlatoonStateEncoded'].iloc[0]
    pitch_group_encoded = synthetic_df['PitchGroupEncoded'].iloc[0]
    pitcher_throws = synthetic_df['PitcherThrows'].iloc[0]

    batter_id = pitches_df[pitches_df['BatterId'] == batter_id]['BatterId'].mode().iloc[0]

    synthetic_df['PlateLocSide'] = (synthetic_df['PlateLocSideBucket'].astype(float))
    synthetic_df['PlateLocHeight'] = (synthetic_df['PlateLocHeightBucket'].astype(float))

    synthetic_df = hf.add_probabilities(synthetic_df)
    batter_df = hf.add_probabilities(pitches_df[(pitches_df['BatterId'] == batter_id) & (pitches_df['PitcherThrows'] == pitcher_throws)])

    _, pivoted_values = hf.calculate_shrunken_means(
        batter_df, global_means
    )

    synthetic_df['BatterId'] = batter_id
    synthetic_df['Model'] = pitcher_throws
    synthetic_df = hf.compute_batter_stuff_value(synthetic_df, pivoted_values)

    return synthetic_df


def generate_figs(pitches_df, rv_model, pitcher_id, batter_id):
    columns_to_drop = [col for col in pitches_df.columns if col.startswith('DeltaRunValue_') or col.startswith('prob_')]
    pitches_df = pitches_df.drop(columns=columns_to_drop)

    if not pitches_df.index.is_unique:
        pitches_df = pitches_df.reset_index(drop=True)

    pitcher_throws_series = pitches_df.loc[pitches_df['PitcherId'] == pitcher_id, 'PitcherThrows']
    batter_side_series = pitches_df.loc[pitches_df['BatterId'] == batter_id, 'BatterSide']

    if not pitcher_throws_series.empty:
        pitcher_throws_mode = pitcher_throws_series.mode().iloc[0]
    else:
        raise ValueError(f"No matching rows found for pitcher: {pitcher_id}")

    if not batter_side_series.empty:
        batter_side_mode = batter_side_series.mode().iloc[0]
    else:
        raise ValueError(f"No matching rows found for batter: {batter_id}")

    platoon_state_encoded = platoon_state_mapping[(pitcher_throws_mode, batter_side_mode)]

    filtered_df = pitches_df[
        (pitches_df['PitcherId'] == pitcher_id) & 
        (pitches_df['TaggedPitchType'] != 'Undefined') & 
        (pitches_df['PlatoonStateEncoded'] == platoon_state_encoded)
    ]

    recent_rows = filtered_df.sort_values(by='UTCDateTime', ascending=False).head(500)

    pitch_type_counts = recent_rows['TaggedPitchType'].value_counts()
    qualifying_pitch_types = pitch_type_counts[pitch_type_counts >= 0.01 * len(recent_rows)].index.tolist()

    pitch_types = (
        filtered_df[filtered_df['TaggedPitchType'].isin(qualifying_pitch_types)]
        .groupby('TaggedPitchType')
        .size()
        .sort_values(ascending=False)
        .index.tolist()
    )

    pitcher_rows = pitches_df[
        (pitches_df['PitcherId'] == pitcher_id) & (pitches_df['TaggedPitchType'] != 'Undefined')
    ].sort_values(by='UTCDateTime', ascending=False).head(1000)

    batter_id = pitches_df[pitches_df['BatterId'] == batter_id]['BatterId'].mode().iloc[0]

    pitch_type_means = []
    pitch_usage = []
    pitch_command = []

    for pitch_type in pitch_types:
        pitch_type_df = (
            pitcher_rows[pitcher_rows['TaggedPitchType'] == pitch_type]
        )

        if pitch_type_df.empty:
            continue

        pitch_group_encoded = (
            pitches_df.loc[
                (pitches_df['PitcherId'] == pitcher_id) & 
                (pitches_df['TaggedPitchType'] == pitch_type), 
                'PitchGroupEncoded'
            ].mode()[0]
        )

        pitch_type_df = calculate_batter_metrics(pitch_type_df, pitches_df, pitcher_id, batter_id)

        if len(pitcher_rows[pitcher_rows['Top/Bottom'] == 'Bottom']) >= 50:
            pitch_usage_value = (
                len(pitch_type_df[(pitch_type_df['Top/Bottom'] == 'Bottom') & (pitch_type_df['PlatoonStateEncoded'] == platoon_state_encoded)])                 / len(pitcher_rows[(pitcher_rows['Top/Bottom'] == 'Bottom') & (pitcher_rows['PlatoonStateEncoded'] == platoon_state_encoded)])
            )
        else:
            pitch_usage_value = len(pitch_type_df) / len(pitcher_rows)
        pitch_usage.append((pitch_type, pitch_usage_value))

        expected_features = rv_model.get_booster().feature_names
        rank_df = pitch_type_df[expected_features].copy()
        rank_df['ExpectedRunValue'] = rv_model.predict(rank_df)
        mean_value = rank_df['ExpectedRunValue'].mean()
        pitch_type_means.append((pitch_type, mean_value))

    return pitch_usage, pitch_type_means


def calculate_weighted_averages(pitches_df, rv_model, pitchers, batters, mean=0.01, stdev=0.015):
    for pitcher_group in pitchers:
        primary_pitcher_id = float(pitcher_group[0])

        for pitcher_id in pitcher_group[1:]:
            pitches_df.loc[pitches_df['PitcherId'] == float(pitcher_id), 'PitcherId'] = primary_pitcher_id

    for batter_group in batters:
        primary_batter_id = float(batter_group[0])

        for batter_id in batter_group[1:]:
            pitches_df.loc[pitches_df['BatterId'] == float(batter_id), 'BatterId'] = primary_batter_id

    pitchers = [float(pitcher_group[0]) for pitcher_group in pitchers]
    batters = [float(batter_group[0]) for batter_group in batters]

    matchup_weighted_averages = {}

    for pitcher_id in pitchers:
        for batter_id in batters:
            pitch_usage, pitch_type_means = generate_figs(pitches_df, rv_model, pitcher_id, batter_id)

            pitch_type_means_dict = dict(pitch_type_means)

            total_weight = 0
            weighted_sum = 0
            for pitch_type, usage in pitch_usage:
                if pitch_type in pitch_type_means_dict:
                    value = pitch_type_means_dict[pitch_type]
                    weighted_sum += usage * value
                    total_weight += usage

            if total_weight > 0:
                weighted_average = weighted_sum / total_weight
            else:
                weighted_average = 0

            z_score = (weighted_average - mean) / stdev
            translated_value = (100 - (z_score * 10)).round(0)
            
            matchup_weighted_averages[(pitcher_id, batter_id)] = translated_value

    return matchup_weighted_averages


def plot_matchup_matrix_to_pdf(matchup_weighted_averages, pitches_df, pitching_team, batting_team, date, output_filename):
    batters = list(set(key[1] for key in matchup_weighted_averages.keys()))
    pitchers = list(set(key[0] for key in matchup_weighted_averages.keys()))

    heatmap_data_names = pd.DataFrame(index=batters, columns=pitchers)

    for (pitcher_id, batter_id), value in matchup_weighted_averages.items():
        pitcher = pitches_df.loc[pitches_df['PitcherId'] == pitcher_id, 'Pitcher'].mode()
        if not pitcher.empty:
            pitcher_name = " ".join(reversed(pitcher.iloc[0].split(", ")))
        else:
            pitcher_name = "Unknown Pitcher"

        batter = pitches_df.loc[pitches_df['BatterId'] == batter_id, 'Batter'].mode()
        if not batter.empty:
            batter_name = " ".join(reversed(batter.iloc[0].split(", ")))
        else:
            batter_name = "Unknown Batter"

        heatmap_data_names.loc[batter_name, pitcher_name] = value

    heatmap_data_names = heatmap_data_names.dropna(axis=0, how='all').dropna(axis=1, how='all')

    plt.figure(figsize=(20, 20))
    sns.heatmap(
        heatmap_data_names,
        annot=True,
        fmt=".0f",
        cmap="coolwarm",
        cbar=False,
        vmin=80,
        vmax=120,
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"size": 20}
    )

    plt.title("", pad=20)
    plt.xlabel("Pitchers", fontsize=20, labelpad=15)
    plt.ylabel("Batters", fontsize=20, labelpad=15)

    plt.yticks(fontsize=14, rotation=0)
    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().xaxis.set_label_position('top')
    plt.xticks(fontsize=14, rotation=10)

    temp_image_path = 'temp_matrix.png'
    plt.savefig(temp_image_path, dpi=300)
    plt.close()

    pdf = FPDF(format='letter')
    pdf.add_page()

    pdf.image("georgia_logo.png", x=170, y=5, w=30)

    pdf.set_font("Arial", size=16, style='B')
    pdf.cell(200, 10, txt="Matchup Matrix", ln=True, align='C')

    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt=f"{batting_team} vs. {pitching_team}", ln=True, align='C')

    pdf.set_font("Arial", size=12)
    y_offset = pdf.get_y()
    pdf.set_xy(10, y_offset - 2.5)
    pdf.cell(200, 10, txt=f"{date}", ln=True, align='C')

    pdf.image(temp_image_path, x=10, y=30, w=190)

    directory = "matchup_matrices"
    output_filename = os.path.join(directory, "output.pdf")

    if not os.path.exists(directory):
        os.makedirs(directory)

    pdf.output(output_filename)

    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)

    print(f"PDF saved as {output_filename}")


### DO NOT TOUCH THE CODE ABOVE

### PUT PITCHER IDS HERE
pitchers = [
            [803287, 1000110595], ### PITCHER 1
            [701368, 1000187635], ### PITCHER 2
            [1000121636], ### PITCHER 3
            [815124, 1000122824], ### PITCHER 4
            [1000127884], ### PITCHER 5
            [90000235966], ### PITCHER 6
            [1000051331, 809709], ### PITCHER 7
            [823229, 1000132878], ### PITCHER 8
            [804607, 1000050067], ### PITCHER 9
            [809712, 1000066681] ### PITCHER 10
]

### PUT BATTER IDS HERE
batters = [
            [695477, 1000052101], ### BATTER 1
            [809707, 1000133690], ### BATTER 2
            [702705, 1000107443], ### BATTER 3
            [692232, 1000064202], ### BATTER 4
            [687408, 1000013128], ### BATTER 5
            [699821, 1000079889], ### BATTER 6
            [690970], ### BATTER 7
            [802119, 1000073064], ### BATTER 8
            [804569, 1000133791], ### BATTER 9
            [674890, 1000051139] ### BATTER 10
]

### PUT PITCHING TEAM HERE
pitching_team = "Georgia"

### PUT HITTING TEAM HERE
batting_team = "Georgia"

### PUT DATE HERE
date = "2025-01-24"

### LEAVE THIS CODE AS IS
matchup_weighted_averages = calculate_weighted_averages(pitches_df, rv_model, pitchers, batters)
plot_matchup_matrix_to_pdf(
    matchup_weighted_averages=matchup_weighted_averages,
    pitches_df=pitches_df,
    pitching_team=pitching_team,
    batting_team=batting_team,
    date=date,
    output_filename = f"{pitching_team.lower().replace(' ', '_')}_{batting_team.lower().replace(' ', '_')}_matchup_matrix.pdf"
)