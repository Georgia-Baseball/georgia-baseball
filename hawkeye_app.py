import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import matplotlib.patches as patches
warnings.filterwarnings('ignore')

PASSWORD = "ugabulldogs2025"

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("🔒 Password Protected App")

    password_input = st.text_input("Enter Password", type="password")

    if st.button("Login"):
        if password_input == PASSWORD:
            st.session_state["authenticated"] = True
            st.button("Enter Site")
        else:
            st.error("❌ Incorrect password. Please try again.")

    st.stop()

@st.cache_data(ttl=900)
def load_and_concat_csvs(directory):
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    if not csv_files:
        st.error("No CSV files found in the directory.")
        return None
    
    df_list = []
    for file in csv_files:
        file_path = os.path.join(directory, file)
        try:
            df = pd.read_csv(file_path)
            df_list.append(df)
        except Exception as e:
            st.error(f"Error reading {file}: {e}")
    
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    else:
        return None

directory = "hawkeye_csvs"
data = load_and_concat_csvs(directory)

st.sidebar.title("Filters")

unique_pitchers = sorted(data['Pitcher'].unique(), key=lambda name: name.split()[-1])

selected_pitcher = st.sidebar.selectbox("Select a Pitcher", unique_pitchers)

pitcher_data = data[data['Pitcher'] == selected_pitcher]

pitcher_data['Year'] = pd.to_datetime(pitcher_data['Date']).dt.year
unique_years = ["All"] + sorted(pitcher_data['Year'].unique(), reverse=True)

has_game_data = not pitcher_data.empty
default_index = 1 if has_game_data else 2

selected_year = st.sidebar.selectbox("Select Season", unique_years, index=default_index)

if selected_year != "All":
    pitcher_data = pitcher_data[pitcher_data['Year'] == selected_year]

if pitcher_data.empty:
    st.markdown(
        "<h1 style='text-align: center; font-size: 40px; color: black;'>No data ☹️</h1>",
        unsafe_allow_html=True
    )
    st.stop()

pitcher_data['Date'] = pd.to_datetime(pitcher_data['Date'], errors='coerce')

min_date = pitcher_data['Date'].min()
max_date = pitcher_data['Date'].max()

selected_date_range = st.sidebar.date_input(
    "Select Date Range",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

unique_game_ids = pitcher_data[['GameID', 'Date', 'AwayTeam']].drop_duplicates()

unique_game_ids = unique_game_ids.sort_values('Date', ascending=False)

unique_game_ids['GameID_Display'] = unique_game_ids.apply(
    lambda row: f"{row['AwayTeam']} ({row['Date'].strftime('%Y-%m-%d')})", axis=1
)

game_id_options = dict(zip(unique_game_ids['GameID_Display'], unique_game_ids['GameID']))

selected_game_id_display = st.sidebar.selectbox(
    "Select a Game", options=["None"] + list(game_id_options.keys())
)

pitcher_data['Date'] = pd.to_datetime(pitcher_data['Date'], errors='coerce')

filtered_data = pitcher_data[
    (pitcher_data['Date'] >= pd.Timestamp(selected_date_range[0])) &
    (pitcher_data['Date'] <= pd.Timestamp(selected_date_range[1]))
]

st.markdown(
    f"<h1 style='text-align: center;'>{selected_pitcher}</h1>",
    unsafe_allow_html=True,
)

hitter_hand_options = ['All', 'R', 'L']
selected_hitter_hand = st.sidebar.selectbox(
    "Select Hitter Handedness",
    hitter_hand_options,
    index=0,
    key="tab1_hitter_hand_selectbox"
)

if selected_hitter_hand == 'R':
    filtered_data = filtered_data[filtered_data['BatterSide'] == 'Right']
elif selected_hitter_hand == 'L':
    filtered_data = filtered_data[filtered_data['BatterSide'] == 'Lefty']

description_text = f"Data From {pitcher_data['Date'].min().date()} to {pitcher_data['Date'].max().date()}"

st.markdown(
    f"<p style='text-align: center; font-size: 18px;'>{description_text}</p>",
    unsafe_allow_html=True,
)

pitch_counts = filtered_data['TaggedPitchType'].value_counts(normalize=True) * 100
grouped_df = filtered_data.groupby('TaggedPitchType')[['RelSpeed', 'SpinRate', 'RelHeight', 'RelSide', 'Extension', 'InducedVertBreak', 'HorzBreak']].mean()
grouped_df.insert(0, 'Usage%', pitch_counts)
grouped_df = grouped_df.fillna(0).round(1)
grouped_df.reset_index(inplace=True)
st.dataframe(grouped_df)
grouped_df = grouped_df.sort_values(by='Usage%', ascending=False)

tab1, tab2 = st.tabs(["Pitch Data", "Pitch Usage"])
        
with tab1:
    st.write("### Pitch Summary:")
    fig, ax = plt.subplots(figsize=(8, len(grouped_df) * 0.5))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=grouped_df.values,
                        colLabels=grouped_df.columns,
                        cellLoc='center',
                        loc='center',
                        colColours=['#f0f0f0']*len(grouped_df.columns),
                        cellColours=[['#f0f0f0']*len(grouped_df.columns) for _ in range(len(grouped_df))],
                        bbox=[0, 0, 1, 1])
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(fontweight='bold', fontsize=6, color='black')
        else:
            cell.set_text_props(fontsize=10, color='black')

    table.auto_set_column_width(col=list(range(len(grouped_df.columns))))
    table.auto_set_font_size(False)
    st.pyplot(fig)
    st.markdown("<br>", unsafe_allow_html=True)

    for pitch_type in pitcher_data['TaggedPitchType'].unique():
        st.write(f"<h3 style='text-align: center;'>{pitch_type}</h3>", unsafe_allow_html=True)
        pitch_df = pitcher_data[pitcher_data['TaggedPitchType'] == pitch_type]
        pitch_df['AdjPlateLocSide'] = pitch_df['PlateLocSide'] * -1
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.scatterplot(data=pitch_df, x='AdjPlateLocSide', y='PlateLocHeight', ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(0, 5)
        ax.add_patch(plt.Rectangle((-0.9, 1.5), 1.8, 1.8, fill=False, edgecolor='black', linewidth=2))
        home_plate_coords = [
            (-0.9, 0.8), (0.9, 0.8),
            (0.9, 0.4), (0, 0.1), (-0.9, 0.4)
        ]
        home_plate = patches.Polygon(
            home_plate_coords,
            closed=True,
            edgecolor='black',
            facecolor='grey',
            linewidth=2
        )
        ax.add_patch(home_plate)
        ax.set_xticks([])
        ax.set_yticks([])
        st.pyplot(fig)
        st.markdown("<br>", unsafe_allow_html=True)

with tab2:
    overall_pitch_data = (
        filtered_data.groupby('TaggedPitchType').size().reset_index(name='Pitch Count')
    )
    overall_pitch_data['Usage (%)'] = (overall_pitch_data['Pitch Count'] / overall_pitch_data['Pitch Count'].sum()) * 100

    pitch_type_order = overall_pitch_data.sort_values('Pitch Count', ascending=False)['TaggedPitchType']
    overall_pitch_data['TaggedPitchType'] = pd.Categorical(overall_pitch_data['TaggedPitchType'], categories=pitch_type_order, ordered=True)
    overall_pitch_data = overall_pitch_data.sort_values('TaggedPitchType')

    st.markdown(
        "<h1 style='text-align: center; font-size: 40px; font-weight: bold;'>Pitch Usage</h1>",
        unsafe_allow_html=True,
    )

    label_distance = 2

    fig, ax = plt.subplots(figsize=(20, 14))

    wedges, texts, autotexts = ax.pie(
        overall_pitch_data['Pitch Count'],
        labels=None,
        autopct='%1.1f%%',
        pctdistance=0.7,
    )

    for autotext in autotexts:
        autotext.set_fontsize(30)

    label_distance = 1.1
    for i, wedge in enumerate(wedges):
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = label_distance * np.cos(np.radians(angle))
        y = label_distance * np.sin(np.radians(angle))
        ax.text(
            x, y,
            overall_pitch_data['TaggedPitchType'].iloc[i],
            ha='center', va='center', fontsize=30,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.6)
        )

    st.pyplot(fig)
    plt.close(fig)

    usage_data = filtered_data.groupby(['Balls', 'Strikes', 'TaggedPitchType']).size().reset_index(name='Pitch Count')
    total_count_data = filtered_data.groupby(['Balls', 'Strikes']).size().reset_index(name='Total Count')

    usage_data = usage_data.merge(total_count_data, on=['Balls', 'Strikes'])
    usage_data['Usage (%)'] = (usage_data['Pitch Count'] / usage_data['Total Count']) * 100

    count_order = ['3-0', '3-1', '2-0', '2-1', '1-0', '3-2', '0-0', '1-1', '2-2', '0-1', '1-2', '0-2']
    usage_data['Count'] = usage_data['Balls'].astype(str) + '-' + usage_data['Strikes'].astype(str)
    usage_data['Count'] = pd.Categorical(usage_data['Count'], categories=count_order, ordered=True)

    pitch_type_order = (
        usage_data.groupby('TaggedPitchType')['Pitch Count']
        .sum()
        .sort_values(ascending=False)
        .index
    )
    usage_data['TaggedPitchType'] = pd.Categorical(usage_data['TaggedPitchType'], categories=pitch_type_order, ordered=True)
    usage_data = usage_data.sort_values(['TaggedPitchType', 'Count'])

    usage_pivot = usage_data.pivot_table(
        index='TaggedPitchType', 
        columns='Count', 
        values='Usage (%)', 
        aggfunc='sum'
    )

    plt.figure(figsize=(12, 16))
    ax = sns.heatmap(
        usage_pivot,
        annot=True,
        fmt=".0f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor='black',
        cbar=False,
        vmin=0,
        vmax=80,
        annot_kws={"size": 42}
    )

    for text in ax.texts:
        text.set_text(f"{text.get_text()}%")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<h1 style='text-align: center; font-size: 40px; font-weight: bold;'>Pitch Usage by Count</h1>",
        unsafe_allow_html=True,
    )

    plt.xlabel("Count", fontsize=50, labelpad=30)
    plt.ylabel("")

    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36, rotation=75)

    plt.subplots_adjust(left=-0.95, right=0.95)

    st.pyplot(plt.gcf())
    plt.close()

    filtered_data = filtered_data.sort_values(['GameID', 'Inning', 'Top/Bottom', 'PAofInning', 'PitchofPA']).reset_index(drop=True)
    filtered_data['Times Through Lineup'] = (
        filtered_data.groupby(['Pitcher', 'Batter'])
        .cumcount() + 1
    )

    filtered_data['Times Through Lineup'] = np.clip(filtered_data['Times Through Lineup'], 1, 3)
    ttl_data = filtered_data.groupby(['Times Through Lineup', 'TaggedPitchType']).size().reset_index(name='Pitch Count')
    ttl_data['Usage (%)'] = (ttl_data['Pitch Count'] / ttl_data.groupby('Times Through Lineup')['Pitch Count'].transform('sum')) * 100

    ttl_1_data = ttl_data[ttl_data['Times Through Lineup'] == 1]
    ttl_2_data = ttl_data[ttl_data['Times Through Lineup'] == 2]
    ttl_3_data = ttl_data[ttl_data['Times Through Lineup'] == 3]

    label_distance = 1.2
    percentage_distance = 0.4

    if not ttl_1_data.empty:
        plt.figure(figsize=(20, 14))
        wedges, texts, autotexts = plt.pie(
            ttl_1_data['Pitch Count'], 
            labels=None,
            autopct='%1.1f%%', 
            pctdistance=percentage_distance,
        )
        plt.title("First Time Through", fontsize=30)

        for autotext in autotexts:
            autotext.set_fontsize(30)

        for i, wedge in enumerate(wedges):
            angle = (wedge.theta2 + wedge.theta1) / 2
            x = label_distance * np.cos(np.radians(angle))
            y = label_distance * np.sin(np.radians(angle))
            plt.text(
                x, y, 
                ttl_1_data['TaggedPitchType'].iloc[i],
                ha='center', va='center', fontsize=30,
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.6)
            )
        st.pyplot(plt.gcf())
        plt.close()

    if not ttl_2_data.empty:
        plt.figure(figsize=(20, 14))
        wedges, texts, autotexts = plt.pie(
            ttl_2_data['Pitch Count'], 
            labels=None, 
            autopct='%1.1f%%', 
            pctdistance=percentage_distance,
        )
        plt.title("Second Time Through", fontsize=30)

        for autotext in autotexts:
            autotext.set_fontsize(30)

        for i, wedge in enumerate(wedges):
            angle = (wedge.theta2 + wedge.theta1) / 2
            x = label_distance * np.cos(np.radians(angle))
            y = label_distance * np.sin(np.radians(angle))
            plt.text(
                x, y, 
                ttl_2_data['TaggedPitchType'].iloc[i],
                ha='center', va='center', fontsize=30,
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.6)
            )
        st.pyplot(plt.gcf())
        plt.close()

    if not ttl_3_data.empty:
        plt.figure(figsize=(20, 14))
        wedges, texts, autotexts = plt.pie(
            ttl_3_data['Pitch Count'], 
            labels=None, 
            autopct='%1.1f%%', 
            pctdistance=percentage_distance,
        )
        plt.title("Third Time Through", fontsize=30)

        for autotext in autotexts:
            autotext.set_fontsize(30)

        for i, wedge in enumerate(wedges):
            angle = (wedge.theta2 + wedge.theta1) / 2
            x = label_distance * np.cos(np.radians(angle))
            y = label_distance * np.sin(np.radians(angle))
            plt.text(
                x, y, 
                ttl_3_data['TaggedPitchType'].iloc[i],
                ha='center', va='center', fontsize=30,
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.6)
            )
        st.pyplot(plt.gcf())
        plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<h1 style='text-align: center; font-size: 40px; font-weight: bold;'>Pitch Usage After Each Pitch</h1>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    filtered_data = filtered_data.sort_values(['GameID', 'Inning', 'Top/Bottom', 'PAofInning', 'PitchofPA']).reset_index(drop=True)
    filtered_tab1_data_no_first_pitch = filtered_data[filtered_data['PitchofPA'] > 1]
    filtered_tab1_data_no_first_pitch['Next Pitch Type'] = filtered_tab1_data_no_first_pitch['TaggedPitchType'].shift(-1)

    filtered_tab1_data_no_first_pitch = filtered_tab1_data_no_first_pitch.dropna(subset=['Next Pitch Type'])

    next_pitch_data = filtered_tab1_data_no_first_pitch.groupby(['TaggedPitchType', 'Next Pitch Type']).size().reset_index(name='Count')
    next_pitch_data['Usage (%)'] = next_pitch_data.groupby('TaggedPitchType')['Count'].transform(lambda x: (x / x.sum()) * 100)

    next_pitch_data['TaggedPitchType'] = pd.Categorical(next_pitch_data['TaggedPitchType'], categories=pitch_type_order, ordered=True)

    unique_pitch_types = next_pitch_data['TaggedPitchType'].cat.categories
    percentage_distance = 0.5

    for pitch_type in unique_pitch_types:
        pitch_data = next_pitch_data[next_pitch_data['TaggedPitchType'] == pitch_type]
        if not pitch_data.empty:
            fig, ax = plt.subplots(figsize=(20, 14), subplot_kw=dict(aspect="equal"))

            wedges, texts, autotexts = ax.pie(
                pitch_data['Count'],
                labels=None,
                autopct='%1.1f%%',
                pctdistance=percentage_distance,
            )
            ax.set_title(f"After {pitch_type}", fontsize=30, pad=10)

            for autotext in autotexts:
                autotext.set_fontsize(30)

            for i, wedge in enumerate(wedges):
                angle = (wedge.theta2 + wedge.theta1) / 2
                x = label_distance * np.cos(np.radians(angle))
                y = label_distance * np.sin(np.radians(angle))
                ax.text(
                    x, y,
                    pitch_data['Next Pitch Type'].iloc[i],
                    ha='center', va='center', fontsize=30,
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.6)
                )

            st.pyplot(fig)
            plt.close(fig)
