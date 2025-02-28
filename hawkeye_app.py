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
        data = pd.concat(df_list, ignore_index=True)
        data.rename(columns={'TaggedPitchType': 'Pitch Type'}, inplace=True)
        return data
    else:
        return None

directory = "hawkeye_csvs"
data = load_and_concat_csvs(directory)

st.sidebar.title("Filters")

selected_pitcher_or_batter = st.sidebar.selectbox(
    "Pitcher or Batter",
    ['Pitcher', 'Batter'],
    index=0,
    key="pitcher_batter"
)

if selected_pitcher_or_batter == 'Pitcher':
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

    if filtered_data.empty:
        st.markdown(
            "<h1 style='text-align: center; font-size: 40px; color: black;'>No data ☹️</h1>",
            unsafe_allow_html=True
        )
        st.stop()

    description_text = f"Data From {pitcher_data['Date'].min().date()} to {pitcher_data['Date'].max().date()}"

    st.markdown(
        f"<p style='text-align: center; font-size: 18px;'>{description_text}</p>",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs(["Pitch Data", "Pitch Usage", "Pitch Type Stats", "Game Logs"])
            
    with tab1:
        st.markdown(
            "<h2 style='text-align: center; font-weight: bold;'>Pitch Summary</h2>",
            unsafe_allow_html=True,
        )

        pitch_counts = filtered_data['Pitch Type'].value_counts(normalize=True) * 100
        grouped_df = filtered_data.groupby('Pitch Type')[['RelSpeed', 'SpinRate', 'InducedVertBreak', 'HorzBreak', 'RelHeight', 'RelSide', 'Extension']].mean()
        grouped_df.insert(0, 'Usage%', pitch_counts)
        grouped_df = grouped_df.fillna(0)
        grouped_df = grouped_df.round({col: 1 for col in grouped_df.columns if col != 'SpinRate'})
        grouped_df['SpinRate'] = grouped_df['SpinRate'].round(0).astype(int)
        grouped_df.reset_index(inplace=True)
        grouped_df = grouped_df.sort_values(by='Usage%', ascending=False)

        grouped_df.rename(columns={
            'RelSpeed': 'Velocity',
            'SpinRate': 'Spin Rate',
            'InducedVertBreak': 'IVB',
            'HorzBreak': 'HB'
        }, inplace=True)

        fig, ax = plt.subplots(figsize=(8, len(grouped_df) * 1))
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
                cell.set_text_props(fontweight='bold', fontsize=8.5, color='black')
            else:
                cell.set_text_props(fontsize=10, color='black')

        table.auto_set_column_width(col=list(range(len(grouped_df.columns))))
        table.auto_set_font_size(False)
        st.pyplot(fig)
        st.markdown("<br>", unsafe_allow_html=True)

        for pitch_type in pitcher_data['Pitch Type'].unique():
            st.write(f"<h3 style='text-align: center;'>{pitch_type}</h3>", unsafe_allow_html=True)
            pitch_df = pitcher_data[pitcher_data['Pitch Type'] == pitch_type]
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
            filtered_data.groupby('Pitch Type').size().reset_index(name='Pitch Count')
        )
        overall_pitch_data['Usage (%)'] = (overall_pitch_data['Pitch Count'] / overall_pitch_data['Pitch Count'].sum()) * 100

        pitch_type_order = overall_pitch_data.sort_values('Pitch Count', ascending=False)['Pitch Type']
        overall_pitch_data['Pitch Type'] = pd.Categorical(overall_pitch_data['Pitch Type'], categories=pitch_type_order, ordered=True)
        overall_pitch_data = overall_pitch_data.sort_values('Pitch Type')

        st.write(f"<h3 style='text-align: center;'>Pitch Usage</h3>", unsafe_allow_html=True)

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
                overall_pitch_data['Pitch Type'].iloc[i],
                ha='center', va='center', fontsize=30,
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.6)
            )

        st.pyplot(fig)
        plt.close(fig)

        usage_data = filtered_data.groupby(['Balls', 'Strikes', 'Pitch Type']).size().reset_index(name='Pitch Count')
        total_count_data = filtered_data.groupby(['Balls', 'Strikes']).size().reset_index(name='Total Count')

        usage_data = usage_data.merge(total_count_data, on=['Balls', 'Strikes'])
        usage_data['Usage (%)'] = (usage_data['Pitch Count'] / usage_data['Total Count']) * 100

        count_order = ['3-0', '3-1', '2-0', '2-1', '1-0', '3-2', '0-0', '1-1', '2-2', '0-1', '1-2', '0-2']
        usage_data['Count'] = usage_data['Balls'].astype(str) + '-' + usage_data['Strikes'].astype(str)
        usage_data['Count'] = pd.Categorical(usage_data['Count'], categories=count_order, ordered=True)

        pitch_type_order = (
            usage_data.groupby('Pitch Type')['Pitch Count']
            .sum()
            .sort_values(ascending=False)
            .index
        )
        usage_data['Pitch Type'] = pd.Categorical(usage_data['Pitch Type'], categories=pitch_type_order, ordered=True)
        usage_data = usage_data.sort_values(['Pitch Type', 'Count'])

        usage_pivot = usage_data.pivot_table(
            index='Pitch Type', 
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
        st.markdown("<br>", unsafe_allow_html=True)

        filtered_data = filtered_data.sort_values(['GameID', 'Inning', 'Top/Bottom', 'PAofInning', 'PitchofPA']).reset_index(drop=True)
        filtered_data['Times Through Lineup'] = (
            filtered_data.groupby(['Pitcher', 'Batter'])
            .cumcount() + 1
        )

        filtered_data['Times Through Lineup'] = np.clip(filtered_data['Times Through Lineup'], 1, 3)
        ttl_data = filtered_data.groupby(['Times Through Lineup', 'Pitch Type']).size().reset_index(name='Pitch Count')
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
                    ttl_1_data['Pitch Type'].iloc[i],
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
                    ttl_2_data['Pitch Type'].iloc[i],
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
                    ttl_3_data['Pitch Type'].iloc[i],
                    ha='center', va='center', fontsize=30,
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.6)
                )
            st.pyplot(plt.gcf())
            plt.close()

        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown(
            "<h1 style='text-align: center; font-size: 40px; font-weight: bold;'>Pitch Usage After Each Pitch</h1>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        filtered_data = filtered_data.sort_values(['GameID', 'Inning', 'Top/Bottom', 'PAofInning', 'PitchofPA']).reset_index(drop=True)
        filtered_tab1_data_no_first_pitch = filtered_data[filtered_data['PitchofPA'] > 1]
        filtered_tab1_data_no_first_pitch['Next Pitch Type'] = filtered_tab1_data_no_first_pitch['Pitch Type'].shift(-1)

        filtered_tab1_data_no_first_pitch = filtered_tab1_data_no_first_pitch.dropna(subset=['Next Pitch Type'])

        next_pitch_data = filtered_tab1_data_no_first_pitch.groupby(['Pitch Type', 'Next Pitch Type']).size().reset_index(name='Count')
        next_pitch_data['Usage (%)'] = next_pitch_data.groupby('Pitch Type')['Count'].transform(lambda x: (x / x.sum()) * 100)

        next_pitch_data['Pitch Type'] = pd.Categorical(next_pitch_data['Pitch Type'], categories=pitch_type_order, ordered=True)

        unique_pitch_types = next_pitch_data['Pitch Type'].cat.categories
        percentage_distance = 0.5

        for pitch_type in unique_pitch_types:
            pitch_data = next_pitch_data[next_pitch_data['Pitch Type'] == pitch_type]
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

    with tab3:
        st.markdown(
            "<h2 style='text-align: center; font-weight: bold;'>Zone & Swing Metrics</h2>",
            unsafe_allow_html=True,
        )

        conditions = [
            (filtered_data['PlateLocHeight'] >= 1.5) & (filtered_data['PlateLocHeight'] <= 3.3) &
            (filtered_data['PlateLocSide'] >= -0.9) & (filtered_data['PlateLocSide'] <= 0.9)
        ]

        values = [1]

        filtered_data['Zone'] = np.select(conditions, values, default=0)

        filtered_data['Swing'] = filtered_data['PitchCall'].isin(['InPlay', 'StrikeSwinging', 'FoulBall']).astype(int)

        grouped = filtered_data.groupby(['Pitcher', 'Pitch Type'])

        result = grouped.size().reset_index(name='Pitches')
        result['Zone%'] = ((grouped.apply(lambda x: (x['Zone'] == 1).sum() / len(x)).values) * 100).round(1)
        result['Z-Swing%'] = ((grouped.apply(lambda x: ((x['Zone'] == 1) & (x['Swing'] == 1)).sum() / (x['Zone'] == 1).sum() if (x['Zone'] == 1).sum() > 0 else 0).values) * 100).round(1)
        result['Chase%'] = ((grouped.apply(lambda x: ((x['Zone'] == 0) & (x['Swing'] == 1)).sum() / (x['Zone'] == 0).sum() if (x['Zone'] == 0).sum() > 0 else 0).values) * 100).round(1)
        result['Whiff%'] = ((grouped.apply(lambda x: (x['PitchCall'] == 'StrikeSwinging').sum() / (x['Swing'] == 1).sum() if (x['Swing'] == 1).sum() > 0 else 0).values) * 100).round(1)

        result = result[['Pitch Type', 'Pitches', 'Zone%', 'Z-Swing%', 'Chase%', 'Whiff%']]
        result = result.sort_values(by='Pitches', ascending=False)

        fig, ax = plt.subplots(figsize=(8, len(result) * 2))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(
            cellText=result.values,
            colLabels=result.columns,
            cellLoc='center',
            loc='center',
            colColours=['#f0f0f0'] * len(result.columns),
            cellColours=[['#f0f0f0'] * len(result.columns) for _ in range(len(result))],
            bbox=[0, 0, 1, 1]
        )

        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(fontweight='bold', fontsize=10, color='black')
            else:
                cell.set_text_props(fontsize=10, color='black')

        table.auto_set_column_width(col=list(range(len(result.columns))))
        table.auto_set_font_size(False)

        st.pyplot(fig)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            "<h2 style='text-align: center; font-weight: bold;'>Batted Ball Metrics</h2>",
            unsafe_allow_html=True,
        )

        filtered_data['Batted Balls'] = (
            (filtered_data['TaggedHitType'].isin(['GroundBall', 'LineDrive', 'FlyBall', 'PopUp'])) &
            (filtered_data['PitchCall'] == 'InPlay')
        ).astype(int)

        grouped = filtered_data.groupby(['Pitcher', 'Pitch Type'])

        result = grouped['Batted Balls'].sum().reset_index(name='Batted Balls')

        hit_types = ['Ground Ball', 'Line Drive', 'Fly Ball', 'Pop Up']
        for hit_type in hit_types:
            result[hit_type + '%'] = grouped.apply(
                lambda x: (x['TaggedHitType'] == hit_type.replace(' ', '')).sum() / x['Batted Balls'].sum() * 100
            ).values.round(1)

        result = result[['Pitch Type', 'Batted Balls', 'Ground Ball%', 'Line Drive%', 'Fly Ball%', 'Pop Up%']]
        result = result.sort_values(by='Batted Balls', ascending=False)
        result = result[result['Batted Balls'] > 0]

        fig, ax = plt.subplots(figsize=(8, len(result) * 2))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(
            cellText=result.values,
            colLabels=result.columns,
            cellLoc='center',
            loc='center',
            colColours=['#f0f0f0'] * len(result.columns),
            cellColours=[['#f0f0f0'] * len(result.columns) for _ in range(len(result))],
            bbox=[0, 0, 1, 1]
        )

        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(fontweight='bold', fontsize=9, color='black')
            else:
                cell.set_text_props(fontsize=10, color='black')

        table.auto_set_column_width(col=list(range(len(result.columns))))
        table.auto_set_font_size(False)

        st.pyplot(fig)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            "<h2 style='text-align: center; font-weight: bold;'>Statcast Metrics</h2>",
            unsafe_allow_html=True,
        )

        grouped = filtered_data.groupby(['Pitcher', 'Pitch Type'])

        filtered_data['Hard Hit'] = (filtered_data['ExitSpeed'] >= 95) & (filtered_data['PitchCall'] == 'InPlay')
        filtered_data['Sweet Spot'] = filtered_data['Angle'].between(8, 32) & (filtered_data['PitchCall'] == 'InPlay')

        batted_balls_data = filtered_data[filtered_data['Batted Balls'] == 1]

        result = batted_balls_data.groupby(['Pitcher', 'Pitch Type']).agg(
            **{
                'Batted Balls': ('Batted Balls', 'sum'),
                'Avg EV': ('ExitSpeed', lambda x: x.mean().round(1)),
                'Avg LA': ('Angle', lambda x: x.mean().round(1)),
                'Hard Hit%': ('Hard Hit', lambda x: (x.mean() * 100).round(1)),
                'Sweet Spot%': ('Sweet Spot', lambda x: (x.mean() * 100).round(1))
            }
        ).reset_index()

        result = result[['Pitch Type', 'Batted Balls', 'Avg EV', 'Avg LA', 'Hard Hit%', 'Sweet Spot%']]
        result = result.sort_values(by='Batted Balls', ascending=False)

        fig, ax = plt.subplots(figsize=(8, len(result) * 2))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(
            cellText=result.values,
            colLabels=result.columns,
            cellLoc='center',
            loc='center',
            colColours=['#f0f0f0'] * len(result.columns),
            cellColours=[['#f0f0f0'] * len(result.columns) for _ in range(len(result))],
            bbox=[0, 0, 1, 1]
        )

        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(fontweight='bold', fontsize=10, color='black')
            else:
                cell.set_text_props(fontsize=10, color='black')

        table.auto_set_column_width(col=list(range(len(result.columns))))
        table.auto_set_font_size(False)

        st.pyplot(fig)

    with tab4:
        st.markdown(
            "<h2 style='text-align: center; font-weight: bold;'>Box Score Statistics</h2>",
            unsafe_allow_html=True,
        )

        filtered_data['Opponent'] = filtered_data['BatterTeam']
        filtered_data['Date'] = pd.to_datetime(filtered_data['Date'])
        filtered_data['Date'] = filtered_data['Date'].dt.date

        strike_condition = (filtered_data['Strikes'] == 2) & (filtered_data['PitchCall'].isin(['StrikeSwinging', 'StrikeCalled']))

        filtered_data['AdditionalOuts'] = strike_condition.astype(int)

        hit_condition = filtered_data['PlayResult'].isin(['Single', 'Double', 'Triple', 'Home Run'])

        filtered_data['Hit'] = hit_condition.astype(int)
        filtered_data['is_bb'] = (filtered_data['KorBB'] == 'Walk').astype(int)
        filtered_data['is_k'] = (filtered_data['KorBB'] == 'Strikeout').astype(int)

        result = filtered_data.groupby(['Pitcher', 'GameID', 'Opponent', 'Date']).agg(
            TotalOuts=('OutsOnPlay', 'sum'),
            AdditionalOuts=('AdditionalOuts', 'sum'),
            Hits=('Hit', 'sum'),
            Runs=('RunsScored', 'sum'),
            K=('is_k', 'sum'),
            BB=('is_bb', 'sum')
        ).reset_index()

        result['TotalOuts'] = result['TotalOuts'] + result['AdditionalOuts']

        result['FullInnings'] = result['TotalOuts'] // 3
        result['RemainingOuts'] = result['TotalOuts'] % 3

        fractional_innings_map = {0: '.0', 1: '.1', 2: '.2'}
        result['FractionalInnings'] = result['RemainingOuts'].map(fractional_innings_map)

        result['IP'] = result['FullInnings'].astype(str) + result['FractionalInnings']

        result = result[['Date', 'Opponent', 'IP', 'Hits', 'Runs', 'BB', 'K']]
        result = result.sort_values(by='Date', ascending=True)

        fig, ax = plt.subplots(figsize=(8, len(result) * 2))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(
            cellText=result.values,
            colLabels=result.columns,
            cellLoc='center',
            loc='center',
            colColours=['#f0f0f0'] * len(result.columns),
            cellColours=[['#f0f0f0'] * len(result.columns) for _ in range(len(result))],
            bbox=[0, 0, 1, 1]
        )

        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(fontweight='bold', fontsize=10, color='black')
            else:
                cell.set_text_props(fontsize=10, color='black')

        table.auto_set_column_width(col=list(range(len(result.columns))))
        table.auto_set_font_size(False)

        st.pyplot(fig)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            "<h2 style='text-align: center; font-weight: bold;'>Zone & Swing Metrics</h2>",
            unsafe_allow_html=True,
        )

        conditions = [
            (filtered_data['PlateLocHeight'] >= 1.5) & (filtered_data['PlateLocHeight'] <= 3.3) &
            (filtered_data['PlateLocSide'] >= -0.9) & (filtered_data['PlateLocSide'] <= 0.9)
        ]

        values = [1]

        filtered_data['Zone'] = np.select(conditions, values, default=0)

        filtered_data['Swing'] = filtered_data['PitchCall'].isin(['InPlay', 'StrikeSwinging', 'FoulBall']).astype(int)

        grouped = filtered_data.groupby(['Pitcher', 'GameID', 'Opponent', 'Date'])

        result = grouped.size().reset_index(name='Pitches')
        result['Zone%'] = ((grouped.apply(lambda x: (x['Zone'] == 1).sum() / len(x)).values) * 100).round(1)
        result['Z-Swing%'] = ((grouped.apply(lambda x: ((x['Zone'] == 1) & (x['Swing'] == 1)).sum() / (x['Zone'] == 1).sum() if (x['Zone'] == 1).sum() > 0 else 0).values) * 100).round(1)
        result['Chase%'] = ((grouped.apply(lambda x: ((x['Zone'] == 0) & (x['Swing'] == 1)).sum() / (x['Zone'] == 0).sum() if (x['Zone'] == 0).sum() > 0 else 0).values) * 100).round(1)
        result['Whiff%'] = ((grouped.apply(lambda x: (x['PitchCall'] == 'StrikeSwinging').sum() / (x['Swing'] == 1).sum() if (x['Swing'] == 1).sum() > 0 else 0).values) * 100).round(1)

        result = result[['Date', 'Opponent', 'Pitches', 'Zone%', 'Z-Swing%', 'Chase%', 'Whiff%']]
        result = result.sort_values(by='Date', ascending=True)

        fig, ax = plt.subplots(figsize=(8, len(result) * 2))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(
            cellText=result.values,
            colLabels=result.columns,
            cellLoc='center',
            loc='center',
            colColours=['#f0f0f0'] * len(result.columns),
            cellColours=[['#f0f0f0'] * len(result.columns) for _ in range(len(result))],
            bbox=[0, 0, 1, 1]
        )

        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(fontweight='bold', fontsize=10, color='black')
            else:
                cell.set_text_props(fontsize=10, color='black')

        table.auto_set_column_width(col=list(range(len(result.columns))))
        table.auto_set_font_size(False)

        st.pyplot(fig)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            "<h2 style='text-align: center; font-weight: bold;'>Batted Ball Metrics</h2>",
            unsafe_allow_html=True,
        )

        filtered_data['Batted Balls'] = filtered_data['TaggedHitType'].isin(['GroundBall', 'LineDrive', 'FlyBall', 'PopUp']).astype(int)

        grouped = filtered_data.groupby(['Pitcher', 'GameID', 'Opponent', 'Date'])

        result = grouped['Batted Balls'].sum().reset_index(name='Batted Balls')

        hit_types = ['Ground Ball', 'Line Drive', 'Fly Ball', 'Pop Up']
        for hit_type in hit_types:
            result[hit_type + '%'] = grouped.apply(
                lambda x: (x['TaggedHitType'] == hit_type.replace(' ', '')).sum() / x['Batted Balls'].sum() * 100
            ).values.round(1)

        result = result[['Date', 'Opponent', 'Batted Balls', 'Ground Ball%', 'Line Drive%', 'Fly Ball%', 'Pop Up%']]
        result = result.sort_values(by='Date', ascending=True)

        fig, ax = plt.subplots(figsize=(8, len(result) * 2))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(
            cellText=result.values,
            colLabels=result.columns,
            cellLoc='center',
            loc='center',
            colColours=['#f0f0f0'] * len(result.columns),
            cellColours=[['#f0f0f0'] * len(result.columns) for _ in range(len(result))],
            bbox=[0, 0, 1, 1]
        )

        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(fontweight='bold', fontsize=9, color='black')
            else:
                cell.set_text_props(fontsize=10, color='black')

        table.auto_set_column_width(col=list(range(len(result.columns))))
        table.auto_set_font_size(False)

        st.pyplot(fig)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            "<h2 style='text-align: center; font-weight: bold;'>Statcast Metrics</h2>",
            unsafe_allow_html=True,
        )

        grouped = filtered_data.groupby(['Pitcher', 'GameID', 'Opponent', 'Date'])

        result = grouped['Batted Balls'].sum().reset_index(name='Batted Balls')

        result['Avg EV'] = grouped['ExitSpeed'].mean().round(1).values
        result['Avg LA'] = grouped['Angle'].mean().round(1).values

        result['Hard Hit%'] = grouped.apply(
            lambda x: (x['ExitSpeed'] >= 95).sum() / x['Batted Balls'].sum() * 100
        ).round(1).values

        result['Sweet Spot%'] = grouped.apply(
            lambda x: ((x['Angle'] >= 8) & (x['Angle'] <= 32)).sum() / x['Batted Balls'].sum() * 100
        ).round(1).values

        result = result[['Date', 'Opponent', 'Batted Balls', 'Avg EV', 'Avg LA', 'Hard Hit%', 'Sweet Spot%']]
        result = result.sort_values(by='Date', ascending=True)

        fig, ax = plt.subplots(figsize=(8, len(result) * 2))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(
            cellText=result.values,
            colLabels=result.columns,
            cellLoc='center',
            loc='center',
            colColours=['#f0f0f0'] * len(result.columns),
            cellColours=[['#f0f0f0'] * len(result.columns) for _ in range(len(result))],
            bbox=[0, 0, 1, 1]
        )

        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(fontweight='bold', fontsize=10, color='black')
            else:
                cell.set_text_props(fontsize=10, color='black')

        table.auto_set_column_width(col=list(range(len(result.columns))))
        table.auto_set_font_size(False)

        st.pyplot(fig)
else:
    unique_batters = sorted(data['Batter'].unique(), key=lambda name: name.split()[-1])
    selected_batter = st.sidebar.selectbox("Select a Batter", unique_batters)

    batter_data = data[data['Batter'] == selected_batter]

    batter_data['Year'] = pd.to_datetime(batter_data['Date']).dt.year
    unique_years = ["All"] + sorted(batter_data['Year'].unique(), reverse=True)

    has_game_data = not batter_data.empty
    default_index = 1 if has_game_data else 2

    selected_year = st.sidebar.selectbox("Select Season", unique_years, index=default_index)

    if selected_year != "All":
        batter_data = batter_data[batter_data['Year'] == selected_year]

    if batter_data.empty:
        st.markdown(
            "<h1 style='text-align: center; font-size: 40px; color: black;'>No data ☹️</h1>",
            unsafe_allow_html=True
        )
        st.stop()

    batter_data['Date'] = pd.to_datetime(batter_data['Date'], errors='coerce')

    min_date = batter_data['Date'].min()
    max_date = batter_data['Date'].max()

    selected_date_range = st.sidebar.date_input(
        "Select Date Range",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

    unique_game_ids = batter_data[['GameID', 'Date', 'AwayTeam']].drop_duplicates()

    unique_game_ids = unique_game_ids.sort_values('Date', ascending=False)

    unique_game_ids['GameID_Display'] = unique_game_ids.apply(
        lambda row: f"{row['AwayTeam']} ({row['Date'].strftime('%Y-%m-%d')})", axis=1
    )

    game_id_options = dict(zip(unique_game_ids['GameID_Display'], unique_game_ids['GameID']))

    selected_game_id_display = st.sidebar.selectbox(
        "Select a Game", options=["None"] + list(game_id_options.keys())
    )

    batter_data['Date'] = pd.to_datetime(batter_data['Date'], errors='coerce')

    filtered_data = batter_data[
        (batter_data['Date'] >= pd.Timestamp(selected_date_range[0])) &
        (batter_data['Date'] <= pd.Timestamp(selected_date_range[1]))
    ]

    st.markdown(
        f"<h1 style='text-align: center;'>{selected_batter}</h1>",
        unsafe_allow_html=True,
    )

    pitcher_hand_options = ['All', 'R', 'L']
    selected_pitcher_hand = st.sidebar.selectbox(
        "Select Pitcher Handedness",
        pitcher_hand_options,
        index=0,
        key="tab1_pitcher_hand_selectbox"
    )

    if selected_pitcher_hand == 'R':
        filtered_data = filtered_data[filtered_data['PitcherThrows'] == 'Right']
    elif selected_pitcher_hand == 'L':
        filtered_data = filtered_data[filtered_data['PitcherThrows'] == 'Lefty']

    if filtered_data.empty:
        st.markdown(
            "<h1 style='text-align: center; font-size: 40px; color: black;'>No data ☹️</h1>",
            unsafe_allow_html=True
        )
        st.stop()

    description_text = f"Data From {batter_data['Date'].min().date()} to {batter_data['Date'].max().date()}"

    st.markdown(
        f"<p style='text-align: center; font-size: 18px;'>{description_text}</p>",
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["Game Logs", "Batter Hot Zones"])
            
    with tab1:
        st.markdown(
            "<h2 style='text-align: center; font-weight: bold;'>Box Score Statistics</h2>",
            unsafe_allow_html=True,
        )

        filtered_data['Opponent'] = filtered_data['PitcherTeam']
        filtered_data['Date'] = pd.to_datetime(filtered_data['Date'])
        filtered_data['Date'] = filtered_data['Date'].dt.date

        strike_condition = (filtered_data['Strikes'] == 2) & (filtered_data['PitchCall'].isin(['StrikeSwinging', 'StrikeCalled']))

        filtered_data['AdditionalOuts'] = strike_condition.astype(int)

        hit_condition = filtered_data['PlayResult'].isin(['Single', 'Double', 'Triple', 'Home Run'])

        filtered_data['Hit'] = hit_condition.astype(int)
        filtered_data['is_bb'] = (filtered_data['KorBB'] == 'Walk').astype(int)
        filtered_data['is_k'] = (filtered_data['KorBB'] == 'Strikeout').astype(int)

        result = filtered_data.groupby(['Batter', 'GameID', 'Opponent', 'Date']).agg(
            PA=('PitchofPA', lambda x: (x == 1).sum()),
            Hits=('Hit', 'sum'),
            Runs=('RunsScored', 'sum'),
            K=('is_k', 'sum'),
            BB=('is_bb', 'sum'),
            Doubles=('PlayResult', lambda x: (x == 'Double').sum()),
            Triples=('PlayResult', lambda x: (x == 'Triple').sum()),
            HR=('PlayResult', lambda x: (x == 'Home Run').sum())
        ).reset_index()

        result = result.rename(columns={
            'Doubles': '2B',
            'Triples': '3B',
        })

        result = result[['Date', 'Opponent', 'PA',  'Hits', 'Runs', 'BB', 'K', '2B', '3B', 'HR']]
        result = result.sort_values(by='Date', ascending=True)

        fig, ax = plt.subplots(figsize=(8, len(result) * 2))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(
            cellText=result.values,
            colLabels=result.columns,
            cellLoc='center',
            loc='center',
            colColours=['#f0f0f0'] * len(result.columns),
            cellColours=[['#f0f0f0'] * len(result.columns) for _ in range(len(result))],
            bbox=[0, 0, 1, 1]
        )

        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(fontweight='bold', fontsize=10, color='black')
            else:
                cell.set_text_props(fontsize=10, color='black')

        table.auto_set_column_width(col=list(range(len(result.columns))))
        table.auto_set_font_size(False)

        st.pyplot(fig)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            "<h2 style='text-align: center; font-weight: bold;'>Zone & Swing Metrics</h2>",
            unsafe_allow_html=True,
        )

        conditions = [
            (filtered_data['PlateLocHeight'] >= 1.5) & (filtered_data['PlateLocHeight'] <= 3.3) &
            (filtered_data['PlateLocSide'] >= -0.9) & (filtered_data['PlateLocSide'] <= 0.9)
        ]

        values = [1]

        filtered_data['Zone'] = np.select(conditions, values, default=0)

        filtered_data['Swing'] = filtered_data['PitchCall'].isin(['InPlay', 'StrikeSwinging', 'FoulBall']).astype(int)

        grouped = filtered_data.groupby(['Batter', 'GameID', 'Opponent', 'Date'])

        result = grouped.size().reset_index(name='Pitches')
        result['Zone%'] = ((grouped.apply(lambda x: (x['Zone'] == 1).sum() / len(x)).values) * 100).round(1)
        result['Z-Swing%'] = ((grouped.apply(lambda x: ((x['Zone'] == 1) & (x['Swing'] == 1)).sum() / (x['Zone'] == 1).sum() if (x['Zone'] == 1).sum() > 0 else 0).values) * 100).round(1)
        result['Chase%'] = ((grouped.apply(lambda x: ((x['Zone'] == 0) & (x['Swing'] == 1)).sum() / (x['Zone'] == 0).sum() if (x['Zone'] == 0).sum() > 0 else 0).values) * 100).round(1)
        result['Whiff%'] = ((grouped.apply(lambda x: (x['PitchCall'] == 'StrikeSwinging').sum() / (x['Swing'] == 1).sum() if (x['Swing'] == 1).sum() > 0 else 0).values) * 100).round(1)

        result = result[['Date', 'Opponent', 'Pitches', 'Zone%', 'Z-Swing%', 'Chase%', 'Whiff%']]
        result = result.sort_values(by='Date', ascending=True)

        fig, ax = plt.subplots(figsize=(8, len(result) * 2))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(
            cellText=result.values,
            colLabels=result.columns,
            cellLoc='center',
            loc='center',
            colColours=['#f0f0f0'] * len(result.columns),
            cellColours=[['#f0f0f0'] * len(result.columns) for _ in range(len(result))],
            bbox=[0, 0, 1, 1]
        )

        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(fontweight='bold', fontsize=10, color='black')
            else:
                cell.set_text_props(fontsize=10, color='black')

        table.auto_set_column_width(col=list(range(len(result.columns))))
        table.auto_set_font_size(False)

        st.pyplot(fig)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            "<h2 style='text-align: center; font-weight: bold;'>Batted Ball Metrics</h2>",
            unsafe_allow_html=True,
        )

        filtered_data['Batted Balls'] = filtered_data['TaggedHitType'].isin(['GroundBall', 'LineDrive', 'FlyBall', 'PopUp']).astype(int)

        grouped = filtered_data.groupby(['Batter', 'GameID', 'Opponent', 'Date'])

        result = grouped['Batted Balls'].sum().reset_index(name='Batted Balls')

        hit_types = ['Ground Ball', 'Line Drive', 'Fly Ball', 'Pop Up']
        for hit_type in hit_types:
            result[hit_type + '%'] = grouped.apply(
                lambda x: (x['TaggedHitType'] == hit_type.replace(' ', '')).sum() / x['Batted Balls'].sum() * 100
            ).values.round(1)

        result = result[['Date', 'Opponent', 'Batted Balls', 'Ground Ball%', 'Line Drive%', 'Fly Ball%', 'Pop Up%']]
        result = result.sort_values(by='Date', ascending=True)

        fig, ax = plt.subplots(figsize=(8, len(result) * 2))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(
            cellText=result.values,
            colLabels=result.columns,
            cellLoc='center',
            loc='center',
            colColours=['#f0f0f0'] * len(result.columns),
            cellColours=[['#f0f0f0'] * len(result.columns) for _ in range(len(result))],
            bbox=[0, 0, 1, 1]
        )

        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(fontweight='bold', fontsize=9, color='black')
            else:
                cell.set_text_props(fontsize=10, color='black')

        table.auto_set_column_width(col=list(range(len(result.columns))))
        table.auto_set_font_size(False)

        st.pyplot(fig)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            "<h2 style='text-align: center; font-weight: bold;'>Statcast Metrics</h2>",
            unsafe_allow_html=True,
        )

        grouped = filtered_data.groupby(['Batter', 'GameID', 'Opponent', 'Date'])

        result = grouped['Batted Balls'].sum().reset_index(name='Batted Balls')

        result['Avg EV'] = grouped['ExitSpeed'].mean().round(1).values
        result['Avg LA'] = grouped['Angle'].mean().round(1).values

        result['Hard Hit%'] = grouped.apply(
            lambda x: (x['ExitSpeed'] >= 95).sum() / x['Batted Balls'].sum() * 100
        ).round(1).values

        result['Sweet Spot%'] = grouped.apply(
            lambda x: ((x['Angle'] >= 8) & (x['Angle'] <= 32)).sum() / x['Batted Balls'].sum() * 100
        ).round(1).values

        result = result[['Date', 'Opponent', 'Batted Balls', 'Avg EV', 'Avg LA', 'Hard Hit%', 'Sweet Spot%']]
        result = result.sort_values(by='Date', ascending=True)

        fig, ax = plt.subplots(figsize=(8, len(result) * 2))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(
            cellText=result.values,
            colLabels=result.columns,
            cellLoc='center',
            loc='center',
            colColours=['#f0f0f0'] * len(result.columns),
            cellColours=[['#f0f0f0'] * len(result.columns) for _ in range(len(result))],
            bbox=[0, 0, 1, 1]
        )

        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(fontweight='bold', fontsize=10, color='black')
            else:
                cell.set_text_props(fontsize=10, color='black')

        table.auto_set_column_width(col=list(range(len(result.columns))))
        table.auto_set_font_size(False)

        st.pyplot(fig)