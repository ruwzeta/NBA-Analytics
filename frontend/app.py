from flask import Flask, render_template
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg') # Set backend before importing pyplot
import matplotlib.pyplot as plt
import io
import base64
import numpy as np # Already imported, but good to note for np.number

app = Flask(__name__)

def load_and_prepare_data():
    """
    Loads, merges, and prepares NBA player statistics from CSV files.
    """
    # Define file paths (relative to app.py in the frontend directory)
    base_path = os.path.join(os.path.dirname(__file__), '..', 'PyScripts')
    per_game_file = os.path.join(base_path, 'nba_per_game_stats_2024_25.csv')
    advanced_file = os.path.join(base_path, 'nba_advanced_stats_2024_25.csv')
    shooting_file = os.path.join(base_path, 'nba_shooting_stats_2024_25.csv')

    df_per_game = None
    df_advanced = None
    df_shooting = None

    try:
        df_per_game = pd.read_csv(per_game_file)
        print(f"Successfully loaded per_game_stats from {per_game_file}")
    except FileNotFoundError:
        print(f"Error: {per_game_file} not found.")
        # Allow continuation if some files are missing, will result in partial data
    
    try:
        df_advanced = pd.read_csv(advanced_file)
        print(f"Successfully loaded advanced_stats from {advanced_file}")
    except FileNotFoundError:
        print(f"Error: {advanced_file} not found.")

    try:
        df_shooting = pd.read_csv(shooting_file)
        print(f"Successfully loaded shooting_stats from {shooting_file}")
    except FileNotFoundError:
        print(f"Error: {shooting_file} not found.")

    # Initialize merged_df with an empty DataFrame or the first successfully loaded one
    if df_per_game is not None:
        merged_df = df_per_game.copy()
        # Rename MP from per_game to distinguish if other MP columns exist
        if 'MP' in merged_df.columns:
            merged_df.rename(columns={'MP': 'MP_per_game'}, inplace=True)
    else:
        merged_df = pd.DataFrame()

    # Merge with Advanced Stats
    if df_advanced is not None:
        # Define common keys for merging with advanced stats
        common_keys_adv = ['Player', 'Pos', 'Age', 'Team', 'G'] 
        # Ensure all keys exist in both dataframes before merging
        keys_for_adv_merge = [key for key in common_keys_adv if key in merged_df.columns and key in df_advanced.columns]
        
        if not keys_for_adv_merge: # If no common keys (e.g., merged_df is empty)
             if merged_df.empty:
                 merged_df = df_advanced.copy()
             else: # merged_df has columns but no common ones with df_advanced for merge
                 print("Warning: No common keys to merge df_per_game and df_advanced. Advanced stats might not be merged correctly.")
        else:
            # Rename advanced_df's MP if it exists, to avoid conflict, assuming it's total MP
            if 'MP' in df_advanced.columns:
                df_advanced = df_advanced.rename(columns={'MP': 'MP_total_adv'})
            
            # Drop 'GS' from advanced if it exists to avoid suffix, as 'G' is already a key
            df_advanced_to_merge = df_advanced.drop(columns=['GS'], errors='ignore')

            if merged_df.empty: # If df_per_game was not loaded
                merged_df = df_advanced_to_merge
            else:
                merged_df = pd.merge(merged_df, df_advanced_to_merge, on=keys_for_adv_merge, how='outer', suffixes=('', '_adv'))
        print(f"Shape after merging advanced stats: {merged_df.shape}")

    # Merge with Shooting Stats
    if df_shooting is not None:
        common_keys_shoot = ['Player', 'Pos', 'Age', 'Team', 'G', 'GS']
        keys_for_shoot_merge = [key for key in common_keys_shoot if key in merged_df.columns and key in df_shooting.columns]

        if not keys_for_shoot_merge:
            if merged_df.empty:
                merged_df = df_shooting.copy()
            else:
                print("Warning: No common keys to merge with df_shooting. Shooting stats might not be merged correctly.")
        else:
            # Rename shooting_df's MP if it exists, to avoid conflict, assuming it's total MP
            if 'MP' in df_shooting.columns:
                 df_shooting = df_shooting.rename(columns={'MP': 'MP_total_shoot'})

            if merged_df.empty: # If per_game and advanced were not loaded
                merged_df = df_shooting
            else:
                merged_df = pd.merge(merged_df, df_shooting, on=keys_for_shoot_merge, how='outer', suffixes=('', '_shoot'))
        print(f"Shape after merging shooting stats: {merged_df.shape}")
    
    if merged_df.empty:
        print("No data loaded. Returning empty DataFrame.")
        return pd.DataFrame()

    # Consolidate MP_total columns
    if 'MP_total_adv' in merged_df.columns and 'MP_total_shoot' in merged_df.columns:
        merged_df['MP_total'] = merged_df['MP_total_adv'].fillna(merged_df['MP_total_shoot'])
        merged_df.drop(columns=['MP_total_adv', 'MP_total_shoot'], inplace=True, errors='ignore')
    elif 'MP_total_adv' in merged_df.columns:
        merged_df.rename(columns={'MP_total_adv': 'MP_total'}, inplace=True)
    elif 'MP_total_shoot' in merged_df.columns:
        merged_df.rename(columns={'MP_total_shoot': 'MP_total'}, inplace=True)

    # Clean up suffixed columns - prefer original if no suffix, then no suffix, then _adv, then _shoot
    # Example: 'Awards' might become 'Awards_adv' or 'Awards_shoot'
    for col_base in ['Awards', 'Rk', 'FG%', '3P%', '2P%', 'FT%']: # Add other potential conflicting columns
        suffixed_cols = [f"{col_base}_adv", f"{col_base}_shoot"]
        if col_base in merged_df.columns: # Original from per_game (or first loaded) is preferred
            for suff_col in suffixed_cols:
                if suff_col in merged_df.columns:
                    merged_df[col_base] = merged_df[col_base].fillna(merged_df[suff_col])
                    merged_df.drop(columns=[suff_col], inplace=True, errors='ignore')
        elif f"{col_base}_adv" in merged_df.columns: # Prefer _adv if original doesn't exist
            merged_df.rename(columns={f"{col_base}_adv": col_base}, inplace=True)
            if f"{col_base}_shoot" in merged_df.columns:
                 merged_df[col_base] = merged_df[col_base].fillna(merged_df[f"{col_base}_shoot"])
                 merged_df.drop(columns=[f"{col_base}_shoot"], inplace=True, errors='ignore')
        elif f"{col_base}_shoot" in merged_df.columns: # Use _shoot if no other
            merged_df.rename(columns={f"{col_base}_shoot": col_base}, inplace=True)
            
    # Identify numeric columns for filling NaNs
    # Exclude known non-numeric columns like 'Player', 'Pos', 'Team', 'Awards'
    potential_numeric_cols = merged_df.select_dtypes(include=np.number).columns.tolist()
    # Or, be more explicit based on expected stats columns:
    known_stat_cols = [ # A subset of expected numeric columns
        'Age', 'G', 'GS', 'MP_per_game', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 
        '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 
        'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'MP_total', 'PER', 'TS%', 
        '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 
        'USG%', 'OWS', 'DWS', 'WS', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'VORP', 
        'Dist.', '%FGA 2P', '%FGA 0-3', '%FGA 3-10', '%FGA 10-16', '%FGA 16-3P', '%FGA 3P', 
        'FG% 0-3', 'FG% 3-10', 'FG% 10-16', 'FG% 16-3P', '%Ast\'d 2P', '%Ast\'d 3P', 
        'Dunk %FGA', 'Dunks #', 'Corner 3s %3PA', 'Corner 3s 3P%', 'Heaves Att.', 'Heaves Md.'
    ]
    
    numeric_cols_to_fill = [col for col in known_stat_cols if col in merged_df.columns]
    
    for col in numeric_cols_to_fill:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').fillna(0)

    # Drop rows where Player name is NaN (can happen with outer merges if a player is in one file but not primary)
    merged_df.dropna(subset=['Player'], inplace=True)
    # Remove duplicate player entries if any, keeping the first one (can happen with TOT if not handled)
    # For players traded mid-season, Basketball-Reference often includes a 'TOT' row for their total stats across teams.
    # We should prioritize 'TOT' rows if they exist for a player, or ensure player-team combination is unique for non-TOT.
    # A simple way is to drop duplicates keeping the one with most games played or total minutes if 'TOT' is consistently used.
    # For now, let's sort by MP_total (if available) and G to keep more complete records first.
    if 'MP_total' in merged_df.columns:
        merged_df.sort_values(by=['Player', 'MP_total'], ascending=[True, False], inplace=True)
    elif 'MP_per_game' in merged_df.columns and 'G' in merged_df.columns:
         merged_df.sort_values(by=['Player', 'MP_per_game', 'G'], ascending=[True, False, False], inplace=True)
    
    # Drop duplicates based on 'Player', keeping the first entry which should be the 'TOT' or most significant record.
    # This is a simplification; more sophisticated handling of traded players might be needed for precise analysis.
    merged_df.drop_duplicates(subset=['Player'], keep='first', inplace=True)
    
    print(f"Final merged DataFrame shape: {merged_df.shape}")
    return merged_df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/leaders')
def leaders():
    df = load_and_prepare_data()

    if df is None or df.empty:
        return "Error: Data could not be loaded or is empty. Cannot display leaders."

    leaderboards = {}
    
    # Ensure columns exist before trying to sort or filter
    if 'PTS' in df.columns:
        leaderboards['pts_leaders'] = df.sort_values(by='PTS', ascending=False).head(10)[['Player', 'Team', 'PTS']]
    else:
        leaderboards['pts_leaders'] = pd.DataFrame(columns=['Player', 'Team', 'PTS']) # Empty df

    if 'AST' in df.columns:
        leaderboards['ast_leaders'] = df.sort_values(by='AST', ascending=False).head(10)[['Player', 'Team', 'AST']]
    else:
        leaderboards['ast_leaders'] = pd.DataFrame(columns=['Player', 'Team', 'AST'])

    if 'TRB' in df.columns:
        leaderboards['trb_leaders'] = df.sort_values(by='TRB', ascending=False).head(10)[['Player', 'Team', 'TRB']]
    else:
        leaderboards['trb_leaders'] = pd.DataFrame(columns=['Player', 'Team', 'TRB'])

    # For PER and WS, filter by minutes played or games played
    # load_and_prepare_data creates 'MP_total' and also has 'G'
    min_mp_total = 500
    min_games = 20 # Fallback if MP_total is not reliable or available

    filtered_df_for_advanced = df.copy()
    if 'MP_total' in filtered_df_for_advanced.columns:
        filtered_df_for_advanced = filtered_df_for_advanced[filtered_df_for_advanced['MP_total'] > min_mp_total]
    elif 'G' in filtered_df_for_advanced.columns: # Fallback to games played
        filtered_df_for_advanced = filtered_df_for_advanced[filtered_df_for_advanced['G'] > min_games]
    else: # No filter applicable
        print("Warning: MP_total and G not available for filtering PER/WS leaders.")
        # leaderboards will be based on unfiltered data if this happens

    if 'PER' in filtered_df_for_advanced.columns:
        leaderboards['per_leaders'] = filtered_df_for_advanced.sort_values(by='PER', ascending=False).head(10)[['Player', 'Team', 'PER', 'MP_total' if 'MP_total' in filtered_df_for_advanced else 'G']]
    else:
        leaderboards['per_leaders'] = pd.DataFrame(columns=['Player', 'Team', 'PER', 'FilterCriteria'])


    if 'WS' in filtered_df_for_advanced.columns:
        leaderboards['ws_leaders'] = filtered_df_for_advanced.sort_values(by='WS', ascending=False).head(10)[['Player', 'Team', 'WS', 'MP_total' if 'MP_total' in filtered_df_for_advanced else 'G']]
    else:
        leaderboards['ws_leaders'] = pd.DataFrame(columns=['Player', 'Team', 'WS', 'FilterCriteria'])
        
    return render_template('leaders.html', leaderboards=leaderboards)

if __name__ == '__main__':
    # Make sure to run from the root directory of the project if PyScripts is a sibling module
    # For development, if running app.py directly from frontend folder, ensure PyScripts is in PYTHONPATH
    # Or adjust path to CSVs in load_and_prepare_data if needed for direct execution context.
    # The os.path.join in load_and_prepare_data should handle relative paths correctly if app.py is run.
    app.run(debug=True, host='0.0.0.0', port=5001)


@app.route('/shooting')
def shooting_performance():
    df = load_and_prepare_data()

    if df is None or df.empty:
        return "Error: Data could not be loaded or is empty. Cannot display shooting performance."

    shooting_data = {}
    plots = {}

    # Calculate Total FGA and Total 3PA for filtering
    if 'FGA' in df.columns and 'G' in df.columns:
        df['Total_FGA'] = df['FGA'] * df['G']
    else:
        df['Total_FGA'] = 0 # Avoids error if columns missing
        print("Warning: 'FGA' or 'G' not available for Total_FGA calculation.")
        
    if '3PA' in df.columns and 'G' in df.columns:
        df['Total_3PA'] = df['3PA'] * df['G']
    else:
        df['Total_3PA'] = 0
        print("Warning: '3PA' or 'G' not available for Total_3PA calculation.")


    # 1. Top 3P% Shooters (Overall)
    if '3P%' in df.columns and 'Total_3PA' in df.columns:
        filtered_3p = df[df['Total_3PA'] > 50]
        shooting_data['top_3p_shooters'] = filtered_3p.sort_values(by='3P%', ascending=False).head(10)[['Player', 'Team', '3P%', 'Total_3PA']]
    else:
        shooting_data['top_3p_shooters'] = pd.DataFrame(columns=['Player', 'Team', '3P%', 'Total_3PA'])

    # 2. Top FG% by Distance (0-3ft)
    # Assumes 'pct_fga_00_03' is like '%FGA 0-3' in shooting_df (decimal)
    if 'FG% 0-3' in df.columns and '%FGA 0-3' in df.columns and 'Total_FGA' in df.columns:
        df['FGA_0_3_Attempts'] = df['%FGA 0-3'] * df['Total_FGA']
        filtered_fg_0_3 = df[df['FGA_0_3_Attempts'] > 30]
        shooting_data['top_fg_0_3'] = filtered_fg_0_3.sort_values(by='FG% 0-3', ascending=False).head(10)[['Player', 'Team', 'FG% 0-3', 'FGA_0_3_Attempts']]
    else:
        shooting_data['top_fg_0_3'] = pd.DataFrame(columns=['Player', 'Team', 'FG% 0-3', 'FGA_0_3_Attempts'])
        print("Warning: Columns for FG% 0-3ft analysis missing.")

    # 3. Top Corner 3P% Shooters
    # Assumes 'pct_fg3a_corner3' is like 'Corner 3s %3PA' (decimal) and 'fg3_pct_corner3' is 'Corner 3s 3P%'
    if 'Corner 3s 3P%' in df.columns and 'Corner 3s %3PA' in df.columns and 'Total_3PA' in df.columns:
        df['Corner_3PA_Attempts'] = df['Corner 3s %3PA'] * df['Total_3PA']
        filtered_corner_3p = df[df['Corner_3PA_Attempts'] > 20]
        shooting_data['top_corner_3p'] = filtered_corner_3p.sort_values(by='Corner 3s 3P%', ascending=False).head(10)[['Player', 'Team', 'Corner 3s 3P%', 'Corner_3PA_Attempts']]
    else:
        shooting_data['top_corner_3p'] = pd.DataFrame(columns=['Player', 'Team', 'Corner 3s 3P%', 'Corner_3PA_Attempts'])
        print("Warning: Columns for Corner 3P% analysis missing.")

    # 4. Top eFG% Shooters
    if 'eFG%' in df.columns and 'Total_FGA' in df.columns:
        filtered_efg = df[df['Total_FGA'] > 100]
        shooting_data['top_efg'] = filtered_efg.sort_values(by='eFG%', ascending=False).head(10)[['Player', 'Team', 'eFG%', 'Total_FGA']]
    else:
        shooting_data['top_efg'] = pd.DataFrame(columns=['Player', 'Team', 'eFG%', 'Total_FGA'])

    # 5. Top TS% Shooters
    if 'TS%' in df.columns and 'Total_FGA' in df.columns:
        filtered_ts = df[df['Total_FGA'] > 100]
        shooting_data['top_ts'] = filtered_ts.sort_values(by='TS%', ascending=False).head(10)[['Player', 'Team', 'TS%', 'Total_FGA']]
    else:
        shooting_data['top_ts'] = pd.DataFrame(columns=['Player', 'Team', 'TS%', 'Total_FGA'])

    # 6. Average Shot Distance Distribution Plot
    if 'Dist.' in df.columns and 'G' in df.columns:
        dist_data = df[df['G'] > 10]['Dist.'].dropna()
        if not dist_data.empty:
            img = io.BytesIO()
            plt.figure(figsize=(8, 5))
            sns.histplot(dist_data, kde=True, bins=20)
            plt.title('Distribution of Average Shot Distance (ft)')
            plt.xlabel('Average Shot Distance (ft)')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(img, format='png')
            plt.close() 
            img.seek(0)
            plots['avg_shot_dist_plot'] = base64.b64encode(img.getvalue()).decode('utf8')
        else:
            plots['avg_shot_dist_plot'] = None
            print("Warning: Not enough data for Average Shot Distance plot after filtering.")
    else:
        plots['avg_shot_dist_plot'] = None
        print("Warning: 'Dist.' or 'G' column not found for plot generation.")

    return render_template('shooting.html', shooting_data=shooting_data, plots=plots)
