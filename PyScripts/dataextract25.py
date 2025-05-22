import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re
from pymongo.errors import ConnectionFailure, OperationFailure
from MongoDB import get_mongo_db # Import the function to get db object

def scrape_advanced_stats(year='2025'):
    """
    Scrape advanced NBA stats from Basketball Reference for a specific season.
    
    Args:
        year (str): The year representing the end of the season (e.g., '2025' for the 2024-25 season)
    
    Returns:
        DataFrame: A pandas DataFrame containing the advanced stats
    """
    # Construct the URL based on the provided year
    url = f'https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html'
    
    # Add headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Make the request
    print(f"Fetching data from {url}...")
    response = requests.get(url, headers=headers)
    
    # Check if the request was successful
    if response.status_code != 200:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
        return None
    
    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the main table with the advanced stats
    table = soup.find('table', id='advanced')
    
    if not table:
        print("Could not find the advanced stats table on the page.")
        return None
    
    # Extract the table headers
    headers = []
    for th in table.find('thead').find_all('th'):
        if th.get('data-stat'):
            # Clean up header text (remove asterisks and other special characters)
            header_text = re.sub(r'[^a-zA-Z0-9_]', '', th.text.strip())
            if not header_text:  # If empty after cleaning, use the data-stat attribute
                header_text = th.get('data-stat')
            headers.append(header_text)
    
    # Extract the data rows
    rows_data = []
    
    # Process all tbody elements (may contain multiple sections)
    for tbody in table.find_all('tbody'):
        rows = tbody.find_all('tr')
        
        for row in rows:
            # Skip header rows that might be interspersed in the table
            if row.get('class') and 'thead' in row.get('class'):
                continue
                
            # Extract player data
            player_data = {}
            
            # Go through each cell in the row
            cells = row.find_all(['th', 'td'])
            for i, cell in enumerate(cells):
                stat_category = cell.get('data-stat')
                
                # If it's a valid stat category and we have a corresponding header
                if stat_category and i < len(headers):
                    # Handle player name and link separately
                    if stat_category == 'player':
                        player_link = cell.find('a')
                        if player_link:
                            player_data['Player'] = player_link.text
                            player_data['player_link'] = player_link.get('href', '')
                        else:
                            player_data['Player'] = cell.text
                            player_data['player_link'] = ''
                    elif stat_category == 'pos':
                        # Position - keep as is
                        player_data['Pos'] = cell.text
                    elif stat_category == 'team_id':
                        # Team abbreviation
                        player_data['Team'] = cell.text
                    elif stat_category == 'age':
                        # Player age
                        player_data['Age'] = cell.text
                    elif stat_category == 'g':
                        # Games played
                        player_data['G'] = cell.text
                    elif stat_category == 'gs':
                        # Games started
                        player_data['GS'] = cell.text
                    elif stat_category == 'mp':
                        # Minutes played
                        player_data['MP'] = cell.text
                    elif stat_category == 'per':
                        # Player Efficiency Rating
                        player_data['PER'] = cell.text
                    elif stat_category == 'ts_pct':
                        # True Shooting Percentage
                        player_data['TS%'] = cell.text
                    elif stat_category == 'fg3a_per_fga_pct':
                        # 3-Point Attempt Rate
                        player_data['3PAr'] = cell.text
                    elif stat_category == 'fta_per_fga_pct':
                        # Free Throw Attempt Rate
                        player_data['FTr'] = cell.text
                    elif stat_category == 'orb_pct':
                        # Offensive Rebound Percentage
                        player_data['ORB%'] = cell.text
                    elif stat_category == 'drb_pct':
                        # Defensive Rebound Percentage
                        player_data['DRB%'] = cell.text
                    elif stat_category == 'trb_pct':
                        # Total Rebound Percentage
                        player_data['TRB%'] = cell.text
                    elif stat_category == 'ast_pct':
                        # Assist Percentage
                        player_data['AST%'] = cell.text
                    elif stat_category == 'stl_pct':
                        # Steal Percentage
                        player_data['STL%'] = cell.text
                    elif stat_category == 'blk_pct':
                        # Block Percentage
                        player_data['BLK%'] = cell.text
                    elif stat_category == 'tov_pct':
                        # Turnover Percentage
                        player_data['TOV%'] = cell.text
                    elif stat_category == 'usg_pct':
                        # Usage Percentage
                        player_data['USG%'] = cell.text
                    elif stat_category == 'ows':
                        # Offensive Win Shares
                        player_data['OWS'] = cell.text
                    elif stat_category == 'dws':
                        # Defensive Win Shares
                        player_data['DWS'] = cell.text
                    elif stat_category == 'ws':
                        # Win Shares
                        player_data['WS'] = cell.text
                    elif stat_category == 'ws_per_48':
                        # Win Shares Per 48 Minutes
                        player_data['WS/48'] = cell.text
                    elif stat_category == 'obpm':
                        # Offensive Box Plus/Minus
                        player_data['OBPM'] = cell.text
                    elif stat_category == 'dbpm':
                        # Defensive Box Plus/Minus
                        player_data['DBPM'] = cell.text
                    elif stat_category == 'bpm':
                        # Box Plus/Minus
                        player_data['BPM'] = cell.text
                    elif stat_category == 'vorp':
                        # Value Over Replacement Player
                        player_data['VORP'] = cell.text
                    elif stat_category == 'rank':
                        # Player rank
                        player_data['Rk'] = cell.text
                    elif stat_category == 'awards':
                        # Awards (if any)
                        player_data['Awards'] = cell.text
                    else:
                        # Any other stat, use the header name
                        player_data[headers[i]] = cell.text
            
            # Only add non-empty rows
            if player_data and 'Player' in player_data and player_data['Player'].strip():
                rows_data.append(player_data)
    
    # Create DataFrame and clean up data
    df = pd.DataFrame(rows_data)
    
    # Replace empty strings with NaN
    df = df.replace('', float('nan'))
    
    # Convert numeric columns to float where possible
    numeric_cols = ['Rk', 'Age', 'G', 'GS', 'MP', 'PER', 'TS%', '3PAr', 'FTr', 
                    'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%',
                    'OWS', 'DWS', 'WS', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'VORP']
    
    for col in numeric_cols:
        if col in df.columns:
            try:
                df[col] = df[col].astype(float)
            except (ValueError, TypeError):
                # Try to clean up the column by removing non-numeric characters
                try:
                    df[col] = df[col].str.replace(r'[^\d.-]', '', regex=True).astype(float)
                except (ValueError, TypeError, AttributeError):
                    print(f"Could not convert column {col} to numeric.")
    
    # Reorganize columns to match the desired order
    desired_columns = ['Rk', 'Player', 'Age', 'Team', 'Pos', 'G', 'GS', 'MP', 'PER', 
                      'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 
                      'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS', 'WS', 'WS/48', 'OBPM', 
                      'DBPM', 'BPM', 'VORP', 'Awards']
    
    # Filter to only include columns that actually exist in the dataframe
    # Ensure 'player_link' is handled correctly by not adding it to final_columns if it's not desired.
    final_columns = [col for col in desired_columns if col in df.columns]
    existing_cols = [col for col in df.columns if col not in final_columns and col != 'player_link']
    final_columns.extend(existing_cols)
    
    # Reorder columns
    df = df[final_columns]
    
    print(f"Successfully extracted advanced data for {len(df)} players.")
    return df

def scrape_per_game_stats(year='2025'):
    """
    Scrape per-game NBA stats from Basketball Reference for a specific season.
    
    Args:
        year (str): The year representing the end of the season (e.g., '2025' for the 2024-25 season)
    
    Returns:
        DataFrame: A pandas DataFrame containing the per-game stats
    """
    # Construct the URL based on the provided year
    url = f'https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html'
    
    # Add headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Make the request
    print(f"Fetching data from {url}...")
    response = requests.get(url, headers=headers)
    
    # Check if the request was successful
    if response.status_code != 200:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
        return None
    
    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the main table with the per-game stats
    # Try by ID first, then by caption if ID is not found
    table = soup.find('table', id='pgl_basic')
    if not table:
        print("Could not find the per-game stats table by ID 'pgl_basic'. Trying by caption...")
        table = soup.find('caption', string='Per Game Table')
        if table:
            table = table.find_parent('table') # Get the parent table element
    
    if not table:
        print("Could not find the per-game stats table on the page.")
        return None
        
    # Extract the table headers using data-stat attributes
    # Rk, Player, Pos, Age, Tm, G, GS, MP, FG, FGA, FG%, 3P, 3PA, 3P%, 2P, 2PA, 2P%, eFG%, FT, FTA, FT%, ORB, DRB, TRB, AST, STL, BLK, TOV, PF, PTS, Awards
    header_data_stats = [
        'ranker', 'player', 'pos', 'age', 'team_id', 'g', 'gs', 'mp_per_g', 'fg_per_g', 'fga_per_g', 'fg_pct',
        'fg3_per_g', 'fg3a_per_g', 'fg3_pct', 'fg2_per_g', 'fg2a_per_g', 'fg2_pct', 'efg_pct',
        'ft_per_g', 'fta_per_g', 'ft_pct', 'orb_per_g', 'drb_per_g', 'trb_per_g', 'ast_per_g',
        'stl_per_g', 'blk_per_g', 'tov_per_g', 'pf_per_g', 'pts_per_g', 'awards'
    ]
    
    # Create a mapping from data-stat to desired column names
    # Using a more direct mapping based on typical Basketball-Reference headers
    column_mapping = {
        'ranker': 'Rk', 'player': 'Player', 'pos': 'Pos', 'age': 'Age', 'team_id': 'Team', 'g': 'G', 'gs': 'GS',
        'mp_per_g': 'MP', 'fg_per_g': 'FG', 'fga_per_g': 'FGA', 'fg_pct': 'FG%', 'fg3_per_g': '3P',
        'fg3a_per_g': '3PA', 'fg3_pct': '3P%', 'fg2_per_g': '2P', 'fg2a_per_g': '2PA', 'fg2_pct': '2P%',
        'efg_pct': 'eFG%', 'ft_per_g': 'FT', 'fta_per_g': 'FTA', 'ft_pct': 'FT%', 'orb_per_g': 'ORB',
        'drb_per_g': 'DRB', 'trb_per_g': 'TRB', 'ast_per_g': 'AST', 'stl_per_g': 'STL', 'blk_per_g': 'BLK',
        'tov_per_g': 'TOV', 'pf_per_g': 'PF', 'pts_per_g': 'PTS', 'awards': 'Awards'
    }
    
    headers_list = [column_mapping.get(stat, stat) for stat in header_data_stats]

    # Extract the data rows
    rows_data = []
    
    for tbody in table.find_all('tbody'):
        rows = tbody.find_all('tr')
        for row in rows:
            if row.get('class') and ('thead' in row.get('class') or 'partial_table' in row.get('class')): # Skip header rows
                continue
            
            player_data = {}
            cells = row.find_all(['th', 'td'])
            
            for i, cell in enumerate(cells):
                stat_category = cell.get('data-stat')
                
                if stat_category == 'player':
                    player_link = cell.find('a')
                    if player_link:
                        player_data['Player'] = player_link.text
                        player_data['player_link'] = player_link.get('href', '')
                    else:
                        player_data['Player'] = cell.text
                        player_data['player_link'] = ''
                elif stat_category in column_mapping: # Use mapped column name
                    player_data[column_mapping[stat_category]] = cell.text.strip()
                elif i < len(headers_list): # Fallback for any unmapped but present columns based on order
                     player_data[headers_list[i]] = cell.text.strip()


            if player_data and 'Player' in player_data and player_data['Player'].strip():
                rows_data.append(player_data)
                
    df = pd.DataFrame(rows_data)
    df = df.replace('', float('nan')) # Replace empty strings with NaN
    
    # Convert numeric columns
    numeric_cols = ['Rk', 'Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 
                    '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 
                    'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']

    for col in numeric_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except (ValueError, TypeError):
                print(f"Could not convert column {col} to numeric for per-game stats.")

    # Reorganize columns to match the desired order (similar to advanced stats, but with per-game fields)
    desired_columns_pg = ['Rk', 'Player', 'Pos', 'Age', 'Team', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', 
                          '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 
                          'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Awards']
    
    final_columns_pg = [col for col in desired_columns_pg if col in df.columns]
    existing_cols_pg = [col for col in df.columns if col not in final_columns_pg and col != 'player_link']
    final_columns_pg.extend(existing_cols_pg)
    
    df = df[final_columns_pg]
    
    print(f"Successfully extracted per-game data for {len(df)} players.")
    return df

def save_to_csv(df, filename):
    """
    Save the DataFrame to a CSV file
    
    Args:
        df (DataFrame): The DataFrame to save
        filename (str): The filename to save the data to.
    """
    if df is None:
        print("No data to save.")
        return
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def main():
    """
    Main function to scrape and save the data
    """
    # Get the advanced stats
    advanced_stats_year = '2025' # Example: 2024-25 season
    advanced_stats = scrape_advanced_stats(year=advanced_stats_year)
    
    if advanced_stats is not None:
        # Save the advanced stats data
        save_to_csv(advanced_stats, f'nba_advanced_stats_{int(advanced_stats_year)-1}_{advanced_stats_year[-2:]}.csv')
        
        # Display the first few rows of advanced stats
        print("\nPreview of the Advanced Stats data:")
        print(advanced_stats.head())
        
        # Display some basic advanced stats
        if 'PER' in advanced_stats.columns:
            print("\nTop 10 players by PER:")
            top_per = advanced_stats.sort_values(by='PER', ascending=False).head(10)
            for i, (_, player) in enumerate(top_per[['Player', 'PER']].iterrows(), 1):
                print(f"{i}. {player['Player']}: {player['PER']}")
    
    print("-" * 50) # Separator

    # Get the per-game stats
    per_game_stats_year = '2025' # Example: 2024-25 season
    per_game_stats = scrape_per_game_stats(year=per_game_stats_year)

    if per_game_stats is not None:
        # Save the per-game stats data
        save_to_csv(per_game_stats, f'nba_per_game_stats_{int(per_game_stats_year)-1}_{per_game_stats_year[-2:]}.csv')

        # Display the first few rows of per-game stats
        print("\nPreview of the Per Game Stats data:")
        print(per_game_stats.head())

        # Display some basic per-game stats
        if 'PTS' in per_game_stats.columns:
            print("\nTop 10 players by PTS (Per Game):")
            top_pts = per_game_stats.sort_values(by='PTS', ascending=False).head(10)
            for i, (_, player) in enumerate(top_pts[['Player', 'PTS']].iterrows(), 1):
                print(f"{i}. {player['Player']}: {player['PTS']}")

    print("-" * 50) # Separator

    # Get the shooting stats
    shooting_stats_year = '2025' # Example: 2024-25 season
    shooting_stats = scrape_shooting_stats(year=shooting_stats_year)

    if shooting_stats is not None:
        # Save the shooting stats data
        save_to_csv(shooting_stats, f'nba_shooting_stats_{int(shooting_stats_year)-1}_{shooting_stats_year[-2:]}.csv')

        # Display the first few rows of shooting stats
        print("\nPreview of the Shooting Stats data:")
        print(shooting_stats.head())
        
        # Display some basic shooting stats (e.g., top 3P% players with > 100 3PA if 3PA data is available)
        # Note: For this example, we'll just print head, a more complex sort would require FGA/3PA which are not directly in the shooting table from prompt
        if 'FG%' in shooting_stats.columns: # A general stat to check
            print("\nShooting Stats Data Sample (first 5 rows for FG%):")
            print(shooting_stats[['Player', 'FG%']].head())


def scrape_shooting_stats(year='2025'):
    """
    Scrape shooting NBA stats from Basketball Reference for a specific season.
    
    Args:
        year (str): The year representing the end of the season (e.g., '2025' for the 2024-25 season)
    
    Returns:
        DataFrame: A pandas DataFrame containing the shooting stats
    """
    url = f'https://www.basketball-reference.com/leagues/NBA_{year}_shooting.html'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    print(f"Fetching data from {url}...")
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
        return None
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    table = soup.find('table', id='shooting_stats')
    if not table:
        print("Could not find the shooting stats table by ID 'shooting_stats'. Trying by caption...")
        table_caption = soup.find('caption', string='Shooting Table')
        if table_caption:
            table = table_caption.find_parent('table')
    
    if not table:
        print("Could not find the shooting stats table on the page.")
        return None
        
    # data-stat attributes from prompt and website inspection
    header_data_stats = [
        'ranker', 'player', 'pos', 'age', 'team_id', 'g', 'gs', 'mp', 
        'fg_pct', 'avg_dist', 
        'pct_fga_fg2a', 'pct_fga_00_03', 'pct_fga_03_10', 'pct_fga_10_16', 'pct_fga_16_3p', 'pct_fga_fg3a',
        'fg_pct_fg2', 'fg_pct_00_03', 'fg_pct_03_10', 'fg_pct_10_16', 'fg_pct_16_3p', 'fg_pct_fg3',
        'pct_ast_fg2', 'pct_ast_fg3',
        'pct_fga_dunk', 'dunks', 
        'pct_fg3a_corner3', 'fg3_pct_corner3',
        'fg3a_heave', 'fg3_heave', 'awards'
    ]
    
    column_mapping = {
        'ranker': 'Rk', 'player': 'Player', 'pos': 'Pos', 'age': 'Age', 'team_id': 'Team', 'g': 'G', 'gs': 'GS', 'mp': 'MP',
        'fg_pct': 'FG%', 'avg_dist': 'Dist.',
        'pct_fga_fg2a': '%FGA 2P', 'pct_fga_00_03': '%FGA 0-3', 'pct_fga_03_10': '%FGA 3-10', 
        'pct_fga_10_16': '%FGA 10-16', 'pct_fga_16_3p': '%FGA 16-3P', 'pct_fga_fg3a': '%FGA 3P',
        'fg_pct_fg2': '2P%', 'fg_pct_00_03': 'FG% 0-3', 'fg_pct_03_10': 'FG% 3-10', 
        'fg_pct_10_16': 'FG% 10-16', 'fg_pct_16_3p': 'FG% 16-3P', 'fg_pct_fg3': '3P%',
        'pct_ast_fg2': '%Ast\'d 2P', 'pct_ast_fg3': '%Ast\'d 3P',
        'pct_fga_dunk': 'Dunk %FGA', 'dunks': 'Dunks #',
        'pct_fg3a_corner3': 'Corner 3s %3PA', 'fg3_pct_corner3': 'Corner 3s 3P%',
        'fg3a_heave': 'Heaves Att.', 'fg3_heave': 'Heaves Md.', 'awards': 'Awards'
    }

    headers_list = [column_mapping.get(stat, stat) for stat in header_data_stats]
    
    rows_data = []
    for tbody in table.find_all('tbody'):
        rows = tbody.find_all('tr')
        for row in rows:
            if row.get('class') and ('thead' in row.get('class') or 'partial_table' in row.get('class')):
                continue
            
            player_data = {}
            cells = row.find_all(['th', 'td'])
            
            for i, cell in enumerate(cells):
                stat_category = cell.get('data-stat')
                
                if stat_category == 'player':
                    player_link_tag = cell.find('a')
                    if player_link_tag:
                        player_data['Player'] = player_link_tag.text
                        player_data['player_link'] = player_link_tag.get('href', '')
                    else:
                        player_data['Player'] = cell.text
                        player_data['player_link'] = ''
                elif stat_category in column_mapping:
                    player_data[column_mapping[stat_category]] = cell.text.strip()
                # Fallback for any unmapped columns (though ideally all are mapped)
                elif i < len(headers_list) and headers_list[i] not in player_data: # only if not already populated
                     player_data[headers_list[i]] = cell.text.strip()
            
            if player_data and 'Player' in player_data and player_data['Player'].strip():
                rows_data.append(player_data)
                
    df = pd.DataFrame(rows_data)
    df = df.replace('', float('nan'))
    
    numeric_cols = [
        'Rk', 'Age', 'G', 'GS', 'MP', 'FG%', 'Dist.',
        '%FGA 2P', '%FGA 0-3', '%FGA 3-10', '%FGA 10-16', '%FGA 16-3P', '%FGA 3P',
        '2P%', 'FG% 0-3', 'FG% 3-10', 'FG% 10-16', 'FG% 16-3P', '3P%',
        '%Ast\'d 2P', '%Ast\'d 3P',
        'Dunk %FGA', 'Dunks #', 
        'Corner 3s %3PA', 'Corner 3s 3P%',
        'Heaves Att.', 'Heaves Md.'
    ]

    for col in numeric_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except (ValueError, TypeError):
                print(f"Could not convert column {col} to numeric for shooting stats.")

    # Define desired order of columns
    desired_columns_shooting = [
        'Rk', 'Player', 'Pos', 'Age', 'Team', 'G', 'GS', 'MP', 'FG%', 'Dist.', 
        '%FGA 2P', '%FGA 0-3', '%FGA 3-10', '%FGA 10-16', '%FGA 16-3P', '%FGA 3P',
        '2P%', 'FG% 0-3', 'FG% 3-10', 'FG% 10-16', 'FG% 16-3P', '3P%',
        '%Ast\'d 2P', '%Ast\'d 3P',
        'Dunk %FGA', 'Dunks #', 
        'Corner 3s %3PA', 'Corner 3s 3P%',
        'Heaves Att.', 'Heaves Md.', 'Awards'
    ]
    
    final_columns_shooting = [col for col in desired_columns_shooting if col in df.columns]
    existing_cols_shooting = [col for col in df.columns if col not in final_columns_shooting and col != 'player_link']
    final_columns_shooting.extend(existing_cols_shooting)
    
    df = df[final_columns_shooting]
    
    print(f"Successfully extracted shooting data for {len(df)} players.")
    return df

def save_to_mongodb(df, collection_name_prefix, year_end, db):
    """
    Saves the DataFrame to a MongoDB collection.

    Args:
        df (DataFrame): The DataFrame to save.
        collection_name_prefix (str): The prefix for the collection name (e.g., "advanced_stats").
        year_end (str): The year representing the end of the season (e.g., '2025' for 2024-25 season).
        db (Database): The MongoDB database object.
    """
    if df is None or df.empty:
        print(f"No data to save for {collection_name_prefix} for year {year_end}.")
        return
    
    if db is None:
        print("MongoDB database object is not available. Cannot save data.")
        return

    # Construct collection name, e.g., advanced_stats_2024_25
    collection_name = f"{collection_name_prefix}_{int(year_end)-1}_{year_end[-2:]}"
    
    try:
        collection = db[collection_name]
        # Clear existing data in the collection for the specific year/type
        print(f"Clearing existing data from collection: {collection_name}...")
        delete_result = collection.delete_many({})
        print(f"Deleted {delete_result.deleted_count} existing documents.")

        # Convert DataFrame to list of dictionaries
        records = df.to_dict('records')
        
        if records:
            print(f"Inserting {len(records)} records into collection: {collection_name}...")
            collection.insert_many(records)
            print(f"Successfully inserted {len(records)} records into MongoDB collection: {collection_name}")
        else:
            print(f"No records to insert for {collection_name}.")
            
    except OperationFailure as e:
        print(f"MongoDB operation failed for collection {collection_name}: {e}")
    except Exception as e:
        print(f"An error occurred while saving to MongoDB collection {collection_name}: {e}")


def main():
    """
    Main function to scrape and save the data
    """
    # Establish MongoDB connection first
    print("Connecting to MongoDB...")
    db = get_mongo_db()

    if db is None:
        print("Exiting script as MongoDB connection could not be established.")
        return

    # Get the advanced stats
    advanced_stats_year = '2025' # Example: 2024-25 season
    advanced_stats_df = scrape_advanced_stats(year=advanced_stats_year)
    
    if advanced_stats_df is not None:
        csv_filename_advanced = f'nba_advanced_stats_{int(advanced_stats_year)-1}_{advanced_stats_year[-2:]}.csv'
        save_to_csv(advanced_stats_df, csv_filename_advanced)
        print("\nPreview of the Advanced Stats data:")
        print(advanced_stats_df.head())
        if 'PER' in advanced_stats_df.columns:
            print("\nTop 10 players by PER:")
            top_per = advanced_stats_df.sort_values(by='PER', ascending=False).head(10)
            for i, (_, player) in enumerate(top_per[['Player', 'PER']].iterrows(), 1):
                print(f"{i}. {player['Player']}: {player['PER']}")
        # Save to MongoDB
        save_to_mongodb(advanced_stats_df, "advanced_stats", advanced_stats_year, db)
    
    print("-" * 50) 

    # Get the per-game stats
    per_game_stats_year = '2025' 
    per_game_stats_df = scrape_per_game_stats(year=per_game_stats_year)

    if per_game_stats_df is not None:
        csv_filename_per_game = f'nba_per_game_stats_{int(per_game_stats_year)-1}_{per_game_stats_year[-2:]}.csv'
        save_to_csv(per_game_stats_df, csv_filename_per_game)
        print("\nPreview of the Per Game Stats data:")
        print(per_game_stats_df.head())
        if 'PTS' in per_game_stats_df.columns:
            print("\nTop 10 players by PTS (Per Game):")
            top_pts = per_game_stats_df.sort_values(by='PTS', ascending=False).head(10)
            for i, (_, player) in enumerate(top_pts[['Player', 'PTS']].iterrows(), 1):
                print(f"{i}. {player['Player']}: {player['PTS']}")
        # Save to MongoDB
        save_to_mongodb(per_game_stats_df, "per_game_stats", per_game_stats_year, db)

    print("-" * 50)

    # Get the shooting stats
    shooting_stats_year = '2025' 
    shooting_stats_df = scrape_shooting_stats(year=shooting_stats_year)

    if shooting_stats_df is not None:
        csv_filename_shooting = f'nba_shooting_stats_{int(shooting_stats_year)-1}_{shooting_stats_year[-2:]}.csv'
        save_to_csv(shooting_stats_df, csv_filename_shooting)
        print("\nPreview of the Shooting Stats data:")
        print(shooting_stats_df.head())
        if 'FG%' in shooting_stats_df.columns: 
            print("\nShooting Stats Data Sample (first 5 rows for FG%):")
            print(shooting_stats_df[['Player', 'FG%']].head())
        # Save to MongoDB
        save_to_mongodb(shooting_stats_df, "shooting_stats", shooting_stats_year, db)

if __name__ == "__main__":
    main()