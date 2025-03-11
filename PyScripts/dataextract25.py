import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re

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
    final_columns = [col for col in desired_columns if col in df.columns]
    # Add any columns that exist in the dataframe but aren't in our desired list
    for col in df.columns:
        if col not in final_columns and col != 'player_link':
            final_columns.append(col)
    
    # Reorder columns
    df = df[final_columns]
    
    print(f"Successfully extracted data for {len(df)} players.")
    return df

def save_to_csv(df, filename=None):
    """
    Save the DataFrame to a CSV file
    
    Args:
        df (DataFrame): The DataFrame to save
        filename (str): Optional filename, defaults to 'nba_advanced_stats_YYYY.csv'
    """
    if filename is None:
        filename = f'nba_advanced_stats_2024_25.csv'
    
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def main():
    """
    Main function to scrape and save the data
    """
    # Get the advanced stats
    advanced_stats = scrape_advanced_stats(year='2025')  # 2024-25 season
    
    if advanced_stats is not None:
        # Save the data
        save_to_csv(advanced_stats)
        
        # Display the first few rows
        print("\nPreview of the data:")
        print(advanced_stats.head())
        
        # Display some basic stats
        if 'PER' in advanced_stats.columns:
            print("\nTop 10 players by PER:")
            top_per = advanced_stats.sort_values(by='PER', ascending=False).head(10)
            for i, (_, player) in enumerate(top_per[['Player', 'PER']].iterrows(), 1):
                print(f"{i}. {player['Player']}: {player['PER']}")

if __name__ == "__main__":
    main()