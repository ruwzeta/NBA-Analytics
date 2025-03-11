import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

class NBAMVPPredictor:
    def __init__(self):
        self.current_season_data = None
        self.historical_data = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def scrape_historical_mvp_data(self, start_year=2010, end_year=2024):
        """
        Scrape historical MVP voting data from Basketball Reference
        """
        all_mvp_data = []
        
        for year in range(start_year, end_year + 1):
            print(f"Scraping MVP data for {year-1}-{year} season...")
            url = f'https://www.basketball-reference.com/awards/awards_{year}.html'
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            try:
                response = requests.get(url, headers=headers)
                if response.status_code != 200:
                    print(f"Failed to retrieve MVP data for {year}. Status code: {response.status_code}")
                    continue
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find the MVP voting table
                mvp_table = soup.find('table', id='mvp')
                
                if not mvp_table:
                    print(f"Could not find MVP table for {year}")
                    continue
                
                # Extract MVP data
                rows = mvp_table.find('tbody').find_all('tr')
                
                for row in rows:
                    if 'class' in row.attrs and 'thead' in row.attrs['class']:
                        continue
                        
                    cells = row.find_all(['td', 'th'])
                    if len(cells) < 2:
                        continue
                        
                    mvp_data = {'Season': f"{year-1}-{year}"}
                    
                    for cell in cells:
                        stat = cell.get('data-stat')
                        if stat:
                            if stat == 'player':
                                player_link = cell.find('a')
                                if player_link:
                                    mvp_data['Player'] = player_link.text
                            elif stat == 'team_id':
                                mvp_data['Team'] = cell.text
                            elif stat == 'pts_won':
                                mvp_data['MVP_Points'] = cell.text
                            elif stat == 'pts_max':
                                mvp_data['MVP_Max_Points'] = cell.text
                            elif stat == 'share':
                                mvp_data['MVP_Share'] = cell.text
                            elif stat == 'rank':
                                mvp_data['MVP_Rank'] = cell.text
                    
                    all_mvp_data.append(mvp_data)
                
                # Be nice to the server
                time.sleep(1)
                
            except Exception as e:
                print(f"Error scraping MVP data for {year}: {str(e)}")
        
        mvp_df = pd.DataFrame(all_mvp_data)
        
        # Convert numeric columns
        numeric_cols = ['MVP_Points', 'MVP_Max_Points', 'MVP_Share', 'MVP_Rank']
        for col in numeric_cols:
            if col in mvp_df.columns:
                mvp_df[col] = pd.to_numeric(mvp_df[col], errors='coerce')
        
        return mvp_df
    
    def scrape_historical_advanced_stats(self, start_year=2010, end_year=2024):
        """
        Scrape historical advanced stats from Basketball Reference
        """
        all_stats = []
        
        for year in range(start_year, end_year + 1):
            print(f"Scraping advanced stats for {year-1}-{year} season...")
            url = f'https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html'
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            try:
                response = requests.get(url, headers=headers)
                if response.status_code != 200:
                    print(f"Failed to retrieve advanced stats for {year}. Status code: {response.status_code}")
                    continue
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find the advanced stats table
                table = soup.find('table', id='advanced')
                
                if not table:
                    print(f"Could not find advanced stats table for {year}")
                    continue
                
                # Extract data rows
                rows_data = []
                
                for tbody in table.find_all('tbody'):
                    rows = tbody.find_all('tr')
                    
                    for row in rows:
                        if 'class' in row.attrs and 'thead' in row.attrs['class']:
                            continue
                            
                        player_data = {'Season': f"{year-1}-{year}"}
                        
                        for cell in row.find_all(['th', 'td']):
                            stat = cell.get('data-stat')
                            
                            if stat:
                                if stat == 'player':
                                    player_link = cell.find('a')
                                    if player_link:
                                        player_data['Player'] = player_link.text
                                elif stat == 'team_id':
                                    player_data['Team'] = cell.text
                                elif stat in ['g', 'mp', 'per', 'ts_pct', 'ast_pct', 'usg_pct', 'ws', 'bpm', 'vorp']:
                                    # Important advanced stats
                                    if stat == 'g':
                                        player_data['Games'] = cell.text
                                    elif stat == 'mp':
                                        player_data['Minutes'] = cell.text
                                    elif stat == 'per':
                                        player_data['PER'] = cell.text
                                    elif stat == 'ts_pct':
                                        player_data['TS%'] = cell.text
                                    elif stat == 'ast_pct':
                                        player_data['AST%'] = cell.text
                                    elif stat == 'usg_pct':
                                        player_data['USG%'] = cell.text
                                    elif stat == 'ws':
                                        player_data['WS'] = cell.text
                                    elif stat == 'bpm':
                                        player_data['BPM'] = cell.text
                                    elif stat == 'vorp':
                                        player_data['VORP'] = cell.text
                        
                        if 'Player' in player_data and player_data['Player'].strip():
                            rows_data.append(player_data)
                
                all_stats.extend(rows_data)
                
                # Be nice to the server
                time.sleep(1)
                
            except Exception as e:
                print(f"Error scraping advanced stats for {year}: {str(e)}")
        
        stats_df = pd.DataFrame(all_stats)
        
        # Convert numeric columns
        numeric_cols = ['Games', 'Minutes', 'PER', 'TS%', 'AST%', 'USG%', 'WS', 'BPM', 'VORP']
        for col in numeric_cols:
            if col in stats_df.columns:
                stats_df[col] = pd.to_numeric(stats_df[col], errors='coerce')
        
        return stats_df
    
    def scrape_team_records(self, start_year=2010, end_year=2024):
        """
        Scrape historical team records from Basketball Reference
        """
        all_records = []
        
        for year in range(start_year, end_year + 1):
            print(f"Scraping team records for {year-1}-{year} season...")
            url = f'https://www.basketball-reference.com/leagues/NBA_{year}_standings.html'
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            try:
                response = requests.get(url, headers=headers)
                if response.status_code != 200:
                    print(f"Failed to retrieve team records for {year}. Status code: {response.status_code}")
                    continue
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find the team records tables (East and West)
                for conf in ['E', 'W']:
                    table = soup.find('table', id=f'divs_standings_{conf}')
                    if not table:
                        continue
                    
                    rows = table.find('tbody').find_all('tr')
                    
                    for row in rows:
                        if 'class' in row.attrs and 'thead' in row.attrs['class']:
                            continue
                            
                        team_data = {'Season': f"{year-1}-{year}"}
                        
                        for cell in row.find_all(['th', 'td']):
                            stat = cell.get('data-stat')
                            
                            if stat:
                                if stat == 'team_name':
                                    team_link = cell.find('a')
                                    if team_link:
                                        team_data['Team_Long'] = team_link.text
                                elif stat == 'team_id':
                                    team_data['Team'] = cell.text
                                elif stat == 'wins':
                                    team_data['Wins'] = cell.text
                                elif stat == 'losses':
                                    team_data['Losses'] = cell.text
                                elif stat == 'win_loss_pct':
                                    team_data['Win%'] = cell.text
                        
                        if 'Team' in team_data and team_data['Team'].strip():
                            all_records.append(team_data)
                
                # Be nice to the server
                time.sleep(1)
                
            except Exception as e:
                print(f"Error scraping team records for {year}: {str(e)}")
        
        records_df = pd.DataFrame(all_records)
        
        # Convert numeric columns
        numeric_cols = ['Wins', 'Losses', 'Win%']
        for col in numeric_cols:
            if col in records_df.columns:
                records_df[col] = pd.to_numeric(records_df[col], errors='coerce')
        
        return records_df
    
    def load_current_season_data(self, file_path='nba_advanced_stats_2024_25.csv'):
        """
        Load the current season's advanced stats from CSV
        """
        try:
            self.current_season_data = pd.read_csv(file_path)
            print(f"Loaded {len(self.current_season_data)} player records for the current season.")
        except Exception as e:
            print(f"Error loading current season data: {str(e)}")
    
    def prepare_historical_data(self):
        """
        Prepare historical data for training by merging MVP voting, advanced stats, and team records
        """
        print("Preparing historical MVP training data...")
        
        # Scrape historical data
        mvp_data = self.scrape_historical_mvp_data()
        advanced_stats = self.scrape_historical_advanced_stats()
        team_records = self.scrape_team_records()
        
        # Merge MVP data with advanced stats
        df = pd.merge(advanced_stats, mvp_data, on=['Season', 'Player', 'Team'], how='left')
        
        # Fill missing MVP data with zeros (players who didn't receive MVP votes)
        for col in ['MVP_Points', 'MVP_Max_Points', 'MVP_Share', 'MVP_Rank']:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Merge with team records
        df = pd.merge(df, team_records, on=['Season', 'Team'], how='left')
        
        # Filter to players with significant playing time
        # Typically, MVPs play at least 50 games and 25+ minutes per game
        df = df[(df['Games'] >= 50) & (df['Minutes'] >= 1500)]
        
        # Create target variable: MVP_Share (percentage of max possible points)
        if 'MVP_Share' not in df.columns and 'MVP_Points' in df.columns and 'MVP_Max_Points' in df.columns:
            df['MVP_Share'] = df['MVP_Points'] / df['MVP_Max_Points'].max()
        
        # Create additional features
        df['Minutes_Per_Game'] = df['Minutes'] / df['Games']
        df['Team_Win_Pct'] = df['Wins'] / (df['Wins'] + df['Losses'])
        
        # Store the prepared historical data
        self.historical_data = df
        print(f"Prepared {len(df)} historical player seasons for training.")
        
        return df
    
    def prepare_features(self, data=None):
        """
        Prepare features for training or prediction
        """
        if data is None:
            data = self.historical_data
            
        # Select relevant features for MVP prediction
        features = [
            'Games', 'Minutes', 'Minutes_Per_Game', 'PER', 'TS%', 'AST%', 'USG%', 
            'WS', 'BPM', 'VORP', 'Team_Win_Pct'
        ]
        
        # Ensure all required features exist
        for feature in features:
            if feature not in data.columns:
                if feature == 'Minutes_Per_Game' and 'Minutes' in data.columns and 'Games' in data.columns:
                    data['Minutes_Per_Game'] = data['Minutes'] / data['Games']
                elif feature == 'Team_Win_Pct' and 'Wins' in data.columns and 'Losses' in data.columns:
                    data['Team_Win_Pct'] = data['Wins'] / (data['Wins'] + data['Losses'])
        
        # Filter to only include rows with all features
        data_filtered = data.dropna(subset=[f for f in features if f in data.columns])
        
        # Extract features and target
        X = data_filtered[[f for f in features if f in data_filtered.columns]]
        
        if 'MVP_Share' in data_filtered.columns:
            y = data_filtered['MVP_Share']
        else:
            y = None
        
        return X, y, data_filtered
    
    def train_model(self):
        """
        Train the MVP prediction model using historical data
        """
        if self.historical_data is None:
            self.prepare_historical_data()
            
        X, y, _ = self.prepare_features()
        
        print(f"Training MVP prediction model with {len(X)} historical player seasons...")
        
        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Try multiple models
        models = {
            'Ridge': Ridge(alpha=1.0),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        best_model = None
        best_score = -np.inf
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)
            score = r2_score(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            
            print(f"{name} - Validation R²: {score:.4f}, RMSE: {rmse:.4f}")
            
            if score > best_score:
                best_score = score
                best_model = model
        
        self.model = best_model
        
        # Get feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(X.columns, self.model.feature_importances_))
        elif hasattr(self.model, 'coef_'):
            self.feature_importance = dict(zip(X.columns, np.abs(self.model.coef_)))
        
        print(f"Model training complete. Best validation R²: {best_score:.4f}")
        
        return self.model
    
    def predict_mvp(self, data=None, top_n=10):
        """
        Predict MVP candidates for the current season
        """
        if self.model is None:
            print("Model not trained. Training model first...")
            self.train_model()
            
        if data is None:
            data = self.current_season_data
            
        # Check if data already contains team records
        if 'Wins' not in data.columns or 'Losses' not in data.columns:
            print("Scraping current season team records...")
            # Get current season team records (2025 for 2024-25 season)
            current_records = self.scrape_team_records(start_year=2025, end_year=2025)
            
            # Merge with player data
            data = pd.merge(data, current_records, on='Team', how='left')
        
        # Prepare features
        X, _, filtered_data = self.prepare_features(data)
        
        # Apply scaling
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Add predictions to the dataframe
        filtered_data['Predicted_MVP_Share'] = predictions
        
        # Sort by predicted MVP share
        mvp_candidates = filtered_data.sort_values('Predicted_MVP_Share', ascending=False)
        
        print("\nTop MVP Candidates Prediction for 2024-25 Season:")
        top_candidates = mvp_candidates.head(top_n)
        
        # Print results in a nice format
        for i, (_, player) in enumerate(top_candidates[['Player', 'Team', 'PER', 'WS', 'VORP', 'Predicted_MVP_Share']].iterrows(), 1):
            print(f"{i}. {player['Player']} ({player['Team']}) - " +
                  f"PER: {player['PER']:.1f}, WS: {player['WS']:.1f}, VORP: {player['VORP']:.1f}, " +
                  f"MVP Share: {player['Predicted_MVP_Share']:.3f}")
        
        return top_candidates
    
    def visualize_predictions(self, top_n=10):
        """
        Visualize MVP predictions
        """
        if self.model is None:
            print("Model not trained. Please train the model first.")
            return
            
        # Get MVP predictions
        predictions = self.predict_mvp(top_n=top_n)
        
        # Plot MVP share
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Predicted_MVP_Share', y='Player', data=predictions.head(top_n))
        plt.title('Predicted MVP Share for 2024-25 Season')
        plt.xlabel('Predicted MVP Share')
        plt.ylabel('Player')
        plt.tight_layout()
        plt.savefig('mvp_predictions.png')
        plt.close()
        
        # Plot feature importance if available
        if self.feature_importance:
            plt.figure(figsize=(10, 6))
            importance_df = pd.DataFrame({
                'Feature': list(self.feature_importance.keys()),
                'Importance': list(self.feature_importance.values())
            }).sort_values('Importance', ascending=False)
            
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title('Feature Importance for MVP Prediction')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            plt.close()
        
        # Visualize key metrics for top candidates
        top_players = predictions.head(top_n)['Player'].tolist()
        metrics = ['PER', 'WS', 'BPM', 'VORP', 'USG%']
        
        for metric in metrics:
            if metric in predictions.columns:
                plt.figure(figsize=(12, 6))
                sns.barplot(x=metric, y='Player', data=predictions[predictions['Player'].isin(top_players)].sort_values(metric, ascending=False))
                plt.title(f'{metric} for Top MVP Candidates 2024-25 Season')
                plt.xlabel(metric)
                plt.ylabel('Player')
                plt.tight_layout()
                plt.savefig(f'mvp_{metric.replace("%", "Pct")}.png')
                plt.close()
        
        print("Visualizations saved to current directory.")
        
def main():
    # Initialize the MVP predictor
    predictor = NBAMVPPredictor()
    
    # Load current season data (optional, comment out to scrape fresh data)
    predictor.load_current_season_data()
    
    # Train the model (this will scrape historical data if needed)
    predictor.train_model()
    
    # Predict MVP candidates for the current season
    predictor.predict_mvp(top_n=15)
    
    # Visualize the predictions
    predictor.visualize_predictions(top_n=10)

if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    print(f"\nTotal runtime: {time.time() - start_time:.2f} seconds")