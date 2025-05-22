# NBA Player Statistics Analysis

## Project Overview

This project focuses on scraping, storing, and analyzing NBA player statistics from [Basketball-Reference.com](https://www.basketball-reference.com/). It originally targeted the 2023-2024 season and has now been expanded to include comprehensive data and analytics for the 2024-2025 season.

The project covers:
- **Data Extraction**: Scraping per-game, advanced, and shooting statistics for NBA players.
- **Data Storage**: Saving scraped data into CSV files and optionally into a MongoDB database.
- **Data Analysis**: Performing exploratory data analysis (EDA), visualizing key metrics, and identifying statistical leaders using Jupyter Notebooks.

<center><img src = "NBA Analytics.png" /></center>

## 2024-2025 Season Data Extraction and Analytics

This section details the processes and tools used for the 2024-2025 NBA season data.

### Data Scraping

Player statistics for the 2024-2025 NBA season are scraped using the Python script `PyScripts/dataextract25.py`. This script fetches data directly from [Basketball-Reference.com](https://www.basketball-reference.com/) and processes it into structured formats.

The script generates the following CSV files, which are stored in the `PyScripts/` directory:
- `PyScripts/nba_per_game_stats_2024_25.csv`: Contains per-game player statistics.
- `PyScripts/nba_advanced_stats_2024_25.csv`: Contains advanced player metrics.
- `PyScripts/nba_shooting_stats_2024_25.csv`: Contains detailed player shooting statistics from various distances and situations.

### Database Storage (Optional)

Functionality to store the scraped 2024-2025 season data into a MongoDB database has also been implemented within `PyScripts/dataextract25.py`. The MongoDB connection details are managed by `PyScripts/MongoDB.py`.

The data is organized into collections, dynamically named based on the statistic type and season, for example:
- `per_game_stats_2024_25`
- `advanced_stats_2024_25`
- `shooting_stats_2024_25`

**Note**: During development and testing within the provided sandbox environment, a persistent DNS resolution error (`pymongo.errors.ConfigurationError: The DNS query name does not exist`) prevented successful connection to the specified MongoDB URI. While the code for MongoDB integration is in place and designed to clear and repopulate collections on each run, this storage mechanism could not be fully verified in the test environment.

### Data Analytics

Exploratory data analysis and visualizations for the 2024-2025 season data are performed in the Jupyter Notebook:
- `notebooks/PlayersStatsAnalysis_2024_25.ipynb`

This notebook covers:
- Loading the per-game, advanced, and shooting statistics CSV files.
- Cleaning and inspecting the data.
- Merging the different datasets into a comprehensive DataFrame.
- Performing EDA, including:
    - Generating leaderboards for key statistical categories (e.g., Points Per Game, Assists, Rebounds, PER, True Shooting %).
    - Visualizing data distributions (e.g., average shot distance).
    - Analyzing shooting efficiency from different court areas.
    - Computing and visualizing a correlation matrix of key player statistics.

## Web Frontend (Flask Application)

A Flask-based web application has been developed to provide an interactive way to view player statistics and basic analytics derived from the 2024-2025 season data. The application is located in the `frontend/` directory.

### Features

The web frontend currently offers the following features:

-   **Home Page**: A landing page with navigation to other sections.
-   **League Leaders Page (`/leaders`)**: Displays top 10 leaderboards for key statistical categories:
    -   Points Per Game (PTS)
    -   Assists Per Game (AST)
    -   Total Rebounds Per Game (TRB)
    -   Player Efficiency Rating (PER) (filtered for players with significant playing time)
    -   Win Shares (WS) (filtered for players with significant playing time)
-   **Shooting Performance Page (`/shooting`)**: Presents detailed shooting statistics, including:
    -   Top 10 players by overall 3-Point Percentage (3P%) (filtered by total 3-point attempts).
    -   Top 10 players by Field Goal Percentage (FG%) on shots from 0-3 feet (filtered by attempts from this distance).
    -   Top 10 players by Corner 3-Point Percentage (Corner 3P%) (filtered by corner 3-point attempts).
    -   Top 10 players by Effective Field Goal Percentage (eFG%) (filtered by total field goal attempts).
    -   Top 10 players by True Shooting Percentage (TS%) (filtered by total field goal attempts).
    -   A histogram visualizing the distribution of Average Shot Distance for players.

### How to Run

To run the Flask web application locally, follow these steps:

1.  **Navigate to the project root directory** (e.g., `NBA-Analytics`).
2.  **Create and activate a virtual environment** (recommended):
    ```bash
    python -m venv venv
    ```
    -   On Windows:
        ```bash
        venv\Scripts\activate
        ```
    -   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
3.  **Install dependencies**:
    ```bash
    pip install Flask pandas matplotlib seaborn
    ```
    (Note: `pymongo` would also be needed if the MongoDB connection were active and being used by the frontend directly).
4.  **Navigate to the frontend directory**:
    ```bash
    cd frontend
    ```
5.  **Run the Flask application**:
    ```bash
    python app.py
    ```
6.  Open your web browser and go to `http://0.0.0.0:5001/` (or `http://127.0.0.1:5001/`).

### Data Source for Frontend

The web application currently loads all its data directly from the CSV files generated by the scraping script. These files are expected to be located in the `PyScripts/` directory, relative to the project root:
-   `../PyScripts/nba_per_game_stats_2024_25.csv`
-   `../PyScripts/nba_advanced_stats_2024_25.csv`
-   `../PyScripts/nba_shooting_stats_2024_25.csv`

The `load_and_prepare_data()` function within `frontend/app.py` handles the reading, merging, and basic cleaning of this data before it's displayed on the web pages.

## Future Enhancements
- Incorporate data from more seasons.
- Develop predictive models for player performance.
- Expand the range of analytical queries and visualizations.
