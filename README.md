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

## Future Enhancements
- Incorporate data from more seasons.
- Develop predictive models for player performance.
- Expand the range of analytical queries and visualizations.
