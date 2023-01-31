%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import requests as req
import numpy as np
import pandas as pd
import json
import re
import warnings
import seaborn as sns
import math
import operator
from collections import OrderedDict
from enum import Enum
from datetime import datetime, timedelta
from functools import reduce
from pycaret.classification import *
import tweepy
pd.options.display.max_rows = 100
pd.options.display.max_columns = 999
warnings.filterwarnings('ignore')

def get_data(date_from):

    from nba_api.stats.endpoints import leaguestandings
    from nba_api.stats.endpoints import leaguegamefinder
    from nba_api.stats.endpoints import boxscoretraditionalv2
    from nba_api.stats.endpoints import boxscoreadvancedv2
    from ratelimiter import RateLimiter
    from functools import reduce
    
    current_date = datetime.now().strftime('%m/%d/%Y')
    
    #grab standings
    standingsfinder = leaguestandings.LeagueStandings()
    standings = standingsfinder.get_data_frames()[0]
    team_list = list(standings['TeamID'])

    #grab games since
    games_df = pd.DataFrame(columns = ['SEASON_ID'])
    gamefinder = leaguegamefinder.LeagueGameFinder(date_from_nullable = date_from)
    games = gamefinder.get_data_frames()[0]
    games_df = pd.concat([games_df, games], join = 'outer')

    #filter games since only for NBA games, using team_list from standings
    games_df = games_df[games_df['TEAM_ID'].isin(team_list)]

    #getting the rest of the player and team stats
    #create a new game list since the rest of the API calls require a game_id
    game_list = list(games_df['GAME_ID'])

    #create skeleton dataframes for game concatenation
    player_trad_df = pd.DataFrame(columns = ['GAME_ID'])
    player_adv_df = pd.DataFrame(columns = ['GAME_ID'])
    game_adv_df = pd.DataFrame(columns = ['GAME_ID'])

    #create a ratelimiter function to avoid timeouts
    rate_limiter = RateLimiter(max_calls = 1, period = 7)

    for game in game_list:

        with rate_limiter:

            try:
                #get player stats from traditional boxscore
                player_trad_finder = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id = game)
                new_player_trad = player_trad_finder.get_data_frames()[0]
                player_trad_df = pd.concat([player_trad_df, new_player_trad], join = 'outer')

                #get player stats from advanced boxscore
                player_adv_finder = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id = game)
                new_player_adv = player_adv_finder.get_data_frames()[0]
                player_adv_df = pd.concat([player_adv_df, new_player_adv], join = 'outer')

                #get game stats from advanced boxscore
                game_adv_finder = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id = game)
                new_game_adv = game_adv_finder.get_data_frames()[1]
                game_adv_df = pd.concat([game_adv_df, new_game_adv], join = 'outer')

            except Exception:
                pass
            
    return games_df, player_trad_df, player_adv_df, game_adv_df, standings

def validate_data(date_from):
    
    games_df, player_trad, player_adv, game_adv, standings = get_data(date_from)
    
    games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'], format = '%Y-%m-%d')
    games_df.sort_values(by = 'GAME_DATE', inplace = True)
    games_df.reset_index(drop = True, inplace = True)
    games_df = games_df[(games_df['WL'] == 'W') | (games_df['WL'] == 'L')]

    # if data is correct and merge was successful, all games should have 2 teams

    # creating dataframe to count the number of GAME_ID occurences
    team_counts = games_df.groupby(['GAME_ID'])['GAME_ID'].count()
    team_counts = pd.DataFrame(team_counts)

    # dropping games if GAME_ID is found more or less than twice
    for count in team_counts.iterrows():
        if count[1][0] > 2:
            print(count[0])
            games_df = games_df[games_df['GAME_ID'] != count[0]]

        else:
            pass

        if count[1][0] < 2:
            print(count[0])
            games_df = games_df[games_df['GAME_ID'] != count[0]]

        else:
            continue

    # to ensure only regular season games, remove all SEASON_IDs that do not start with a 2
    games_df = games_df[games_df['SEASON_ID'].str[0] == '2']
    games_df = games_df.drop_duplicates()
    
    return games_df, player_trad, player_adv, game_adv, standings

def game_player_stats(date_from):
    
    games_df, player_trad, player_adv, game_adv, standings = validate_data(date_from)
    
    player_stats = player_adv[['GAME_ID', 'TEAM_ID', 'PLAYER_ID', 'PLAYER_NAME', 'START_POSITION', 'E_OFF_RATING', 'E_DEF_RATING', 'POSS', 'PIE']]
    player_stats = player_stats[player_stats['PIE'].notnull()]
    player_stats['START_POSITION'].fillna('bench', inplace = True)
    player_stats['START_POSITION'].replace({'':'bench'}, inplace = True)
    
    #create separate df and aggregations of start position per team per game
    f = player_stats[(player_stats['START_POSITION'] == 'F')]
    f = f.drop(columns = ['START_POSITION', 'PLAYER_ID', 'PLAYER_NAME', 'POSS'])
    f = f.groupby(['TEAM_ID', 'GAME_ID']).mean()
    f.reset_index(inplace = True)

    g = player_stats[(player_stats['START_POSITION'] == 'G')]
    g = g.drop(columns = ['START_POSITION', 'PLAYER_ID', 'PLAYER_NAME', 'POSS'])
    g = g.groupby(['TEAM_ID', 'GAME_ID']).mean()
    g.reset_index(inplace = True)


    c = player_stats[(player_stats['START_POSITION'] == 'C')]
    c = c.drop(columns = ['START_POSITION', 'PLAYER_ID', 'PLAYER_NAME', 'POSS'])
    c = c.groupby(['TEAM_ID', 'GAME_ID']).mean()
    c.reset_index(inplace = True)


    b = player_stats[(player_stats['START_POSITION'] == 'bench')]
    b = b.drop(columns = ['START_POSITION', 'PLAYER_ID', 'PLAYER_NAME', 'POSS'])
    b = b.groupby(['TEAM_ID', 'GAME_ID']).mean()
    b.reset_index(inplace = True)


    #loop each positional df to create single df, with each position's metrics for each game being 1 line item
    df_list = [g, f, c, b]
    df_name_list = ['G', 'F', 'C', 'BEN']

    for i, df in enumerate(df_list, 0):
        df.columns = [f'{df_name_list[i]}_'.format(i) + col_name for col_name in df.columns]
        df.rename(columns = {f'{df_name_list[i]}_GAME_ID': 'GAME_ID',
                             f'{df_name_list[i]}_TEAM_ID': 'TEAM_ID'}, inplace = True)

    player_df = reduce(lambda left, right: pd.merge(left, right, how = 'outer', on = ['GAME_ID', 'TEAM_ID']), df_list)
    player_df.sort_values(by = ['GAME_ID', 'TEAM_ID'], inplace = True)
    player_df.reset_index(drop = True, inplace = True)
    player_df.drop_duplicates(inplace = True)
    
    #merge new player df with game df by team and game IDs
    game_player = pd.merge(games_df, player_df, how = 'outer', on = ['TEAM_ID', 'GAME_ID'])
    game_player.dropna(inplace = True)
    
    game_player['SEASON_ID'] = game_player['SEASON_ID'].astype(str)
    game_player['TEAM_ID'] = game_player['TEAM_ID'].astype(str)
    game_player['GAME_ID'] = game_player['GAME_ID'].astype(str)
    
    return game_player, standings

def get_upcoming_games(teams_df):
    
    import os
    import argparse
    import requests

    api_key = ''
    sport = 'upcoming'
    regions = 'us'
    markets = 'h2h'
    odds_format = 'decimal'
    date_format = 'iso'

    odds_response = requests.get(f'https://api.the-odds-api.com/v4/sports/basketball_nba/odds', params={
        'api_key': api_key,
        'regions': regions,
        'markets': markets,
        'oddsFormat': odds_format,
        'dateFormat': date_format,
    })

    if odds_response.status_code != 200:
        print(f'Failed to get odds: status_code {odds_response.status_code}, response body {odds_response.text}')

    else:
        odds_json = odds_response.json()

        # Check the usage quota
        print('Remaining requests', odds_response.headers['x-requests-remaining'])
        print('Used requests', odds_response.headers['x-requests-used'])

    rows = []

    for item in odds_json:
        for book in item['bookmakers']:
            for market in book['markets']:
                for ml in market['outcomes']:
                    rows.append(
                    {
                        'GAME_ID': item['id'],
                        'GAME_DATE': item['commence_time'],
                        'home_team': item['home_team'],
                        'away_team': item['away_team'],
                        'book': book['title'],
                        'price': f"{ml['name']}, {ml['price']}"
                    })

    full_upcoming = pd.DataFrame(rows)
    full_upcoming[['team', 'price']] = full_upcoming['price'].str.split(',', expand = True)

    upcoming_games = full_upcoming[['GAME_ID', 'GAME_DATE', 'home_team', 'away_team']]
    # change datetime to EST date
    upcoming_games['GAME_DATE'] = pd.to_datetime(upcoming_games['GAME_DATE'], format = '%Y-%m-%d')
    output = (upcoming_games.set_index('GAME_DATE')
                            .tz_convert("US/Eastern")
                            .reset_index()
              )
    upcoming_games['GAME_DATE'] = output['GAME_DATE']
    upcoming_games['GAME_DATE'] = upcoming_games['GAME_DATE'].apply(lambda x: x.date().strftime('%Y-%m-%d'))

    #filter df for today's games
    current_date = datetime.now().strftime('%Y-%m-%d')
    upcoming_games = upcoming_games[upcoming_games['GAME_DATE'] == current_date]

    #split df into home and away dataframes to get relevant team metadata
    teams = teams_df.drop_duplicates().reset_index(drop = True)

    h_df = upcoming_games[['GAME_ID', 'GAME_DATE', 'home_team']]
    h_df = h_df.rename(columns = {'home_team': 'TEAM_NAME'})
    h_df = pd.merge(h_df, teams, how = 'inner', on = 'TEAM_NAME')
    h_df = h_df.assign(SEASON_ID = '22022')
    h_df = h_df.assign(MATCHUP = 'X vs. Y')
    h_df = h_df[['SEASON_ID', 'TEAM_ID', 'TEAM_NAME', 'GAME_ID', 'GAME_DATE', 'MATCHUP']]
    h_df = h_df.drop_duplicates()

    a_df = upcoming_games[['GAME_ID', 'GAME_DATE', 'away_team']]
    a_df = a_df.rename(columns = {'away_team': 'TEAM_NAME'})
    a_df = pd.merge(a_df, teams, how = 'inner', on = 'TEAM_NAME')
    a_df = a_df.assign(SEASON_ID = '22022')
    a_df = a_df.assign(MATCHUP = 'X @ Y')
    a_df = a_df[['SEASON_ID', 'TEAM_ID', 'TEAM_NAME', 'GAME_ID', 'GAME_DATE', 'MATCHUP']]
    a_df = a_df.drop_duplicates()

    upcoming_games = pd.concat([h_df, a_df], join = 'outer')
    upcoming_games = upcoming_games.drop_duplicates()
    upcoming_games.reset_index(drop = True, inplace = True)

    upcoming_games['SEASON_ID'] = upcoming_games['SEASON_ID'].astype(int)
    upcoming_games['SEASON_ID'] = upcoming_games['SEASON_ID'].astype(str)
    upcoming_games['TEAM_ID'] = upcoming_games['TEAM_ID'].astype(int)
    upcoming_games['TEAM_ID'] = upcoming_games['TEAM_ID'].astype(str)
    upcoming_games['GAME_ID'] = upcoming_games['GAME_ID'].astype(str)
    
    return upcoming_games, full_upcoming

def ma_transform(season_game_player, standings, teams_df):
    
    upcoming_games, full_upcoming = get_upcoming_games(teams_df)
    
    full_df = pd.concat([season_game_player, upcoming_games], join = 'outer')
    full_df.reset_index(drop = True, inplace = True)
    full_df.sort_values(by = ['GAME_DATE', 'GAME_ID', 'TEAM_ID'], inplace = True)
    full_df.drop(columns = ['MIN'], inplace = True)

    full_df['GAME_DATE'] = pd.to_datetime(full_df['GAME_DATE'], format = '%Y-%m-%d')
    full_df['SEASON_ID'] = full_df['SEASON_ID'].astype(int)
    full_df['SEASON_ID'] = full_df['SEASON_ID'].astype(str)
    full_df['TEAM_ID'] = pd.to_numeric(full_df['TEAM_ID'], downcast = 'integer')
    full_df['TEAM_ID'] = full_df['TEAM_ID'].astype(str)
    full_df['GAME_ID'] = full_df['GAME_ID'].astype(str)

    for col in full_df.select_dtypes(['int', 'float']):
        full_df[f'{col}_MA'] = full_df.groupby(['SEASON_ID', 'TEAM_ID'])[col].apply(lambda y: y.shift().expanding().mean())
        full_df[f'5DAY_MA_{col}'] = full_df.groupby(['SEASON_ID', 'TEAM_ID'])[col].apply(lambda y: y.shift().rolling(5, min_periods = 5).mean())

    full_df = full_df.drop(columns = ['FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS', 'G_E_OFF_RATING', 'G_E_DEF_RATING', 'F_E_OFF_RATING', 'F_E_DEF_RATING', 'C_E_OFF_RATING', 'C_E_DEF_RATING', 'BEN_E_OFF_RATING', 'BEN_E_DEF_RATING', 'F_PIE', 'G_PIE', 'C_PIE', 'BEN_PIE'])
    
    return full_df, upcoming_games, standings, full_upcoming

def transform_standings(season_game_player, standings, teams_df):
    
    full_df, full_upcoming, standings, full_upcoming = ma_transform(season_game_player, standings, teams_df)
    
    league_standings = standings[['SeasonID', 'TeamID', 'WINS', 'LOSSES', 'WinPCT', 'HOME', 'ROAD', 'L10', 'Last10Home', 'Last10Road']]
    league_standings['GAMES_PLAYED'] = league_standings['WINS'] + league_standings['LOSSES']
    league_standings.rename(columns = {'SeasonID': 'SEASON_ID',
                                       'TeamID': 'TEAM_ID',
                                       'WINS': 'WIN_TOTAL',
                                       'LOSSES': 'LOSS_TOTAL',
                                       'WinPCT': 'WIN_PCT',
                                       }, inplace = True)

    league_standings[['HOME_WIN_TOTAL', 'HOME_LOSS_TOTAL']] = league_standings['HOME'].str.split('-', expand = True)
    league_standings['HOME_WIN_PCT'] = league_standings['HOME_WIN_TOTAL'].astype(int) / league_standings['HOME_WIN_TOTAL'].astype(int) + league_standings['HOME_LOSS_TOTAL'].astype(int)
    league_standings[['HOME_LAST_10_W', 'HOME_LAST_10_L']] = league_standings['Last10Home'].str.split('-', expand = True)
    league_standings['HOME_LAST_10_PCT'] = league_standings['HOME_LAST_10_W'].astype(int) / 10

    league_standings[['ROAD_WIN_TOTAL', 'ROAD_LOSS_TOTAL']] = league_standings['ROAD'].str.split('-', expand = True)
    league_standings['ROAD_WIN_PCT'] = league_standings['ROAD_WIN_TOTAL'].astype(int) / league_standings['ROAD_WIN_TOTAL'].astype(int) + league_standings['ROAD_LOSS_TOTAL'].astype(int)
    league_standings[['ROAD_LAST_10_W', 'ROAD_LAST_10_L']] = league_standings['Last10Road'].str.split('-', expand = True)
    league_standings['ROAD_LAST_10_PCT'] = league_standings['ROAD_LAST_10_W'].astype(int) / 10

    league_standings[['LAST_10_W', 'LAST_10_L']] = league_standings['L10'].str.split('-', expand = True)
    league_standings['LAST_10_PCT'] = league_standings['LAST_10_W'].astype(int) / 10

    league_standings = league_standings.assign(GAME_DATE = current_date)
    league_standings = league_standings[['TEAM_ID', 'SEASON_ID', 'GAME_DATE', 'GAMES_PLAYED', 'WIN_TOTAL', 'LOSS_TOTAL', 'WIN_PCT', 'LAST_10_PCT', 'HOME_WIN_TOTAL', 'HOME_LOSS_TOTAL', 'HOME_WIN_PCT', 'HOME_LAST_10_PCT', 'ROAD_WIN_TOTAL', 'ROAD_LOSS_TOTAL', 'ROAD_WIN_PCT', 'ROAD_LAST_10_PCT']]

    league_standings['GAME_DATE'] = pd.to_datetime(league_standings['GAME_DATE'], format = '%Y-%m-%d')

    for col in league_standings.iloc[:,3:].columns:
        league_standings[col] = league_standings[col].astype('float')

    league_standings.drop_duplicates(inplace = True)
    league_standings.reset_index(drop = True, inplace = True)
    
    league_standings['SEASON_ID'] = league_standings['SEASON_ID'].astype(int)
    league_standings['SEASON_ID'] = league_standings['SEASON_ID'].astype(str)
    league_standings['TEAM_ID'] = league_standings['TEAM_ID'].astype(int)
    league_standings['TEAM_ID'] = league_standings['TEAM_ID'].astype(str)
    
    full_w_standings = pd.merge(full_df, league_standings, how = 'outer', on = ['SEASON_ID', 'TEAM_ID', 'GAME_DATE'])
    
    return full_w_standings, full_upcoming

def split_merge(season_game_player, standings, teams_df):
    
    full_w_standings, full_upcoming = transform_standings(season_game_player, standings, teams_df)
    
    full_df = full_w_standings.sort_values(by = ['GAME_DATE', 'GAME_ID', 'TEAM_ID'])
    full_df[['team', 'home_away', 'opp']] = full_df['MATCHUP'].str.split(' ', expand = True)
    full_df['home_away'].replace('vs.', 'H', inplace = True)
    full_df['home_away'].replace('@', 'V', inplace = True)
    full_df.drop(columns = ['team', 'opp', 'MATCHUP'], inplace = True)

    home_team_df = full_df[full_df['home_away'] == 'H']
    away_team_df = full_df[full_df['home_away'] == 'V']
    full_df = pd.merge(home_team_df, away_team_df, how = 'outer', on = ['GAME_ID'])

    full_df['WL_outcome'] = full_df['WL_x'].map({'W': 1, 'L': 0})
    full_df['WL_outcome'].fillna(0, inplace = True)
    full_df.rename(columns = {'SEASON_ID_x': 'SEASON_ID', 'GAME_DATE_x': 'GAME_DATE', 'WL_outcome_x': 'WL_outcome'}, inplace = True)
    full_df = full_df.drop(columns = ['WL_x', 'WL_y', 'home_away_x', 'home_away_y', 'SEASON_ID_y', 'TEAM_ABBREVIATION_x', 'TEAM_ABBREVIATION_y', 'TEAM_NAME_x', 'TEAM_NAME_y', 'GAME_DATE_y'])
    full_df.drop_duplicates(inplace = True)
    full_df = full_df[(full_df['TEAM_ID_x'].notnull()) & (full_df['TEAM_ID_y'].notnull())]
    full_df['PTS_x'].fillna(0, inplace = True)
    full_df['PTS_y'].fillna(0, inplace = True)
    full_df['WL_outcome'].fillna(0, inplace = True)
    
    return full_df, full_upcoming

def win_probs(*, home_elo, road_elo, hca_elo):
    """Home and road team win probabilities implied by Elo ratings and home court adjustment."""
    h = math.pow(10, home_elo/400)
    r = math.pow(10, road_elo/400)
    a = math.pow(10, hca_elo/400)
    denom = r + a*h
    home_prob = a*h / denom
    road_prob = r / denom
    return home_prob, road_prob

def home_odds_on(*, home_elo, road_elo, hca_elo):
    """Odds in favor of home team implied by Elo ratings and home court adjustment."""
    h = math.pow(10, home_elo/400)
    r = math.pow(10, road_elo/400)
    a = math.pow(10, hca_elo/400)
    return a*h/r

def hca_calibrate(*, home_win_prob):
    """Calibrate Elo home court adjustment to a given historical home team win percentage."""
    if home_win_prob <= 0 or home_win_prob >= 1:
        raise ValueError('invalid home win probability', home_win_prob)
    a = home_win_prob / (1 - home_win_prob)
    print(f'a = {a}')
    hca = 400 * math.log10(a)
    return hca

def update(winner, home_elo, road_elo, games_played, home_pts, road_pts, probs=False):
    """Update Elo ratings for a given match up."""
    home_prob, road_prob = win_probs(home_elo=home_elo, road_elo=road_elo, hca_elo=100)
    if winner == 1:
        home_win = 1
        road_win = 0
    elif winner == 0:
        home_win = 0
        road_win = 1
    elif winner == 'None':
        home_wim = 0
        road_win = 1
    else:
        raise ValueError('unrecognized winner string', winner)
        
    K_home, K_away = k_multiplier(home_elo, road_elo, home_pts, road_pts)
        
    if games_played != 0:
        new_home_elo = home_elo + K_home * (home_win - home_prob)
        new_road_elo = road_elo + K_away * (road_win - road_prob)
    else:
        new_home_elo = (0.75 * home_elo) + (0.25 * 1505)
        new_road_elo = (0.75 * road_elo) + (0.25 * 1505)
    
    if probs:
        return new_home_elo, new_road_elo, home_prob, road_prob
    else:
        return new_home_elo, new_road_elo
    
def k_multiplier(home_elo, road_elo, home_pts, away_pts):
    
    MOV = home_pts - away_pts
    elo_diff = home_elo - road_elo
    K_0 = 20
    
    if MOV>0:
        multiplier=(MOV+3)**(0.8)/(7.5+0.006*(elo_diff))
    else:
        multiplier=(-MOV+3)**(0.8)/(7.5+0.006*(-elo_diff))
    
    return K_0*multiplier,K_0*multiplier
    
        
def simple_nba_elo(box_scores):
    """Compute simple Elo ratings over the course of an NBA season."""
    teams = box_scores['TEAM_ID_x'].unique()
    latest_elos = {team_id: 1500 for team_id in teams}
    matchups = box_scores.sort_values(by='GAME_DATE', ascending=True).copy()
    home_probs = []
    road_probs = []
    home_elos = []
    road_elos = []
    index_check = []
    games_played = []
    elo_ts = []
    for game in box_scores.itertuples(index=True):
        index = game.Index
        home_game_count = game.GAMES_PLAYED_x
        home_team = game.TEAM_ID_x
        home_pts = game.PTS_x
        road_game_count = game.GAMES_PLAYED_y
        road_team = game.TEAM_ID_y
        road_pts = game.PTS_y
        winner = game.WL_outcome
        home_elo = latest_elos[home_team]
        road_elo = latest_elos[road_team]
        (new_home_elo, new_road_elo, home_prob, road_prob) = update(
            games_played=games_played,
            winner=winner,
            home_elo=home_elo,
            road_elo=road_elo,
            home_pts = home_pts,
            road_pts = road_pts,
            probs=True
        )
        home_info = OrderedDict({
            'GAME_DATE': game.GAME_DATE,
            'GAME_ID': game.GAME_ID,
            'home_road': 'H',
            'win_prob': home_prob,
            'prior_elo': latest_elos[home_team]
        })
        elo_ts.append(home_info)
        road_info = OrderedDict({
            'GAME_DATE': game.GAME_DATE,
            'GAME_ID': game.GAME_ID,
            'home_road': 'R',
            'win_prob': road_prob,
            'prior_elo': latest_elos[road_team]
        })
        elo_ts.append(road_info)
        latest_elos[home_team] = new_home_elo
        latest_elos[road_team] = new_road_elo
        home_probs.append(home_prob)
        road_probs.append(road_prob)
        home_elos.append(new_home_elo)
        road_elos.append(new_road_elo)
        index_check.append(index)
    matchups['home_prob'] = home_probs
    matchups['road_prob'] = road_probs
    matchups['home_elos'] = home_elos
    matchups['road_elos'] = road_elos
    matchups['index_check'] = index_check
    matchups = matchups.drop(columns=['index_check'])
    return matchups, pd.DataFrame(elo_ts), latest_elos

def with_elo(season_game_player, standings, teams_df):
    
    box_scores, full_upcoming = split_merge(season_game_player, standings, teams_df)
    matchups, elo_list, curr_elos = simple_nba_elo(box_scores = box_scores)

    home_elo = elo_list[elo_list['home_road'] == 'H']
    away_elo = elo_list[elo_list['home_road'] == 'R']
    
    home_elo.drop(columns = 'home_road', inplace = True)
    away_elo.drop(columns = 'home_road', inplace = True)
    
    df = pd.merge(box_scores, home_elo, how = 'left', on = ['GAME_ID', 'GAME_DATE'])
    df = pd.merge(df, away_elo, how = 'left', on = ['GAME_ID', 'GAME_DATE'])
    df.drop_duplicates(inplace = True)
    
    return df, full_upcoming

def pred_set(season_game_player, standings, team_df):
    
    game_set, full_upcoming = with_elo(season_game_player, standings, team_df)
    
    game_set = game_set[game_set['GAME_ID'].str.len() > 15]
    game_set =  game_set[[col for col in game_set if col not in ['WL_outcome']] + ['WL_outcome']]
    for_pred = game_set.drop(columns = ['SEASON_ID', 'TEAM_ID_x', 'TEAM_ID_y', 'GAME_ID', 'GAME_DATE', 'PTS_x', 'PTS_y', 'GAMES_PLAYED_x', 'GAMES_PLAYED_y'])
    for_display = game_set.drop(columns = ['SEASON_ID', 'TEAM_ID_x', 'TEAM_ID_y', 'PTS_x', 'PTS_y', 'GAMES_PLAYED_x', 'GAMES_PLAYED_y'])
    
    return for_pred, for_display, full_upcoming

def save_set(season_game_player):
    
    season_game_player.drop_duplicates()
    season_game_player.to_csv(f'Z:\\OneDrive\\C\\Documents\\modeling\\nba\\data\\concat stats\\game_player_2223_{current_date}.csv')
    
    return print('Saved!')
    

def run_predictions(pred_df, odds, lgb_runner, map_df):
    
    output = predict_model(data = pred_df, estimator = lgb_runner, verbose = False)

    output = output[['prediction_label', 'prediction_score']].reset_index(drop = False)
    output.reset_index(drop = False, inplace = True)
    team_pred = map_df[['GAME_ID']].reset_index(drop = False)
    team_pred.reset_index(drop = False, inplace = True)

    joined_output = pd.merge(team_pred, output, how = 'outer', on = 'level_0')
    joined_output.drop(columns = ['level_0', 'index_x', 'index_y'], inplace = True)

    final_output = pd.merge(odds, joined_output, how = 'outer', on = 'GAME_ID').drop(columns = ['GAME_ID', 'GAME_DATE'])
    
    book_price = final_output[['team', 'book', 'price']]

    final_output['winning_team'] = np.where(final_output['prediction_label'] == 1, final_output['home_team'], final_output['away_team'])
    final_output['losing_team'] = np.where(final_output['prediction_label'] == 0, final_output['home_team'], final_output['away_team'])

    results = final_output[['winning_team', 'losing_team', 'prediction_score']]

    home_book = pd.merge(results, book_price, left_on = 'winning_team', right_on = 'team', how = 'left').rename(columns = {'price': 'winners_odds'})
    away_book = pd.merge(home_book, book_price, left_on = 'losing_team', right_on = 'team', how = 'left').rename(columns = {'price': 'losers_odds', 'book_x':'book_winner_odds', 'book_y': 'book_loser_odds'}).drop(columns = ['team_x', 'team_y'])

    final_book_results = away_book.drop_duplicates().dropna().reset_index(drop = True)

    final_book_results = final_book_results[(final_book_results['book_winner_odds'] == 'DraftKings') & (final_book_results['book_loser_odds'] == 'DraftKings')]
    
    return final_book_results



current_date = datetime.now().strftime('%Y-%m-%d')
past_date = (pd.to_datetime(current_date) - pd.Timedelta(1, unit = 'D')).strftime('%Y-%m-%d')

hist_set = pd.read_csv('data/hist_set.csv')

teams_df = pd.read_csv('data/teams_df.csv')

last_date = pd.to_datetime(max(hist_set['GAME_DATE'])).strftime('%m/%d/%Y')

if last_date <= current_date:
    game_player, standings = game_player_stats(last_date)

else:
    print('DateError: Maximum date in dataset is greater than current_date. Check current_date function')

season_game_player = pd.concat([hist_set, game_player], join = 'outer')
save_set(season_game_player)

pred_df, map_df, odds = pred_set(season_game_player, standings, teams_df)
lgb_runner = load_model('lgb_v1_runner')

predictions = run_predictions(pred_df, odds, lgb_runner, map_df)

for index, row in predictions.iterrows():
    
    consumer_key = ''
    consumer_secret = ''
    access_token = ''
    access_token_secret = ''

    client = tweepy.Client(consumer_key = consumer_key,
                           consumer_secret = consumer_secret,
                           access_token = access_token,
                           access_token_secret = access_token_secret)

    response = client.create_tweet(text = f"{row['winning_team']} ({round(row['prediction_score'] * 100,1)}%) vs. {row['losing_team']}: {row['winners_odds']} ({row['book_winner_odds']})")
    print(f"{row['winning_team']} ({round(row['prediction_score'] * 100,1)}%) vs. {row['losing_team']}: {row['winners_odds']} ({row['book_winner_odds']})")
    print('winner winner chicken dinner')
