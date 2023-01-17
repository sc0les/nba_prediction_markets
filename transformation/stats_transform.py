def add_new_games(date_from):

    from nba_api.stats.endpoints import leaguegamefinder

    df = pd.DataFrame(columns = ['SEASON_ID'])
    gamefinder = leaguegamefinder.LeagueGameFinder(date_from_nullable = date_from)
    games = gamefinder.get_data_frames()[0]
    games['SEASON_ID'] = games['SEASON_ID'].astype(str)
    games['TEAM_ID'] = games['TEAM_ID'].astype(str)
    games['GAME_ID'] = games['GAME_ID'].astype(str)
    
    df = pd.concat([df, games], join = 'outer')
    df = pd.merge(teams, df, how = 'left', on = ['TEAM_ID'])
    df.drop(columns = ['TEAM_NAME_x', 'TEAM_ABBREVIATION_x', 'SEASON_ID_x'], inplace = True)
    df.rename(columns = {'TEAM_NAME_y': 'TEAM_NAME', 'TEAM_ABBREVIATION_y': 'TEAM_ABBREVIATION', 'SEASON_ID_y': 'SEASON_ID'}, inplace = True)
    df.drop_duplicates(inplace = True)
    df = df[(df['WL'] == 'W') | (df['WL'] == 'L')]
    
    return df
    
def get_upcoming_games(standings):
    
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

    #create dataframe from API response
    df = pd.DataFrame(odds_json)
    
    # change datetime to EST date
    df = df[['id', 'commence_time', 'home_team', 'away_team']]
    df['commence_time'] = pd.to_datetime(df['commence_time'], format = '%Y-%m-%d')
    output = (df.set_index('commence_time')
                         .tz_convert("US/Eastern")
                         .reset_index()
              )
    df['commence_time'] = output['commence_time']
    df['commence_time'] = df['commence_time'].apply(lambda x: x.date().strftime('%Y-%m-%d'))
    df.rename(columns = {'id': 'GAME_ID', 'commence_time': 'GAME_DATE'}, inplace = True)
    
    #split df into home and away dataframes to get relevant team metadata
    h_df = df[['GAME_ID', 'GAME_DATE', 'home_team']]
    h_df.rename(columns = {'home_team': 'TEAM_NAME'}, inplace = True)
    teams_temp = teams[['TEAM_NAME', 'TEAM_ID']]
    abbr_temp = teams[teams['SEASON_ID'] == max(standings['SEASON_ID'])][['TEAM_ABBREVIATION', 'TEAM_ID']]
    h_df = pd.merge(h_df, teams_temp, how = 'left', on = 'TEAM_NAME')
    h_df = pd.merge(h_df, abbr_temp, how = 'left', on = 'TEAM_ID')
    h_df.drop_duplicates(inplace = True)
    h_df = h_df.assign(SEASON_ID = max(standings['SEASON_ID']))
    h_df = h_df.assign(MATCHUP = 'X vs. Y')
    h_df = h_df[['TEAM_ID', 'SEASON_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'GAME_ID', 'GAME_DATE', 'MATCHUP']]
    h_df.reset_index(drop = True, inplace = True)
    
    a_df = df[['GAME_ID', 'GAME_DATE', 'away_team']]
    a_df.rename(columns = {'away_team': 'TEAM_NAME'}, inplace = True)
    a_df = pd.merge(a_df, teams_temp, how = 'left', on = 'TEAM_NAME')
    a_df = pd.merge(a_df, abbr_temp, how = 'left', on = 'TEAM_ID')
    a_df.drop_duplicates(inplace = True)
    a_df = a_df.assign(SEASON_ID = max(standings['SEASON_ID']))
    a_df = a_df.assign(MATCHUP = 'X @ Y')
    a_df = a_df[['TEAM_ID', 'SEASON_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'GAME_ID', 'GAME_DATE', 'MATCHUP']]
    a_df.reset_index(drop = True, inplace = True)
    
    upcoming_df = pd.concat([h_df, a_df], join = 'outer')
    upcoming_df['GAME_DATE'] = pd.to_datetime(upcoming_df['GAME_DATE'], format = '%Y-%m-%d')
    upcoming_df['TEAM_ID'] = upcoming_df['TEAM_ID'].astype(str)
    upcoming_df.drop_duplicates(inplace = True)
    upcoming_df.reset_index(drop = True, inplace = True)
    
    return upcoming_df
    
def final_game_set(hist_game_logs, new_game_logs, upcoming_games):

    h = hist_game_logs
    n = new_game_logs
    u = upcoming_games
    
    h['GAME_ID'] = h['GAME_ID'].astype(int)
    h['GAME_ID'] = h['GAME_ID'].astype(str)
    n['GAME_ID'] = n['GAME_ID'].astype(int)
    n['GAME_ID'] = n['GAME_ID'].astype(str)

    df = pd.concat([h, n, u], join = 'outer')
    df = df[pd.notnull(df['GAME_DATE'])]
    df.drop_duplicates(inplace = True)
    df.reset_index(drop = True, inplace = True)
    
    # if data is correct and merge was successful, all games should have 2 teams

    # creating dataframe to count the number of GAME_ID occurences
    team_counts = df.groupby(['GAME_ID'])['GAME_ID'].count()
    team_counts = pd.DataFrame(team_counts)

    # dropping games if GAME_ID is found more or less than twice
    for count in team_counts.iterrows():
        if count[1][0] > 2:
            print(count[0])
            df = df[df['GAME_ID'] != count[0]]

        else:
            pass

        if count[1][0] < 2:
            print(count[0])
            df = df[df['GAME_ID'] != count[0]]

        else:
            continue
            
    # to ensure only regular season games, remove all SEASON_IDs that do not start with a 2
    df = df[df['SEASON_ID'].str[0] == '2']
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format = '%Y-%m-%d')
    
    df.drop_duplicates(inplace = True)
    
    return df
    
def update_player_data(game_stats, player_data_set):
    
    game_stats = game_stats
    o_player = player_data_set
    f_player = pd.DataFrame()
    
    from nba_api.stats.endpoints import boxscoreadvancedv2
    from nba_api.stats.endpoints import boxscoretraditionalv2
    from ratelimiter import RateLimiter
    from functools import reduce

    merged_player_game = game_stats.merge(o_player.drop_duplicates(), on = ['TEAM_ID', 'GAME_ID'], how = 'left', indicator = True)
    missing_games = merged_player_game[(merged_player_game['_merge'] == 'left_only') & (merged_player_game['GAME_DATE_x'] < current_date)]['GAME_ID']
    game_list = list(missing_games)
    
    players_advanced = pd.DataFrame(columns = ['GAME_ID'])
    players_traditional = pd.DataFrame(columns = ['GAME_ID'])

    rate_limiter = RateLimiter(max_calls = 1, period = 10)
    
    x = 0

    for game in game_list:
        if len(game_list) > 0:
            with rate_limiter:
                game = f'00{game}'
                #for the old details
                get = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id = game)
                player_adv = get.get_data_frames()[0]
                players_advanced = pd.concat([players_advanced, player_adv], join = 'outer')

                #for the new details
                get = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id = game)
                player_trad = get.get_data_frames()[0]
                players_traditional = pd.concat([players_traditional, player_trad], join = 'outer')
                
            x = 1
        else:
            x = 0
            pass

    if x == 1:
        players_advanced['PLAYER_ID'] = players_advanced['PLAYER_ID'].astype(int)
        players_advanced['PLAYER_ID'] = players_advanced['PLAYER_ID'].astype(str)
        players_advanced['TEAM_ID'] = players_advanced['TEAM_ID'].astype(int)
        players_advanced['TEAM_ID'] = players_advanced['TEAM_ID'].astype(str)
        players_advanced['GAME_ID'] = players_advanced['GAME_ID'].astype(int)
        players_advanced['GAME_ID'] = players_advanced['GAME_ID'].astype(str)

        players_traditional['PLAYER_ID'] = players_traditional['PLAYER_ID'].astype(int)
        players_traditional['PLAYER_ID'] = players_traditional['PLAYER_ID'].astype(str)
        players_traditional['TEAM_ID'] = players_traditional['TEAM_ID'].astype(int)
        players_traditional['TEAM_ID'] = players_traditional['TEAM_ID'].astype(str)
        players_traditional['GAME_ID'] = players_traditional['GAME_ID'].astype(int)
        players_traditional['GAME_ID'] = players_traditional['GAME_ID'].astype(str)

        new_players = pd.merge(players_traditional, players_advanced, how = 'inner', on = ['GAME_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_CITY', 'PLAYER_ID', 'PLAYER_NAME', 'START_POSITION', 'COMMENT'])
        new_players.drop(columns = 'MIN_y', inplace = True)
        new_players.rename(columns = {'MIN_x': 'MIN'}, inplace = True)

        a_player = pd.concat([o_player, new_players], join = 'outer')
        a_player.drop_duplicates(inplace = True)
        a_player.reset_index(drop = True, inplace = True)
        
        f_player = a_player
        
    else:
        f_player = o_player

    player_stats = f_player[['GAME_ID', 'TEAM_ID', 'START_POSITION', 'MIN', 'E_OFF_RATING', 'E_DEF_RATING', 'PIE']]
    player_stats = player_stats[player_stats['MIN'].notnull()]
    player_stats['START_POSITION'].fillna('bench', inplace = True)

    player_stats[['MIN', 'SEC']] = player_stats['MIN'].str.split(':', expand = True)
    player_stats['MIN'] = pd.to_numeric(player_stats['MIN'])
    player_stats['SEC'] = pd.to_numeric(player_stats['SEC']) / 60
    player_stats['PLAYING_TIME_M'] = player_stats['MIN'] + player_stats['SEC']
    player_stats.drop(columns = ['MIN', 'SEC'], inplace = True)
    player_stats = player_stats[player_stats['PLAYING_TIME_M'] >= 12]

    f = player_stats[(player_stats['START_POSITION'] == 'F')]
    f['E_OFF_RATING_W'] = f['E_OFF_RATING'] * f['PLAYING_TIME_M']
    f['E_DEF_RATING_W'] = f['E_DEF_RATING'] * f['PLAYING_TIME_M']
    f['PIE_W'] = f['PIE'] * f['PLAYING_TIME_M']
    f.drop(columns = ['START_POSITION', 'E_OFF_RATING', 'E_DEF_RATING', 'PLAYING_TIME_M', 'PIE'], inplace = True)
    f = f.groupby(['TEAM_ID', 'GAME_ID']).mean()
    f.reset_index(inplace = True)

    g = player_stats[(player_stats['START_POSITION'] == 'G')]
    g['E_OFF_RATING_W'] = g['E_OFF_RATING'] * g['PLAYING_TIME_M']
    g['E_DEF_RATING_W'] = g['E_DEF_RATING'] * g['PLAYING_TIME_M']
    g['PIE_W'] = g['PIE'] * g['PLAYING_TIME_M']
    g.drop(columns = ['START_POSITION', 'E_OFF_RATING', 'E_DEF_RATING', 'PLAYING_TIME_M', 'PIE'], inplace = True)
    g = g.groupby(['TEAM_ID', 'GAME_ID']).mean()
    g.reset_index(inplace = True)


    c = player_stats[(player_stats['START_POSITION'] == 'C')]
    c['E_OFF_RATING_W'] = c['E_OFF_RATING'] * c['PLAYING_TIME_M']
    c['E_DEF_RATING_W'] = c['E_DEF_RATING'] * c['PLAYING_TIME_M']
    c['PIE_W'] = c['PIE'] * c['PLAYING_TIME_M']
    c.drop(columns = ['START_POSITION', 'E_OFF_RATING', 'E_DEF_RATING', 'PLAYING_TIME_M', 'PIE'], inplace = True)
    c = c.groupby(['TEAM_ID', 'GAME_ID']).mean()
    c.reset_index(inplace = True)


    b = player_stats[(player_stats['START_POSITION'] == 'bench')]
    b['E_OFF_RATING_W'] = b['E_OFF_RATING'] * b['PLAYING_TIME_M']
    b['E_DEF_RATING_W'] = b['E_DEF_RATING'] * b['PLAYING_TIME_M']
    b['PIE_W'] = b['PIE'] * b['PLAYING_TIME_M']
    b.drop(columns = ['START_POSITION', 'E_OFF_RATING', 'E_DEF_RATING', 'PLAYING_TIME_M', 'PIE'], inplace = True)
    b = b.groupby(['TEAM_ID', 'GAME_ID']).mean()
    b.reset_index(inplace = True)

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
        
    return f_player, player_df
    
def poss_reg(game_set, player_set):
    
    game_set = game_set
    player = player_set
    
    poss_stat = player[['SEASON_ID', 'TEAM_ID', 'GAME_ID', 'GAME_DATE', 'POSS']]
    poss_stat.drop_duplicates(inplace = True)
    
    poss_stat['SEASON_ID'] = poss_stat['SEASON_ID'].astype(str)
    poss_stat['TEAM_ID'] = poss_stat['TEAM_ID'].astype(str)
    poss_stat['GAME_ID'] = poss_stat['GAME_ID'].astype(str)

    poss_stat = pd.DataFrame(poss_stat.groupby(['TEAM_ID', 'GAME_ID'])['POSS'].sum())
    poss_stat.reset_index(drop = False, inplace = True)
    poss_stat['POSS'] = poss_stat['POSS'] / 5
    
    poss_df = pd.merge(game_set, poss_stat, how = 'outer', on = ['TEAM_ID', 'GAME_ID'])
    poss_df['POSS'].fillna(100, inplace = True)
    
    for col in poss_df.select_dtypes(['int', 'float']):
        poss_df[col] = (poss_df[col] / poss_df['POSS']) * 100
        poss_df[col].round(4)
        
    poss_df.drop_duplicates(inplace = True)
    poss_df = poss_df[poss_df['GAME_DATE'].notnull()]
    poss_df.reset_index(drop = True, inplace = True)
    
    return poss_df
    
def moving_avg_transform(poss_reg):
    
    df = poss_reg.sort_values(by = ['GAME_DATE', 'GAME_ID', 'TEAM_ID'])
    x = [5]
    df = df.drop(columns = 'MIN')
    
    for col in df.select_dtypes(['int', 'float']):
        df[f'{col}_MA'] = df.groupby(['SEASON_ID', 'TEAM_ID'])[col].apply(lambda y: y.shift().expanding().mean())
        for per in x:
            df[f'{per}DAY_MA_{col}'] = df.groupby(['SEASON_ID', 'TEAM_ID'])[col].apply(lambda y: y.shift().rolling(per, min_periods = 5).mean())
    
    df = df.drop(columns = ['PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS', 'G_E_OFF_RATING_W', 'G_E_DEF_RATING_W', 'F_E_OFF_RATING_W', 'F_E_DEF_RATING_W', 'C_E_OFF_RATING_W', 'C_E_DEF_RATING_W', 'BEN_E_OFF_RATING_W', 'BEN_E_DEF_RATING_W', 'POSS', 'F_PIE_W', 'G_PIE_W', 'C_PIE_W', 'BEN_PIE_W'])
    
    df.drop_duplicates(inplace = True)
    df = df[df['GAME_DATE'].notnull()]
    df.sort_values(by = ['GAME_DATE', 'GAME_ID', 'TEAM_ID'], inplace = True)
    df.reset_index(drop = True, inplace = True)
    
    return df
    
def stat_rank(moving_avg_transform):
    
    df = moving_avg_transform.sort_values(by = ['GAME_DATE', 'GAME_ID', 'TEAM_ID'])
    
    for col in df.select_dtypes(['int', 'float']):
        df[f'{col}_rank'] = df.groupby(['SEASON_ID', 'GAMES_PLAYED'])[col].rank('max')
    
    df.drop_duplicates(inplace = True)
    df.reset_index(drop = True, inplace = True)
    
    return df
    
def game_merge(stat_rank):
    
    full_df = stat_rank.sort_values(by = ['GAME_DATE', 'GAME_ID', 'TEAM_ID'])
    
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
    
    return full_df
    
