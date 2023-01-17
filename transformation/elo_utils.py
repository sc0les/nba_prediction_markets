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
        home_game_count = game.GAME_COUNT_x
        home_team = game.TEAM_ID_x
        home_pts = game.PTS_x
        road_game_count = game.GAME_COUNT_y
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
            'abbr': home_team,
            'matchup_index': index,
            'opp_abbr': road_team,
            'home_road': 'H',
            'win_prob': home_prob,
            'opp_prior_elo': latest_elos[road_team],
            'prior_elo': latest_elos[home_team],
            'new_elo': new_home_elo,
        })
        elo_ts.append(home_info)
        road_info = OrderedDict({
            'GAME_DATE': game.GAME_DATE,
            'GAME_ID': game.GAME_ID,
            'abbr': road_team,
            'matchup_index': index,
            'opp_abbr': home_team,
            'home_road': 'R',
            'win_prob': road_prob,
            'opp_prior_elo': latest_elos[home_team],
            'prior_elo': latest_elos[road_team],
            'new_elo': new_road_elo,
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
