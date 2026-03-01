import streamlit as st
import requests
import pandas as pd
import joblib
from datetime import datetime, UTC
import os

st.set_page_config(
    page_title="DOTD Predictor",
    page_icon="🏆",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    background-color: #0d0d0d;
    color: #e8e8e8;
}

h1, h2, h3 {
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 2px;
}

.stApp {
    background-color: #0d0d0d;
}

.title-block {
    text-align: center;
    padding: 2rem 0 1rem;
}

.title-block h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.5rem;
    color: #f5f5f5;
    letter-spacing: 6px;
    margin: 0;
    line-height: 1;
}

.title-block p {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #666;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

.accent-line {
    height: 2px;
    background: linear-gradient(90deg, transparent, #ff4b00, transparent);
    margin: 1rem auto 2rem;
    width: 60%;
}

.rank-card {
    background: #161616;
    border: 1px solid #222;
    border-left: 3px solid #ff4b00;
    border-radius: 4px;
    padding: 1rem 1.4rem;
    margin-bottom: 0.6rem;
    font-family: 'DM Mono', monospace;
    display: flex;
    align-items: center;
    gap: 1.2rem;
}

.rank-num {
    font-size: 1.6rem;
    font-family: 'Bebas Neue', sans-serif;
    color: #ff4b00;
    min-width: 2rem;
    text-align: center;
}

.rank-num.gold   { color: #ffd700; }
.rank-num.silver { color: #b0b0b0; }
.rank-num.bronze { color: #cd7f32; }

.team-label {
    font-size: 1.1rem;
    font-weight: 500;
    color: #f0f0f0;
    min-width: 3.5rem;
    letter-spacing: 2px;
}

.vote-info {
    font-size: 0.7rem;
    color: #666;
    line-height: 1.6;
}

.vote-info span {
    color: #aaa;
}

.move-up   { color: #00cc66 !important; }
.move-down { color: #ff4b00 !important; }
.move-same { color: #555 !important; }

.stSelectbox > div > div {
    background-color: #161616 !important;
    border: 1px solid #333 !important;
    color: #e8e8e8 !important;
    font-family: 'DM Mono', monospace !important;
}

div[data-testid="stButton"] > button {
    background: #ff4b00;
    color: #fff;
    border: none;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1rem;
    letter-spacing: 3px;
    padding: 0.6rem 2.5rem;
    border-radius: 3px;
    width: 100%;
    cursor: pointer;
    transition: background 0.2s;
}

div[data-testid="stButton"] > button:hover {
    background: #e04000;
}

.error-box {
    background: #1a0a0a;
    border: 1px solid #ff4b00;
    border-radius: 4px;
    padding: 1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #ff7755;
}

.info-chip {
    display: inline-block;
    background: #1c1c1c;
    border: 1px solid #333;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #888;
    margin-bottom: 1.5rem;
}

footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

ID_URL = 'https://dotd-ids.tomfconreal.workers.dev/'
POLL_URL_TPL = 'https://dotd.tomfconreal.workers.dev/?pollId={}'

EXPECTED_COLUMNS = {
    "nba": [
        'placement', 'votes', 'odds', 'spread',
        'snapshot_time_ms', 'start_time_ms',
        'team_name_ATL', 'team_name_BKN', 'team_name_BOS', 'team_name_CHA',
        'team_name_CHI', 'team_name_CLE', 'team_name_DAL', 'team_name_DEN',
        'team_name_DET', 'team_name_GSW', 'team_name_HOU', 'team_name_IND',
        'team_name_LAC', 'team_name_LAL', 'team_name_MEM', 'team_name_MIA',
        'team_name_MIL', 'team_name_MIN', 'team_name_NOP', 'team_name_NYK',
        'team_name_ORL', 'team_name_PHI', 'team_name_PHX', 'team_name_POR',
        'team_name_SAC', 'team_name_SAS', 'team_name_TOR', 'team_name_UTA',
        'team_name_WAS', 'time_till_close',
    ],
    "nhl": [
        'placement', 'votes', 'odds',
        'snapshot_time_ms', 'start_time_ms',
        'team_name_ANA', 'team_name_BOS', 'team_name_BUF', 'team_name_CAR',
        'team_name_CBJ', 'team_name_CGY', 'team_name_CHI', 'team_name_DAL',
        'team_name_DET', 'team_name_FLA', 'team_name_LAK', 'team_name_MIN',
        'team_name_MTL', 'team_name_NJD', 'team_name_NSH', 'team_name_NYI',
        'team_name_NYR', 'team_name_OTT', 'team_name_PHI', 'team_name_PIT',
        'team_name_SEA', 'team_name_SJS', 'team_name_STL', 'team_name_TBL',
        'team_name_TOR', 'team_name_UTA', 'team_name_VAN', 'team_name_VGK',
        'team_name_WPG', 'team_name_WSH', 'time_till_close',
    ],
    "cbb": [
      'placement', 'votes', 'spread',
      'snapshot_time_ms', 'start_time_ms',
      'team_name_ALA', 'team_name_ARK', 'team_name_ASU', 'team_name_AUB',
      'team_name_BAY', 'team_name_BC ', 'team_name_BUT', 'team_name_BYU',
      'team_name_CAL', 'team_name_CIN', 'team_name_COL', 'team_name_CRE',
      'team_name_DAY', 'team_name_DEP', 'team_name_DUQ', 'team_name_FLA',
      'team_name_GCU', 'team_name_GT ', 'team_name_GTW', 'team_name_GW ',
      'team_name_HAL', 'team_name_ILL', 'team_name_IOW', 'team_name_IU ',
      'team_name_KEN', 'team_name_KSU', 'team_name_KU ', 'team_name_LOU',
      'team_name_LSU', 'team_name_MAS', 'team_name_MD ', 'team_name_MIA',
      'team_name_MIN', 'team_name_MIS', 'team_name_MIZ', 'team_name_MSS',
      'team_name_MSU', 'team_name_NCS', 'team_name_ND ', 'team_name_NEB',
      'team_name_NEV', 'team_name_NIU', 'team_name_NU ', 'team_name_OKS',
      'team_name_ORE', 'team_name_OSU', 'team_name_OU ', 'team_name_PEP',
      'team_name_PIT', 'team_name_PRO', 'team_name_PSU', 'team_name_RUT',
      'team_name_SBU', 'team_name_SC ', 'team_name_SCU', 'team_name_SEA',
      'team_name_SF ', 'team_name_SMC', 'team_name_SMU', 'team_name_STA',
      'team_name_SYR', 'team_name_TA&', 'team_name_TCU', 'team_name_TEN',
      'team_name_TEX', 'team_name_TTU', 'team_name_UCF', 'team_name_UCL',
      'team_name_UGA', 'team_name_UK ', 'team_name_UNC', 'team_name_USC',
      'team_name_UTA', 'team_name_UVA', 'team_name_VAN', 'team_name_VIL',
      'team_name_VT ', 'team_name_WAK', 'team_name_WAS', 'team_name_WIS',
      'team_name_WVU', 'time_till_close'
    ]
}


@st.cache_data(ttl=60)
def fetch_poll_data(sport: str):
    r = requests.get(ID_URL, timeout=10)
    r.raise_for_status()
    if sport == 'cbb':
        dotd_id = r.json()['ncaam']
    else:
        dotd_id = r.json()[sport]

    r2 = requests.get(POLL_URL_TPL.format(dotd_id), timeout=10)
    r2.raise_for_status()
    poll = r2.json()['poll']

    now = datetime.now(UTC)
    rows = []
    for option in poll['options']:
        ai = option['additionalInfo']
        rows.append({
            'team_name': option['label'][:3],
            'current_utc_time': now,
            'utc_start_time': option['locksAt'],
            'votes': option['count'],
            'odds': ai.get('odds', 'N/A'),
            'spread_value': ai.get('spread', 'N/A'),
        })

    df = pd.DataFrame(rows)

    # placement (rank by current votes, 1 = most)
    sorted_votes = sorted(df['votes'].tolist(), reverse=True)
    df['placement'] = df['votes'].apply(lambda v: sorted_votes.index(v) + 1)

    return df


def make_preds(sport: str, df: pd.DataFrame, model):
    df = df.copy()

    df['snapshot_time_ms'] = pd.to_datetime(df['current_utc_time']).astype('int64') // 10**6
    df['start_time_ms'] = pd.to_datetime(df['utc_start_time']).astype('int64') // 10**6
    df['time_till_close'] = (df['start_time_ms'] - df['snapshot_time_ms']) / 1000

    # CBB uses 'spread' column name instead of 'spread_value'
    if sport == 'cbb':
        df = df.rename(columns={'spread_value': 'spread'})

    df_dummies = pd.get_dummies(df, columns=['team_name'])
    x = df_dummies.reindex(columns=EXPECTED_COLUMNS[sport], fill_value=0)

    preds = model.predict(x)
    df['predicted_final_votes'] = preds.astype(int)
    df['rank'] = df['predicted_final_votes'].rank(ascending=False, method='min').astype(int)
    return df[['team_name', 'votes', 'placement', 'predicted_final_votes', 'rank']].sort_values('rank')


def load_model(sport: str):
    path = f'{sport}_dotd_ml.pkl'
    if not os.path.exists(path):
        return None, f"Model file `{path}` not found. Make sure it's in the same directory as this app."
    return joblib.load(path), None


def rank_color(r):
    return {1: 'gold', 2: 'silver', 3: 'bronze'}.get(r, '')

st.markdown("""
<div class="title-block">
    <h1>DOTD PREDICTOR</h1>
    <p>Dog of the Day · Vote Forecast</p>
</div>
<div class="accent-line"></div>
""", unsafe_allow_html=True)

sport_map = {"🏀  NBA": "nba", "🏒  NHL": "nhl", "🏫  CBB": "cbb"}
sport_label = st.selectbox("Select sport", list(sport_map.keys()), label_visibility="collapsed")
sport = sport_map[sport_label]

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    run = st.button("RUN PREDICTION")

if run:
    model, err = load_model(sport)
    if err:
        st.markdown(f'<div class="error-box">⚠ {err}</div>', unsafe_allow_html=True)
    else:
        with st.spinner("Fetching live poll data…"):
            try:
                df = fetch_poll_data(sport)
            except Exception as e:
                st.markdown(f'<div class="error-box">⚠ Could not fetch poll data: {e}</div>', unsafe_allow_html=True)
                st.stop()

        snapshot_time = df['current_utc_time'].iloc[0]
        st.markdown(f'<div class="info-chip">snapshot · {snapshot_time.strftime("%H:%M:%S UTC")}</div>', unsafe_allow_html=True)

        try:
            results = make_preds(sport, df, model)
        except Exception as e:
            st.markdown(f'<div class="error-box">⚠ Prediction error: {e}</div>', unsafe_allow_html=True)
            st.stop()

        st.markdown(f"### PREDICTED RANKINGS")

        for _, row in results.iterrows():
            rank = int(row['rank'])
            current_rank = int(row['placement'])
            rc = rank_color(rank)

            diff = current_rank - rank  # positive = moving up, negative = moving down
            if diff > 0:
                movement = f'<span class="move-up">▲ {diff}</span>'
            elif diff < 0:
                movement = f'<span class="move-down">▼ {abs(diff)}</span>'
            else:
                movement = f'<span class="move-same">—</span>'

            st.markdown(f"""
            <div class="rank-card">
                <div class="rank-num {rc}">#{rank}</div>
                <div class="team-label">{row['team_name']}</div>
                <div class="vote-info">
                    Now: <span>#{current_rank}</span> &nbsp;→&nbsp; Predicted: <span>#{rank}</span> &nbsp;{movement}<br>
                    Current votes: <span>{int(row['votes']):,}</span><br>
                    Predicted final: <span>{int(row['predicted_final_votes']):,}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)