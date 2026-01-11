import streamlit as st
import streamlit_authenticator as stauth
import bcrypt
import base64
import pandas as pd
import numpy as np
import os
import altair as alt
from nba_api.stats.endpoints import leaguegamelog

# ==========================================
# 1. Configuration & Path Management
# ==========================================
st.set_page_config(page_title="Shot Decision Optimizer", layout="wide", page_icon="🏀")

current_dir = os.path.dirname(os.path.abspath(__file__))

def get_img_path(filename):
    """Holt den absoluten Pfad für ein Bild im aktuellen Ordner."""
    return os.path.join(current_dir, filename)

@st.cache_data
def get_base64_of_bin_file(bin_file):
    """Wandelt Bild in Base64 um für HTML-Embedding."""
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

# ==========================================
# 2. Design & Layout (CSS)
# ==========================================

def set_modern_nba_style():
    style = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

    :root {
        --nba-blue: #1D428A;
        --nba-red: #C8102E;
        --dark-bg: #0b1116;
        --card-bg: rgba(255, 255, 255, 0.05);
        --card-border: rgba(255, 255, 255, 0.1);
        --text-primary: #ffffff;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }

    .stApp {
        background-color: var(--dark-bg);
        background-image: 
            radial-gradient(circle at 10% 10%, rgba(29, 66, 138, 0.4) 0%, transparent 40%),
            radial-gradient(circle at 90% 90%, rgba(200, 16, 46, 0.15) 0%, transparent 40%),
            radial-gradient(circle at 50% 50%, rgba(255, 255, 255, 0.02) 0%, transparent 60%);
        background-attachment: fixed;
    }

    /* Standard Container Styling */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: var(--card-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Button Styling */
    div[data-testid="column"] .stButton > button {
        background: rgba(255,255,255,0.1);
        color: white;
        border: 1px solid rgba(255,255,255,0.2);
        width: 100%;
        transition: all 0.3s ease;
    }
    div[data-testid="column"] .stButton > button:hover {
        background: var(--nba-blue);
        border-color: var(--nba-blue);
        transform: translateY(-2px);
    }

    /* FIX: Alignment für Header Buttons */
    div[data-testid="stHorizontalBlock"] {
        align-items: center;
    }

    /* Typography helpers */
    .card-date { font-size: 0.75rem; color: #8a9ba8; text-align: center; text-transform: uppercase; margin-bottom: 10px; }
    .card-score { font-size: 2.2rem; font-weight: 900; text-align: center; color: white; line-height: 1; text-shadow: 0 4px 15px rgba(0,0,0,0.5); }
    
    /* --- LOGIN PAGE CSS --- */
    .login-hero-container {
        position: relative;
        height: 75vh;
        width: 100%;
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 20px 50px rgba(0,0,0,0.5);
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        padding: 40px;
    }
    .hero-overlay {
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background: linear-gradient(to bottom, rgba(0,0,0,0.1), rgba(11, 17, 22, 0.95));
        z-index: 1;
    }
    .hero-content { position: relative; z-index: 2; }
    .hero-title {
        font-size: 3.5rem; font-weight: 900; line-height: 1.1; margin-bottom: 15px;
        background: -webkit-linear-gradient(45deg, #fff, #A8DADC);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero-subtitle { font-size: 1.1rem; color: #ccc; font-weight: 300; line-height: 1.5; max-width: 80%; }

    /* Login Form Clean up */
    div[data-testid="stForm"] h1, div[data-testid="stForm"] h2, div[data-testid="stForm"] h3 { display: none !important; }
    div[data-testid="stForm"] { border: none !important; background-color: transparent !important; box-shadow: none !important; padding: 0 !important; }
    
    /* Header Spacing Fix */
    .block-container { padding-top: 1rem !important; padding-bottom: 0 !important; }
    
    /* Custom Back Button Style */
    div[data-testid="column"]:nth-of-type(1) .stButton > button {
        background-color: rgba(255, 255, 255, 0.05);
        color: #e0e0e0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 30px;
        padding: 0px 20px;
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 1px;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        height: 40px;
    }
    div[data-testid="column"]:nth-of-type(1) .stButton > button:hover {
        background-color: rgba(255, 255, 255, 0.15);
        border-color: #ffffff;
        color: #ffffff;
        box-shadow: 0 0 15px rgba(255, 255, 255, 0.1);
        transform: translateX(-5px);
    }
    div[data-testid="column"]:nth-of-type(1) .stButton > button:active {
        background-color: rgba(255, 255, 255, 0.25);
        transform: translateX(-2px);
    }
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)

# ==========================================
# 3. Helper Functions
# ==========================================

def render_comparison_stats(home_stats, away_stats):
    metrics = [
        ("FG_PCT", "FG%", True), ("FG3_PCT", "3P%", True), ("FT_PCT", "FT%", True),
        ("REB", "REB", False), ("AST", "AST", False), ("STL", "STL", False),
        ("BLK", "BLK", False), ("TOV", "TOV", False), ("PTS", "PTS", False)
    ]

    html = """
    <style>
        .comp-container { display: flex; flex-direction: column; width: 100%; height: 100%; justify-content: center; }
        .comp-row { display: flex; align-items: center; justify-content: space-between; padding: 12px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.05); }
        .comp-val { flex: 1; font-size: 1.1rem; font-weight: 700; color: white; width: 35%; }
        .comp-val.left { text-align: right; padding-right: 15px; }
        .comp-val.right { text-align: left; padding-left: 15px; }
        .comp-label { width: 60px; text-align: center; font-size: 0.75rem; color: #8a9ba8; font-weight: 600; text-transform: uppercase; background: rgba(255,255,255,0.05); padding: 4px 0; border-radius: 4px; }
    </style>
    <div class="comp-container">
    """

    for key, label, is_pct in metrics:
        val_home = home_stats[key]
        val_away = away_stats[key]
        str_home = f"{val_home * 100:.1f}%" if is_pct else str(int(val_home))
        str_away = f"{val_away * 100:.1f}%" if is_pct else str(int(val_away))

        color_home, color_away = "", ""
        if key == "TOV": 
            if val_home < val_away: color_home = "color: #4CAF50;" 
            elif val_away < val_home: color_away = "color: #4CAF50;"
        else: 
            if val_home > val_away: color_home = "color: #4CAF50;"
            elif val_away > val_home: color_away = "color: #4CAF50;"

        row_html = f"<div class='comp-row'><div class='comp-val left' style='{color_home}'>{str_home}</div><div class='comp-label'>{label}</div><div class='comp-val right' style='{color_away}'>{str_away}</div></div>"
        html += row_html
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def get_court_lines_data():
    lines = []
    lines.append(pd.DataFrame({'LOC_X': [-250, 250, 250, -250, -250], 'LOC_Y': [-47.5, -47.5, 422.5, 422.5, -47.5], 'group': 'outer'}))
    lines.append(pd.DataFrame({'LOC_X': [-80, -80, 80, 80], 'LOC_Y': [-47.5, 142.5, 142.5, -47.5], 'group': 'paint_outer'}))
    lines.append(pd.DataFrame({'LOC_X': [-60, -60, 60, 60], 'LOC_Y': [-47.5, 142.5, 142.5, -47.5], 'group': 'paint_inner'}))
    lines.append(pd.DataFrame({'LOC_X': [-30, 30], 'LOC_Y': [-7.5, -7.5], 'group': 'backboard'}))
    theta = np.linspace(0, np.pi, 50)
    lines.append(pd.DataFrame({'LOC_X': 60 * np.cos(theta), 'LOC_Y': 142.5 + 60 * np.sin(theta), 'group': 'ft_circle_top'}))
    theta2 = np.linspace(np.pi, 2*np.pi, 50)
    lines.append(pd.DataFrame({'LOC_X': 60 * np.cos(theta2), 'LOC_Y': 142.5 + 60 * np.sin(theta2), 'group': 'ft_circle_bottom'}))
    theta_ra = np.linspace(0, np.pi, 50)
    lines.append(pd.DataFrame({'LOC_X': 40 * np.cos(theta_ra), 'LOC_Y': 40 * np.sin(theta_ra), 'group': 'restricted_area'}))
    lines.append(pd.DataFrame({'LOC_X': [-220, -220], 'LOC_Y': [-47.5, 92.5], 'group': '3pt_side_l'}))
    lines.append(pd.DataFrame({'LOC_X': [220, 220], 'LOC_Y': [-47.5, 92.5], 'group': '3pt_side_r'}))
    angle = np.arccos(220 / 237.5)
    theta_3pt = np.linspace(angle, np.pi - angle, 60)
    lines.append(pd.DataFrame({'LOC_X': 237.5 * np.cos(theta_3pt), 'LOC_Y': 237.5 * np.sin(theta_3pt), 'group': '3pt_arc'}))
    theta_center = np.linspace(np.pi, 2*np.pi, 50)
    lines.append(pd.DataFrame({'LOC_X': 60 * np.cos(theta_center), 'LOC_Y': 422.5 + 60 * np.sin(theta_center), 'group': 'center_circle'}))
    return pd.concat(lines, ignore_index=True)

def create_full_court_plot(df_shots):
    df_plot = df_shots.copy()
    
    # X-Achse spiegeln (gegen Seitenverkehrtheit)
    df_plot['LOC_X'] = -df_plot['LOC_X']
    
    # Text-Labels für Tooltips
    df_plot['Result'] = df_plot['SHOT_MADE_FLAG'].apply(lambda x: 'Made' if x == 1 else 'Missed')
    
    # Zeit-Formatierung
    if 'PERIOD' in df_plot.columns and 'MINUTES_REMAINING' in df_plot.columns:
        df_plot['Time_Display'] = df_plot.apply(lambda x: f"Q{int(x['PERIOD'])} {int(x['MINUTES_REMAINING'])}:{int(x['SECONDS_REMAINING']):02d}", axis=1)
    else:
        df_plot['Time_Display'] = "N/A"
    
    # Spielfeld-Daten holen
    court_data = get_court_lines_data()
    hoop_data = pd.DataFrame({'LOC_X': [0], 'LOC_Y': [0]})

    # Layer 1: Spielfeld-Linien
    court_layer = alt.Chart(court_data).mark_line(color='white', opacity=0.5, strokeWidth=2).encode(
        x=alt.X('LOC_X', axis=None, scale=alt.Scale(domain=[-250, 250])),
        y=alt.Y('LOC_Y', axis=None, scale=alt.Scale(domain=[-50, 420])),
        detail='group'
    )
    
    # Layer 2: Korb
    hoop_layer = alt.Chart(hoop_data).mark_circle(size=150, color='orange', opacity=1, stroke='white').encode(x='LOC_X', y='LOC_Y')

    # Tooltip-Konfiguration (Bereinigt)
    tooltips = [
        alt.Tooltip('PLAYER_NAME', title='Player'),
        alt.Tooltip('predicted_shot_probability', title='eFG%', format='.2%'),
        alt.Tooltip('Result', title='Result'),
        alt.Tooltip('SHOTCLOCK', title='Shot Clock'),
        alt.Tooltip('Time_Display', title='Time')
    ]

    # --- FIX: Wir nutzen wieder href, da Streamlit on_select nicht für Layered Charts unterstützt ---
    # Layer 3: Würfe
    shots_layer = alt.Chart(df_plot).mark_circle(
        size=100, 
        stroke='black', 
        strokeWidth=1,
        cursor='pointer' # Cursor gehört HIER hin, nicht in encode
    ).encode(
        x='LOC_X', 
        y='LOC_Y',
        color=alt.Color('SHOT_MADE_FLAG:Q', scale=alt.Scale(domain=[0, 1], range=['#C8102E', '#4CAF50']), legend=None),
        tooltip=tooltips,
        href='video_url:N' # Direkter Link auf dem Punkt
    )

    # Zusammenfügen
    combined_chart = (court_layer + hoop_layer + shots_layer).properties(width='container', height=600).configure_view(strokeWidth=0)
    return combined_chart

# ==========================================
# 4. Data Loading & Credentials
# ==========================================
NBA_TEAMS = [
    "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
    "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets",
    "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers",
    "Los Angeles Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat",
    "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
    "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns",
    "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors",
    "Utah Jazz", "Washington Wizards"
]

@st.cache_data
def get_credentials():
    users_dict = {}
    for team in NBA_TEAMS:
        username = team.replace(" ", "_").lower()
        password_plain = f"{username}_123"
        hashed_pw = bcrypt.hashpw(password_plain.encode(), bcrypt.gensalt()).decode()
        users_dict[username] = {"name": team, "password": hashed_pw, "email": f"{username}@nba.com"}
    return {"usernames": users_dict}

@st.cache_data(ttl=3600) 
def load_gamelog_data():
    try:
        game_log = leaguegamelog.LeagueGameLog(season='2024-25', season_type_all_star='Regular Season').get_data_frames()[0]
        if game_log.empty:
            game_log = leaguegamelog.LeagueGameLog(season='2023-24', season_type_all_star='Regular Season').get_data_frames()[0]
        if not game_log.empty:
            game_log['IS_HOME'] = game_log.apply(lambda row: 1 if ' vs.' in row['MATCHUP'] else 0, axis=1)
        return game_log
    except Exception:
        return pd.DataFrame()

def process_gamelog_for_dashboard(df_gamelog, my_team_name):
    if df_gamelog.empty: return pd.DataFrame()
    my_games = df_gamelog[df_gamelog["TEAM_NAME"] == my_team_name].copy()
    dashboard_data = []
    for _, my_row in my_games.iterrows():
        game_id = my_row["GAME_ID"]
        is_home = my_row["IS_HOME"]
        opponent_row = df_gamelog[(df_gamelog["GAME_ID"] == game_id) & (df_gamelog["TEAM_NAME"] != my_team_name)]
        if not opponent_row.empty:
            opp_row = opponent_row.iloc[0]
            dashboard_data.append({
                "GameID": game_id,
                "Date": my_row["GAME_DATE"], 
                "Home Team": my_team_name if is_home else opp_row["TEAM_NAME"],
                "Away Team": opp_row["TEAM_NAME"] if is_home else my_team_name,
                "Home Score": my_row["PTS"] if is_home else opp_row["PTS"],
                "Away Score": opp_row["PTS"] if is_home else my_row["PTS"],
                "WL": my_row["WL"]
            })
    df_final = pd.DataFrame(dashboard_data)
    if not df_final.empty: df_final = df_final.sort_values(by="Date", ascending=False)
    return df_final

def get_team_logo_url(team_name):
    try:
        slug_map = {
            "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA",
            "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
            "Detroit Pistons": "DET", "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
            "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM", "Miami Heat": "MIA",
            "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NO", "New York Knicks": "NYK",
            "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
            "Portland Trail Blazers": "POR", "Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
            "Toronto Raptors": "TOR", "Utah Jazz": "utah", "Washington Wizards": "WAS"
        }
        code = slug_map.get(team_name, None)
        if not code:
            key = team_name.split(" ")[-1]
            code = slug_map.get(key, "NBA")
        return f"https://a.espncdn.com/i/teamlogos/nba/500/{code.lower()}.png"
    except:
        return "https://a.espncdn.com/i/teamlogos/nba/500/nba.png"

# ==========================================
# 5. Main Application Logic
# ==========================================

# Apply styles
set_modern_nba_style()

# Auth Setup
credentials = get_credentials()
authenticator = stauth.Authenticate(credentials, "nba_cookie_final", "nba_key_final", cookie_expiry_days=1)

if 'selected_game' not in st.session_state:
    st.session_state.selected_game = None

# --- PART A: LOGIN SCREEN ---
if not st.session_state.get("authentication_status"):
    
    intuition_path = get_img_path("intuition.png")
    bg_b64 = get_base64_of_bin_file(intuition_path)
    hm_b64 = get_base64_of_bin_file(get_img_path("hm_logo.png"))
    nba_b64 = get_base64_of_bin_file(get_img_path("nba_logo.png"))
    
    bg_style = f"background-image: url('data:image/png;base64,{bg_b64}'); background-size: cover; background-position: center;" if bg_b64 else "background-color: #0b1116;"

    st.write("") 
    col_hero, col_login = st.columns([1.5, 1], gap="large")
    
    with col_hero:
        st.markdown(f"""
        <div class="login-hero-container" style="{bg_style}">
            <div class="hero-overlay"></div>
            <div class="hero-content">
                <div class="hero-title">From Intuition<br>to Efficiency.</div>
                <div class="hero-subtitle">
                    A Data-Driven Approach to NBA Shot Selection.<br>
                    Increase your eFG% with AI-powered insights.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_login:
        img_hm = f'<img src="data:image/png;base64,{hm_b64}" style="height: 50px; opacity: 0.9;">' if hm_b64 else "HM Logo"
        img_nba = f'<img src="data:image/png;base64,{nba_b64}" style="height: 100px;">' if nba_b64 else "NBA Logo"

        st.markdown(f"""
        <div style="display: flex; justify-content: center; align-items: center; gap: 30px; margin-top: 10vh; margin-bottom: 30px;">
            {img_hm}
            <div style="width: 1px; height: 40px; background-color: rgba(255,255,255,0.2);"></div>
            {img_nba}
        </div>
        """, unsafe_allow_html=True)
        
        with st.container(border=False):
            st.markdown("""
            <div style='text-align: center; margin-bottom: 30px;'>
                <h2 style='font-family: "Inter", sans-serif; font-weight: 900; font-size: 1.8rem; text-transform: uppercase; letter-spacing: 3px; margin: 0; background: -webkit-linear-gradient(45deg, #ffffff, #a8dadc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 10px 20px rgba(0,0,0,0.5);'>
                    Success is Calculated
                </h2>
            </div>
            """, unsafe_allow_html=True)

            authenticator.login(captcha=True, location="main")
            
            if st.session_state["authentication_status"] is False:
                st.error('Incorrect credentials.') 
            elif st.session_state["authentication_status"] is None:
                st.markdown("<div style='text-align: center; color: #888; font-size: 0.8rem; margin-top: 15px;'>Please log in with your team account.</div>", unsafe_allow_html=True)

# --- PART B: DASHBOARD (LOGGED IN) ---
elif st.session_state["authentication_status"]:

    # 1. Bilder laden
    hm_b64 = get_base64_of_bin_file(get_img_path("hm_logo.png"))
    nba_b64 = get_base64_of_bin_file(get_img_path("nba_logo.png"))
    img_hm = f'<img src="data:image/png;base64,{hm_b64}" style="height: 50px; opacity: 0.9;">' if hm_b64 else "HM Logo"
    img_nba = f'<img src="data:image/png;base64,{nba_b64}" style="height: 90px;">' if nba_b64 else "NBA Logo"

    username = st.session_state["username"]
    user_data = credentials["usernames"][username]
    my_team_name = user_data['name']

    # ==========================================
    # NAVIGATION BAR
    # ==========================================
    nav_col1, nav_col2, nav_col3 = st.columns([1, 6, 1], gap="medium")

    # Spalte 1: Back Button
    with nav_col1:
        if st.session_state.selected_game is not None:
            if st.button(" く ", key="nav_back", use_container_width=False):
                st.session_state.selected_game = None
                st.rerun()
        else:
            st.write("")

    # Spalte 2: Logos (Zentriert)
    with nav_col2:
        st.markdown(f"""
        <div style="display: flex; justify-content: center; align-items: center; gap: 30px; height: 80px;">
            {img_hm}
            <div style="width: 1px; height: 40px; background-color: rgba(255,255,255,0.2);"></div>
            {img_nba}
        </div>
        """, unsafe_allow_html=True)

    # Spalte 3: Logout Button
    with nav_col3:
        authenticator.logout("LOGOUT", location="main")

    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin-top: 20px; margin-bottom: 20px;'>", unsafe_allow_html=True)

    # Data Loading
    with st.spinner('Loading data...'): 
        raw_gamelog = load_gamelog_data()
        df_dashboard = process_gamelog_for_dashboard(raw_gamelog, my_team_name)

    # --- VIEW 1: DETAIL (Specific Game Selected) ---
    if st.session_state.selected_game is not None:
        
        st.markdown("""
        <style>
        div[data-testid="stVerticalBlockBorderWrapper"] { min-height: 850px; height: 100%; display: flex; flex-direction: column; }
        </style>
        """, unsafe_allow_html=True)

        game = st.session_state.selected_game
        game_id = game["GameID"] 
        
        col_left_stats, col_right_court = st.columns([1.1, 2.5], gap="large")

        # --- LEFT: STATS ---
        with col_left_stats:
            with st.container(border=False):
                home_logo = get_team_logo_url(game['Home Team'])
                away_logo = get_team_logo_url(game['Away Team'])
                
                scoreboard_html = f"""
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 25px; width: 100%; padding-top: 10px;">
                    <div style="text-align: center; width: 30%;">
                        <img src="{home_logo}" style="width: 110px; height: auto; object-fit: contain; max-height: 110px;">
                        <div style="font-size: 0.75rem; font-weight: bold; margin-top: 10px; color: #aaa;">HOME</div>
                    </div>
                    <div style="text-align: center; width: 40%;">
                        <div style="font-size: 2.0rem; font-weight: 900; line-height: 1; color: white;">{game['Home Score']} : {game['Away Score']}</div>
                        <div style="font-size: 0.8rem; color: #555; margin-top: 5px;">FINAL</div>
                    </div>
                    <div style="text-align: center; width: 30%;">
                        <img src="{away_logo}" style="width: 110px; height: auto; object-fit: contain; max-height: 110px;">
                        <div style="font-size: 0.75rem; font-weight: bold; margin-top: 10px; color: #aaa;">AWAY</div>
                    </div>
                </div>
                """
                st.markdown(scoreboard_html, unsafe_allow_html=True)
                st.divider()

                st.markdown("<h5 style='text-align: center; color: #888; margin-bottom: 20px;'>Matchup Stats</h5>", unsafe_allow_html=True)
                try:
                    search_id = str(game_id).zfill(10)
                    game_stats = raw_gamelog[raw_gamelog['GAME_ID'].astype(str) == search_id]
                    if game_stats.empty: game_stats = raw_gamelog[raw_gamelog['GAME_ID'].astype(str) == str(game_id)]

                    if not game_stats.empty and len(game_stats) >= 2:
                        home_stats_row = game_stats[game_stats['TEAM_NAME'] == game['Home Team']]
                        away_stats_row = game_stats[game_stats['TEAM_NAME'] == game['Away Team']]
                        if not home_stats_row.empty and not away_stats_row.empty:
                            render_comparison_stats(home_stats_row.iloc[0], away_stats_row.iloc[0])
                        else: st.warning("Stats incomplete.") 
                    else: st.warning("No stats found.") 
                except Exception as e: st.error(f"Error: {e}") 

        # --- RIGHT: COURT ---
        with col_right_court:
            with st.container(border=False):
                st.markdown(f"<h3 style='text-align: center;'>Analytical High Risk Shot Analysis</h3>", unsafe_allow_html=True)
                
                # Check Data Path
                parquet_path = os.path.join(current_dir, '..', 'data', 'predict', 'xgboost_preds_2024-25.parquet')
                
                if os.path.exists(parquet_path):
                    df_shots_all = pd.read_parquet(parquet_path)
                    df_shots_all['GAME_ID'] = df_shots_all['GAME_ID'].astype(str)
                    search_game_id = str(game_id).zfill(10)
                    current_game_shots = df_shots_all[df_shots_all['GAME_ID'] == search_game_id]
                    if current_game_shots.empty: current_game_shots = df_shots_all[df_shots_all['GAME_ID'] == str(game_id)]

                    if not current_game_shots.empty:
                        my_team_shots = current_game_shots[current_game_shots['TEAM_NAME'] == my_team_name].copy()
                        if not my_team_shots.empty:
                            f_col1, f_col2 = st.columns(2)
                            with f_col1:
                                risk_threshold = st.slider("High Risk Threshold (Shot Probability < X%)", 0, 100, 35, 1, format="%d%%")
                                risk_factor = risk_threshold / 100.0
                            
                            df_risk_filtered = my_team_shots[my_team_shots['predicted_shot_probability'] < risk_factor]
                            available_players = sorted(df_risk_filtered['PLAYER_NAME'].unique())
                            
                            with f_col2:
                                selected_players = st.multiselect("Player Filter", options=available_players, placeholder="All Players")

                            if selected_players: df_final_plot = df_risk_filtered[df_risk_filtered['PLAYER_NAME'].isin(selected_players)]
                            else: df_final_plot = df_risk_filtered
                        
                            if not df_final_plot.empty:
                                final_chart = create_full_court_plot(df_final_plot)
                                
                                # --- FIX: width="stretch" für use_container_width Ersatz ---
                                # --- RÜCKGANG: on_select entfernt, da es bei Layern crasht ---
                                st.altair_chart(final_chart, width="stretch", theme="streamlit")
                                
                                st.caption(f"Showing {len(df_final_plot)} shots with win probability < {risk_threshold}%") 
                            else: st.info(f"No shots found below {risk_threshold}% probability for the selected players.") 
                        else: st.warning("No shot data for your team in this game.") 
                    else: st.warning("No data found for this GameID.") 
                else: st.error(f"Database file not found at {parquet_path}") 
            
    # --- VIEW 2: GAME GRID (Overview) ---
    else:
        if not df_dashboard.empty:
            
            # --- FILTER LOGIC START ---
            
            # 1. Datenaufbereitung für Filter
            df_dashboard['Date_Obj'] = pd.to_datetime(df_dashboard['Date'])
            df_dashboard['Month'] = df_dashboard['Date_Obj'].dt.strftime('%B')
            df_dashboard['Opponent'] = df_dashboard.apply(lambda x: x['Away Team'] if x['Home Team'] == my_team_name else x['Home Team'], axis=1)
            
            unique_opponents = sorted(df_dashboard['Opponent'].unique())
            unique_months = sorted(df_dashboard['Month'].unique(), key=lambda m: pd.to_datetime(m, format='%B').month)

            # 2. Header Layout: Titel links, Filter Button rechts
            col_title, col_filter = st.columns([6, 1], vertical_alignment="center")

            with col_filter:
                # Das Popover "Trichter"
                with st.popover("🌪️ Filter", use_container_width=True):
                    st.markdown("**Filter Options**")
                    
                    # Split for Location and Result radios
                    f_col1, f_col2 = st.columns(2)
                    with f_col1:
                        filter_loc = st.radio("Location", ["All", "Home", "Away"], index=0)
                    with f_col2:
                        filter_result = st.radio("Result", ["All", "Wins", "Losses"], index=0)
                        
                    st.divider()
                    filter_opp = st.multiselect("Opponent", unique_opponents)
                    filter_month = st.multiselect("Month", unique_months)

            # 3. Filter anwenden
            df_filtered = df_dashboard.copy()

            # Location Filter
            if filter_loc == "Home":
                df_filtered = df_filtered[df_filtered['Home Team'] == my_team_name]
            elif filter_loc == "Away":
                df_filtered = df_filtered[df_filtered['Away Team'] == my_team_name]
            
            # Result Filter (New!)
            if filter_result == "Wins":
                df_filtered = df_filtered[df_filtered['WL'] == 'W']
            elif filter_result == "Losses":
                df_filtered = df_filtered[df_filtered['WL'] == 'L']

            # Opponent Filter
            if filter_opp:
                df_filtered = df_filtered[df_filtered['Opponent'].isin(filter_opp)]
            
            # Month Filter
            if filter_month:
                df_filtered = df_filtered[df_filtered['Month'].isin(filter_month)]

            # Titel rendern mit aktualisierter Anzahl
            with col_title:
                st.markdown(f"<h3 style='margin-bottom: 0px;'>Season 2024-25 ({len(df_filtered)} Games)</h3>", unsafe_allow_html=True)
            
            st.write("") # Spacer

            # --- FILTER LOGIC END ---

            # Grid rendern mit df_filtered statt df_dashboard
            if not df_filtered.empty:
                COLS_PER_ROW = 3 
                rows = [df_filtered.iloc[i:i + COLS_PER_ROW] for i in range(0, len(df_filtered), COLS_PER_ROW)]

                for row_chunk in rows:
                    cols = st.columns(COLS_PER_ROW)
                    for idx, (index, game_row) in enumerate(row_chunk.iterrows()):
                        with cols[idx]:
                            is_home = (game_row["Home Team"] == my_team_name)
                            win = game_row["WL"] == "W"
                            color = "#4CAF50" if win else "#C8102E"
                            
                            with st.container(border=True):
                                st.markdown(f'<div style="background-color:{color};height:4px;border-radius:4px;margin-bottom:10px;"></div>', unsafe_allow_html=True)
                                date_str = pd.to_datetime(game_row['Date']).strftime("%d. %b").upper() if isinstance(game_row['Date'], str) else str(game_row['Date'])
                                st.markdown(f"<div class='card-date'>{date_str} • {'HOME' if is_home else 'AWAY'}</div>", unsafe_allow_html=True)
                                st.markdown(f"""
                                    <div style="display:flex;justify-content:space-between;align-items:center;margin:15px 0;">
                                        <img src="{get_team_logo_url(game_row['Home Team'])}" style="max-height:80px;">
                                        <div class='card-score'>{game_row['Home Score']}:{game_row['Away Score']}</div>
                                        <img src="{get_team_logo_url(game_row['Away Team'])}" style="max-height:80px;">
                                    </div>
                                """, unsafe_allow_html=True)
                                st.write("") 
                                if st.button("Analyze", key=f"btn_{game_row['GameID']}", use_container_width=True):
                                    st.session_state.selected_game = game_row.to_dict()
                                    st.rerun()
            else:
                st.info("No games found with the selected filters.")