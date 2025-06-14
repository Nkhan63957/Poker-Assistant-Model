import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from treys import Card, Evaluator
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math
import traceback
import logging

# Set up Streamlit page (MUST be first Streamlit command)
st.set_page_config(page_title="Advanced Poker AI Predictor", page_icon="🃏", layout="wide")

# Debug: Confirm Plotly import
try:
    import plotly
    logging.info("Plotly imported successfully!")
except ImportError:
    st.error("Plotly import failed. Please ensure 'plotly==5.22.0' is in requirements.txt.")
    st.stop()

# Define PokerQNetwork class
class PokerQNetwork(nn.Module):
    def __init__(self):
        super(PokerQNetwork, self).__init__()
        self.card_embedding = nn.Embedding(53, 16)
        self.hole_conv = nn.Conv1d(16, 32, kernel_size=2)
        self.community_conv = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        self.state_fc = nn.Linear(10, 64)
        total_features = 32 + 32 + 64
        self.fc1 = nn.Linear(total_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 3)
        self.dropout = nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(total_features)

    def forward(self, card_input, state_input):
        batch_size = card_input.size(0)
        card_ids = torch.zeros(batch_size, 7, dtype=torch.long, device=card_input.device)
        suits = ['h', 's', 'd', 'c']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        for b in range(batch_size):
            for c in range(7):
                if card_input[b, 0, c] > 0 or card_input[b, 1, c] > 0:
                    suit = int(card_input[b, 0, c] * 3)
                    rank = int(card_input[b, 1, c] * 12)
                    card_ids[b, c] = rank * 4 + suit + 1
        card_embeds = self.card_embedding(card_ids)
        hole_embeds = card_embeds[:, :2, :].transpose(1, 2)
        community_embeds = card_embeds[:, 2:, :].transpose(1, 2)
        hole_features = self.hole_conv(hole_embeds).squeeze(-1)
        community_features = self.community_conv(community_embeds)
        community_features = torch.nn.functional.adaptive_avg_pool1d(community_features, 1).squeeze(-1)
        combined_cards = torch.stack([hole_features, community_features], dim=1)
        attended_cards, _ = self.attention(combined_cards, combined_cards, combined_cards)
        attended_cards = attended_cards.mean(dim=1)
        state_features = torch.nn.functional.relu(self.state_fc(state_input))
        combined = torch.cat([hole_features, attended_cards, state_features], dim=1)
        combined = self.layer_norm(combined)
        x = torch.nn.functional.relu(self.fc1(combined))
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Enhanced evaluate_hand function
def evaluate_hand(hole_cards, community_cards, num_opponents=1):
    evaluator = Evaluator()
    if len(community_cards) < 3:  # Preflop
        card1_str = Card.int_to_str(hole_cards[0])
        card2_str = Card.int_to_str(hole_cards[1])
        rank1 = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'].index(card1_str[0])
        rank2 = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'].index(card2_str[0])
        if rank1 == rank2:
            equity = 0.5 + (rank1 / 26)
        elif card1_str[1] == card2_str[1]:
            equity = 0.3 + (max(rank1, rank2) / 26) + 0.1
        else:
            equity = 0.2 + (max(rank1, rank2) / 26)
        equity = equity * (0.85 ** num_opponents)
        equity = min(equity, 0.85)
        all_hands = 1326
        rank = min(rank1, rank2) if rank1 != rank2 else rank1
        suited = card1_str[1] == card2_str[1]
        if rank1 == rank2:
            percentile = 100 * (1 - (6 * (12 - rank) / all_hands))
        elif suited:
            percentile = 100 * (1 - (4 * (169 - (rank1 + rank2 + 10)) / all_hands))
        else:
            percentile = 100 * (1 - (12 * (169 - (rank1 + rank2)) / all_hands))
        return 0, 7462, equity, max(0, min(100, percentile))
    rank = evaluator.evaluate(hole_cards, community_cards)
    rank_class = evaluator.get_rank_class(rank)
    equity_map = {1: 0.95, 2: 0.90, 3: 0.85, 4: 0.80, 5: 0.75, 6: 0.65, 7: 0.55, 8: 0.40, 9: 0.25}
    base_equity = equity_map.get(rank_class, 0.25)
    relative_strength = (7462 - rank) / 7462
    equity = base_equity + (relative_strength * 0.15)
    equity = equity * (0.9 ** num_opponents)
    equity = min(equity, 0.98)
    percentile = 100 * (1 - rank / 7462)
    return rank_class, rank, equity, max(0, min(100, percentile))

# App title and description
st.title('Advanced Poker AI Predictor')
st.write('This app uses a deep Q-learning model to predict the best action in Texas Hold\'em, supporting multiple opponents, visual card selection, and detailed analysis.')

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'dealer_idx' not in st.session_state:
    st.session_state.dealer_idx = 0
if 'user_seat_idx' not in st.session_state:
    st.session_state.user_seat_idx = 0
if 'table_setup' not in st.session_state:
    st.session_state.table_setup = False
if 'opp_aggressions' not in st.session_state:
    st.session_state.opp_aggressions = {}
if 'opp_stacks' not in st.session_state:
    st.session_state.opp_stacks = []

# Load the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = PokerQNetwork().to(device)
try:
    model.load_state_dict(torch.load('best_poker_model.pth', map_location=device))
    model.eval()
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Model file 'best_poker_model.pth' not found. Please ensure it's in the same directory.")
    st.stop()

# Calculate positions
def calculate_positions(dealer_idx, total_players):
    positions = []
    for i in range(total_players):
        rel_pos = (i - dealer_idx) % total_players
        if total_players == 2:
            pos = 'Small Blind' if rel_pos == 1 else 'Big Blind/Button'
        elif total_players == 3:
            pos = 'Small Blind' if rel_pos == 1 else 'Big Blind' if rel_pos == 2 else 'Button'
        elif total_players == 4:
            pos = 'Small Blind' if rel_pos == 1 else 'Big Blind' if rel_pos == 2 else 'UTG' if rel_pos == 3 else 'Button'
        elif total_players == 5:
            pos = 'Small Blind' if rel_pos == 1 else 'Big Blind' if rel_pos == 2 else 'UTG' if rel_pos == 3 else 'Cutoff' if rel_pos == 4 else 'Button'
        else:
            pos = 'Small Blind' if rel_pos == 1 else 'Big Blind' if rel_pos == 2 else 'UTG' if rel_pos == 3 else 'UTG+1' if rel_pos == 4 else 'Cutoff' if rel_pos == 5 else 'Button'
        positions.append(pos)
    return positions

# Table visualization
def create_table_visualization(dealer_idx, user_seat_idx, total_players, positions):
    try:
        fig = go.Figure()
        radius = 2
        center_x, center_y = 0, 0
        theta = np.linspace(0, 2 * np.pi, total_players, endpoint=False)
        x = [center_x + radius * math.cos(t) for t in theta]
        y = [center_y + radius * math.sin(t) for t in theta]
        fig.add_shape(
            type="circle",
            xref="x", yref="y",
            x0=-radius, y0=-radius, x1=radius, y1=radius,
            fillcolor="rgba(0, 128, 0, 0.5)",  # Semi-transparent green
            line_color="black"
        )
        seat_to_opp = {}
        opp_idx = 0
        for i in range(total_players):
            if i != user_seat_idx:
                seat_to_opp[i] = opp_idx
                opp_idx += 1
        sb_idx = (dealer_idx + 1) % total_players
        bb_idx = (dealer_idx + 2) % total_players
        for i in range(total_players):
            label = 'User' if i == user_seat_idx else f'Opp {seat_to_opp.get(i)+1}'
            color = 'yellow' if i == user_seat_idx else 'blue'
            markers = []
            if i == dealer_idx:
                markers.append('D')
            if i == sb_idx:
                markers.append('SB')
            if i == bb_idx:
                markers.append('BB')
            marker_text = ', '.join(markers) if markers else ''
            fig.add_trace(go.Scatter(
                x=[x[i]], y=[y[i]], mode='markers+text',
                marker=dict(size=25, color=color),
                text=f"{label}<br>{positions[i]}<br>{marker_text}",
                textposition="middle center",
                textfont=dict(color="white", size=12),
                showlegend=False
            ))
        fig.update_layout(
            title="Table Layout",
            xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
            yaxis=dict(visible=False),
            width=400,
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        return fig, None
    except Exception as e:
        return None, f"Failed to render table: {str(e)}\n{traceback.format_exc()}"

# Sidebar: Table Setup
st.sidebar.header('Table Setup')
st.sidebar.write("Configure the number of opponents, your seat, dealer, blinds, and stacks.")
num_opponents = st.sidebar.selectbox('Number of Opponents', [1, 2, 3, 4, 5], index=0)
total_players = num_opponents + 1

user_seat = st.sidebar.selectbox('Your Seat', [f'Seat {i+1}' for i in range(total_players)], index=st.session_state.user_seat_idx)
user_seat_idx = int(user_seat.split()[-1]) - 1
st.session_state.user_seat_idx = user_seat_idx

dealer_options = ['User'] + [f'Opp {i+1}' for i in range(num_opponents)]
dealer = st.sidebar.selectbox('Dealer', dealer_options, index=st.session_state.dealer_idx)
dealer_idx = 0 if dealer == 'User' else int(dealer.split()[-1])
st.session_state.dealer_idx = dealer_idx

small_blind = st.sidebar.number_input('Small Blind ($)', min_value=0, max_value=100, value=5, step=1)
big_blind = st.sidebar.number_input('Big Blind ($)', min_value=small_blind, max_value=200, value=10, step=1)
if big_blind < small_blind:
    st.error("Big blind must be at least as large as small blind.")
    st.stop()

player_stack = st.sidebar.number_input('Your Stack ($)', min_value=0, max_value=10000, value=1000, step=1)
opp_stacks = []
for i in range(num_opponents):
    opp_stack = st.sidebar.number_input(f'Opp {i+1} Stack ($)', min_value=0, max_value=10000, value=1000, step=1, key=f'opp_stack_setup_{i}')
    opp_stacks.append(opp_stack)

# Validate blinds against stacks
sb_idx = (dealer_idx + 1) % total_players
bb_idx = (dealer_idx + 2) % total_players
if user_seat_idx == sb_idx and small_blind > player_stack:
    st.error("Small blind exceeds your stack.")
    st.stop()
if user_seat_idx == bb_idx and big_blind > player_stack:
    st.error("Big blind exceeds your stack.")
    st.stop()
for i, opp_stack in enumerate(opp_stacks):
    opp_seat = [j for j in range(total_players) if j != user_seat_idx][i]
    if opp_seat == sb_idx and small_blind > opp_stack:
        st.error(f"Small blind exceeds Opp {i+1}'s stack.")
        st.stop()
    if opp_seat == bb_idx and big_blind > opp_stack:
        st.error(f"Big blind exceeds Opp {i+1}'s stack.")
        st.stop()

# Display table preview
st.subheader('Table Preview')
positions = calculate_positions(dealer_idx, total_players)
fig, error = create_table_visualization(dealer_idx, user_seat_idx, total_players, positions)
if fig:
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error(f"Unable to render circular table. Error: {error}")
    st.write("Fallback: Seat positions (clockwise):")
    seat_to_opp = {}
    opp_idx = 0
    for i in range(total_players):
        if i != user_seat_idx:
            seat_to_opp[i] = opp_idx
            opp_idx += 1
    for i in range(total_players):
        label = 'User' if i == user_seat_idx else f'Opp {seat_to_opp.get(i)+1}'
        markers = []
        if i == dealer_idx:
            markers.append('D')
        if i == sb_idx:
            markers.append('SB')
        if i == bb_idx:
            markers.append('BB')
        marker_text = ', '.join(markers) if markers else ''
        st.write(f"Seat {i+1}: {label}, {positions[i]}, {marker_text}")

# Confirm setup
if st.sidebar.button('Confirm Setup'):
    st.session_state.table_setup = True
    st.session_state.opp_aggressions = {f'opp_{i}': 0.5 for i in range(num_opponents)}
    st.session_state.opp_stacks = opp_stacks
    st.rerun()

if st.sidebar.button('Reset Table'):
    st.session_state.table_setup = False
    st.session_state.dealer_idx = 0
    st.session_state.user_seat_idx = 0
    st.session_state.opp_aggressions = {}
    st.session_state.opp_stacks = []
    st.rerun()

# Game Setup and Analysis
if st.session_state.table_setup:
    st.sidebar.header('Game Setup')
    suits = ['♥', '♠', '♦', '♣']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    deck = [f"{r}{s}" for r in ranks for s in suits]

    def card_image(card_str):
        if card_str == "None":
            return None
        rank = card_str[:-1].lower().replace('10', 'T').replace('j', 'J').replace('q', 'Q').replace('k', 'K').replace('a', 'A')
        suit = {'♥': 'H', '♠': 'S', '♦': 'D', '♣': 'C'}[card_str[-1]]
        image_url = f'https://deckofcardsapi.com/static/img/{rank}{suit}.png'
        return image_url

    # Hole cards
    st.sidebar.subheader('Hole Cards')
    hole_cols = st.sidebar.columns(2)
    hole_cards = []
    selected_cards = []
    for i in range(2):
        with hole_cols[i]:
            card_key = f"hole_card_{i}"
            selected = st.selectbox(f"Card {i+1}", ['None'] + deck, key=card_key)
            if selected != 'None':
                hole_cards.append(Card.new(selected[:-1] + {'♥': 'h', '♠': 's', '♦': 'd', '♣': 'c'}[selected[-1]]))
                selected_cards.append(selected)
                st.image(card_image(selected), width=80)

    # Community cards
    st.sidebar.subheader('Community Cards')
    stage = st.sidebar.selectbox('Game Stage', ['Preflop', 'Flop', 'Turn', 'River'])
    community_cards = []
    if stage != 'Preflop':
        num_comm_cards = 3 if stage == 'Flop' else 4 if stage == 'Turn' else 5
        comm_cols = st.sidebar.columns(num_comm_cards)
        for i in range(num_comm_cards):
            with comm_cols[i]:
                card_key = f"comm_card_{i}"
                selected = st.selectbox(f"Card {i+1}", ['None'] + deck, key=card_key)
                if selected != 'None':
                    community_cards.append(Card.new(selected[:-1] + {'♥': 'h', '♠': 's', '♦': 'd', '♣': 'c'}[selected[-1]]))
                    selected_cards.append(selected)
                    st.image(card_image(selected), width=80)

    # Validate card selections
    if len(selected_cards) != len(set(selected_cards)):
        st.error("Duplicate cards selected. Please choose unique cards.")
        st.stop()
    if len(hole_cards) != 2:
        st.error("Please select both hole cards.")
        st.stop()

    # Game state
    st.sidebar.subheader('Game State')
    pot = st.sidebar.number_input('Pot Size ($)', min_value=0, max_value=5000, value=50, step=1)
    player_bet = st.sidebar.number_input('Your Current Bet ($)', min_value=0, max_value=player_stack, value=0, step=1)
    if player_bet > player_stack:
        st.error("Your bet cannot exceed your stack.")
        st.stop()

    # Opponent actions
    opponent_data = []
    seat_to_opp = {}
    opp_idx = 0
    for i in range(total_players):
        if i != user_seat_idx:
            seat_to_opp[i] = opp_idx
            with st.sidebar.expander(f"Opp {opp_idx+1} ({positions[i]})"):
                opp_stack = st.session_state.opp_stacks[opp_idx]
                opp_bet = st.number_input(f'Opp {opp_idx+1} Bet ($)', min_value=0, max_value=opp_stack, value=10, step=1, key=f'opp_bet_{opp_idx}')
                opp_folded = st.checkbox(f'Opp {opp_idx+1} Folded', value=False, key=f'opp_folded_{opp_idx}')
                if opp_bet > opp_stack:
                    st.error(f"Opp {opp_idx+1}'s bet cannot exceed their stack.")
                    st.stop()
                opponent_data.append({
                    'stack': opp_stack,
                    'bet': opp_bet,
                    'folded': opp_folded,
                    'position': positions[i]
                })
                opp_idx += 1

    # Update aggressions
    avg_opp_bet = sum(opp['bet'] for opp in opponent_data if not opp['folded']) / max(1, sum(1 for opp in opponent_data if not opp['folded']))
    for i in range(num_opponents):
        opp_key = f'opp_{i}'
        if opponent_data[i]['folded']:
            st.session_state.opp_aggressions[opp_key] = max(0.0, st.session_state.opp_aggressions[opp_key] - 0.1)
        elif opponent_data[i]['bet'] > avg_opp_bet:
            st.session_state.opp_aggressions[opp_key] = min(1.0, st.session_state.opp_aggressions[opp_key] + 0.1)
        elif opponent_data[i]['bet'] == avg_opp_bet and opponent_data[i]['bet'] > 0:
            st.session_state.opp_aggressions[opp_key] = min(1.0, st.session_state.opp_aggressions[opp_key] + 0.01)
        opponent_data[i]['aggression'] = st.session_state.opp_aggressions[opp_key]

    # Adjust for blinds in preflop
    if stage == 'Preflop':
        if user_seat_idx == sb_idx:
            player_bet = max(player_bet, small_blind)
            player_stack = max(0, player_stack - small_blind)
            pot += small_blind
        elif user_seat_idx == bb_idx:
            player_bet = max(player_bet, big_blind)
            player_stack = max(0, player_stack - big_blind)
            pot += big_blind
        for i, opp in enumerate(opponent_data):
            opp_seat = [j for j in range(total_players) if j != user_seat_idx][i]
            if opp_seat == sb_idx:
                opp['bet'] = max(opp['bet'], small_blind)
                opp['stack'] = max(0, opp['stack'] - small_blind)
                pot += small_blind
            elif opp_seat == bb_idx:
                opp['bet'] = max(opp['bet'], big_blind)
                opp['stack'] = max(0, opp['stack'] - big_blind)
                pot += big_blind

    # Action order
    def get_action_order():
        active_players = [i for i in range(total_players) if i == user_seat_idx or not opponent_data[seat_to_opp.get(i, 0)]['folded']]
        preflop_order = []
        postflop_order = []
        if total_players == 2:
            start_idx = sb_idx
        else:
            start_idx = (bb_idx + 1) % total_players
        for i in range(len(active_players)):
            idx = (start_idx + i) % total_players
            if idx in active_players:
                preflop_order.append(idx)
        start_idx = sb_idx
        for i in range(len(active_players)):
            idx = (start_idx + i) % total_players
            if idx in active_players:
                postflop_order.append(idx)
        return preflop_order, postflop_order

    preflop_order, postflop_order = get_action_order()
    next_to_act = preflop_order[0] if stage == 'Preflop' else postflop_order[0]

    # Prepare input tensors
    card_input = np.zeros((1, 2, 7))
    suits_map = {'h': 0, 's': 1, 'd': 2, 'c': 3}
    ranks_map = {r: i for i, r in enumerate(['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'])}
    all_cards = hole_cards + community_cards
    while len(all_cards) < 7:
        all_cards.append(0)
    for i, card in enumerate(all_cards[:7]):
        if card != 0:
            card_str = Card.int_to_str(card)
            card_input[0, 0, i] = suits_map[card_str[-1]] / 3.0
            card_input[0, 1, i] = ranks_map[card_str[0]] / 12.0
    card_input = torch.FloatTensor(card_input).to(device)

    # Encode state
    stage_onehot = np.zeros(4)
    stage_idx = ['Preflop', 'Flop', 'Turn', 'River'].index(stage)
    stage_onehot[stage_idx] = 1
    avg_opp_stack = sum(opp['stack'] for opp in opponent_data) / max(1, num_opponents)
    avg_opp_bet = sum(opp['bet'] for opp in opponent_data if not opp['folded']) / max(1, sum(1 for opp in opponent_data if not opp['folded']))
    avg_opp_aggression = sum(opp['aggression'] for opp in opponent_data) / max(1, num_opponents)
    any_opp_folded = any(opp['folded'] for opp in opponent_data)
    state_features = [
        pot / 1000,
        player_stack / 1000,
        avg_opp_stack / 1000,
        player_bet / 1000,
        avg_opp_bet / 1000,
        float(any_opp_folded),
        avg_opp_aggression,
        stage_onehot[0],
        stage_onehot[1],
        position_bonus := (0.0 if positions[user_seat_idx] in ['Small Blind', 'Big Blind'] else 0.1 if positions[user_seat_idx].startswith('UTG') else 0.2)
    ]
    state_input = torch.FloatTensor([state_features]).to(device)

    # Main content
    st.subheader('Game Setup')
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write('**Table Layout**')
        fig, error = create_table_visualization(dealer_idx, user_seat_idx, total_players, positions)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Unable to render circular table. Error: {error}")
        st.write('**Game State**')
        input_data = {
            'Pot Size ($)': pot,
            'Your Stack ($)': player_stack,
            'Your Bet ($)': player_bet,
            'Small Blind ($)': small_blind,
            'Big Blind ($)': big_blind,
            'Avg Opponent Stack ($)': avg_opp_stack,
            'Avg Opponent Bet ($)': avg_opp_bet,
            'Avg Opponent Aggression': avg_opp_aggression,
            'Any Opponent Folded': any_opp_folded,
            'Game Stage': stage,
            'Your Position': positions[user_seat_idx],
            'Number of Opponents': num_opponents,
            'Dealer': dealer
        }
        st.write(pd.DataFrame([input_data]))
        st.write('**Action Order**')
        preflop_labels = ['User' if i == user_seat_idx else f'Opp {seat_to_opp[i]+1} ({positions[i]})' for i in preflop_order]
        postflop_labels = ['User' if i == user_seat_idx else f'Opp {seat_to_opp[i]+1} ({positions[i]})' for i in postflop_order]
        next_label = 'User' if next_to_act == user_seat_idx else f'Opp {seat_to_opp[next_to_act]+1} ({positions[next_to_act]})'
        st.write(f"- **Preflop**: {', '.join(preflop_labels)}")
        st.write(f"- **Postflop**: {', '.join(postflop_labels)}")
        st.write(f"- **Next to Act**: {next_label}")

    with col2:
        st.write('**Your Cards**')
        cols = st.columns(2)
        for i, card in enumerate(hole_cards):
            cols[i].image(card_image(Card.int_to_pretty_str(card)), caption=Card.int_to_pretty_str(card), width=80)
        if community_cards:
            st.write('**Community Cards**')
            cols = st.columns(len(community_cards))
            for i, card in enumerate(community_cards):
                cols[i].image(card_image(Card.int_to_pretty_str(card)), caption=Card.int_to_pretty_str(card), width=80)

    # Make prediction
    with torch.no_grad():
        q_values = model(card_input, state_input)
        action_probs = torch.softmax(q_values, dim=1).cpu().numpy()[0]
        action = torch.argmax(q_values, dim=1).item()

    # Evaluate hand strength
    rank_class, rank, equity, percentile = evaluate_hand(hole_cards, community_cards, num_opponents)

    # Store prediction
    prediction_entry = {
        'Timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Hole Cards': ', '.join([Card.int_to_pretty_str(c) for c in hole_cards]),
        'Community Cards': ', '.join([Card.int_to_pretty_str(c) for c in community_cards]) if community_cards else 'None',
        'Stage': stage,
        'Your Position': positions[user_seat_idx],
        'Recommended Action': ['Fold', 'Call', 'Raise'][action],
        'Equity': f'{equity:.2%}',
        'Hand Strength Percentile': f'{percentile:.1f}%'
    }
    st.session_state.prediction_history.append(prediction_entry)
    if len(st.session_state.prediction_history) > 10:
        st.session_state.prediction_history = st.session_state.prediction_history[-10:]

    # Display results
    st.subheader('Prediction')
    action_names = ['Fold', 'Call', 'Raise']
    st.write(f'The model recommends to **{action_names[action]}**.')
    st.write(f'Estimated Hand Equity: **{equity:.2%}** (against {num_opponents} opponent(s))')
    st.write(f'Hand Strength Percentile: **{percentile:.1f}%** (top {100-percentile:.1f}% of possible hands)')

    st.subheader('Action Probabilities')
    prob_df = pd.DataFrame({
        'Action': action_names,
        'Probability': [f'{p:.2%}' for p in action_probs]
    })
    st.write(prob_df)
    fig = px.bar(
        x=action_names,
        y=action_probs,
        title='Action Probability Distribution',
        labels={'x': 'Action', 'y': 'Probability'},
        text=[f'{p:.2%}' for p in action_probs]
    )
    fig.update_traces(textposition='auto')
    fig.update_yaxes(range=[0, 1], tickformat='.0%')
    st.plotly_chart(fig, use_container_width=True)

    # In-depth analysis
    st.subheader('In-Depth Analysis')
    pot_odds = avg_opp_bet / (pot + avg_opp_bet) if avg_opp_bet > 0 else 0
    st.write(f'**Pot Odds**: {pot_odds:.2%} (you need {pot_odds:.2%} equity to break even on a call)')

    position_insights = {
        'Small Blind': 'As the Small Blind, you’ve invested half the big blind, making calls more attractive, but you act first postflop, which is a disadvantage.',
        'Big Blind': 'As the Big Blind, you’ve invested the full big blind, incentivizing you to defend against raises, but you act second postflop.',
        'UTG': 'As Under the Gun, you act first preflop and early postflop, requiring stronger hands to compensate for the positional disadvantage.',
        'UTG+1': 'As UTG+1, you act early preflop and postflop, needing strong hands due to limited information.',
        'Cutoff': 'As the Cutoff, you act late preflop and often postflop, allowing wider ranges and more aggressive plays.',
        'Button': 'As the Button, you act last postflop, giving a significant advantage to play a wider range and control the pot.',
        'Big Blind/Button': 'As the Big Blind and Button (heads-up), you’ve invested the big blind and act last postflop.'
    }
    st.write(f'**Position Insight ({positions[user_seat_idx]})**: {position_insights.get(positions[user_seat_idx], "Your position influences your strategy.")}')

    stage_insights = {
        'Preflop': f"In the preflop stage, your hand's equity ({equity:.2%}) is based on starting hand strength against {num_opponents} opponent(s). With a percentile of {percentile:.1f}%, your hand is {'strong' if percentile > 80 else 'medium' if percentile > 50 else 'weak'}. {'Consider raising with strong hands to build the pot.' if percentile > 80 else 'Play cautiously unless position or odds favor you.'}",
        'Flop': f"On the flop, your equity ({equity:.2%}) reflects your hand's strength with three community cards. A percentile of {percentile:.1f}% indicates {'a strong made hand or draw' if percentile > 70 else 'a moderate hand' if percentile > 40 else 'a weak hand or weak draw'}. {'Aggression may be warranted with strong hands or draws.' if percentile > 70 else 'Evaluate draws carefully against pot odds.'}",
        'Turn': f"On the turn, with four community cards, your equity ({equity:.2%}) is more defined. Your hand's percentile ({percentile:.1f}%) suggests {'a strong hand or draw' if percentile > 65 else 'a marginal hand' if percentile > 35 else 'a weak hand'}. {'Protect strong hands with bets; consider folding weak hands unless odds are favorable.' if percentile > 65 else 'Be cautious with marginal hands.'}",
        'River': f"On the river, your hand is fully defined with equity ({equity:.2%}) and percentile ({percentile:.1f}%). This indicates {'a strong hand' if percentile > 60 else 'a medium hand' if percentile > 30 else 'a weak hand'}. {'Value bet strong hands; bluff selectively with weak hands.' if percentile > 60 else 'Check or fold unless pot odds justify a call.'}"
    }
    st.write(f'**Stage Insight ({stage})**: {stage_insights[stage]}')

    st.write('**Action Analysis**')
    for i, action_name in enumerate(action_names):
        expected_value = 0
        if action_name == 'Fold':
            expected_value = 0
            analysis = "Folding avoids further risk but forfeits the current pot. Recommended with weak hands or poor pot odds."
        elif action_name == 'Call':
            if equity > pot_odds + 0.1:
                expected_value = (equity * pot) - ((1 - equity) * avg_opp_bet)
                analysis = f"Calling is profitable with {equity:.2%} equity against {pot_odds:.2%} pot odds. Expected value: ${expected_value:.2f}."
            elif equity > pot_odds:
                expected_value = (equity * pot) - ((1 - equity) * avg_opp_bet)
                analysis = f"Calling is marginal with {equity:.2%} equity close to {pot_odds:.2%} pot odds. Expected value: ${expected_value:.2f}."
            else:
                expected_value = (equity * pot) - ((1 - equity) * avg_opp_bet)
                analysis = f"Calling may be unprofitable with {equity:.2%} equity below {pot_odds:.2%} pot odds. Expected value: ${expected_value:.2f}."
        else:  # Raise
            raise_amount = avg_opp_bet * 2 if avg_opp_bet > 0 else 20
            if equity > 0.6:
                expected_value = (equity * (pot + raise_amount)) - ((1 - equity) * (avg_opp_bet + raise_amount))
                analysis = f"Raising with a strong hand ({equity:.2%} equity) can build the pot or force folds. Expected value: ${expected_value:.2f}."
            elif equity > 0.4 and avg_opp_aggression < 0.5:
                expected_value = (equity * (pot + raise_amount)) - ((1 - equity) * (avg_opp_bet + raise_amount))
                analysis = f"Raising as a semi-bluff with {equity:.2%} equity may induce folds from less aggressive opponents. Expected value: ${expected_value:.2f}."
            else:
                expected_value = (equity * (pot + raise_amount)) - ((1 - equity) * (avg_opp_bet + raise_amount))
                analysis = f"Raising with {equity:.2%} equity is risky against aggressive opponents. Expected value: ${expected_value:.2f}."
        if action_name == 'Call' and positions[user_seat_idx] in ['Small Blind', 'Big Blind']:
            analysis += f" As {positions[user_seat_idx]}, you’ve invested {'half the big blind' if positions[user_seat_idx] == 'Small Blind' else 'the big blind'}, making calling more attractive."
        elif action_name == 'Raise' and positions[user_seat_idx] in ['Cutoff', 'Button']:
            analysis += f" As {positions[user_seat_idx]}, your late position allows more aggressive raises due to acting last postflop."
        st.write(f"- **{action_name}**: {analysis}")

    st.subheader('Prediction History (Last 10)')
    if st.session_state.prediction_history:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.write(history_df)
    else:
        st.write("No predictions yet.")
else:
    st.write("Configure the table setup in the sidebar and click 'Confirm Setup' to proceed.")
