"""
Streamlit Supply Chain Simulation
--------------------------------

This script reimplements the core logic of the interactive supply chain game
originally written as a React application.  It is designed to run on
Streamlit, enabling you to deploy the simulation easily via GitHub.  To run the
app locally use:

    streamlit run streamlit_supply_chain_app.py

The user interface features a sidebar for selecting one of several case
studies, a setup page for choosing whether each role in the supply chain
is driven by AI or manual input, a play loop where inventory levels,
backlogs and costs are updated week by week, and a results page summarizing
the total cost once the simulation has completed.

The AI logic optionally calls Google's Generative Language API when
`apiKey` is set.  If the API call fails or the key is not provided the
simulation falls back to a simple heuristic that orders exactly as much as
the incoming demand.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pandas as pd  # type: ignore
import requests  # type: ignore
import streamlit as st  # type: ignore
import altair as alt  # type: ignore
import streamlit.components.v1 as components



# -----------------------------------------------------------------------------
# Configuration

TOTAL_WEEKS = 20
HOLDING_COST = 0.50
BACKLOG_COST = 2.00

# Set your Google Generative Language API key here or via environment variable.
# If left empty, the AI players will use a simple fallback heuristic.
apiKey: str = os.getenv("GENERATIVE_LANGUAGE_API_KEY", "")


@dataclass
class GameMode:
    id: str
    title: str
    primary_color: str
    demand: List[float]
    initial_inv: int
    unit: str


@dataclass
class RoleState:
    inv: int
    backlog: int
    incoming: List[int] = field(default_factory=lambda: [0, 0])
    last_order: int = 0


# Demand definitions for the various scenarios.  These values mirror those
# used in the original React application.  Some values have been scaled for
# readability (e.g. NVIDIA demand divided by 100, McDonald's demand divided by
# 10).
BEER_DEMAND = [4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
NVIDIA_DEMAND = [135, 136.5, 135.9, 136.2, 136.6, 137.8, 138.1, 138.3, 140, 140,
                 131.2, 129.9, 132.5, 135, 136.2, 138.1, 140.5, 142, 145, 148.5]
MCDONALDS_DEMAND = [77, 80, 81, 85, 83, 79, 82, 80, 84, 86, 88, 85, 82, 80, 78,
                     81, 83, 85, 87, 90]


# Define the possible game modes, mirroring the React configuration.
GAME_MODES: Dict[str, GameMode] = {
    "BEER": GameMode(
        id="BEER",
        title="Beer Game",
        primary_color="#f59e0b",
        demand=BEER_DEMAND,
        initial_inv=12,
        unit="Cases",
    ),
    "SEMICONDUCTOR": GameMode(
        id="SEMICONDUCTOR",
        title="NVIDIA CoWoS",
        primary_color="#10b981",
        demand=NVIDIA_DEMAND,
        initial_inv=150,
        unit="Units (x100)",
    ),
    "FAST_FOOD": GameMode(
        id="FAST_FOOD",
        title="McDonald's Big Mac",
        primary_color="#ef4444",
        demand=MCDONALDS_DEMAND,
        initial_inv=100,
        unit="Patties (x10)",
    ),
}

# Role definitions.  Each role can be played by the AI or by a human.
ROLE_IDS = ["factory", "distributor", "wholesaler", "retailer"]


def initialize_state(game_mode: GameMode) -> Tuple[Dict[str, RoleState], Dict[str, float]]:
    """Return fresh game state and cost trackers for a new simulation."""
    state = {
        rid: RoleState(inv=game_mode.initial_inv, backlog=0) for rid in ROLE_IDS
    }
    costs = {rid: 0.0 for rid in ROLE_IDS}
    return state, costs


def call_ai_for_order(
    role_id: str,
    state: RoleState,
    incoming_demand: int,
    api_key: str,
) -> Tuple[int, str]:
    """Ask the generative language model for an order quantity and reasoning.

    If an API key is not provided or the API call fails for any reason, this
    function returns a simple heuristic decision along with a brief reason.
    """
    # Fallback: order exactly the incoming demand.
    fallback = (incoming_demand, "Smoothing demand.")
    if not api_key:
        return fallback

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.5-flash-preview-09-2025:generateContent?key=" + api_key
    )
    # Compose the prompt similar to the React app.  The AI is asked to output
    # JSON with an `order` number and a `reasoning` string.
    system_instruction = {
        "parts": [
            {
                "text": "Supply chain AI. JSON: {\"order\": number, \"reasoning\": \"string\"}"
            }
        ]
    }
    contents = [
        {
            "parts": [
                {
                    "text": (
                        f"Role: {role_id}, Inv: {state.inv}, Backlog: {state.backlog}, "
                        f"Demand: {incoming_demand}. Order?"
                    )
                }
            ]
        }
    ]
    body = {
        "contents": contents,
        "systemInstruction": system_instruction,
        "generationConfig": {"responseMimeType": "application/json"},
    }
    try:
        response = requests.post(url, headers={"Content-Type": "application/json"}, json=body)
        response.raise_for_status()
        data = response.json()
        # Extract the JSON string from the response.  The structure follows
        # data["candidates"][0]["content"]["parts"][0]["text"].  If parsing
        # fails, fall back to the heuristic.
        candidate = data.get("candidates", [{}])[0]
        content = candidate.get("content", {}).get("parts", [{}])[0].get("text", "")
        decision = json.loads(content)
        order = int(decision.get("order", incoming_demand))
        reasoning = str(decision.get("reasoning", "AI reasoning unavailable."))
        return order, reasoning
    except Exception:
        return fallback


def process_turn(
    active_game: GameMode,
    game_state: Dict[str, RoleState],
    total_costs: Dict[str, float],
    manual_orders: Dict[str, int],
    role_configs: Dict[str, str],
    current_week: int,
    api_key: str,
) -> Tuple[int, Dict[str, RoleState], Dict[str, float], Dict[str, Dict[str, str]], List[Dict]]:
    """Advance the simulation by one week.

    Returns updated values for the current week counter, game state, total
    costs, AI thoughts and history.  The history list will be appended with
    this week's summary for plotting inventory health.
    """
    external_demand = active_game.demand[current_week] if current_week < len(active_game.demand) else active_game.demand[-1]
    decisions: Dict[str, Tuple[int, str]] = {}
    # Acquire decisions from AI or manual input.  We work from the bottom of the
    # supply chain up since each decision depends on the downstream order.
    # Retailer receives external demand.
    for role in reversed(ROLE_IDS):
        if role == "retailer":
            incoming = external_demand
        else:
            # The incoming demand for this role is the order placed by the downstream role.
            downstream = ROLE_IDS[ROLE_IDS.index(role) + 1]
            incoming = decisions[downstream][0]
        if role_configs.get(role, "AI") == "MANUAL":
            # Use manual order if provided; default to zero.
            order = int(manual_orders.get(role, 0))
            decisions[role] = (order, "Manual input")
        else:
            order, reasoning = call_ai_for_order(role, game_state[role], incoming, api_key)
            decisions[role] = (order, reasoning)

    # Apply inventory flows and update costs.  We use a copy of the state to avoid
    # mutating in-place while computing shipments.
    ai_thoughts: Dict[str, Dict[str, str]] = {}
    history_entry: Dict[str, float] = {"week": current_week + 1, "demand": external_demand}
    next_state = {rid: RoleState(inv=st.inv, backlog=st.backlog, incoming=list(st.incoming)) for rid, st in game_state.items()}
    next_costs = total_costs.copy()

    # --- FLOW SECTION (replace your calculate_flow + its calls with this) ---

    def step(role_id: str, demand_from_below: int) -> int:
        state = next_state[role_id]
    
        # Receive incoming deliveries (arrive after delay)
        arrived = state.incoming[0]
        state.inv += arrived
    
        # Shift incoming queue forward; we'll fill incoming[1] after we compute shipments
        state.incoming = [state.incoming[1], 0]
    
        # Ship to downstream as much as possible
        total_needed = demand_from_below + state.backlog
        shipped = min(state.inv, total_needed)
        state.inv -= shipped
        state.backlog = total_needed - shipped
    
        # Update costs
        next_costs[role_id] += state.inv * HOLDING_COST + state.backlog * BACKLOG_COST
        return shipped
    
    # Run flows top -> bottom
    ship_to_dist = step("factory", decisions["distributor"][0])
    ship_to_whole = step("distributor", decisions["wholesaler"][0])
    ship_to_retail = step("wholesaler", decisions["retailer"][0])
    _ = step("retailer", external_demand)  # shipped to customer (unused)
    
    # Push shipments into downstream incoming queues (2-week delay)
    next_state["distributor"].incoming[1] = ship_to_dist
    next_state["wholesaler"].incoming[1] = ship_to_whole
    next_state["retailer"].incoming[1] = ship_to_retail
    
    # Factory production pipeline uses factory order decision as "production started"
    next_state["factory"].incoming[1] = decisions["factory"][0]


    # Record AI reasoning for each role.
    for rid in ROLE_IDS:
        ai_thoughts[rid] = {
            "order": decisions[rid][0],
            "reasoning": decisions[rid][1],
        }

    # Append inventory levels for the history chart.
    history_entry["retailerInv"] = next_state["retailer"].inv
    history_entry["factoryInv"] = next_state["factory"].inv
    history_entry["wholesalerInv"] = next_state["wholesaler"].inv
    history_entry["distributorInv"] = next_state["distributor"].inv

    return (current_week + 1, next_state, next_costs, ai_thoughts, [history_entry])


def demand_chart(game_mode: GameMode, current_week: int, interactive: bool = False) -> alt.Chart:
    """Create an Altair area chart of the demand curve.

    When `interactive` is True, a vertical rule marks the current week.
    """
    df = pd.DataFrame({"week": list(range(1, len(game_mode.demand) + 1)), "demand": game_mode.demand})
    base = alt.Chart(df).encode(
        x=alt.X("week:O", title="Week"),
        y=alt.Y("demand:Q", title="Demand"),
    )
    # Use a simple fill and line colour.  Vega-Lite v6 does not allow a gradient
    # definition as the colour value for marks, so specifying a Gradient here
    # raises a SchemaValidationError.  Instead we set a fixed colour and
    # adjust the opacity for the filled area.  If you prefer a gradient
    # appearance you can overlay a semi-transparent rectangle or customise
    # styling further, but a constant opacity yields a clean result.
    area = base.mark_area(
        line={"color": game_mode.primary_color},
        color=game_mode.primary_color,
        opacity=0.3,
    )
    chart = area
    if interactive:
        rule = alt.Chart(pd.DataFrame({"week": [current_week + 1]})).mark_rule(
            color="#1e293b"
        ).encode(x="week:O")
        chart = area + rule
    return chart.properties(height=200)


def inventory_health_chart(history: List[Dict], game_mode: GameMode) -> alt.Chart:
    """Create an Altair line chart showing inventory health over time."""
    if not history:
        # Return a valid empty chart
        return alt.Chart(pd.DataFrame({"week": [], "value": [], "metric": []})).mark_line()

    hist_df = pd.DataFrame(history)
    long_df = pd.melt(
        hist_df,
        id_vars=["week"],
        value_vars=["retailerInv", "factoryInv", "demand"],
        var_name="metric",
        value_name="value",
    )

    color_map = {
        "retailerInv": game_mode.primary_color,
        "factoryInv": "#1e293b",
        "demand": "#cbd5e1",
    }

    base = alt.Chart(long_df).encode(
        x=alt.X("week:Q", title="Week"),
        y=alt.Y("value:Q", title="Value"),
        color=alt.Color(
            "metric:N",
            scale=alt.Scale(
                domain=list(color_map.keys()),
                range=[color_map[k] for k in color_map.keys()],
            ),
            legend=alt.Legend(title="Metric"),
        ),
    )

    line = base.mark_line()
    points = base.mark_point(filled=True, size=30)

    return (line + points).properties(height=250)

def render_generated_supply_chain_diagram(
    game_state: Dict[str, RoleState],
    ai_thoughts: Dict[str, Dict[str, str]],
    current_demand: float,
    unit_label: str,
) -> None:
    """
    Draws a supply-chain diagram (no external image) using SVG.
    Numbers update automatically from `game_state` + `ai_thoughts`.
    """

    def last_order(role: str) -> int:
        v = ai_thoughts.get(role, {}).get("order", 0)
        try:
            return int(v)
        except Exception:
            return 0

    # Values from state
    r = game_state["retailer"]
    w = game_state["wholesaler"]
    d = game_state["distributor"]
    f = game_state["factory"]

    # Orders placed (top row)
    retailer_order = last_order("retailer")
    wholesaler_order = last_order("wholesaler")
    distributor_order = last_order("distributor")
    factory_order = last_order("factory")

    # Incoming queues (shipment/production delays)
    r0, r1 = r.incoming
    w0, w1 = w.incoming
    d0, d1 = d.incoming
    f0, f1 = f.incoming

    # Inventories + backlogs
    retailer_inv, retailer_bl = r.inv, r.backlog
    wholesaler_inv, wholesaler_bl = w.inv, w.backlog
    distributor_inv, distributor_bl = d.inv, d.backlog
    factory_inv, factory_bl = f.inv, f.backlog

    # Simple palette
    C_RETAIL = "#bfdbfe"   # light blue
    C_WHOLE  = "#bae6fd"   # cyan-ish
    C_DIST   = "#d9f99d"   # light green
    C_FACT   = "#fecaca"   # light red
    C_TEXT   = "#111827"
    C_LINE   = "#6b7280"
    C_BOX    = "#f3f4f6"

    # SVG canvas size (viewBox); scales responsively
    W, H = 1200, 300

    # Helper snippets
    def circle(x, y, r, fill, label, value=None):
        t_value = f'<text x="{x}" y="{y+6}" text-anchor="middle" font-size="18" font-weight="800" fill="{C_TEXT}">{value}</text>' if value is not None else ""
        return f"""
        <g>
          <circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="rgba(17,24,39,0.15)" stroke-width="2"></circle>
          <text x="{x}" y="{y-r-10}" text-anchor="middle" font-size="14" font-weight="700" fill="{C_TEXT}">{label}</text>
          {t_value}
        </g>
        """

    def rect(x, y, w, h, fill, label, value=None, sub=None):
        t_value = f'<text x="{x+w/2}" y="{y+h/2+6}" text-anchor="middle" font-size="18" font-weight="800" fill="{C_TEXT}">{value}</text>' if value is not None else ""
        t_sub = f'<text x="{x+w/2}" y="{y+h+18}" text-anchor="middle" font-size="12" fill="{C_TEXT}" opacity="0.75">{sub}</text>' if sub else ""
        return f"""
        <g>
          <rect x="{x}" y="{y}" rx="14" ry="14" width="{w}" height="{h}" fill="{fill}" stroke="rgba(17,24,39,0.15)" stroke-width="2"></rect>
          <text x="{x+w/2}" y="{y-10}" text-anchor="middle" font-size="14" font-weight="700" fill="{C_TEXT}">{label}</text>
          {t_value}
          {t_sub}
        </g>
        """

    def arrow(x1, y1, x2, y2):
        return f"""
        <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{C_LINE}" stroke-width="3" marker-end="url(#arrow)"/>
        """

    def small_delay_box(x, y, a, b):
        # Two small circles inside a rounded rectangle (for 2-week delay queue)
        return f"""
        <g>
          <rect x="{x}" y="{y}" rx="16" ry="16" width="140" height="55" fill="{C_BOX}" stroke="rgba(17,24,39,0.15)" stroke-width="2"/>
          <circle cx="{x+60}" cy="{y+35}" r="18" fill="white" stroke="rgba(17,24,39,0.15)" stroke-width="2"/>
          <circle cx="{x+110}" cy="{y+35}" r="18" fill="white" stroke="rgba(17,24,39,0.15)" stroke-width="2"/>
          <text x="{x+60}" y="{y+41}" text-anchor="middle" font-size="16" font-weight="800" fill="{C_TEXT}">{a}</text>
          <text x="{x+110}" y="{y+41}" text-anchor="middle" font-size="16" font-weight="800" fill="{C_TEXT}">{b}</text>
          <text x="{x+85}" y="{y+92}" text-anchor="middle" font-size="12" fill="{C_TEXT}" opacity="0.75">shipment delays</text>
        </g>
        """

    # Layout coordinates
    top_y = 70
    bot_y = 210

    x_customer = 90
    x_retail   = 310
    x_whole    = 530
    x_dist     = 750
    x_fact     = 970

    svg = f"""
    <div style="width:100%; max-width:100%; overflow:hidden; margin:0 auto;">
      <svg viewBox="0 0 {W} {H}"
           width="100%"
           preserveAspectRatio="xMidYMid meet"
           xmlns="http://www.w3.org/2000/svg">
        <defs>
          <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
            <path d="M0,0 L0,6 L9,3 z" fill="{C_LINE}"/>
          </marker>
        </defs>
    
        <!-- Top flow: Orders -->
        {circle(x_customer, top_y, 38, "white", "Customer Orders", int(current_demand))}
        {arrow(x_customer+55, top_y, x_retail-55, top_y)}
        {circle(x_retail, top_y, 38, C_RETAIL, "Retailer Orders", retailer_order)}
        {arrow(x_retail+55, top_y, x_whole-55, top_y)}
        {circle(x_whole, top_y, 38, C_WHOLE, "Wholesaler Orders", wholesaler_order)}
        {arrow(x_whole+55, top_y, x_dist-55, top_y)}
        {circle(x_dist, top_y, 38, C_DIST, "Distributor Orders", distributor_order)}
        {arrow(x_dist+55, top_y, x_fact-55, top_y)}
        {circle(x_fact, top_y, 38, C_FACT, "Factory Request", factory_order)}
    
        <!-- Bottom flow: Shipments / Inventories -->
        {rect(x_retail-60, bot_y-28, 120, 55, C_RETAIL, "Retailer Inventory", retailer_inv, sub=f"backlog: {retailer_bl}")}
        {arrow(x_retail-70, bot_y, x_customer+55, bot_y)}
    
        {small_delay_box(x_retail+85, bot_y-35, r0, r1)}
        {arrow(x_retail+60, bot_y, x_retail+85, bot_y)}
        {arrow(x_retail+225, bot_y, x_whole-60, bot_y)}
    
        {rect(x_whole-60, bot_y-28, 120, 55, C_WHOLE, "Wholesaler Inventory", wholesaler_inv, sub=f"backlog: {wholesaler_bl}")}
        {small_delay_box(x_whole+85, bot_y-35, w0, w1)}
        {arrow(x_whole+60, bot_y, x_whole+85, bot_y)}
        {arrow(x_whole+225, bot_y, x_dist-60, bot_y)}
    
        {rect(x_dist-60, bot_y-28, 120, 55, C_DIST, "Distributor Inventory", distributor_inv, sub=f"backlog: {distributor_bl}")}
        {small_delay_box(x_dist+85, bot_y-35, d0, d1)}
        {arrow(x_dist+60, bot_y, x_dist+85, bot_y)}
        {arrow(x_dist+225, bot_y, x_fact-60, bot_y)}
    
        {rect(x_fact-60, bot_y-28, 120, 55, C_FACT, "Factory Inventory", factory_inv, sub=f"backlog: {factory_bl}")}
    
        <!-- Production delays to the right -->
        <g>
          <rect x="{x_fact+120}" y="{top_y+35}" rx="16" ry="16" width="130" height="190"
                fill="{C_BOX}" stroke="rgba(17,24,39,0.15)" stroke-width="2"/>
          <text x="{x_fact+185}" y="{top_y+25}" text-anchor="middle" font-size="14" font-weight="700" fill="{C_TEXT}">
            Production delays
          </text>
          <circle cx="{x_fact+185}" cy="{top_y+90}" r="22" fill="white" stroke="rgba(17,24,39,0.15)" stroke-width="2"/>
          <circle cx="{x_fact+185}" cy="{top_y+150}" r="22" fill="white" stroke="rgba(17,24,39,0.15)" stroke-width="2"/>
          <text x="{x_fact+185}" y="{top_y+96}" text-anchor="middle" font-size="14" font-weight="800" fill="{C_TEXT}">{f0}</text>
          <text x="{x_fact+185}" y="{top_y+156}" text-anchor="middle" font-size="14" font-weight="800" fill="{C_TEXT}">{f1}</text>
        </g>
    
        <!-- Footer -->
        <text x="{W/2}" y="{H-12}" text-anchor="middle" font-size="12" fill="{C_TEXT}" opacity="0.75">
          Units: {unit_label} - Numbers update each submitted week (inventory / delays / orders)
        </text>
      </svg>
    </div>
    """

    components.html(svg, height=320, scrolling=False)



def main() -> None:
    st.set_page_config(page_title="Supply Chain Simulation", layout="wide")
    st.title("ChainMaster Simulation")
    # Initialize session state variables only on first run
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        # Default to Beer game
        st.session_state.active_game_id = "BEER"
        st.session_state.game_state, st.session_state.total_costs = initialize_state(GAME_MODES[st.session_state.active_game_id])
        st.session_state.role_configs = {rid: ("MANUAL" if rid == "retailer" else "AI") for rid in ROLE_IDS}
        st.session_state.history: List[Dict] = []
        st.session_state.current_week = 0
        st.session_state.ai_thoughts: Dict[str, Dict[str, str]] = {}
        st.session_state.view = "intro"

    # Retrieve the current game mode object
    active_game = GAME_MODES[st.session_state.active_game_id]
    # Sidebar for selecting the game mode
    with st.sidebar:
        st.header("Case Studies")
        # Use a radio button for selecting among the available game modes
        game_selection = st.radio(
            label="Select a case study", options=list(GAME_MODES.keys()),
            index=list(GAME_MODES.keys()).index(st.session_state.active_game_id),
            format_func=lambda k: GAME_MODES[k].title,
        )
        if game_selection != st.session_state.active_game_id:
            # Reset state when the user switches case studies
            st.session_state.active_game_id = game_selection
            st.session_state.game_state, st.session_state.total_costs = initialize_state(GAME_MODES[game_selection])
            st.session_state.history = []
            st.session_state.current_week = 0
            st.session_state.ai_thoughts = {}
            st.session_state.view = "intro"
        # Display unit information
        st.markdown(f"**Scale**: {active_game.unit} per unit")

    # Render the appropriate page based on the current view
    view = st.session_state.view
    current_week = st.session_state.current_week
    game_state = st.session_state.game_state
    total_costs = st.session_state.total_costs
    role_configs = st.session_state.role_configs
    history = st.session_state.history
    ai_thoughts = st.session_state.ai_thoughts

    if view == "intro":
        st.subheader(active_game.title)
        st.write("This simulation models a simple supply chain with four stages: factory, distributor, wholesaler and retailer.  Each role can be controlled by you or by an AI assistant.  The demand curve is predefined for each case study.")
        st.altair_chart(demand_chart(active_game, current_week, interactive=False), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Initialize Simulation", key="init_button"):
                st.session_state.view = "setup"

    elif view == "setup":
        st.subheader("Configure the chain")
        # Role configurations
        for rid in ROLE_IDS:
            col_ai, col_manual = st.columns(2)
            with col_ai:
                if st.button(f"AI ({rid.capitalize()})", key=f"ai_{rid}", disabled=(role_configs[rid] == "AI")):
                    st.session_state.role_configs[rid] = "AI"
                    st.rerun()

            with col_manual:
                if st.button(f"Manual ({rid.capitalize()})", key=f"manual_{rid}", disabled=(role_configs[rid] == "MANUAL")):
                    st.session_state.role_configs[rid] = "MANUAL"
                    st.rerun()

            st.write(f"Current mode for **{rid}**: **{st.session_state.role_configs[rid]}**")
        st.altair_chart(demand_chart(active_game, current_week, interactive=False), use_container_width=True)
        if st.button("Launch Game Loop", key="launch_button"):
            st.session_state.view = "play"

    elif view == "play":
        # Display current week and demand
        st.subheader(f"Week {current_week + 1} / {TOTAL_WEEKS}")
        current_demand = active_game.demand[current_week] if current_week < len(active_game.demand) else active_game.demand[-1]
        st.write(f"Current external demand: **{current_demand}** {active_game.unit}")
        st.altair_chart(demand_chart(active_game, current_week, interactive=True), use_container_width=True)


        # Display total chain cost
        total_chain_cost = sum(total_costs.values())
        st.metric(label="Total Chain Cost", value=f"${total_chain_cost:.0f}")
        st.markdown("### Live Supply Chain Flow")
        render_generated_supply_chain_diagram(
            game_state=game_state,
            ai_thoughts=ai_thoughts,
            current_demand=current_demand,
            unit_label=active_game.unit,
        )

        
        # Display role states and allow manual order entry
        manual_orders: Dict[str, int] = {}

        for rid in ROLE_IDS:
            ...
            if role_configs[rid] == "MANUAL":
                k = f"order_input_{rid}_{current_week}"
                st.number_input(
                    label=f"Order quantity for {rid}",
                    min_value=0,
                    value=int(current_demand),
                    step=1,
                    key=k,
                )
                manual_orders[rid] = int(st.session_state.get(k, 0))

            with col3:
                if role_configs[rid] == "AI":
                    reasoning = ai_thoughts.get(rid, {}).get("reasoning", "Awaiting AI...")
                    st.write(f"AI reasoning: *{reasoning}*")
        # Submit button
        disabled = (current_week >= TOTAL_WEEKS)
        if st.button(
            f"Submit Week {current_week + 1}", key=f"submit_week_{current_week}", disabled=disabled
        ):
            new_week, new_state, new_costs, new_thoughts, new_history = process_turn(
                active_game,
                game_state,
                total_costs,
                manual_orders,
                role_configs,
                current_week,
                apiKey,
            )
            st.session_state.current_week = new_week
            st.session_state.game_state = new_state
            st.session_state.total_costs = new_costs
            st.session_state.ai_thoughts = new_thoughts
            st.session_state.history += new_history
            if new_week >= TOTAL_WEEKS:
                st.session_state.view = "results"

        # Display inventory health chart
        st.markdown("### Inventory Health")
        st.altair_chart(inventory_health_chart(history, active_game), use_container_width=True)

    elif view == "results":
        st.header("Simulation Results")
        total_chain_cost = sum(total_costs.values())
        st.markdown(f"**Total Network Operational Cost:** **${total_chain_cost:.0f}**")
        # Display individual costs
        for rid in ROLE_IDS:
            st.markdown(f"- **{rid.capitalize()}**: ${total_costs[rid]:.0f}")
        # Restart or reconfigure
        col1, col2 = st.columns(2)
        with col1:
            if st.button("New Simulation", key="new_sim"):
                st.session_state.game_state, st.session_state.total_costs = initialize_state(active_game)
                st.session_state.history = []
                st.session_state.current_week = 0
                st.session_state.ai_thoughts = {}
                st.session_state.view = "intro"
        with col2:
            if st.button("Reconfigure", key="reconfig"):
                st.session_state.view = "setup"


if __name__ == "__main__":
    main()
