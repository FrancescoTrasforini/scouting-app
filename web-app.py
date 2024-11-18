from flask import Flask, render_template, request, jsonify, send_file, url_for
import pg8000
import pandas as pd
from scipy.stats import percentileofscore
import os
from dotenv import load_dotenv
import dotenv
import matplotlib.pyplot as plt
import matplotlib.colors 
import numpy as np
from mplsoccer import PyPizza, add_image
import io
import base64

# Manually specify the path to .env file
load_dotenv()

app = Flask(__name__)

# Connect to PostgreSQL database
# Establish the connection
conn = pg8000.connect(
    user=os.getenv("DB_USER"),         
    password=os.getenv("DB_PASSWORD"),    
    host=os.getenv("DB_HOST"),             
    port=os.getenv("DB_PORT"),            
    database=os.getenv("DB_NAME")
)

# Create a cursor to execute SQL queries
cursor = conn.cursor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/players', methods=['GET'])
def get_players():
    role = request.args.get('role') 
    players = []
    # Fetch players based on the role
    if role:
        players = get_players_by_role(role)
    print(f"Players for {role}: {players}")  # Print the fetched players data
    # Return data as JSON
    return jsonify(players)

def get_players_by_role(role):
    if role == 'RWB':
        cursor.execute("""
            SELECT "Player", "Team", "Age", "Minutes played"
            FROM players
            WHERE "Position" && ARRAY['RCB', 'RB', 'RWB', 'RCMF', 'RW', 'RAMF', 'RWF']
        """)
    elif role == 'CF':
        cursor.execute("""
            SELECT "Player", "Team", "Age", "Minutes played"
            FROM players
            WHERE "Position" && ARRAY ['CF']
        """)
    elif role == 'GK':
        cursor.execute("""
            SELECT "Player", "Team", "Age", "Minutes played"
            FROM players
            WHERE "Position" && ARRAY ['GK']
        """)
    else:
        return []
    
    # Fetch results as a list of dictionaries
    results = cursor.fetchall()
    # Convert results into a list of dictionaries
    players = []
    for row in results:
        player = {
            "Player": row[0],        # First column: Player
            "Team": row[1],          # Second column: Team
            "Age": row[2] if row[2] is not None else 'Unavailable',  # Replace None with 'Unknown'
            "Minutes played": row[3] # Fourth column: Minutes played
        }
        players.append(player)
    
    return players

@app.route('/shortlist', methods=['GET'])
def create_shortlist():
    role = request.args.get('role')
    age_condition = request.args.get('ageCondition')
    age_value = request.args.get('ageValue')
    minutes = request.args.get('minutes')

    # Validate and parse input
    age_value = int(age_value) if age_value and age_value.isdigit() else None
    minutes = int(minutes) if minutes and minutes.isdigit() else 0

    if not role:
        return jsonify([])

    # Base query (atm base query hardcoded for RWB)
    query = """
        SELECT "Player", "Team", "Age", "Minutes played", "Position",
               "Successful defensive actions per 90", 
               "PAdj Interceptions",
               "Progressive runs per 90", 
               "Accurate crosses, %",
               "Progressive passes per 90"
        FROM players
        WHERE "Position" && ARRAY['RCB', 'RB', 'RWB', 'RCMF', 'RW', 'RAMF', 'RWF']
    """

    # Apply filters
    filters = []
    # Fetch players for the role from the database
    if age_value:
        if age_condition == 'under':
            filters.append(f""" "Age" < {age_value} """)
        elif age_condition == 'over':
            filters.append(f""" "Age" > {age_value} """)
        elif age_condition == 'equal':
            filters.append(f""" "Age" = {age_value} """)
    if minutes:
        filters.append(f""" "Minutes played" >= {minutes} """)

    # Append filters to query
    if filters:
        query += " AND " + " AND ".join(filters)

    # Execute query
    cursor.execute(query)
    players = cursor.fetchall()

    if not players:
        return jsonify({"players": [], "count": 0})
    
    # Convert to DataFrame with all necessary metrics
    df = pd.DataFrame(players, columns=["Player", "Team", "Age", "Minutes played", "Position",
                                        "Successful defensive actions per 90", "PAdj Interceptions",
                                        "Progressive runs per 90", "Accurate crosses, %", "Progressive passes per 90"])

    # Calculate percentiles and total scores
    metrics = ["Successful defensive actions per 90", "PAdj Interceptions",
               "Progressive runs per 90", "Accurate crosses, %",
               "Progressive passes per 90"]
    
    for metric in metrics:
        valid_values = df[metric].dropna()  # drop NaN
        df[f"{metric} Percentile"] = df[metric].apply(lambda x: int(percentileofscore(valid_values, x)) if pd.notna(x) else None)

    # Fill NaN values with 0 in all percentile columns before summing them
    df[[f"{metric} Percentile" for metric in metrics]] = df[[f"{metric} Percentile" for metric in metrics]].fillna(0)

    # Calculate the total score by summing the percentiles
    df['Total Score'] = df[[f"{metric} Percentile" for metric in metrics]].sum(axis=1)
    #df['Total Score'] = df[[f"{metric} Percentile" for metric in metrics]].sum(axis=1)
    # Replace NaN values with 0 in the Percentile columns before summing
    #df['Total Score'] = df[[f"{metric} Percentile" for metric in metrics]].apply(
    #    lambda row: sum([value if pd.notna(value) else 0 for value in row]), axis=1)

    # Get the top 5 players including ties for 5th place
    df = df.sort_values(by='Total Score', ascending=False)
    top_5_threshold = df['Total Score'].iloc[4] if len(df) > 4 else df['Total Score'].iloc[-1]
    shortlist = df[df['Total Score'] >= top_5_threshold]

    # Prepare shortlist data for response
    shortlist_data = shortlist[["Player", "Team", "Age", "Minutes played", "Position"] +  # Include basic player info
                               metrics +  # Include the original metrics
                               [f"{metric} Percentile" for metric in metrics] +  # Include the percentiles
                               ["Total Score"]]  # Include the total score
    
    # Convert to dictionary and return as JSON response
    shortlist_data = shortlist.to_dict(orient='records')
    # Log the data to check if values are correct before sending them
    print(shortlist_data)  
    return jsonify({"shortlist_data": shortlist_data, "count": len(df)})

@app.route('/players', methods=['GET'])
def get_all_players():
    # Fetch all players from the database
    cursor.execute('SELECT "Player" FROM players')
    players = cursor.fetchall()
    return jsonify([{"Player": player[0]} for player in players])

# Route to serve the player profile page
@app.route('/player-profile.html')
def player_profile():
    player_name = request.args.get('player')  # Get player from URL query param
    compare_to = request.args.get('compareTo', 'S. Minihan')  # Default to 'S.Minihan' if not provided
    # Fetch all player names for the comparison dropdown
    cursor.execute('SELECT "Player" FROM players')
    players = cursor.fetchall()
    #player_data = get_player_profile_data(player_name)  
    # Render the page with player data and list of players for comparison
    return render_template('player-profile.html', player_name=player_name, players=players)

# Pizza chart code 
def create_pizza_chart(name, percentiles, chart_name="pizza_chart"):
    
    print(percentiles) #Debug
    params = [
        "Successful defensive actions per 90", 
        "PAdj Interceptions",
        "Progressive runs per 90", 
        "Accurate crosses, %",
        "Progressive passes per 90"
    ]
    # Extract the values from the dictionary into a list
    values = list(percentiles.values())
    print(values)  # For debugging purposes
    #values = [int(float(percentiles[metric])) for metric in metrics]

    # Change this line to specify 5 different colors
    slice_colors = ["red"] + ["cyan"] + ["orange"] + ["green"] + ["blue"]
    text_colors = ["black"] * 3 + ["white"] * 2

    # instantiate PyPizza class
    baker = PyPizza(
        params=params,                 # list of parameters
        background_color="white",       # background color
        straight_line_color="white",    # color for straight lines
        straight_line_lw=1,             # linewidth for straight lines
        last_circle_lw=0,               # linewidth of last circle
        other_circle_lw=0,              # linewidth for other circles
        inner_circle_size=20            # size of inner circle
    )
    # plot pizza
    fig, ax = baker.make_pizza(
        values,                          # list of values
        figsize=(8, 8.5),                # adjust figsize 
        color_blank_space="same",        # use same color to fill blank space
        slice_colors=slice_colors,       # color for individual slices
        value_colors=text_colors,            # color for the value-text
        value_bck_colors=slice_colors,   # color for the blank spaces
        blank_alpha=0.4,                 # alpha for blank-space colors
        kwargs_slices=dict(
            edgecolor="white", zorder=2, linewidth=1
        ),                               # values to be used when plotting slices
        kwargs_params=dict(
            color="black", fontsize=11,
            fontproperties='DejaVu Sans',
            va="center"
        ),                               # values to be used when adding parameter
        kwargs_values=dict(
            color="white", fontsize=12,
            fontproperties='DejaVu Sans',
            zorder=3,
            bbox=dict(
                edgecolor="white", facecolor="white",
                boxstyle="round,pad=0.1", lw=0.2,
            )
        )                               
    )

    # add title
    fig.text(
        0.515, 0.975, f"Pizza Plot - {name}", size=16,
        ha="center", color="white"
    )
    # add subtitle
    fig.text(
        0.515, 0.953,
        "Percentile Rank vs Entire dataset",
        size=13,
        ha="center", color="white"
    )

    # Ensure the folder exists
    output_dir = 'static/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the chart as an image
    chart_path = os.path.join(output_dir, f"{chart_name}.png")
    chart_path = chart_path.replace("\\", "/")
    plt.savefig(chart_path, bbox_inches='tight')
    plt.close(fig)

    # Return just the relative path (excluding 'static/')
    return f"{chart_name}.png"
    #return chart_path

# Function to calculate percentiles
def calculate_percentiles(player_data, metrics, cursor):
    percentiles = {}
    
    for metric in metrics:
        # Fetch all values for the metric from the database
        cursor.execute(f'SELECT "{metric}" FROM players WHERE "{metric}" IS NOT NULL')
        all_values = [row[0] for row in cursor.fetchall()]  # Flatten results

        # Calculate percentile for the player's value
        player_value = player_data[metric]
        if player_value is not None:
            percentiles[metric] = int(percentileofscore(all_values, player_value))
        else:
            percentiles[metric] = None  # Handle missing values

    return percentiles

@app.route('/player-profile-data', methods=['GET'])
def get_player_profile_data():
    player_name = request.args.get('player')
    compare_to = request.args.get('compareTo')

    metrics = [
            "Successful defensive actions per 90", "PAdj Interceptions",
            "Progressive runs per 90", "Accurate crosses, %",
            "Progressive passes per 90"
        ]
    
    # Fetch metrics for the selected player
    cursor.execute('SELECT "Successful defensive actions per 90", "PAdj Interceptions", "Progressive runs per 90", "Accurate crosses, %", "Progressive passes per 90" FROM players WHERE "Player" = %s', (player_name,))
    player_data_raw = cursor.fetchone()
    player_data = dict(zip(metrics, player_data_raw))  # Create dictionary of player data

    # Fetch metrics for the comparison player
    cursor.execute('SELECT "Successful defensive actions per 90", "PAdj Interceptions", "Progressive runs per 90", "Accurate crosses, %", "Progressive passes per 90" FROM players WHERE "Player" = %s', (compare_to,))
    compare_data_raw = cursor.fetchone()
    compare_data = dict(zip(metrics, compare_data_raw))  # Create dictionary of comparison player data

    # Compute percentiles
    player_percentiles = calculate_percentiles(player_data, metrics, cursor)
    print(f"Player percentiles: {player_percentiles}")
    compare_percentiles = calculate_percentiles(compare_data, metrics, cursor)
    print(f"Comparison Player percentiles: {compare_percentiles}")

    player_pizza_chart = create_pizza_chart(player_name,player_percentiles,chart_name=f"{player_name}")
    compare_pizza_chart = create_pizza_chart(compare_to,compare_percentiles,chart_name=f"{compare_to}")

    # Generate the URLs dynamically using url_for
    player_pizza_chart_url = url_for('static', filename=player_pizza_chart)
    compare_pizza_chart_url = url_for('static', filename=compare_pizza_chart)  #f'pizza_charts/{compare_to}.png'

    # Log to verify the data being sent
    print("Sending data:", player_pizza_chart_url, compare_pizza_chart_url)

    return jsonify({
        'playerPizzaChart': player_pizza_chart_url,
        'comparePizzaChart': compare_pizza_chart_url
    })
    
def get_all_players():
    # Retrieve all player names for dropdown 
    cursor.execute('SELECT "Player" FROM players')
    player_names = [row[0] for row in cursor.fetchall()]  # Extract player names from tuples
    return player_names

@app.route('/compare.html', methods=['GET'])
def create_bar_chart():
    print("create_bar_chart() called")  # Add this for debugging
    # Get the selected player and the shortlist from the form
    compare_to = request.args.get('compareTo', 'S. Minihan')
    shortlist_param = request.args.get('shortlist', '')
    shortlist = shortlist_param.split(',') if shortlist_param else []

    print(f"Shortlist: {shortlist}")
    # Retrieve all player names for dropdown 
    player_names = get_all_players()

    # Combine the selected player with the shortlist
    players_to_compare = [compare_to] + shortlist

    # Retrieve the relevant player data for the comparison
    cursor.execute("""
        SELECT "Player", 
               "Successful defensive actions per 90", 
               "PAdj Interceptions", 
               "Progressive runs per 90", 
               "Accurate crosses, %", 
               "Progressive passes per 90"
        FROM players
        WHERE "Player" IN (%s)
    """, (', '.join(['%s'] * len(players_to_compare)),), tuple(players_to_compare))
    
    players_data = cursor.fetchall()

    # Convert the data into a dictionary for easier processing
    players_dict = {player[0]: player[1:] for player in players_data}

    # Metrics to calculate percentiles for
    metrics = ["Successful defensive actions per 90", "PAdj Interceptions",
               "Progressive runs per 90", "Accurate crosses, %",
               "Progressive passes per 90"]

    # Mapping for shorter metric labels
    metric_labels = {
        "Successful defensive actions per 90": "Defensive Actions",
        "PAdj Interceptions": "PAdj Interceptions",
        "Progressive runs per 90": "Progressive Runs",
        "Accurate crosses, %": "Crossing Accuracy",
        "Progressive passes per 90": "Progressive Passes"
    }

    # Fetch the data for percentile calculation for the entire database
    cursor.execute('SELECT "Player", "Successful defensive actions per 90", "PAdj Interceptions", '
                   '"Progressive runs per 90", "Accurate crosses, %", "Progressive passes per 90" FROM players')
    all_players_data = cursor.fetchall()

    # Convert to a DataFrame to calculate percentiles
    import pandas as pd
    df = pd.DataFrame(all_players_data, columns=["Player"] + metrics)

    # Calculate percentiles for each metric
    for metric in metrics:
        valid_values = df[metric].dropna()  # drop NaN
        df[f"{metric} Percentile"] = df[metric].apply(lambda x: int(percentileofscore(valid_values, x)) if pd.notna(x) else None)

    # Get the percentiles for the players we are comparing
    players_percentiles = {}
    for player in players_to_compare:
        player_data = df[df['Player'] == player]
        players_percentiles[player] = {metric: player_data[f"{metric} Percentile"].values[0] for metric in metrics}

    # Generate the bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    # Bar chart for percentiles
    bar_width = 0.10
    positions = range(len(metrics))

    for i, player in enumerate(players_to_compare):
        ax.bar([p + i * bar_width for p in positions], 
               [players_percentiles[player].get(metric, 0) for metric in metrics],
               width=bar_width, label=player)

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Percentile')
    ax.set_title('Player Comparison by Percentile')

    # Set the x-ticks to the shorter metric labels
    ax.set_xticks([p + bar_width * (len(players_to_compare) - 1) / 2 for p in positions])
    ax.set_xticklabels([metric_labels[metric] for metric in metrics])  # Use shorter labels

    # Ensure the y-axis always goes from 0 to 100
    ax.set_ylim(0, 100)  # Set the y-axis limits

    # Explicitly set the y-ticks to appear every 10 units
    ax.set_yticks(range(0, 101, 10))  # This will create ticks at 0, 10, 20, ..., 100
    
    # Add legend outside the plot (using bbox_to_anchor)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Players")
    # Adjust layout to make room for the legend
    plt.tight_layout()

    # Save the plot to a BytesIO object to embed in HTML
    import io
    import base64
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    #plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Render the compare.html template with the chart and player data
    return render_template(
        'compare.html',  # Use the existing template
        players_data=players_dict,   # Player data
        players_percentiles=players_percentiles,  # Percentile data for each player
        img_base64=img_base64,  # Base64 encoded image for embedding
        player_names=player_names, # Player names for dropdown menu
        shortlist=shortlist
    )
    
if __name__ == '__main__':
    app.run(debug=True)