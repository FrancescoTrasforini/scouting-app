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
from urllib.parse import unquote
import io
import base64

# Manually specify the path to .env file
# Only load .env in development
load_dotenv()

app = Flask(__name__)

print(f'user: {os.getenv("DB_USER")}')
print(f'user: {os.getenv("DB_PASSWORD")}')
print(f'user: {os.getenv("DB_HOST")}')
print(f'user: {os.getenv("DB_PORT")}')
print(f'user: {os.getenv("DB_NAME")}')

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

# Define role-to-position mapping
role_position_mapping = {
    'GK': ['GK'],
    'RWB': ['RCB', 'RB', 'RWB', 'RCMF', 'RW', 'RAMF', 'RWF'],
    'LWB': ['LCB', 'LB', 'LWB', 'LCMF', 'LW', 'LAMF', 'LWF'],
    'CM': ['RCMF', 'CM', 'LCMF'],
    'CF': ['CF'],
}

# Define role-to-columns mapping
role_column_mapping = {
    'GK': [
        "Prevented goals per 90", "Aerial duels won, %", "Accurate long passes, %",
        "Accurate passes, %", "PAdj Interceptions"
    ],
    'RWB': [
        "Successful defensive actions per 90", "PAdj Interceptions",
        "Progressive runs per 90", "Accurate crosses, %", "Progressive passes per 90"
    ],
}

# Role-agnostic mapping for shorter metric labels
metric_labels = {
    "Successful defensive actions per 90": "Defensive Actions",
    "PAdj Interceptions": "PAdj Interceptions",
    "Progressive runs per 90": "Progressive Runs",
    "Accurate crosses, %": "Crossing Accuracy",
    "Progressive passes per 90": "Progressive Passes",
    "Prevented goals per 90": "Prevented Goals",
    "Aerial duels won, %": "Aerial Duels Won",
    "Accurate long passes, %": "Long Pass Accuracy",
    "Accurate passes, %": "Pass Accuracy",
}

def get_metric_labels_for_role(role_columns):
    # Filter the metric labels dictionary
    filtered_labels = {metric: metric_labels[metric] for metric in role_columns if metric in metric_labels}
    
    return filtered_labels

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/players', methods=['GET'])
def get_players():
    role = request.args.get('role') 
    tier = request.args.get('tier') 
    players = []
    # Fetch players based on the role
    if role:
        positions = role_position_mapping.get(role)
        players = get_players_homepage(positions,tier)
    print(f"Players for {role}: {players}")  # Print the fetched players data
    # Return data as JSON
    return jsonify(players)

def get_players_homepage(positions,tier):
    # Build the SQL query dynamically
    query = f"""
        SELECT "Player", "Team", "Age", "Minutes played"
        FROM players
        WHERE "Position" && %s AND "Tier" = ARRAY[%s]
    """
    if positions:
        cursor.execute(query,(positions,tier))
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
            "Age": row[2] if row[2] is not None else 'Unavailable',  # Replace None with 'Unavailable'
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
    tier = request.args.get('tier')

    # Validate and parse input
    age_value = int(age_value) if age_value and age_value.isdigit() else None
    minutes = int(minutes) if minutes and minutes.isdigit() else 0

    if not role or not tier:
        return jsonify([])

    # Get the position array for the selected role
    positions = role_position_mapping.get(role)
    if not positions:
        return jsonify([])  # Return empty response if the role is invalid

    # Columns common to all roles
    base_columns = ["Player", "Team", "Age", "Minutes played", "Position"]

    # Get the role-specific columns
    role_columns = role_column_mapping.get(role, [])

    # Combine base columns and role-specific columns
    columns_to_select = base_columns + role_columns

    # Build the SQL query dynamically
    query = f"""
        SELECT {', '.join(f'"{col}"' for col in columns_to_select)}
        FROM players
        WHERE "Position" && %s AND "Tier" = ARRAY[%s]
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

     # Log the query and parameters for debugging
    print("Executing query:", query)
    print("With parameters:", (positions, tier))

    # Execute query
    try:
        cursor.execute(query,(positions, tier))
        players = cursor.fetchall()
    except Exception as e:
        print(f"Database error: {e}")
        return jsonify({"error": "Database query failed"})

    if not players:
        return jsonify({"players": [], "count": 0})
    
    # Convert to DataFrame
    df = pd.DataFrame(players, columns=columns_to_select)

    # Calculate percentiles and total scores for role-specific columns
    for metric in role_columns:
        valid_values = df[metric].dropna()
        df[f"{metric} Percentile"] = df[metric].apply(lambda x: int(percentileofscore(valid_values, x)) if pd.notna(x) else None)
    
    # Fill NaN values with 0 in all percentile columns before summing them
    percentile_columns = [f"{metric} Percentile" for metric in role_columns]
    df[percentile_columns] = df[percentile_columns].fillna(0)

    # Calculate the total score by summing the percentiles
    df['Total Score'] = df[percentile_columns].sum(axis=1)

    # Get the top 5 players including ties for 5th place
    df = df.sort_values(by='Total Score', ascending=False)
    top_5_threshold = df['Total Score'].iloc[4] if len(df) > 4 else df['Total Score'].iloc[-1]
    shortlist = df[df['Total Score'] >= top_5_threshold]

    # Prepare shortlist data for response
    shortlist_data = shortlist.to_dict(orient='records')
    print(shortlist_data)
    return jsonify({"shortlist_data": shortlist_data, "columns": columns_to_select, "count": len(df)})



# Route to serve the player profile page
@app.route('/player-profile.html')
def player_profile():
    player_name = request.args.get('player')  # Get player from URL query param
    compare_to = request.args.get('compareTo') 
    role = request.args.get('role')
    tier = request.args.get('tier')
    # Get the position array for the selected role
    positions = role_position_mapping.get(role)

    # Build the SQL query dynamically
    query = f"""
        SELECT "Player"
        FROM players
        WHERE "Position" && %s AND "Tier" = ARRAY[%s]
    """

    # Fetch all player names for the comparison dropdown
    cursor.execute(query,(positions, tier))
    players = cursor.fetchall()  
    # Render the page with player data and list of players for comparison
    return render_template('player-profile.html', player_name=player_name, players=players)

# Pizza chart code 
def create_pizza_chart(name, percentiles, role_columns, chart_name="pizza_chart"):
    
    print(percentiles) #Debug
    params = role_columns
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
def calculate_percentiles(player_data, role_columns, cursor, positions, tier):
    percentiles = {}
    
    for column in role_columns:
        # Build the SQL query dynamically
        query = f"""
            SELECT "{column}"
            FROM players
            WHERE "Position" && %s AND "Tier" = ARRAY[%s] AND "{column}" IS NOT NULL
        """
        print(f'Position: {positions}')
        print(f'Tier: {tier}')
        print(f"Query: {query}")
        # Fetch all player names for the comparison dropdown
        # Fetch all values for the metric from the database
        cursor.execute(query,(positions, tier))
        all_values = [row[0] for row in cursor.fetchall()]  # Flatten results 
        
        # Calculate percentile for the player's value
        player_value = player_data[column]
        if player_value is not None:
            percentiles[column] = int(percentileofscore(all_values, player_value))
        else:
            percentiles[column] = None  # Handle missing values

    return percentiles

@app.route('/player-profile-data', methods=['GET'])
def get_player_profile_data():
    player_name = request.args.get('player')
    compare_to = request.args.get('compareTo')
    role = request.args.get('role')
    tier= request.args.get('tier')

    # Get the position array for the selected role
    positions = role_position_mapping.get(role)
    print(f"Positions: {positions}")
    # Get the role-specific columns
    role_columns = role_column_mapping.get(role, [])

    # Fetch metrics for the selected player
    column_names = ', '.join(f'"{col}"' for col in role_columns)
    query = f'SELECT {column_names} FROM players WHERE "Player" = %s'
    cursor.execute(query, (player_name,))
    player_data_raw = cursor.fetchone()
    player_data = dict(zip(role_columns, player_data_raw))  # Create dictionary of player data
    print(f"Player data: {player_data}")
    # Fetch metrics for the comparison player
    cursor.execute(query, (compare_to,))
    compare_data_raw = cursor.fetchone()
    compare_data = dict(zip(role_columns, compare_data_raw))  # Create dictionary of comparison player data

    # Compute percentiles
    player_percentiles = calculate_percentiles(player_data, role_columns, cursor, positions, tier)
    print(f"Player percentiles: {player_percentiles}")
    compare_percentiles = calculate_percentiles(compare_data, role_columns, cursor, positions, tier)
    print(f"Comparison Player percentiles: {compare_percentiles}")

    player_pizza_chart = create_pizza_chart(player_name,player_percentiles,role_columns,chart_name=f"{player_name}")
    compare_pizza_chart = create_pizza_chart(compare_to,compare_percentiles,role_columns,chart_name=f"{compare_to}")

    # Generate the URLs dynamically using url_for
    player_pizza_chart_url = url_for('static', filename=player_pizza_chart)
    compare_pizza_chart_url = url_for('static', filename=compare_pizza_chart)  #f'pizza_charts/{compare_to}.png'

    # Log to verify the data being sent
    print("Sending data:", player_pizza_chart_url, compare_pizza_chart_url)

    return jsonify({
        'playerPizzaChart': player_pizza_chart_url,
        'comparePizzaChart': compare_pizza_chart_url
    })
    
def get_players_by_role(positions,tier):
    # Retrieve all player names for dropdown 

    # Build the SQL query dynamically
    query = f"""
        SELECT "Player"
        FROM players
        WHERE "Position" && %s AND "Tier" = ARRAY[%s]
    """
    print(f"get_players_by_role query: {query}") #debug
    # Fetch all player names for the comparison dropdown
    cursor.execute(query,(positions, tier))
    player_names = [row[0] for row in cursor.fetchall()]  # Extract player names from tuples
    return player_names

@app.route('/compare.html', methods=['GET'])
def create_bar_chart():
    print("create_bar_chart() called")  # Add this for debugging
    # Get the selected player and the shortlist from the form
    compare_to = request.args.get('compareTo','')
    shortlist_param = request.args.get('shortlist', '')
    role = request.args.get('role')
    tier = request.args.get('tier')
    # Decode the URL-encoded shortlist parameter and split it into individual player names
    shortlist = unquote(shortlist_param).split(',') if shortlist_param else []
    #shortlist = shortlist_param.split(',') if shortlist_param else []

    print(f"Shortlist: {shortlist}")

    # Get the position array for the selected role
    positions = role_position_mapping.get(role)
    print(f"Positions: {positions}")
    # Get the role-specific columns
    role_columns = role_column_mapping.get(role, [])

    # Retrieve all player names for dropdown 
    player_names = get_players_by_role(positions,tier)
    print(f"Player names: {player_names}") #debug

    # Combine the selected player with the shortlist, removing any empty values
    players_to_compare = [p for p in ([compare_to] + shortlist) if p]  # Filter out empty strings
    print(f"Players to compare: {players_to_compare}")

    # Check if players_to_compare is empty to avoid constructing an invalid query
    if not players_to_compare:
        print("No players to compare. Exiting.")
        return jsonify({"error": "No players to compare"}), 400
    
    #players_to_compare = [compare_to] + shortlist

    # Dynamically generate placeholders for the IN clause
    in_placeholders = ', '.join(['%s'] * len(players_to_compare))

    # Build the SQL query dynamically
    query = f"""
        SELECT "Player", {', '.join(f'"{col}"' for col in role_columns)}
        FROM players
        WHERE "Position" && %s 
          AND "Tier" = ARRAY[%s] 
          AND "Player" IN ({in_placeholders})
    """
    print(f"Query: {query}")  # Debugging
    # Execute the query with the correct parameters
    cursor.execute(query, (positions, tier, *players_to_compare))
    players_data = cursor.fetchall()
    print(f"Results: {players_data}")  # Debugging

    # Convert the data into a dictionary for easier processing
    players_dict = {player[0]: player[1:] for player in players_data}

    # Mapping for shorter metric labels
    metric_labels = get_metric_labels_for_role(role_columns)

    # Fetch the data for percentile calculation for all players in the same role and tier
    # Build the SQL query dynamically
    query = f"""
        SELECT "Player", {', '.join(f'"{col}"' for col in role_columns)}
        FROM players
        WHERE "Position" && %s 
          AND "Tier" = ARRAY[%s]
    """
    print(f"Players Query for percentile calculation: {query}")  # Debugging
    cursor.execute(query, (positions, tier))
    all_players_data = cursor.fetchall()
    print(f"all_players_data: {all_players_data}") #Debugging
    # Convert to a DataFrame to calculate percentiles
    import pandas as pd
    df = pd.DataFrame(all_players_data, columns=["Player"] + role_columns)

    # Calculate percentiles for each metric
    for metric in role_columns:
        valid_values = df[metric].dropna()  # drop NaN
        df[f"{metric} Percentile"] = df[metric].apply(lambda x: int(percentileofscore(valid_values, x)) if pd.notna(x) else None)

    # Get the percentiles for the players we are comparing
    players_percentiles = {}
    for player in players_to_compare:
        player_data = df[df['Player'] == player]
        players_percentiles[player] = {metric: player_data[f"{metric} Percentile"].values[0] for metric in role_columns}

    # Generate the bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    # Bar chart for percentiles
    bar_width = 0.10
    positions = range(len(role_columns))

    for i, player in enumerate(players_to_compare):
        ax.bar([p + i * bar_width for p in positions], 
               [players_percentiles[player].get(metric, 0) for metric in role_columns],
               width=bar_width, label=player)

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Percentile')
    ax.set_title('Player Comparison by Percentile')

    # Set the x-ticks to the shorter metric labels
    ax.set_xticks([p + bar_width * (len(players_to_compare) - 1) / 2 for p in positions])
    ax.set_xticklabels([metric_labels[metric] for metric in role_columns])  # Use shorter labels

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