# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 18:48:46 2024

@author: dforc
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.colors import qualitative
import os
import pickle
import ast
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO
from PIL import Image
import base64
import networkx as nx


## Download VADER
nltk.download('vader_lexicon')

###################
## Core Functions
###################
LOGO_PATH = "assets/P_logo_white.png"



###################
## Load Data
###################
@st.cache_data
def load_data(csv_path):
    """
    Load the labeled data from a CSV file.
    This function is cached to prevent reloading data unnecessarily.
    """
    if not os.path.exists(csv_path):
        st.error(f"The file {csv_path} does not exist.")
        return None
    df = pd.read_csv(csv_path, low_memory = False)
    return df


###################
## Compute Sentiment
###################
@st.cache_data
def compute_sentiment(df, text_column='combined_text'):
    """
    Compute sentiment scores using VADER and add them to the DataFrame.
    This function is cached to avoid recomputing sentiments unless the data changes.
    """
    sia = SentimentIntensityAnalyzer()
    sentiments = df[text_column].astype(str).apply(lambda x: sia.polarity_scores(x)['compound'])
    df = df.copy()  ## avoid SettingWithCopyWarning
    df['sentiment'] = sentiments
    return df


###################
## Normalize Topics
###################
def normalize_topic(topic):
    """
    Normalize topic strings to ensure consistency.
    """
    if isinstance(topic, str):
        topic = topic.strip().lower()
        topic = topic.replace("\\/", "/")
    return topic



###################
## Preprocess Topics
###################
def preprocess_topics(df, mlb=None):
    """
    Convert assigned_label from string representations of tuples to lists and normalize topics.
    Standardizes topic labels to match the format used in preprocessing.
    """
    if 'assigned_label' in df.columns and df['assigned_label'].dtype == object:
        # Safely evaluate the string to convert it into a list
        df = df.copy()  # Avoid SettingWithCopyWarning
        df['assigned_label'] = df['assigned_label'].apply(
            lambda x: [normalize_topic(topic) for topic in ast.literal_eval(x)] if isinstance(x, str) else []
        )

    ## Multi-label binarization logic
    if mlb is not None:
        df['assigned_label_binarized'] = mlb.transform(df['assigned_label'])
    
    return df
    
    ## Normalize topics
    df['assigned_label'] = df['assigned_label'].apply(lambda topics: [normalize_topic(t) for t in topics])
    return df



###################
## Load mlbbinarizer
###################
@st.cache_resource
def load_or_refit_mlbbinarizer(df, mlb_path='./artifacts/mlb_multi_label.pkl'):
    """
    Load the MultiLabelBinarizer from a pickle file.
    If the file does not exist, fit a new one and save it.
    This function uses cache_resource since the model is a resource.
    """
    if os.path.exists(mlb_path):
        with open(mlb_path, 'rb') as f:
            mlb = pickle.load(f)
    else:
        mlb = MultiLabelBinarizer()
        mlb.fit(df['assigned_label'])
        os.makedirs(os.path.dirname(mlb_path), exist_ok=True)
        with open(mlb_path, 'wb') as f:
            pickle.dump(mlb, f)
        st.success("MultiLabelBinarizer has been fitted and saved.")
    return mlb



###################
## Calculate Topic Distributions
###################
def calculate_topic_distribution(df, mlb, state=None):
    """
    Calculate the distribution of topics either nationally or for a specific state.
    """
    if state:
        subset = df[df['state'] == state]
    else:
        subset = df

    topic_counts = subset['assigned_label'].explode().value_counts()
    total = topic_counts.sum()
    distribution = (topic_counts / total).reindex(mlb.classes_, fill_value=0)
    return distribution



###################
## Get Topic Colors
###################
def get_topic_colors(topics):
    """
    Assign consistent colors to each topic using Plotly's qualitative color palette.
    """
    color_palette = qualitative.Plotly
    colors = [color_palette[i % len(color_palette)] for i in range(len(topics))]
    color_map = dict(zip(topics, colors))
    return color_map




###################
## Sankey Create
###################
def create_ordered_sankey(national_dist, state_dist, topics, selected_state):
    """
    Create an ordered Sankey diagram with consistent colors for each topic.
    Topics are ordered based on national distribution.
    """
    ## Sort topics by national distribution
    sorted_topics = national_dist.sort_values(ascending=False).index.tolist()
    
    ## Get colors for each topic
    color_map = get_topic_colors(sorted_topics)
    
    ## Nodes
    labels = ['National'] + sorted_topics + [selected_state]
    node_colors = ['lightgrey'] + [color_map[topic] for topic in sorted_topics] + ['lightgrey']
    
    ## Indices
    national_idx = 0
    topic_indices_national = list(range(1, len(sorted_topics)+1))
    state_idx = len(sorted_topics) + 1

    ## Sources: National to Topics, Topics to State
    sources = []
    targets = []
    values = []
    link_colors = []
    hover_text = []

    ## National to Topics
    for i, topic in enumerate(sorted_topics):
        sources.append(national_idx)
        targets.append(topic_indices_national[i])
        values.append(national_dist[topic])
        link_colors.append(color_map[topic])  # Use topic color for links
        hover_text.append(f"National to {topic.capitalize()}: {national_dist[topic]:.2%}")

    ## Topics to State
    for i, topic in enumerate(sorted_topics):
        sources.append(topic_indices_national[i])
        targets.append(state_idx)
        values.append(state_dist[topic])
        link_colors.append(color_map[topic])  # Use topic color for links
        hover_text.append(f"{topic.capitalize()} to {selected_state}: {state_dist[topic]:.2%}")

    sankey = go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            customdata=hover_text,
            hovertemplate='%{customdata}<extra></extra>'
        )
    )

    fig = go.Figure(sankey)
    fig.update_layout(title_text="Ordered Sankey Diagram of National and Selected State Topics", font_size=10)
    return fig



###################
## Calcualte Prevalence
###################
@st.cache_resource
def calculate_prevalence(national_dist, state_dist):
    """
    Calculate which topics are more prevalent in the state compared to national.
    Returns a DataFrame with prevalence metrics.
    """
    prevalence = pd.DataFrame({
        'Topic': national_dist.index,
        'National_Proportion': national_dist.values,
        'State_Proportion': state_dist.values
    })
    prevalence['Difference'] = prevalence['State_Proportion'] - prevalence['National_Proportion']
    prevalence['More_Prevalent'] = prevalence['Difference'] > 0
    prevalence = prevalence.sort_values(by='Difference', ascending=False)
    return prevalence




###################
## Plot Side by Sides
###################
def plot_side_by_side_bars(national_dist, state_dist, selected_topics=None):
    """
    Create side-by-side bar charts to compare national and state topic distributions.
    If selected_topics is None, display all topics.
    """
    if selected_topics:
        topics_to_plot = selected_topics
    else:
        topics_to_plot = national_dist.index.tolist()
    
    df_plot = pd.DataFrame({
        'Topic': topics_to_plot,
        'National': national_dist[topics_to_plot].values,
        'State': state_dist[topics_to_plot].values
    })

    fig = px.bar(
        df_plot,
        x='Topic',
        y=['National', 'State'],
        barmode='group',
        title="National vs State Topic Distribution",
        labels={'value': 'Proportion', 'Topic': 'Topic'},
        color_discrete_sequence=['#636EFA', '#EF553B']
    )
    return fig



###################
## Plot Heatmap
###################
def plot_heatmap(prevalence_df):
    """
    Create a heatmap to display the magnitude of differences between state and national topic proportions.
    """
    fig = px.imshow(
        prevalence_df[['Difference']].T,
        labels=dict(x="Topic", y="Metric", color="Difference"),
        x=prevalence_df['Topic'],
        y=['Difference'],
        color_continuous_scale='RdBu',
        title="Heatmap of Topic Differences (State - National)"
    )
    return fig




###################
## Plot Sentiment Boxplot
###################
def plot_sentiment_box(national_sentiment, state_sentiment, selected_state):
    """
    Create box plots to compare sentiment score distributions between national and selected state.
    """
    df_sentiment = pd.DataFrame({
        'Sentiment': np.concatenate([national_sentiment, state_sentiment]),
        'Category': ['National'] * len(national_sentiment) + [selected_state] * len(state_sentiment)
    })
    
    fig = px.box(
        df_sentiment,
        x='Category',
        y='Sentiment',
        title=f'Sentiment Score Distribution Across All Topics for {selected_state}',
        labels={'Sentiment': 'Sentiment Score'},
        color='Category',
        color_discrete_sequence=['#636EFA', '#EF553B']
    )
    return fig



###################
## Calculate Sentiment per topic
###################
def calculate_sentiment_per_topic(df, mlb, state=None):
    """
    Calculate average sentiment scores per topic.
    """
    if state:
        subset = df[df['state'] == state]
    else:
        subset = df
    
    ## Expand the topics
    subset_expanded = subset.explode('assigned_label')
    
    ## Group by topic and calculate mean sentiment
    sentiment_per_topic = subset_expanded.groupby('assigned_label')['sentiment'].mean().reindex(mlb.classes_, fill_value=0)
    
    return sentiment_per_topic




###################
## Plot Sentiment Bar
###################
def plot_sentiment_bar(national_sentiment, state_sentiment, topics):
    """
    Plot a bar chart comparing average sentiment scores between national and state.
    """
    df_plot = pd.DataFrame({
        'Topic': topics,
        'National': national_sentiment.values,
        'State': state_sentiment.values
    })
    
    df_melt = df_plot.melt(id_vars='Topic', value_vars=['National', 'State'], var_name='Category', value_name='Average Sentiment')

    fig = px.bar(
        df_melt,
        x='Topic',
        y='Average Sentiment',
        color='Category',
        barmode='group',
        title="Average Sentiment Score for selected state per Topic compared to National Avg.",
        labels={'Average Sentiment': 'Average Sentiment Score', 'Topic': 'Topic'},
        color_discrete_sequence=['#636EFA', '#EF553B']
    )
    return fig




###################
## Calculate Sentiment Prevalence
###################
def calculate_sentiment_prevalence(national_sentiment, state_sentiment):
    """
    Calculate the difference in average sentiment between state and national for each topic.
    
    Args:
        national_sentiment (pd.Series): Average sentiment per topic nationally.
        state_sentiment (pd.Series): Average sentiment per topic for the selected state.
    
    Returns:
        pd.DataFrame: DataFrame with sentiment differences.
    """
    difference = state_sentiment - national_sentiment
    sentiment_prevalence = pd.DataFrame({
        'Topic': national_sentiment.index,
        'National_Average_Sentiment': national_sentiment.values,
        'State_Average_Sentiment': state_sentiment.values,
        'Difference': difference.values
    })
    sentiment_prevalence['More_Positive'] = sentiment_prevalence['Difference'] > 0
    sentiment_prevalence = sentiment_prevalence.sort_values(by='Difference', ascending=False)
    return sentiment_prevalence




###################
## Calculate Topic Sent Correlation
###################
def calculate_topic_sentiment_correlation(df, mlb):
    """
    Calculate correlation between each topic and sentiment scores.
    
    Args:
        df (pd.DataFrame): The DataFrame containing data.
        mlb (MultiLabelBinarizer): The fitted MultiLabelBinarizer.
    
    Returns:
        pd.Series: Correlation coefficients per topic.
    """
    ## Create binary matrix for topics
    topic_matrix = mlb.transform(df['assigned_label'])
    df_topics = pd.DataFrame(topic_matrix, columns=mlb.classes_, index=df.index)
    
    ## Calculate correlation with sentiment
    correlations = df_topics.corrwith(df['sentiment'])
    
    return correlations




###################
## Wordcloud Gen
###################

@st.cache_data
def generate_wordcloud(text, title=None, color='white'):
    """
    Generate a word cloud image from text.

    Args:
        text (str): The input text for the word cloud.
        title (str, optional): Title for the word cloud.
        color (str, optional): Background color for the word cloud.

    Returns:
        str: Base64-encoded image string.
    """
    if not text.strip():
        return None
    
    wordcloud = WordCloud(width=800, height=400, background_color=color, max_words=200).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    if title:
        plt.title(title, fontsize=20)
    plt.tight_layout(pad=0)
    
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    img_bytes = buf.read()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded



###################
## Get Topic Texts
###################
def get_text_for_topic(df, topic, state=None):
    """
    Aggregate text data for a given topic, optionally filtered by state.

    Args:
        df (pd.DataFrame): The DataFrame containing data.
        topic (str): The topic for which to aggregate text.
        state (str, optional): The state to filter by.

    Returns:
        str: Aggregated text.
    """
    if state:
        subset = df[(df['assigned_label'].apply(lambda topics: topic in topics)) & (df['state'] == state)]
    else:
        subset = df[df['assigned_label'].apply(lambda topics: topic in topics)]
        
    aggregated_text = ' '.join(subset['combined_text'].astype(str).tolist())
    return aggregated_text


###################
## Wordlcloud Print
###################
def display_wordcloud(image_encoded, caption):
    """
    Display a word cloud image from base64-encoded string with a caption.

    Args:
        image_encoded (str): Base64-encoded image string.
        caption (str): Caption to display below the image.
    """
    if image_encoded:
        st.markdown(f"**{caption}**")
        st.image(f"data:image/png;base64,{image_encoded}", use_container_width=True)
    else:
        st.markdown(f"**{caption}**")
        st.write("No data available to generate word cloud.")


###################
## Clustering Function
####################

def cluster_states(df, mlb, num_clusters=5):
    """
    Cluster states based on their topic distributions.

    Args:
        df (pd.DataFrame): The DataFrame containing data.
        mlb (MultiLabelBinarizer): The fitted MultiLabelBinarizer.
        num_clusters (int): Number of clusters.

    Returns:
        pd.DataFrame: DataFrame with states and their cluster assignments.
    """
    ## Calculate topic distributions per state
    state_topic_dist = df.groupby('state')['assigned_label'].apply(lambda topics: mlb.transform(topics).mean(axis=0))
    state_topic_dist_df = pd.DataFrame(state_topic_dist.tolist(), index=state_topic_dist.index, columns=mlb.classes_)
    
    ## Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(state_topic_dist_df)
    
    ## Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=99)
    clusters = kmeans.fit_predict(scaled_data)
    
    ## Create a DataFrame with cluster assignments
    cluster_df = pd.DataFrame({
        'State': state_topic_dist_df.index,
        'Cluster': clusters
    })
    
    return cluster_df




###################
##  State Clusters
################### 
def plot_state_clusters(cluster_df):
    """
    Plot a scatter map with states colored by their cluster assignments.

    Args:
        cluster_df (pd.DataFrame): DataFrame with states and cluster labels.

    Returns:
        Plotly Figure: Scatter map.
    """
    # State abbreviations and coordinates
    state_coords = {
        'AL': {'lat': 32.806671, 'lon': -86.791130},
        'AK': {'lat': 61.370716, 'lon': -152.404419},
        'AZ': {'lat': 33.729759, 'lon': -111.431221},
        'AR': {'lat': 34.969704, 'lon': -92.373123},
        'CA': {'lat': 36.116203, 'lon': -119.681564},
        'CO': {'lat': 39.059811, 'lon': -105.311104},
        'CT': {'lat': 41.597782, 'lon': -72.755371},
        'DE': {'lat': 39.318523, 'lon': -75.507141},
        'FL': {'lat': 27.766279, 'lon': -81.686783},
        'GA': {'lat': 33.040619, 'lon': -83.643074},
        'HI': {'lat': 21.094318, 'lon': -157.498337},
        'ID': {'lat': 44.240459, 'lon': -114.478828},
        'IL': {'lat': 40.349457, 'lon': -88.986137},
        'IN': {'lat': 39.849426, 'lon': -86.258278},
        'IA': {'lat': 42.011539, 'lon': -93.210526},
        'KS': {'lat': 38.526600, 'lon': -96.726486},
        'KY': {'lat': 37.668140, 'lon': -84.670067},
        'LA': {'lat': 31.169546, 'lon': -91.867805},
        'ME': {'lat': 44.693947, 'lon': -69.381927},
        'MD': {'lat': 39.063946, 'lon': -76.802101},
        'MA': {'lat': 42.230171, 'lon': -71.530106},
        'MI': {'lat': 43.326618, 'lon': -84.536095},
        'MN': {'lat': 45.694454, 'lon': -93.900192},
        'MS': {'lat': 32.741646, 'lon': -89.678696},
        'MO': {'lat': 38.456085, 'lon': -92.288368},
        'MT': {'lat': 46.921925, 'lon': -110.454353},
        'NE': {'lat': 41.125370, 'lon': -98.268082},
        'NV': {'lat': 38.313515, 'lon': -117.055374},
        'NH': {'lat': 43.452492, 'lon': -71.563896},
        'NJ': {'lat': 40.298904, 'lon': -74.521011},
        'NM': {'lat': 34.840515, 'lon': -106.248482},
        'NY': {'lat': 42.165726, 'lon': -74.948051},
        'NC': {'lat': 35.630066, 'lon': -79.806419},
        'ND': {'lat': 47.528912, 'lon': -99.784012},
        'OH': {'lat': 40.388783, 'lon': -82.764915},
        'OK': {'lat': 35.565342, 'lon': -96.928917},
        'OR': {'lat': 44.572021, 'lon': -122.070938},
        'PA': {'lat': 40.590752, 'lon': -77.209755},
        'RI': {'lat': 41.680893, 'lon': -71.511780},
        'SC': {'lat': 33.856892, 'lon': -80.945007},
        'SD': {'lat': 44.299782, 'lon': -99.438828},
        'TN': {'lat': 35.747845, 'lon': -86.692345},
        'TX': {'lat': 31.054487, 'lon': -97.563461},
        'UT': {'lat': 40.150032, 'lon': -111.862434},
        'VT': {'lat': 44.045876, 'lon': -72.710686},
        'VA': {'lat': 37.769337, 'lon': -78.169968},
        'WA': {'lat': 47.400902, 'lon': -121.490494},
        'WV': {'lat': 38.491226, 'lon': -80.954453},
        'WI': {'lat': 44.268543, 'lon': -89.616508},
        'WY': {'lat': 42.755966, 'lon': -107.302490}
    }
    
    ## Add state abbreviations and coordinates
    cluster_df['State_Abbrev'] = cluster_df['State'].map(lambda x: x.upper())
    cluster_df['lat'] = cluster_df['State_Abbrev'].map(lambda x: state_coords.get(x, {}).get('lat', None))
    cluster_df['lon'] = cluster_df['State_Abbrev'].map(lambda x: state_coords.get(x, {}).get('lon', None))
    
    ## Drop states with missing coordinates
    cluster_df = cluster_df.dropna(subset=['lat', 'lon'])
    
    ## Plot the map
    fig = px.scatter_geo(
        cluster_df,
        lat='lat',
        lon='lon',
        color='Cluster',
        hover_name='State',
        size_max=15,
        title='State Clusters Based on Topic Distribution',
        color_continuous_scale=px.colors.qualitative.Plotly,
        scope='usa'  ## Set scope to USA
    )
    return fig




###################
## Calc Cooccurrence Matrix
###################
def calculate_cooccurrence_matrix(df, mlb):
    """
    Calculate the co-occurrence matrix for topics.

    Args:
        df (pd.DataFrame): The DataFrame containing data.
        mlb (MultiLabelBinarizer): The fitted MultiLabelBinarizer.

    Returns:
        pd.DataFrame: Co-occurrence matrix.
    """
    topic_matrix = mlb.transform(df['assigned_label'])
    cooccurrence = pd.DataFrame(topic_matrix, columns=mlb.classes_).T.dot(pd.DataFrame(topic_matrix, columns=mlb.classes_))
    
    ## Zero out the diagonal
    np.fill_diagonal(cooccurrence.values, 0)
    return cooccurrence



###################
## Plot Cooccurrence
###################
def plot_cooccurrence_network(cooccurrence_df, threshold=10):
    """
    Plot a network graph of topic co-occurrences.

    Args:
        cooccurrence_df (pd.DataFrame): Co-occurrence matrix.
        threshold (int): Minimum number of co-occurrences to display an edge.

    Returns:
        Plotly Figure: Network graph.
    """
    ## Create graph
    G = nx.Graph()

    ## Add nodes
    for topic in cooccurrence_df.index:
        G.add_node(topic)

    # Add edges with weight above threshold
    for topic1 in cooccurrence_df.index:
        for topic2 in cooccurrence_df.columns:
            if cooccurrence_df.loc[topic1, topic2] >= threshold:
                G.add_edge(topic1, topic2, weight=cooccurrence_df.loc[topic1, topic2])

    if G.number_of_edges() == 0:
        st.write("No co-occurring topics meet the threshold.")
        return None

    ## Get positions
    pos = nx.spring_layout(G, k=0.5, iterations=100)

    ## Extract edge attributes
    edge_x = []
    edge_y = []
    edge_weights = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_weights.append(edge[2]['weight'])

    ## Create edge trace
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    ## Create node trace
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color='blue',
            size=10,
            line_width=2
        ),
        text=list(G.nodes()),
        textposition="top center"
    )

    ## Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Topic Co-occurrence Network',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper") ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )

    return fig





###################
## Calculate Sentiment Heatmap
###################
def calculate_sentiment_heatmap(df, mlb):
    """
    Calculate average sentiment scores for each topic in each state.

    Args:
        df (pd.DataFrame): The DataFrame containing data.
        mlb (MultiLabelBinarizer): The fitted MultiLabelBinarizer.

    Returns:
        pd.DataFrame: DataFrame with states as rows, topics as columns, and average sentiment scores.
    """
    ## Expand the topics
    subset_expanded = df.explode('assigned_label')
    
    ## Group by state and topic, then calculate mean sentiment
    sentiment_heatmap = subset_expanded.groupby(['state', 'assigned_label'])['sentiment'].mean().unstack(fill_value=0)
    
    return sentiment_heatmap




###################
## Plot SEnt Heatmap
###################
def plot_sentiment_heatmap(sentiment_heatmap_df):
    """
    Plot a heatmap of sentiment scores across states and topics.

    Args:
        sentiment_heatmap_df (pd.DataFrame): DataFrame with sentiment scores.

    Returns:
        Plotly Figure: Heatmap.
    """
    fig = px.imshow(
        sentiment_heatmap_df,
        labels=dict(x="Topic", y="State", color="Average Sentiment"),
        x=sentiment_heatmap_df.columns,
        y=sentiment_heatmap_df.index,
        color_continuous_scale='RdBu',
        title="Sentiment Scores Across Topics and States",
        width=1000, 
        height=800 
    )
    return fig



###################
## Plot SEnt Choro
###################
def plot_sentiment_choropleth(sentiment_per_state, topic):
    """
    Plot a choropleth map for sentiment scores of a specific topic across states.

    Args:
        sentiment_per_state (pd.Series): Average sentiment per state.
        topic (str): The topic being plotted.

    Returns:
        Plotly Figure: Choropleth map.
    """
    df_sentiment = sentiment_per_state.reset_index()
    df_sentiment.columns = ['state', 'sentiment']
    
    fig = px.choropleth(
        df_sentiment,
        locations='state',
        locationmode="USA-states",
        color='sentiment',
        scope="usa",
        color_continuous_scale='RdBu',
        labels={'sentiment': f"Average Sentiment for '{topic.capitalize()}'"},
        title=f"Sentiment Choropleth Map of '{topic.capitalize()}' Across States"
    )
    
    ## Data check
    if df_sentiment['sentiment'].sum() == 0:
        st.write(f"No sentiment data available for the topic '{topic}'.")
        return None
    else:
        return fig



###################
## Calc Topic Setn Correlation
###################
def calculate_topic_sentiment_correlation(df, mlb):
    """
    Calculate correlation between each topic and sentiment scores.

    Args:
        df (pd.DataFrame): The DataFrame containing data.
        mlb (MultiLabelBinarizer): The fitted MultiLabelBinarizer.

    Returns:
        pd.Series: Correlation coefficients per topic.
    """
    ## Create binary matrix for topics
    topic_matrix = mlb.transform(df['assigned_label'])
    df_topics = pd.DataFrame(topic_matrix, columns=mlb.classes_, index=df.index)
    
    ## Calculate correlation with sentiment
    correlations = df_topics.corrwith(df['sentiment'])
    
    return correlations



###################
## Plot Corr Heatmap
###################
def plot_correlation_heatmap(correlations):
    """
    Plot a heatmap of correlations between topics and sentiment.

    Args:
        correlations (pd.Series): Correlation coefficients.

    Returns:
        Plotly Figure: Heatmap.
    """
    df_corr = correlations.reset_index()
    df_corr.columns = ['Topic', 'Correlation with Sentiment']
    
    fig = px.imshow(
        [df_corr['Correlation with Sentiment']],
        labels=dict(x="Topic", y="", color="Correlation"),
        x=df_corr['Topic'],
        y=['Correlation'],
        color_continuous_scale='RdBu',
        title="Correlation Between Topics and Sentiment"
    )
    return fig


###################
##   State name to Abbreviation
###################
@st.cache_data
def get_state_abbrev():
    """
    Returns a dictionary mapping full state names (CamelCase, no spaces) to their abbreviations.
    """
    return {
        'Alabama': 'AL',
        'Alaska': 'AK',
        'Arizona': 'AZ',
        'Arkansas': 'AR',
        'California': 'CA',
        'Colorado': 'CO',
        'Connecticut': 'CT',
        'Delaware': 'DE',
        'Florida': 'FL',
        'Georgia': 'GA',
        'Hawaii': 'HI',
        'Idaho': 'ID',
        'Illinois': 'IL',
        'Indiana': 'IN',
        'Iowa': 'IA',
        'Kansas': 'KS',
        'Kentucky': 'KY',
        'Louisiana': 'LA',
        'Maine': 'ME',
        'Maryland': 'MD',
        'Massachusetts': 'MA',
        'Michigan': 'MI',
        'Minnesota': 'MN',
        'Mississippi': 'MS',
        'Missouri': 'MO',
        'Montana': 'MT',
        'Nebraska': 'NE',
        'Nevada': 'NV',
        'NewHampshire': 'NH',
        'NewJersey': 'NJ',
        'NewMexico': 'NM',
        'NewYork': 'NY',
        'NorthCarolina': 'NC',
        'NorthDakota': 'ND',
        'Ohio': 'OH',
        'Oklahoma': 'OK',
        'Oregon': 'OR',
        'Pennsylvania': 'PA',
        'RhodeIsland': 'RI',
        'SouthCarolina': 'SC',
        'SouthDakota': 'SD',
        'Tennessee': 'TN',
        'Texas': 'TX',
        'Utah': 'UT',
        'Vermont': 'VT',
        'Virginia': 'VA',
        'Washington': 'WA',
        'WestVirginia': 'WV',
        'Wisconsin': 'WI',
        'Wyoming': 'WY'
    }

##############################################################
# ------------------- Streamlit Dashboard -------------------#
##############################################################

def main():
    

    
    
    st.set_page_config(page_title="State-wise Topic and Sentiment Analysis", 
                       layout="wide",
                       page_icon = LOGO_PATH)
    
    st.title("State-wise Topic and Sentiment Analysis Dashboard v.3")
    
    # ---------- sidebar branding ----------
    st.sidebar.image(LOGO_PATH, use_column_width=True)
    st.sidebar.markdown("---")  # thin divider under the logo
    # --------------------------------------

    ## Sidebar Configuration
    st.sidebar.header("Configuration")
    
    ## Dropdown to select which dataset to use
    dataset_option = st.sidebar.selectbox(
        "Select Dataset",
        options=["Manual Labeled Data", 
                 "Predicted Data", 
                 "Combined Data"],
        index=2  # Default to "Combined Data"
    )
    
    ##### Intro 
    st.markdown(
        """
        ### Dashboard Overview
    
        This dashboard explores the distribution of Reddit post political topics and sentiments across different U.S. states, providing insights into state-specific trends compared to national averages. <br>
        It is primarily being used as a model-performance evaluation tool for futher fine-tuning of the models.  
        
        
        #### Data and Methods:
        - **Topics and Labels:** Topics were mined from Congressional Bill Information, and our team manually labeled 2,500 posts with multi-topic classifications.  
        - **Data Collection:** The dataset includes the top 600 posts from each state's subreddit over the past year, alongside all comments. This results in a comprehensive dataset of 30,000 posts and approximately 4.5 million comments.  
        - **Modeling Approach:** A custom **RoBERTa-based classification ensemble transformer** was developed and trained using 5-fold cross-validation on the manually labeled dataset to predict topics for the remaining data.  
        - Currently these are the results of the non-fusion model.  The Fusion Model is still being developed and tested.   
    
        #### Model Performance:
        The model achieved the following performance metrics on the test set, and had constant performance through all 5 folds in validation:
        - **Micro F1 Score:** 0.7359  
        - **Macro F1 Score:** 0.7042  
        - **Weighted F1 Score:** 0.7359  
        - **Hamming Loss:** 0.0637  
        - **Jaccard Index:** 0.6711  
    
        #### About:
        This interactive dashboard enables users to:
        - Compare topic prevalence and sentiment between states and national averages.
        - Explore state-specific distributions and trends in political discourse.
        - Visualize co-occurrence networks, state clusters, and sentiment variations across topics and regions.
        - Compare the training dataset to the predictions.  Note: The 4.5 Million Comments set is not availalbe in this dashboard due to hosting limitations.
        
    
        Use the sidebar to configure data and visualizations for your analysis.
        """,
        unsafe_allow_html=True
    )
    
    
    
    ## Map dataset option to corresponding CSV file
    dataset_map = {
        "Manual Labeled Data": "dashboard_data/manual_labeled_data.csv",
        "Predicted Data": ".dashboard_data/predicted_data.csv",
        "Combined Data": "dashboard_data/combined_data.csv"
    }
    
    ## Get the selected dataset path
    csv_path = dataset_map[dataset_option]
    
    sentiment_option = st.sidebar.checkbox("Compute Sentiment Scores", value=True)

    ## Load Data
    df = load_data(csv_path)
    if df is None:
        st.stop()

    ## Check for required columns
    required_columns = ['assigned_label', 'state', 'combined_text']
    for col in required_columns:
        if col not in df.columns:
            st.error(f"The DataFrame does not contain the required '{col}' column.")
            st.stop()



    ## Preprocess Topics BEFORE fitting MLB
    df = preprocess_topics(df, mlb=None)


    ## Load or Fit MultiLabelBinarizer AFTER preprocessing
    mlb = load_or_refit_mlbbinarizer(df)



    ## Display Data Sample
    st.subheader("Data Sample")
    st.write(df.head())


    ## Compute Sentiment if Enabled
    if sentiment_option and 'sentiment' not in df.columns:
        with st.spinner("Computing sentiment scores..."):
            df = compute_sentiment(df, text_column='combined_text')
        st.success("Sentiment scores computed.")



    ## Convert Full State Names to Abbreviations
    state_abbrev_dict = get_state_abbrev()
    if df['state'].dtype == object:
        
        df['state'] = df['state'].map(state_abbrev_dict)
        df = df.dropna(subset=['state'])


    ## Filter out entries without a valid state
    initial_count = len(df)
    df = df[df['state'].notna()]
    filtered_count = len(df)
    st.write(f"Filtered out {initial_count - filtered_count} entries without a valid state.")
    
    if filtered_count == 0:
        st.error("No data available after filtering out entries without a valid state.")
        st.stop()

    ## State Selection
    states = sorted(df['state'].unique())
    selected_state = st.sidebar.selectbox("Select a State", options=states)



    ## Calculate Distributions
    with st.spinner("Calculating topic distributions..."):
        national_dist = calculate_topic_distribution(df, mlb, state=None)
        state_dist = calculate_topic_distribution(df, mlb, state=selected_state)
    st.success("Topic distributions calculated.")



    ## Display Distributions
    st.subheader("National Topic Distribution")
    national_dist_df = national_dist.reset_index().rename(columns={'index': 'assigned_label', 0: 'count'})
    fig_national_dist = px.bar(
        national_dist_df,
        x='assigned_label',
        y='count',
        title="National Topic Distribution",
        labels={'assigned_label': 'Topic', 'count': 'Proportion'},
        width=1000,  
        height=400   
    )
    st.plotly_chart(fig_national_dist, use_container_width=True)
    
    st.subheader(f"{selected_state} Topic Distribution")
    state_dist_df = state_dist.reset_index().rename(columns={'index': 'assigned_label', 0: 'count'})
    fig_state_dist = px.bar(
        state_dist_df,
        x='assigned_label',
        y='count',
        title=f"{selected_state} Topic Distribution",
        labels={'assigned_label': 'Topic', 'count': 'Proportion'},
        width=1000,  
        height=400   
    )
    st.plotly_chart(fig_state_dist, use_container_width=True)



    ## Verify Distributions Sum to 1
    st.write(f"Sum of National Distribution: {national_dist.sum():.4f}")
    st.write(f"Sum of {selected_state} Distribution: {state_dist.sum():.4f}")

    if national_dist.sum() == 0 or state_dist.sum() == 0:
        st.error("One of the distributions sums to zero. Please check data processing steps.")
        st.stop()

    ## Create Ordered Sankey Diagram
    with st.spinner("Creating ordered Sankey diagram..."):
        fig = create_ordered_sankey(national_dist, state_dist, topics=mlb.classes_, selected_state=selected_state)
    st.plotly_chart(fig, use_container_width=True)



    ## Side-by-Side Bar Charts
    st.markdown("#### Proportions of the selected states topic distribution, compared with the National Averages")
    fig_bars_filtered = plot_side_by_side_bars(national_dist, state_dist, selected_topics=None)
    st.plotly_chart(fig_bars_filtered, use_container_width=True)



    ## Calculate Prevalence
    prevalence_df = calculate_prevalence(national_dist, state_dist)


    ## Heatmap
    with st.spinner("Creating heatmap of topic differences..."):
        fig_heatmap = plot_heatmap(prevalence_df)
    st.plotly_chart(fig_heatmap, use_container_width=True)


    ## Display Prevalence Table
    st.subheader(f"Topic Prevalence in {selected_state} Compared to National")
    st.markdown("Selected State Key Differences vs National Averages")
    st.dataframe(prevalence_df.style.highlight_max(subset=['Difference'], color='lightgreen').format({
        'National_Proportion': "{:.2%}",
        'State_Proportion': "{:.2%}",
        'Difference': "{:+.2%}"
    }))


############

    ## Highlight Top 5 More Prevalent Topics
    st.markdown(f"#### Top 5 More Prevalent Topics in {selected_state} Compared to National Avg.")
    top_prevalent = prevalence_df[prevalence_df['More_Prevalent']].head(5)
    if not top_prevalent.empty:
        st.write(top_prevalent[['Topic', 'Difference']].to_html(index=False, float_format="{:+.2%}".format), unsafe_allow_html=True)
    else:
        st.write("No topics are more prevalent in the selected state compared to national average.")


    ## Highlight Top 5 Less Prevalent Topics
    st.markdown(f"#### Top 5 Less Prevalent Topics in {selected_state} Compared To National Avg.")
    top_less_prevalent = prevalence_df[~prevalence_df['More_Prevalent']].head(5)
    if not top_less_prevalent.empty:
        st.write(top_less_prevalent[['Topic', 'Difference']].to_html(index=False, float_format="{:+.2%}".format), unsafe_allow_html=True)
    else:
        st.write("No topics are less prevalent in the selected state compared to national average.")


#############


    ## Sentiment Analysis
    if sentiment_option and 'sentiment' in df.columns:
        st.markdown("### Sentiment Analysis")
        with st.spinner("Calculating sentiment distributions..."):
            national_sentiment = df['sentiment']
            state_sentiment = df[df['state'] == selected_state]['sentiment']
        st.success("Sentiment distributions calculated.")


        ## Box Plot for Sentiment
        fig_sentiment_box = plot_sentiment_box(national_sentiment, state_sentiment, selected_state)
        st.plotly_chart(fig_sentiment_box, use_container_width=True)





        ## Sentiment by Topic Bar Chart
        with st.spinner("Calculating average sentiment by topic..."):
            sentiment_by_topic_national = calculate_sentiment_per_topic(df, mlb, state=None)
            sentiment_by_topic_state = calculate_sentiment_per_topic(df, mlb, state=selected_state)
        fig_sentiment_bar = plot_sentiment_bar(sentiment_by_topic_national, sentiment_by_topic_state, mlb.classes_)
        st.plotly_chart(fig_sentiment_bar, use_container_width=True)



        ## Sentiment Prevalence

        with st.spinner("Calculating sentiment prevalence..."):
            sentiment_prevalence_df = calculate_sentiment_prevalence(sentiment_by_topic_national, sentiment_by_topic_state)

        # Display Sentiment Prevalence Table
        st.subheader(f"Sentiment Prevalence in {selected_state} Compared to National")
        st.dataframe(sentiment_prevalence_df.style.highlight_max(subset=['Difference'], color='lightgreen').format({
            'National_Average_Sentiment': "{:.4f}",
            'State_Average_Sentiment': "{:.4f}",
            'Difference': "{:+.4f}"
        }))


        ## Highlight Top 5 More Positive Sentiment Topics
        st.markdown(f"#### Top 5 Topics with Higher Sentiment in {selected_state}")
        top_sentiment = sentiment_prevalence_df[sentiment_prevalence_df['Difference'] > 0].head(5)
        if not top_sentiment.empty:
            st.write(top_sentiment[['Topic', 'Difference']].to_html(index=False, float_format="{:+.4f}".format), unsafe_allow_html=True)
        else:
            st.write("No topics have higher sentiment in the selected state compared to national average.")



        ## Highlight Top 5 More Negative Sentiment Topics
        st.markdown(f"#### Top 5 Topics with Lower Sentiment in {selected_state}")
        bottom_sentiment = sentiment_prevalence_df[sentiment_prevalence_df['Difference'] < 0].tail(5)
        if not bottom_sentiment.empty:
            st.write(bottom_sentiment[['Topic', 'Difference']].to_html(index=False, float_format="{:+.4f}".format), unsafe_allow_html=True)
        else:
            st.write("No topics have lower sentiment in the selected state compared to national average.")

        st.markdown("# All States / National Section")

        ## Sentiment Heatmap Across Topics and States
        st.markdown("#### Sentiment Heatmap Across Topics and States")
        with st.spinner("Calculating sentiment heatmap..."):
            sentiment_heatmap_df = calculate_sentiment_heatmap(df, mlb)
        fig_sentiment_heatmap = plot_sentiment_heatmap(sentiment_heatmap_df)
        st.plotly_chart(fig_sentiment_heatmap, use_container_width=True)



        ## Correlation Between Topics and Sentiment
        st.markdown("#### Topic-Sentiment Correlation")
        with st.spinner("Calculating topic-sentiment correlations..."):
            correlations = calculate_topic_sentiment_correlation(df, mlb)
        fig_correlation = plot_correlation_heatmap(correlations)
        st.plotly_chart(fig_correlation, use_container_width=True)

    ## Geographical Choropleth Map
        st.markdown("#### Geographical Distribution of Topics")
        with st.spinner("Calculating geographical topic distribution..."):
            
                    
            ## Aggregate topic distribution per state
            state_topic_dist = df.groupby('state')['assigned_label'].apply(lambda topics: mlb.transform(topics).mean(axis=0))
            state_topic_dist_df = pd.DataFrame(state_topic_dist.tolist(), index=state_topic_dist.index, columns=mlb.classes_)
        selected_topic_map = st.selectbox("Select a Topic for Choropleth Map", options=mlb.classes_)
        if selected_topic_map:
            
            
            ## Ensure that 'state' column contains valid two-letter abbreviations
            fig_choropleth = px.choropleth(
                state_topic_dist_df.reset_index(),
                locations='state',
                locationmode="USA-states",
                color=selected_topic_map,
                scope="usa",
                color_continuous_scale="Viridis",
                labels={selected_topic_map: f"{selected_topic_map.capitalize()} Proportion"},
                title=f"Choropleth Map of '{selected_topic_map.capitalize()}' Intensity Across States"
            )
            
            ## Check if any data is mapped
            if state_topic_dist_df[selected_topic_map].sum() == 0:
                st.write(f"No data available for the topic '{selected_topic_map}'.")
            else:
                st.plotly_chart(fig_choropleth, use_container_width=True)

            ## Add Sentiment Choropleth Map for Topic
            st.markdown("#### Geographical Distribution of Sentiment for Selected Topic")
            with st.spinner("Calculating sentiment distribution for selected topic..."):
                
                
                ## Calculate average sentiment per state for the selected topic
                sentiment_per_state = df[df['assigned_label'].apply(lambda topics: selected_topic_map in topics)].groupby('state')['sentiment'].mean()
            fig_sentiment_choropleth = plot_sentiment_choropleth(sentiment_per_state, selected_topic_map)
            if fig_sentiment_choropleth:
                st.plotly_chart(fig_sentiment_choropleth, use_container_width=True)

    # ## Topic Co-occurrence Network
    #     st.markdown("#### Topic Co-occurrence Network")
    #     with st.spinner("Calculating topic co-occurrence matrix..."):
    #         cooccurrence_df = calculate_cooccurrence_matrix(df, mlb)
    #     cooccurrence_threshold = st.slider("Minimum Co-occurrence Threshold for Network Edges", min_value=1, max_value=50, value=10)
    #     fig_cooccurrence = plot_cooccurrence_network(cooccurrence_df, threshold=cooccurrence_threshold)
    #     if fig_cooccurrence:
    #         st.plotly_chart(fig_cooccurrence, use_container_width=True)


    ## Cluster Analysis of States Based on Topic Distribution
        st.markdown("#### State Clusters Based on Topic Distribution")
        with st.spinner("Performing cluster analysis on states..."):
            num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=2)
            cluster_df = cluster_states(df, mlb, num_clusters=num_clusters)
        fig_clusters = plot_state_clusters(cluster_df)
        if fig_clusters:
            st.plotly_chart(fig_clusters, use_container_width=True)

    ## Word Cloud Visualization
        st.markdown(f"#### Word Cloud Comparison for {selected_state} vs National Avg.")
        
        ## Select Topic for Word Cloud
        topics = mlb.classes_
        selected_topic_wc = st.selectbox("Select a Topic for Word Cloud Comparison", options=topics)
        
        if selected_topic_wc:
            with st.spinner(f"Generating word clouds for topic: {selected_topic_wc}..."):
                # Aggregate text data
                national_text_wc = get_text_for_topic(df, selected_topic_wc, state=None)
                state_text_wc = get_text_for_topic(df, selected_topic_wc, state=selected_state)
                
                ## Generate word clouds
                national_wc = generate_wordcloud(national_text_wc, title=f"National - {selected_topic_wc}", color='white')
                state_wc = generate_wordcloud(state_text_wc, title=f"{selected_state} - {selected_topic_wc}", color='white')
            
            st.success("Word clouds generated.")
            
            ## Display word clouds side by side
            col1, col2 = st.columns(2)
            with col1:
                display_wordcloud(national_wc, f"National - {selected_topic_wc}")
            with col2:
                display_wordcloud(state_wc, f"{selected_state} - {selected_topic_wc}")

    ## Display word clouds for top N topics
        st.markdown("### Top 5 Topics Word Clouds Comparison")
        top_n = 5
        top_topics = prevalence_df['Topic'].head(top_n).tolist()
        
        for topic in top_topics:
            st.markdown(f"**Topic: {topic}**")
            with st.spinner(f"Generating word clouds for topic: {topic}..."):
                national_text = get_text_for_topic(df, topic, state=None)
                state_text = get_text_for_topic(df, topic, state=selected_state)
                national_wc = generate_wordcloud(national_text, title=f"National - {topic}", color='white')
                state_wc = generate_wordcloud(state_text, title=f"{selected_state} - {topic}", color='white')
            
            col1, col2 = st.columns(2)
            with col1:
                display_wordcloud(national_wc, f"National - {topic}")
            with col2:
                display_wordcloud(state_wc, f"{selected_state} - {topic}")

        st.markdown("---")
        st.markdown("### About")
        st.markdown(
    "By Daniel Forcade.  dforcade@gatech.edu"
)


##### Please god run
if __name__ == "__main__":
    main()





