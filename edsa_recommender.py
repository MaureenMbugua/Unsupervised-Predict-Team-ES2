"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import random
from PIL import Image
from wordcloud import WordCloud
import time

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model


# Data Loading
@st.cache
def data_loader():
    genre_df = pd.read_csv('./streamlit_dataset/genre.csv')
    ratings = pd.read_csv('./streamlit_dataset/ratings.csv')
    # Group all 2 star and lower ratings together
    ratings.rating[ratings.rating == 0.5] = 2
    ratings.rating[ratings.rating == 1] = 2
    ratings.rating[ratings.rating == 1.5] = 2
    return genre_df, ratings


def get_title(genre: str):
    genre_df = data_loader()[0]
    title_g = genre_df.groupby('genre')
    titles = title_g.get_group(genre)
    titles = titles.title.to_list()
    return titles


genres_df, ratings = data_loader()
genres = genres_df.genre.unique()
trend_df = pd.read_csv('./streamlit_dataset/movie_trend.csv')
trend_df.drop("Unnamed: 0", axis=1, inplace=True)


def viza(title: str):
    # Extract selected movie ratings
    top = ratings[ratings['title'] == title]

    # Count how many ratings are in each category: 1 star, 2 star, ect
    grouped = pd.DataFrame(top.groupby(['rating'])['title'].count())
    grouped.rename(columns={'title': 'rating_count'}, inplace=True)

    # Create the figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(122)

    # Create the colour palette
    temp = grouped['rating_count']
    temp_mod = [f'{round((x / sum(temp)) * 100)}%' for x in temp]
    labels = [f'{x} Stars - {y}' for x, y in zip(grouped.index, temp_mod)]
    theme = plt.get_cmap('Reds')
    ax.set_prop_cycle("color", [theme(1. * i / len(labels))
                                for i in range(len(labels))])
    sns.set(font_scale=1.25)

    # Create the pie chart
    pie = ax.pie(grouped['rating_count'],
                 # autopct='%1.1f%%',
                 labels=labels,
                 shadow=True,
                 startangle=10,
                 pctdistance=1.115,
                 explode=(random.choice([0.1], size=(len(grouped['rating_count']))))
                 )

    # Turn the pie chart into a donut chart
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Display the donut chart with a legend
    ax.set_title(f'Rating distribution for the {title}\n', fontsize=15)
    plt.tight_layout()
    return fig


def bag_of_words_count(words, word_dict={}):
    """ this function takes in a list of words and returns a dictionary
        with each word as a key, and the value represents the number of
        times that word appeared"""
    words = words.split()
    for word in words:
        if word in word_dict.keys():
            word_dict[word] += 1
        else:
            word_dict[word] = 1
    return word_dict


def title_extract():
    """This use to compile all the word and there frequency
    in the individual category found in a particular column"""
    count = 0
    result = {}
    type_labels = genres_df.genre.unique()
    genres = {}
    df_grp = genres_df.groupby('genre')
    for pp in type_labels:
        genres[pp] = {}
        for row in df_grp.get_group(pp)['title']:
            genres[pp] = bag_of_words_count(row, genres[pp])
    return genres


title_list = load_movie_titles('./streamlit_dataset/recommender_dataset.csv')


# App declaration
def main():
    """Movie Recommender App with Streamlit """
    # st.set_page_config(page_title="Movie Recommender", page_icon=":hash:", layout="centered")

    # Creates a main title and subheader on your page -
    logo = Image.open("movies_suggestmovie.png")
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image(logo)
    # st.title("Eco")
    # st.subheader("Climate change tweet classification")

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Landing Page", "Recommender System", "Data Exploration",
                    "Solution Overview", "About team"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png', use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write(f'### Enter Your Three Favorite Movies {len(title_list)}')
        movie_1 = st.selectbox('First Option', title_list[0:8000])
        movie_2 = st.selectbox('Second Option', title_list[8001:16000])
        movie_3 = st.selectbox('Third Option', title_list[16001:24866])
        fav_movies = [movie_1, movie_2, movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies, top_n=10)
                    st.title("We think you'll like:")
                    for i, j in enumerate(top_recommendations):
                        st.subheader(str(i + 1) + '. ' + j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")

        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i, j in enumerate(top_recommendations):
                        st.subheader(str(i + 1) + '. ' + j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")

    # -------------------------------------------------------------------

    # Landing page
    landing = Image.open("recommendation_engine.png")
    if page_selection == "Landing Page":
        st.image(landing)  # , height=1500)
        time.sleep(3)
        st.subheader("Movie Recommender App")
        st.button("Go to next page")

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.write("Describe your winning approach on this page")

    # -------------------------------------------------------------------

    # Exploration section
    if page_selection == "Data Exploration":
        st.subheader("Exploratory Data Analysis")
        col1, pick, col3 = st.columns(3)
        with col1:
            st.write('Pick subject you want an insight on:')
        with pick:
            ratings_cat = st.checkbox('Film ratings')
            genres_cat = st.checkbox('All about Genre')
            trending_cat = st.checkbox('Trending Movies')

        if ratings_cat:
            st.subheader("Film ratings")
            genre_select = st.selectbox("pick genre category", genres[1:])
            titles = get_title(genre_select)
            option = st.selectbox('What are your favorite movies', titles)
            result = viza(option)
            st.pyplot(result, use_container_width=True)

            # distribution of mean rating
            rating_grp = ratings.groupby("movieId")[["rating"]].mean()
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.histplot(data=rating_grp, x='rating', bins=10)
            ax.set_title(f'General Rating Distribution', fontsize=30)
            st.pyplot(fig, use_container_width=True)

        if genres_cat:
            st.subheader("Genre Fair")
            if ratings_cat:
                pass
            else:
                genre_select = st.selectbox("pick genre category", genres[1:])
            # visualizing how the genres of the movies are distributed
            df_grp_gens = genres_df.groupby("genre").agg({"title": "count"}).reset_index()
            fig1, ax = plt.subplots(figsize=(20, 15))
            ax = sns.barplot(data=df_grp_gens, x="genre", y="title")
            ax.set_xticks(ax.get_xticks())
            ax.set_ylabel('Frequency', fontsize=30)
            ax.set_xlabel('genre', fontsize=30)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=30)
            ax.set_title(f'Frequency of Genres', fontsize=60)

            # wordcloud plot
            d = title_extract()
            fig2, ax = plt.subplots(figsize=(20, 15))
            word_cloud_genre = WordCloud(width=512, height=384, background_color='black', min_font_size=2,
                                         min_word_length=3).generate(str(d[genre_select]))
            ax.imshow(word_cloud_genre)
            ax.axis("off")
            ax.set_title(f'Movie keywords for {genre_select} movies', fontsize=60)

            st.pyplot(fig2, use_container_width=True)
            st.pyplot(fig1, use_container_width=True)

        if trending_cat:
            st.subheader("Trending Movies")
            st.markdown('Trendy stuff Coming up')
            df_grp = trend_df.groupby('genre')

            if ratings_cat or genres_cat:
                pass
            else:
                genre_select = st.selectbox("pick genre category", genres[1:])

            num = st.slider("Number of movies", 1, 50, 5)
            filter_key = st.radio(
                "Filter trends based on:",
                ('Recent', 'Popular', 'Hot-List'))

            if filter_key == 'Popular':
                temp = df_grp.get_group(genre_select).sort_values(['popularity', 'year_made'], ascending=False)[:num]
                st.dataframe(
                    temp.reset_index(drop=True)[["title", "year_made", "vote_count", "vote_average", "popularity"]])

            elif filter_key == 'Recent':
                temp = df_grp.get_group(genre_select).sort_values(['year_made', 'weighted_rating'], ascending=False)[
                       :num]
                st.dataframe(temp.reset_index(drop=True)[
                                 ["title", "vote_count", "vote_average", "weighted_rating", "year_made"]])

            else:
                temp = df_grp.get_group(genre_select).sort_values('weighted_rating', ascending=False)[:num]
                st.dataframe(temp.reset_index(drop=True)[
                                 ["title", "year_made", "vote_count", "vote_average", "weighted_rating"]])
    # -------------------------------------------------------------------

    # Team profiling
    if page_selection == "About team":
        # col1, col2, col3 = st.columns(3)
        st.subheader("Meet our awesome seasoned professionals behind the great APP")

        st.markdown(" ")
        eliza_pic = Image.open("funny_head.jpg")
        olusola_pic = Image.open("funny_head.jpg")
        emma_pic = Image.open("funny_head.jpg")

        eliza, olusola, emma = st.columns(3)

        eliza.success("IT PROJECT MANAGER")
        olusola.success("Business Analyst")
        emma.success("Systems analyst")

        with eliza:
            st.header("Elizabeth")
            st.image(eliza_pic)

            with st.expander("Brief Bio"):
                st.write("""
                Elizabeth is an IT product manager with a background in user experience design and tons of experience in building
                high quality softwares. She has experience with building high quality products and scaling them. Her attention to 
                details is crucial as it has helped to work through models, visualizations, prototypes, requirements and manage across
                functional team. 
                
                She works consistently with Data Scientists, Data Engineers, creatiives and other business-oriented 
                people. She has gathered experience in data analytics, engineering, entrepreneurship, conversion optimization, internet 
                marketing and UX. Using that experience, she has developed a deep understanding of customer journey and product lifecycle.
                """)

        with olusola:
            st.header("Olusola")
            st.image(olusola_pic)

            with st.expander("Brief Bio"):
                st.write("""
                Founder of TechNation.Inc. Ken has over 10 years experience as a Business Growth manager possessing additional
                expertis in Product Develpoment. Proficient in facilitating business growth and enhancing market share of 
                the company by leading in-depth market research and competitor analysis, liasing eith senior management and
                conceptualizing new product development. 
                
                Highly skilled in functioning across multiple digital platforms and overseeing
                product design to optimize process. Adept at building businesses and teams from scratch and spearheading Strategy, P&L 
                Management, Marketing and Operations to lead data-driven decision making, render consumer impact analysis and achieve
                astronomical growth with respect to profitability and customer acquisition.
                """)

        with emma:
            st.header("Emmanuel")
            st.image(emma_pic)

            with st.expander("Brief Bio"):
                st.write("""
                Emmanuel is an accomplished Quality Assurance tester with over 3 years experience in Software Testing and Quality Assurance.
                He has a solid understanding in Software Development Life Cycle, Software Testing Lifecycle, bug lifecycle and testing
                diiferent procedure.
                """)

        mohamed, maureen, wasiu = st.columns(3)
        maureen.success("Lead Software Developer")
        mohamed.success("Machine Learning Engineer")
        wasiu.success("IT coordinator")

        maureen_pics = Image.open("funny_head.jpg")
        mohamed_pics = Image.open("funny_head.jpg")
        wasiu_pics = Image.open("funny_head.jpg")

        with maureen:
            st.header("Maureen")
            st.image(maureen_pics)

            with st.expander("Brief Bio"):
                st.write("""Maureen is a seasoned forward looking software engineer with 5+ years background in 
                creating and executing innovative software solution to enhance business productivity. Highly 
                experienced in all aspect of the software development lifecycle and end-to-end project management 
                from concept through to development and delivery. 

                He is consistently recognized as a hands-on competent leader, skilled at coordinating cross 
                functional team in a fast paced deadline driven environment to steer timely project completion.""")

        with mohamed:
            st.header("Mohamed")
            st.image(mohamed_pics)

            with st.expander("Brief Bio"):
                st.write("""
                Mohamed is a Senor Machine Learning engineer with around 8 years of professional IT experience in Machine Learning
                statistics modelling, Predictive modelling, Data Analytics, Data modelling, Data Architecture, Data Analysis, Data
                mining, Text mining, Natural Language Processing(NLP), Artificial Intelligence algorithms, Business intelligence (BI),
                analytics module (like Decision Trees, Linear and Logistics regression), Hadoop, R, Python, Spark, Scala, MS Excel and SQL.

                He is proficient in managing the entire Data Science project lifecycle and actively involved in the phase of project
                lifecycle including data acquisition, data cleaning, features engineering and statistical modelling.
                """)

        with wasiu:
            st.header("Wasiu")
            st.image(wasiu_pics)

            with st.expander("Brief Bio"):
                st.write("""Wasiu is a certified Project Management professional and certified Scrum master with over 
                5 years experience in project management, project process management, customer service management, 
                marketing and sales. 

                Being a highly motivated and team-oriented professional, she has successfully led large 
                cross-functional team to achieve strategic objectives and have managed a team of project managers 
                responsible for implementing a project portfolio.""")

                # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()
