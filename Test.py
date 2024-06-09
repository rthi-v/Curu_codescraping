#Importing all the libraries necessary to perform the below activities:
import pandas as pd                                #Used for loading and reading the dataframe
import re                                          #Used for to extract the data for preprocessing using regex functions
from collections import Counter                    #For count the term and the frequencys
#Used for dataprocessing activity for stop words and word tokenization
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import streamlit as st
#Used to plot the graphs
import matplotlib.pyplot as mpt

#Used to plot the word cloud
from wordcloud import WordCloud

#Used for Sentimental Analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

import pandas as pd
import requests
from bs4 import BeautifulSoup

from nltk.tag import pos_tag
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('wordnet')
from sklearn.feature_extraction.text import CountVectorizer

headers = {
    'authority': 'www.amazon.in',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'accept-language': 'en-US,en;q=0.9',
    'cache-control': 'max-age=0',
    'device-memory': '8',
    'downlink': '10',
    'dpr': '0.8',
    'ect': '4g',
    'referer': 'https://www.amazon.in/OnePlus-Nord-Black-128GB-Storage/dp/B09WQY65HN/ref=sr_1_4?crid=1D99WHM86WX80&keywords=oneplus&qid=1656009113&sprefix=onep%2Caps%2C315&sr=8-4&th=1',
    'rtt': '0',
    'sec-ch-device-memory': '8',
    'sec-ch-dpr': '0.8',
    'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="102", "Google Chrome";v="102"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-ch-viewport-width': '2400',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'same-origin',
    'sec-fetch-user': '?1',
    'service-worker-navigation-preload': 'true',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36',
    'viewport-width': '2400',
}

def get_soup(url):
    r = requests.get(url, headers=headers,
    params={'url': url, 'wait': 2})
    soup = BeautifulSoup(r.content, 'html.parser')
    return soup

def get_reviews(soup):
    review_list = []
    reviews = soup.find_all('div', {'data-hook': 'review'})
    try:
        for item in reviews:
            review = {
            'Review': item.find('span', {'data-hook': 'review-body'}).text.strip(),
            }
            review_list.append(review)
            #print(review_list)
    except:
        pass
    return review_list

def fetch_reviews_page(asin, star_rating, page_number):
    page_url = f'https://www.amazon.com.au/product-reviews/{asin}/ref=cm_cr_arp_d_viewopt_sr?ie=UTF8&filterByStar={star_rating}&reviewerType=all_reviews&pageNumber={page_number}'
    soup = get_soup(page_url)
    return get_reviews(soup)

#Sentiment Analysis
def sentiment_analysis(df):
    # Loading the dataframe
    df = df
    #print(df)
    # tokenization of the review column
    reviews_token = df['Review'].apply(word_tokenize)

    # Assigning the variable against all the token created above and ultimately forming a list of the words
    token_review = [token for tokens in reviews_token for token in tokens]

    # calculating the count of each token against it
    review_count = Counter(token_review)

    # calculating the top 20 common words
    token_list = review_count.most_common(20)

    # Sepeating a list of all the terms and their respective frequency from the above calculated list of tokens
    token_term = [term for term, _ in token_list]
    token_frequencies = [frequency for _, frequency in token_list]

    # Custom contraction mapping to handle common variations (if needed)
    contraction_mapping = {
        "I've": "I have",
        "ive": "I have",
        "didn't": "did not",
        "cant": "cannot",
        "can't": "cannot",
        "don't": "do not",
        "dont": "do not",
        "it's": "it is",
        "its": "it is",
        "I’ll": "I will",
        # Add more contractions as needed
    }

    def expand_contractions(text, contraction_mapping):
        for contraction, expansion in contraction_mapping.items():
            text = re.sub(contraction, expansion, text, flags=re.IGNORECASE)
        return text
    #Method to preprocess custom stopwords

    def read_custom_stopwords(file_path):
        with open(file_path, 'r') as file:
            stopwords_list = [line.strip() for line in file.readlines()]
        return set(stopwords_list)

    # Step 1: Read custom stopwords from file
    custom_stopwords_file = 'custom_stopwords.txt'  # Specify your custom stopwords file
    custom_stop_words = read_custom_stopwords(custom_stopwords_file)

    # Combine NLTK stopwords with custom stopwords
    default_stop_words = set(nltk.corpus.stopwords.words('english'))
    combined_stop_words = default_stop_words.union(custom_stop_words)

    def preprocess_text(text, contraction_mapping, stop_words):
        # Expand contractions using the custom mapping
        text = expand_contractions(text, contraction_mapping)

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # Remove punctuation (excluding commas) and other non-alphanumeric characters
        text = re.sub(r'[^\w\s]', '', text)

        # Remove any remaining commas
        text = re.sub(',', '', text)

        # Remove numbers
        text = re.sub(r'\d+', '', text)

        # Remove excessive whitespace
        text = ' '.join(text.split())

        # Tokenize the text
        tokens = word_tokenize(text)

        # Remove stop words
        tokens = [word for word in tokens if word not in stop_words]

        return ' '.join(tokens)

    # Initialize the stop words, stemmer, and lemmatizer
    #stop_words = set(stopwords.words('english'))

    df['Clean_Review'] = df['Review'].apply(
        lambda text: preprocess_text(text, contraction_mapping, combined_stop_words))
    df = df.dropna()

    # tokenization of the reddit column
    review_token = df['Clean_Review'].apply(word_tokenize)

    # Assigning the variable against all the token created above and ultimately forming a list of the words
    token_review = [token for tokens in review_token for token in tokens]

    # calculating the count of each token against it
    review_count = Counter(token_review)

    # calculating the top 20 common words
    token_list = review_count.most_common(20)

    # Sepeating a list of all the terms and their respective frequency from the above calculated list of tokens
    token_term = [term for term, _ in token_list]
    token_frequencies = [frequency for _, frequency in token_list]

    # Assigning this variable to consider the Reddit Column as a string
    final_text = ' '.join(df['Clean_Review'])

    # Generate word cloud for the whole dataset
    final_cloud = WordCloud(width=800, height=400, background_color='white').generate(final_text)

    # variable to hold the VADER function
    final_ana = SentimentIntensityAnalyzer()

    # Creating a list to later append the compound score & sentiment against it
    final_senti = []

    # In order to calculate the compound score against the review column
    for index, row in df.iterrows():
        text = row['Clean_Review']
        senti_score = final_ana.polarity_scores(text)
        score = senti_score['compound']

        # Parameter to calcualte the score
        if score >= 0.05:
            sentiment = 'positive'
        elif score <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        # Appending the sentiment to the list
        final_senti.append(sentiment)

    # Finally creating a new column in the dataset to hold the sentiment against each value
    df['sentiment'] = final_senti
    sentiment_count = df['sentiment'].value_counts()

    df['sentiment_score'] = df['Clean_Review'].apply(lambda x: final_ana.polarity_scores(x)['compound'])
    positive_reviews_df = df[df['sentiment_score'] > 0]

    ####Old code####
    # Step 2: POS Tagging and Keyword Extraction

    stop_words = set(stopwords.words('english'))

    def extract_keywords(review):
        words = word_tokenize(review)
        tagged_words = pos_tag(words)
        keywords = [word for word, tag in tagged_words if tag.startswith('JJ') or tag.startswith('NN')]
        filtered_keywords = [word.lower() for word in keywords if word.lower() not in stop_words]
        return filtered_keywords

    df['keywords'] = df['Clean_Review'].apply(extract_keywords)

    # Step 3: Filter and Rank Keywords
    all_keywords = [keyword for keywords_list in df['keywords'] for keyword in keywords_list]
    filtered_keywords = [keyword for keyword in all_keywords if len(keyword) > 2]  # Filter short words
    keyword_counts = Counter(filtered_keywords)
    top_keywords = keyword_counts.most_common(10)  # Adjust the number based on your preference

    # Print Top Keywords as Highlights
    print("Highlights for the Product:")
    for keyword, count in top_keywords:
        print(f"{keyword.capitalize()} ({count} mentions)")

    # List of emotions
    emotions = ["Happiness", "Sadness", "Anger", "Surprise", "Anticipation"]

    # to calculate for each emotions in the ist
    emotion_counts = {emotion: 0 for emotion in set(emotions)}

    # Tokenize, count words, and associate with emotions
    for text, emotion in zip(df['Clean_Review'], emotions):
        words = text.split()  # Tokenize the text
        for word in words:
            emotion_counts[emotion] += 1

    # Calculcate the sum of emotions
    total_word_count = sum(emotion_counts.values())

    emotion_proportions = {emotion: count / total_word_count for emotion, count in emotion_counts.items()}


    # Filter emotions with non-zero counts
    filtered_emotions = [emotion for emotion, count in emotion_counts.items() if count > 0]
    filtered_proportions = [emotion_proportions[emotion] for emotion in filtered_emotions]

    # Use columns to split the page into two columns
    col1, col2 = st.columns(2)

    # Section 1: highlights (left column)
    with col1:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.header("Product Highlights")
        highlights_text = "\n".join(
            [f"- **{keyword.capitalize()}** ({count} mentions)" for keyword, count in top_keywords])
        st.markdown(highlights_text)

        st.markdown('</div>', unsafe_allow_html=True)
    # Section 2: Pie chart Chart (right column)
    with col2:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.header("Sentiment Distribution")
        fig, ax = plt.subplots(figsize=(16, 10))
        labels = sentiment_count.keys()
        sizes = sentiment_count.values
        colors = ['green', 'blue', 'red']

        #fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140,
                                          textprops=dict(fontsize=14, fontweight='bold'))

        ax.axis('equal')
        # Make the labels and percentages bold
        for text in texts:
            text.set_fontweight('bold')
        for autotext in autotexts:
            autotext.set_fontweight('bold')
        st.pyplot(fig)

        st.markdown('</div>', unsafe_allow_html=True)

    # Section 3: Word cloud (left column)
    with col1:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.header("Wordcloud ")
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.imshow(final_cloud, interpolation='bilinear')
        ax.set_title("Normal Word Cloud for Final DataFrame")
        ax.axis("off")
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    # Section 4: Emotion proportion(Bar chart)
    with col2:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.header("Emotions distribution")
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.bar(filtered_emotions, filtered_proportions)
        ax.set_xlabel("Emotion")
        ax.set_ylabel("Proportion", fontweight='bold')
        # Set x-axis labels bold
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

def get_product_image(asin):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    url = f"https://www.amazon.com.au/dp/{asin}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        img_tag = soup.find('img', class_='a-dynamic-image')
        img_src = img_tag['src'] if img_tag and 'src' in img_tag.attrs else None

        # Extract total review count
        review_count_tag = soup.find('span', {'data-hook': 'total-review-count'})
        review_count = review_count_tag.text.strip() if review_count_tag else None

        # Extract rating out of text
        rating_tag = soup.find('span', {'data-hook': 'rating-out-of-text'})
        rating_text = rating_tag.text.strip() if rating_tag else None

        return {
            'image_src': img_src,
            'review_count': review_count,
            'rating_text': rating_text
        }
    else:
        return None
def main():
    #asin = input('Enter Asin Number: ')
    st.title('Curu App Review Analysis')
    st.sidebar.title('Search')
    asin = st.sidebar.text_input('Enter the ASIN Number').strip()
    star_ratings = ['five_star', 'four_star', 'three_star', 'two_star', 'one_star']
    combined_df = []
    for star_rating in star_ratings:
        #print(f'Fetching reviews for {star_rating}...')
        page_number = 1
        reviews_list = []
        while True:
            reviews = fetch_reviews_page(asin, star_rating, page_number)
            if not reviews:
                break
            reviews_list.extend(reviews)
            page_number += 1
        dfs = pd.DataFrame(reviews_list)
        combined_df.append(dfs)
    df = pd.concat(combined_df, ignore_index=True)
    product_details = get_product_image(asin)
    if product_details:
        # Get product details
        image = product_details['image_src']
        review_count = product_details['review_count']
        rating_text = product_details['rating_text']
        link = "Go to Site"
        url = f"https://www.amazon.com.au/dp/{asin}"
        if image:
            st.sidebar.image(image, use_column_width=False)
            st.sidebar.markdown(f"[{link}]({url})")
        else:
            st.error("Image could not be retrieved.")
        st.markdown(f"Below insights are from the **{len(df)} reviews** available in **Amazon.**")
        st.markdown(f"**Overall Product Rating:** {rating_text} (based on total **{review_count}).**")
        sentiment_analysis(df)

if __name__ == "__main__":
    main()