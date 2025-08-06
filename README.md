# automated-review-rating-system
The Automated Review Rating System is a Python-based application designed to analyze user-generated reviews and predict corresponding ratings using natural language processing and machine learning techniques. It streamlines the feedback analysis process for platforms such as e-commerce websites, hospitality services, or educational portals.

### 🎯 Example Use Case

Imagine a hotel receives hundreds of guest reviews daily. Manually reading and scoring each one is time-consuming and inconsistent. With this system:

- Textual reviews like “The rooms were spotless and staff was friendly” are automatically interpreted using sentiment analysis.
- The system predicts a 4.5-star rating based on language tone and keyword intensity.
- Review summaries and rating predictions are visualized for the management team to improve service.

This helps businesses uncover hidden insights, maintain consistent feedback metrics, and improve customer experience based on real-time review data.
Project Structure

Project Overview and Objective
This project aims to develop an automated system that predicts product review ratings (1 to 5 stars) based on the text of the review. Using machine learning techniques, the system analyzes textual review data to learn patterns and accurately estimate the star rating.
The objective is to build a clean, balanced dataset, apply appropriate preprocessing, and train baseline models using text vectorization methods for initial prototyping.

Dataset Description
The dataset consists of customer product reviews collected from  CSV file. Each review contains a text field (Text) and an associated rating (1 to 5 stars). The data varies in quality with noise such as URLs, HTML tags, emojis, and variable review lengths.

Preprocessing Steps
Text Cleaning: Convert reviews to lowercase; remove URLs, HTML tags, punctuation, emojis, and special characters.

Initial Cleaning
Remove missing ratings or reviews
  df = df.dropna(subset=['Text', 'Score'])

Lowercase, strip punctuation, remove short reviews
  df['Text'] = df['Text'].str.lower().str.replace('[^a-z ]', '', regex=True)
  df = df[df['Text'].str.len() > 10]
  df = df.drop_duplicates()

  # Remove duplicates
df = df.drop_duplicates()

# Clean review text: lowercase, remove non-letters
df['Text'] = df['Text'].str.lower().str.replace('[^a-z ]', '', regex=True)

# Remove reviews shorter than 10 characters (optional)
df = df[df['Text'].str.len() > 10]

# Convert ratings to integer, if they aren't already
df['Score'] = df['Score'].astype(int)


Save cleaned data
  df.to_csv('data/cleaned_dataset/cleaned_data.csv', index=False)


5 Example Reviews per Rating
        for rating in sorted(df['Score'].unique()):
        print(f"\n--- 5 sample reviews for rating {rating} ---")
        for review in df[df['Score'] == rating]['review_text'].sample(5, random_state=42):
        print('-', review)


Stopwords Removal: Removed common English stopwords using SpaCy to reduce noise.

Lemmatization: Applied lemmatization (word normalization) using SpaCy to convert words to their base form, improving semantic consistency.

Filtering: Discarded reviews with fewer than 3 words or excessively long reviews (>200 words) to improve data quality.

Visualizations Used
Bar Plot: Displays the count of reviews per rating to check dataset balance.

Box Plot: Shows the distribution of review word counts per rating class.

Sample Reviews: Printed sample reviews per rating class to qualitatively inspect data content and formatting.

Balancing Strategy
To avoid class imbalance that could bias model training, the dataset was balanced by sampling an equal number of reviews per rating class (e.g., 2,000 reviews each). If the original data per class was insufficient, sampling with replacement was applied.

Train-Test Split Methodology
The balanced dataset was split into training (80%) and testing (20%) sets.

A stratified split was used to maintain proportional representation of all rating classes in both sets.

Shuffling was applied before splitting to randomize the order of samples.

Notes on Decisions Taken
Lemmatization vs Stemming: Lemmatization was favored as it preserves word meaning better than stemming, which can be more aggressive and less accurate.

Vectorizer Choice: TF-IDF vectorization was selected to capture both term frequency and inverse document frequency, giving more importance to distinctive words and improving feature representation over simple Bag of Words.

Modular and Reusable Code: The preprocessing pipeline and splitting methods were organized into reusable functions with clear inline comments and docstrings to facilitate maintainability and extensibility.





