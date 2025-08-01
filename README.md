# automated-review-rating-system
The Automated Review Rating System is a Python-based application designed to analyze user-generated reviews and predict corresponding ratings using natural language processing and machine learning techniques. It streamlines the feedback analysis process for platforms such as e-commerce websites, hospitality services, or educational portals.

### 🎯 Example Use Case

Imagine a hotel receives hundreds of guest reviews daily. Manually reading and scoring each one is time-consuming and inconsistent. With this system:

- Textual reviews like “The rooms were spotless and staff was friendly” are automatically interpreted using sentiment analysis.
- The system predicts a 4.5-star rating based on language tone and keyword intensity.
- Review summaries and rating predictions are visualized for the management team to improve service.

This helps businesses uncover hidden insights, maintain consistent feedback metrics, and improve customer experience based on real-time review data.
Project Structure

Workflow
  Advanced Data Collection
  Data Preprocessing
  Data Visualization
  Creating Imbalanced and Balanced Datasets

Project Overview
This project predicts review star ratings (1–5 stars) from review text. You’ll work with real datasets, practice robust data cleaning, explore class imbalance, and use both classic ML and simple NLP techniques in Python, all documented for easy learning and reproducibility.

Project Structure
automated-review-rating-system/
├data/
│   Day3 Data/
│     Reviews-4.csv
│     cleaned_dataset/
│      cleaned_data.csv
│       balanced_data.csv
│       imbalanced_data.csv
|        
├notebooks/
│    [automated-review-rating-system]
|scripts/
│   []
├README.md
└.gitignore
Workflow
1. Advanced Data Collection
Download raw review data (Reviews-1.csv,Reviews-2.csv,Reviews-3.csv,Reviews-4.csv etc.) and save in data/Day3 Data/.

Inspect data

2. Data Preprocessing
Remove rows with missing reviews or ratings
Remove duplicate reviews
Text cleanup (lowercase, remove punctuation)
Remove short reviews
Save the cleaned data

3. Data Visualization
Show rating distribution (%)

4. Creating Imbalanced and Balanced Datasets, ensuring  no overlap

