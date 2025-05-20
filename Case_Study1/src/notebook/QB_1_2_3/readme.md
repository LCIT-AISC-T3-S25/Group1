# Stemming and Lemmatization Analysis

## Task Description
Performed text processing on social media profile data from Data.csv file using stemming and lemmatization techniques to analyze how these methods affect vocabulary size and word frequencies.

## What I Did

### B.1 - Porter Stemmer Implementation
- Applied Porter stemming algorithm to all tokens in the dataset
- Counted unique tokens before and after stemming
- Calculated the reduction in vocabulary size

### B.2 - NLTK Lemmatization
- Applied NLTK's WordNet lemmatizer to all tokens in the dataset
- Counted unique tokens before and after lemmatization
- Calculated the reduction in vocabulary size

### B.3 - Word Frequency Comparison
- Identified the top 10 most frequent words in:
  - Original text
  - After stemming
  - After lemmatization
- Analyzed how word frequencies change after applying these techniques

## Results Summary
The analysis showed that both stemming and lemmatization reduce vocabulary size by converting different forms of the same word to a common base form. The Porter stemmer tends to produce stems that aren't always real words (e.g., "calgari" for "calgary"), while lemmatization produces dictionary words (e.g., "child" for "children").

## NLTK Issues and Workarounds
I encountered issues with NLTK's word_tokenize function, which was giving a "Resource punkt_tab not found" error in Google Colab. To solve this:
- Implemented a custom tokenization function using simple string splitting instead of NLTK's word_tokenize
- Maintained the use of NLTK's Porter stemmer and WordNet lemmatizer which worked correctly
- Added fallback mechanisms for stopwords in case of NLTK resource issues
- These workarounds allowed me to complete the required analysis despite the NLTK limitations

## Code Implementation
- Preprocessing: Converted text to lowercase, removed punctuation and special characters
- Custom tokenization: Split text into individual words and removed stopwords
- Applied Porter stemmer and WordNet lemmatizer
- Performed frequency analysis using Counter to identify top words
- Added robust error handling for file reading with multiple encoding support