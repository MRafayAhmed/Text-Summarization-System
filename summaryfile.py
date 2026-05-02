import pandas as pd
import spacy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from transformers import pipeline
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK tokenizer if not already
nltk.download('punkt')

file_path = 'C:/Users/Administrator/.cache/kagglehub/datasets/shubham3112/cnn-daily-mail-dataset/versions/1/CNN_Daily_Mail_Dataset.csv'

# 1. Data Loading and Preprocessing
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['article'] = df['article'].str.replace(r'\s+', ' ', regex=True).str.strip()
    df['highlights'] = df['highlights'].str.replace(r'\s+', ' ', regex=True).str.strip()
    return df.head(100)  # Use first 100 rows

# 2. Extractive Summarization
def extractive_summarization(text, num_sentences=3):
    parser = PlaintextParser.from_string(text, Tokenizer('english'))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return ' '.join(str(sentence) for sentence in summary)

# 3. Abstractive Summarization
def abstractive_summarization(text, model_name='facebook/bart-large-cnn'):
    summarizer = pipeline('summarization', model=model_name, tokenizer=model_name)
    text = text[:1024]  # BART max input length
    summary = summarizer(text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
    return summary

# 4. Evaluation
def evaluate_summary(reference, generated):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, generated)

# 5. Visualization
def visualize_rouge_scores(scores_list, model_name):
    plt.figure(figsize=(10, 5))
    rouge_types = ['rouge1', 'rouge2', 'rougeL']
    scores = [scores_list[model_name][rt].fmeasure for rt in rouge_types]
    sns.barplot(x=rouge_types, y=scores)
    plt.title(f'ROUGE Scores for {model_name}')
    plt.ylabel('F1 Score')
    plt.savefig(f'rouge_scores_{model_name}.png')
    plt.close()

# 6. Main Execution
if __name__ == '__main__':
    df = load_and_preprocess_data(file_path)
    sample_article = df['article'].iloc[0]
    reference_summary = df['highlights'].iloc[0]

    # Extractive
    extractive_summary = extractive_summarization(sample_article)
    print("Extractive Summary:\n", extractive_summary)

    # Abstractive
    abstractive_summary = abstractive_summarization(sample_article)
    print("\nAbstractive Summary:\n", abstractive_summary)

    # Evaluate
    extractive_scores = evaluate_summary(reference_summary, extractive_summary)
    abstractive_scores = evaluate_summary(reference_summary, abstractive_summary)

    print("\nExtractive ROUGE Scores:\n", extractive_scores)
    print("\nAbstractive ROUGE Scores:\n", abstractive_scores)

    # Visualize
    visualize_rouge_scores({'extractive': extractive_scores, 'abstractive': abstractive_scores}, 'extractive')
    visualize_rouge_scores({'extractive': extractive_scores, 'abstractive': abstractive_scores}, 'abstractive')

    # Save summaries
    with open('sample_summaries.txt', 'w', encoding='utf-8') as f:
        f.write(f"Extractive Summary:\n{extractive_summary}\n\n")
        f.write(f"Abstractive Summary:\n{abstractive_summary}\n")
