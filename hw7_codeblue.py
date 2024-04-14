'''
File: textastic.py
Description: A reusable library for text analysis and comparison
'''

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import random as rnd
from collections import Counter, defaultdict
from textblob import TextBlob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import plotly.graph_objects as go
import re

class CodeBlue:

    def __init__(self):
        '''constructor'''
        self.data = defaultdict(dict)
        self.stopwords = []

    @staticmethod
    def _default_parser(filename):
        '''this should probably be a default text parser
        for processing simple unformatted text files.'''

        results = {
            'wordcount': Counter("to be or not to be".split(" ")),
            'numwords': rnd.randrange(10, 50)
        }
        print("Parsed: ", filename, ": ", results)
        return results

    def load_text(self, filename, label=None, parser=None, stopwords=None):
        '''load text file into the library'''
        if parser is None:
            results = self._default_parser(filename)
        else:
            results = parser(filename)

        if label is None:
            label = filename

        if stopwords:
            self.data[label] = parser(filename, stopwords=stopwords)
        else:
            self.data[label] = parser(filename)

        self.data[label] = results

    def load_stop_words(self, stopfile):
        '''Load a list of common or stop words.'''
        with open(stopfile, 'r') as infile:
            stop_words = infile.read().splitlines()

        stop_words = [re.sub(r'[^\w\s]', '', word).lower() for word in stop_words]

        self.stopwords.extend(stop_words)
    def wordcount_sankey(self, word_list=None, k=5):
        '''Generate a Sankey diagram with the top 10 words (excluding stop words) for each TV show.'''
        src= []
        targ= []
        val = []

        word_index = {}
        next_word_index = len(self.data)

        stopwords = set(self.stopwords)

        for label, data_dict in self.data.items():
            text_index = list(self.data.keys()).index(label)
            word_counts = data_dict['word_count']

            if word_list:
                selected_words = [word for word in word_list if word in word_counts]
            else:
                counts = Counter({word: count for word, count in word_counts.items() if word not in stopwords})
                selected_words = [word for word, _ in counts.most_common(k)]

            for word in selected_words:
                if word not in word_index:
                    word_index[word] = next_word_index
                    next_word_index += 1

                src.append(text_index)
                targ.append(word_index[word])
                val.append(word_counts[word])

        labels = list(self.data.keys()) + list(word_index.keys())

        fig = go.Figure(data=[go.Sankey(node=dict(pad=15,
                                                  thickness=20,
                                                  line=dict(color='black', width=0.5),
                                                  label=labels,
                                                  ),
                                        link=dict(source=src,
                                        target=targ,
                                        value=val,
                                        )
        )])

        fig.update_layout(title_text=f'Top {k} Words in Medical Drama TV Shows')
        fig.show()

    def sentiment_analysis(self, k=10):
        '''Calculate average sentiment scores for the top k characters in each TV show and overall sentiment'''
        all_lines= []
        for label, data_dict in self.data.items():
            character_lines = data_dict['dialogues']
            for lines in character_lines.values():
                sentiments = [TextBlob(line).sentiment.polarity for line in lines]
                all_lines.extend(sentiments)

        overall_sentiment = np.mean(all_lines)

        for label, data_dict in self.data.items():
            character_lines = data_dict['dialogues']
            top_characters = sorted(character_lines.items(), key=lambda x: len(x[1]), reverse=True)[:k] 

            character_sentiments = {}
            for character, lines in top_characters:
                sentiments = [TextBlob(line).sentiment.polarity for line in lines]
                avg_sentiment = np.mean(sentiments) if sentiments else 0
                character_sentiments[character] = avg_sentiment

            self.data[label]['character_sentiments'] = character_sentiments
            self.data[label]['overall_sentiment'] = overall_sentiment

    def plot_sentiments(self):
        '''subplots of sentiment scores for characters in tv shows'''
        num_shows = len(self.data)
        cols = 2
        rows = np.ceil(num_shows / cols)
        fig, axs = plt.subplots(int(rows), int(cols), figsize=(15, rows * 4), constrained_layout=True)

        if num_shows == 1:
            axs = np.array([[axs]])
        elif cols == 1 or rows == 1:
            axs = axs.reshape((rows, cols))

        for ax, (label, data_dict) in zip(axs.flat, self.data.items()):
            character_sentiments = data_dict.get('character_sentiments', {})
            overall_sentiment = data_dict.get('overall_sentiment', 0)
            characters = list(character_sentiments.keys())
            sentiments = list(character_sentiments.values())

            ax.bar(characters, sentiments, color='cornflowerblue', label='Character Sentiments')
            baseline = ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, label='Baseline')
            overall_line = ax.axhline(y=overall_sentiment, color='blue', linestyle='--', linewidth=1,
                                      label='Overall Sentiment')
            ax.set_title(f'{label} - Top {len(characters)} Characters Sentiment')
            ax.set_ylabel('Average Sentiment Score')
            ax.set_xlabel('Character')
            ax.tick_params(axis='x', rotation=45)
            baseline_legend = mlines.Line2D([], [], color='gray', linestyle='--', label='Baseline = 0')
            overall_legend = mlines.Line2D([], [], color='blue', linestyle='--', label='Overall Sentiment')
            ax.legend(handles=[baseline_legend, overall_legend])

        plt.show()

    def cosine_similarity(self):
        '''Calculate the cosine similarity between each TV show'''
        text = []
        labels = []

        for label, data_dict in self.data.items():
            text.append(' '.join([' '.join(map(str, lines)) for lines in data_dict['dialogues'].values()]))
            labels.append(label)

        vectorizer = TfidfVectorizer(stop_words=self.stopwords)
        matrix = vectorizer.fit_transform(text)
        cosine_similarities = cosine_similarity(matrix, matrix)

        return cosine_similarities, labels

    def cosine_heatmap(self, scores, labels):
        '''Heatmap of cosine similarity'''
        cmap = sns.color_palette('Blues', as_cmap=True)
        sns.heatmap(scores, cmap=cmap, xticklabels=labels, yticklabels=labels)

        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j + 0.5, i + 0.5, '{:.2f}'.format(scores[i][j]), ha='center', va='center')

        plt.title('Cosine Similarity between TV Shows')
        plt.xlabel('TV Shows')
        plt.ylabel('TV Shows')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()