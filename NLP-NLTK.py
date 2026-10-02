import nltk
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords

# --- PREREQUISITES DOWNLOAD ---
# NLTK datasets auto-download agar pehle se majood na hon
nltk.download('punkt')
nltk.download('stopwords')

# --- STEP 1: DATA LOAD KARNA ---
# Agar aapke paas text file majood ha to isko use karein:
# with open("Natural_Language_Processing_Text.txt", "r") as file:
#     raw_text = file.read()

# Testing ke liye sample text (Aap iski jagah apna koi b text rakh sakte hain):
raw_text = """
Once upon a time there was an old mother pig who had three little pigs and not enough food to feed them. 
So when they were old enough, she sent them out into the world to seek their fortunes. 
Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence 
concerned with the interactions between computers and human language, in particular how to program computers 
to process and analyze large amounts of natural language data.
"""

print(f"--- Raw Text Length: {len(raw_text)} characters ---")

# --- STEP 2: SENTENCE TOKENIZATION ---
sentences = sent_tokenize(raw_text)
print(f"\nTotal Sentences: {len(sentences)}")
print("First Sentence Example:", sentences[0])

# --- STEP 3: WORD TOKENIZATION ---
words = word_tokenize(raw_text)
print(f"\nTotal Raw Words/Tokens: {len(words)}")

# --- STEP 4: RAW FREQUENCY DISTRIBUTION ---
fdist_raw = FreqDist(words)
print("\nTop 10 Most Common Tokens (Uncleaned):")
print(fdist_raw.most_common(10))

# --- STEP 5: REMOVE PUNCTUATION & LOWERCASE ---
# Faqat alphabetic words ko rakhein aur lowercase ma convert karein
words_no_punc = [word.lower() for word in words if word.isalpha()]
print(f"\nTotal Words (Without Punctuation): {len(words_no_punc)}")

# --- STEP 6: REMOVE STOPWORDS ---
stop_words = set(stopwords.words("english"))
clean_words = [word for word in words_no_punc if word not in stop_words]

print(f"Total Clean Words (Without Stopwords): {len(clean_words)}")

# --- STEP 7: FINAL FREQUENCY ANALYSIS & PLOTTING ---
fdist_clean = FreqDist(clean_words)
print("\nTop 10 Meaningful Keywords:")
print(fdist_clean.most_common(10))

# Frequency Plot Graph Display Karna
plt.figure(figsize=(10, 5))
fdist_clean.plot(10, cumulative=False, title="Top 10 Clean Word Frequencies")
plt.show()