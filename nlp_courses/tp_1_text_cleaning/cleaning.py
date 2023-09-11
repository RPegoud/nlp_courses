import re
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from utils import emojis_unicode, emoticons, slang_words


def lowercase(text: str) -> str:
    """
    Convert the input text to lowercase.

    Args:
        text (str): The input text to be converted to lowercase.

    Returns:
        str: A new string with all characters in lowercase.

    Example:
        >>> lowercase("Hello World")
        'hello world'
    """
    out = text.lower()
    return out


def remove_punctuation(text: str) -> str:
    """
    Remove punctuation characters from the input text.

    Args:
        text (str): The input text from which punctuation characters will be removed.

    Returns:
        str: A new string with punctuation characters removed.

    Example:
        >>> remove_punctuation("Hello, World!")
        'Hello World'
    """
    PUNCT_TO_REMOVE = string.punctuation
    out = text.translate(str.maketrans("", "", PUNCT_TO_REMOVE))
    return out


def stopwords_removal(text: str) -> str:
    """
    Remove stopwords from the input text.

    Args:
        text (str): The input text from which stopwords will be removed.

    Returns:
        str: A new string with stopwords removed.

    Example:
        >>> stopwords_removal("This is a sample text with some common stopwords.")
        'This sample text common stopwords.'
    """
    STOPWORDS = set(stopwords.words("english"))
    out = " ".join(
        [word for word in str(text).split() if word.lower() not in STOPWORDS]
    )
    return out


def lemmatize_words(text: str) -> str:
    """
    Lemmatize words in the input text.

    Args:
        text (str): The input text to be lemmatized.

    Returns:
        str: A new string with words lemmatized.

    Example:
        >>> lemmatize_words("Dogs are running quickly. Cats are jumping high.")
        'Dogs be run quickly . Cats be jump high .'
    """
    lemmatizer = WordNetLemmatizer()
    wordnet_map = {
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "J": wordnet.ADJ,
        "R": wordnet.ADV,
    }
    pos_tagged_text = nltk.pos_tag(text.split())
    out = " ".join(
        [
            lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN))
            for word, pos in pos_tagged_text
        ]
    )
    return out


def convert_emojis(text: str) -> str:
    """
    Convert emojis in the input text to their corresponding textual descriptions.

    Args:
        text (str): The input text containing emojis to be converted.

    Returns:
        str: A new string with emojis replaced by their textual descriptions.

    Example:
        >>> convert_emojis("I'm feeling 😊 and excited! 😀")
        "I'm feeling Smiling_Face_with_Smiling_Eyes and excited! Grinning_Face_with_Big_Eyes"
    """
    EMO_UNICODE = emojis_unicode()
    UNICODE_EMO = {v: k for k, v in EMO_UNICODE.items()}
    for emoticon, description in UNICODE_EMO.items():
        cleaned_description = description.replace(",", "").replace(":", "").split()
        replacement = "_".join(cleaned_description)
        text = text.replace(emoticon, replacement)
    return text


def convert_emoticons(text: str) -> str:
    """
    Convert emoticons in the input text to their corresponding textual descriptions.

    Args:
        text (str): The input text containing emoticons to be converted.

    Returns:
        str: A new string with emoticons replaced by their textual descriptions.

    Example:
        >>> convert_emoticons("Hello! :-)")
        'Hello! Happy_face'
    """
    EMOTICONS = emoticons()
    for emoticon, description in EMOTICONS.items():
        cleaned_description = description.replace(",", "").split()
        cleaned_description_joined = "_".join(cleaned_description)
        out = re.sub("(" + emoticon + ")", cleaned_description_joined, text)
    return out


def remove_urls(text: str) -> str:
    """
    Remove URLs (web links) from the input text.

    Args:
        text (str): The input text from which URLs will be removed.

    Returns:
        str: A new string with URLs removed.

    Example:
        >>> remove_urls("Visit our website at https://www.example.com to learn more.")
        "Visit our website at to learn more."
    """
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    out = url_pattern.sub(r"", text)
    return out


def remove_html(text: str) -> str:
    """
    Remove HTML tags and content from the input text.

    Args:
        text (str): The input text containing HTML tags to be removed.

    Returns:
        str: A new string with HTML tags and content removed.

    Example:
        >>> remove_html("<p>This is <b>bold</b> text.</p>")
        "This is bold text."
    """
    html_pattern = re.compile("<.*?>")
    out = html_pattern.sub(r"", text)
    return out


def chat_words_conversion(text: str) -> str:
    """
    Convert chat slang words in the input text to their full forms.

    This function takes the input text and searches for known chat
    slang words within it.
    For each detected slang word, it replaces it with its full form.

    Args:
        text (str): The input text containing chat slang words to be converted.

    Returns:
        str: A new string with chat slang words replaced by their full forms.

    Example:
        >>> chat_words_conversion("lol brb gtg")
        'laugh out loud be right back got to go'
    """
    slang_words_list = slang_words()
    chat_words_list = list(slang_words_list.keys())
    new_text = []
    for w in text.split():
        if w.upper() in chat_words_list:
            new_text.append(slang_words_list[w.upper()])
        else:
            new_text.append(w)
    out = " ".join(new_text)
    return out


# Create FunctionTransformer for each custom function
lowercase_transformer = FunctionTransformer(func=lowercase, validate=False)
remove_punctuation_transformer = FunctionTransformer(
    func=remove_punctuation, validate=False
)
stopwords_removal_transformer = FunctionTransformer(
    func=stopwords_removal, validate=False
)
lemmatize_words_transformer = FunctionTransformer(func=lemmatize_words, validate=False)
convert_emojis_transformer = FunctionTransformer(func=convert_emojis, validate=False)
convert_emoticons_transformer = FunctionTransformer(
    func=convert_emoticons, validate=False
)
remove_urls_transformer = FunctionTransformer(func=remove_urls, validate=False)
remove_html_transformer = FunctionTransformer(func=remove_html, validate=False)
chat_words_conversion_transformer = FunctionTransformer(
    func=chat_words_conversion, validate=False
)

# Create an scikit-learn pipeline
pipeline = Pipeline(
    [
        ("remove_urls", remove_urls_transformer),
        ("remove_html", remove_html_transformer),
        ("convert_emojis", convert_emojis_transformer),
        ("convert_emoticons", convert_emoticons_transformer),
        ("chat_words_conversion", chat_words_conversion_transformer),
        ("lowercase", lowercase_transformer),
        ("remove_punctuation", remove_punctuation_transformer),
        ("stopwords_removal", stopwords_removal_transformer),
        ("lemmatize_words", lemmatize_words_transformer),
    ]
)


def clean(text, pipeline=pipeline):
    return pipeline.transform(text)


if __name__ == "__main__":
    df = pd.read_csv("nlp_courses/tp_1_text_cleaning/to_clean.csv", index_col=0)
    df["cleaned_text"] = df.text.apply(lambda x: clean(x))
    for idx, row in df.iterrows():
        print(f"\nBase text: {row.text}")
        print(f"Cleaned text: {row.cleaned_text}\n")
