import re
import string


def process_input(text_list):
    new_list = []
    dele_symbol = string.punctuation + string.digits
    for text in text_list:
        new_text = ""
        for char in text:
            if char not in dele_symbol:
                new_text += char.lower()
        new_list.append(new_text)
    return new_list