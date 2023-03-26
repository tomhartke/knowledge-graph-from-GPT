import os
import openai
import re
import json
import numpy as np
import pandas as pd
from collections import Counter
from typing import Any, Optional, Tuple, Dict, List, NamedTuple, Set
import scipy
import time

from basic_utils import *

# Convert dataframe to json and save to file in human readable format. 
def save_cards_df_to_json(cards_df, save_file_name):
    cards_df_as_json = cards_df.to_json(orient="index")
    with open(save_file_name + '.json', 'w') as f:
        json.dump(cards_df_as_json, f)
    
def read_cards_df_from_json(save_file_name):
    with open(save_file_name + '.json', 'r') as f:
        cards_df_reloaded = pd.read_json(json.load(f), orient="index")
    return cards_df_reloaded


# Generate initial text descriptions from front and back of a flashcard 

def get_cards_df_text_descriptions_from_front_and_back(flashcardExamples_front, 
                                                   flashcardExamples_back,
                                                  verbose=False):

    """
    Returns a pandas dataframe with the text descriptions of card content
    including:
        Front (question)
        Back (answer)
        Key ideas (LM summary in its own words)
        Subject list (component subjects)
        Expanded explanation (higher abstractions)
    """
    
    flashcard_text_descriptions = [] 
    global_tokens_used_for_card_reading = 0  # to keep track

    for card_ind in range(len(flashcardExamples_front)): 
        if card_ind % 20 == 0 :
            print('Card index: ', card_ind)
            print("global_tokens_used_for_card_reading: ", global_tokens_used_for_card_reading)

        total_used_tokens = 0
        ##########################
        flashcardQuestion = flashcardExamples_front[card_ind]
        flashcardAnswer = flashcardExamples_back[card_ind]
        flashcardIntroduction = "Here is a flashcard written by Professor Smith for the students in his class to study: \n"
        flashcardStatement = ("\nFlashcard Content:\nQuestion: { " +  flashcardQuestion + " } \n" + 
                           "Answer: { " + flashcardAnswer + " } \n" )
        flashcardRephraseRequestKeyIdeas = ("\nProfessor Smith has made a list of the key facts and ideas a student must know " +
                                    "in order to fully understand this flashcard. " + 
                                   "He has taken care to explain all acronyms and abbreviations, and and made no assumptions about the knowledge of the student. " +
                                    "To be even more helpful, he has formatted the list of ideas in a structured way to convey the hierarchy of ideas being tested, "+
                                    "such as into numbered lists, when applicable. Overall, he has tried to make the answer brief and concise, " + 
                                    "while maintaining completeness and not introducing ambiguity.\n\n" + 
                                   "Professor Smith's numbered list of key ideas necessary to understand the flashcard:\n")
        flashcardPrompt = (flashcardIntroduction + 
                           flashcardStatement + 
                           flashcardRephraseRequestKeyIdeas)

        response_text, used_tokens = gen_response_text_with_backoff(flashcardPrompt, max_tokens = 400)
        total_used_tokens += used_tokens  # print("Used tokens:",used_tokens)
        if verbose:
            print(flashcardPrompt + response_text)

        flashcardKeyIdeas = "\nKey ideas necessary to understand the flashcard:\n" + response_text + "\n"
        flashcardKeyIdeas_list = response_text

        ##########################
        flashcardRephraseRequestSubjects = ("\nProfessor Smith has also compiled a numbered list of the minor subjects discussed within the flashcard. " +
                                            "These are the component objects which make up the major ideas of the flashcard. "
                                            "These are, for example, the nouns and objects discussed in both the question and answer. "
                                            "The names of subjects in the list are reported in extremely brief form (less than 3 words, preferably 1 word), "
                                            "since we want to compare them to other flashcards.\n\n" +
                                            "Professor Smith's numbered list of subjects discussed in the flashcard:\n")

        flashcardPrompt = (flashcardIntroduction + 
                           flashcardStatement + 
                           flashcardKeyIdeas +  
                           flashcardRephraseRequestSubjects)

        response_text, used_tokens = gen_response_text_with_backoff(flashcardPrompt, max_tokens = 400)
        total_used_tokens += used_tokens  # print("Used tokens:",used_tokens)
        if verbose:
            print(flashcardPrompt + response_text)

        flashcardSubjects = "\nSubjects discussed in the flashcard:\n" + response_text + "\n"
        flashcardSubjects_list = response_text

        ##########################
        flashcardRephraseRequestSummary = (
        "\nProfessor Smith has also written a summary of the contents of the flashcard, to help his students understand the context of its information. " +
        "He categorizes the topic of the flashcard, and then the abstract category that topic is contained in, followed by more abstract categories " + 
        "that these categories are contained in, in increasing order of abstraction.\n\n" +
        "His explanation uses the following format:\n" + 
        "Specific topic: [text]  # names should be brief, less than 3 words, but preferably one word, and capitalized. Names should not be abbreviations.\n" +
        "General category: [[text], [text], ... ]\n" +
        "More general categories: [[text], [text], ... ]\n" +
        "Even more general categories: [[text], [text], ... ]\n" +
        "Most general category: [[text], [text], ... ]\n\n" +
        "Professor Smith's summary:\n")

        flashcardPrompt = (flashcardIntroduction + 
                           flashcardStatement + 
                           flashcardKeyIdeas +
                           flashcardSubjects + 
                           flashcardRephraseRequestSummary)

        response_text, used_tokens = gen_response_text_with_backoff(flashcardPrompt, max_tokens = 400)
        total_used_tokens += used_tokens  # print("Used tokens:",used_tokens)
        if verbose:
            print(flashcardPrompt + response_text)

        flashcardExpandedExplanation = response_text

        # Save text data for later parsing and conversion to dictionary
        combined_dict = {}
        combined_dict["Question"] = flashcardQuestion
        combined_dict["Answer"] = flashcardAnswer
        combined_dict["Key ideas"] = flashcardKeyIdeas_list
        combined_dict["flashcardSubjects_list"] = flashcardSubjects_list
        combined_dict["flashcardExpandedExplanation"] = flashcardExpandedExplanation
        # pprint(combined_dict)
        flashcard_text_descriptions.append(combined_dict)

        print("Total used tokens:",total_used_tokens, " for card index ", card_ind)
        global_tokens_used_for_card_reading += total_used_tokens
        
    cards_df_text_descriptions = pd.DataFrame(flashcard_text_descriptions)

    return cards_df_text_descriptions

# Extract text descriptions to JSON then to a python dataframe 

def get_cards_df_meta_data_from_text_descriptions(cards_df_text_descriptions,
                                                  verbose=False):
    """
    Returns a pandas dataframe with the text descriptions converted into dictionaries
    which contain the separated out key words at various levels of abstraction.
    """

    flashcard_list_of_dicts = []
    global_tokens_used_for_card_reading = 0  # to keep track

    for card_ind in range(len(cards_df_text_descriptions)):
        if card_ind % 20 == 0 :
            print('Card index: ', card_ind)
            print("global_tokens_used_for_card_reading: ", global_tokens_used_for_card_reading)

        total_used_tokens = 0

        ##########################

        flashcardSubjects_list = cards_df_text_descriptions["flashcardSubjects_list"].values[card_ind]
        flashcardExpandedExplanation = cards_df_text_descriptions["flashcardExpandedExplanation"].values[card_ind]

        ##########################

        # Now extract meta data to JSON files 
        jsonConversionFailure = False
        flashcardPrompt = ("Reformat the following list into a JSON array. Capitalize the first character of each word. " +
                           'Set all plural words to singular form. For example "car mechanics" should become "Car Mechanic" (plural to singular, and capitalization), ' +
                           '"Flights" should become "Flight" (plural to singular), and "new World Machines" should become "New World Machine"\n\n' +
                           "List:\n" + 
                           flashcardSubjects_list + 
                           "\n\nResult:\n")
        response_text, used_tokens = gen_response_text_with_backoff(flashcardPrompt, max_tokens = 400)
        total_used_tokens += used_tokens  # print("Used tokens:",used_tokens)
        if verbose:
            print(flashcardPrompt + response_text)

        try:
            subject_list = json.loads(response_text)

            # check datatype for everything 
            if not isinstance(subject_list, list):
                jsonConversionFailure = True
                print("   !!!!!!!! JSON conversion failed (not a real list of subjects)", " for card index ", card_ind)

        except:
            jsonConversionFailure = True
            print("   !!!!!!!! JSON conversion failed", " for card index ", card_ind)


        flashcardPrompt = ("Reformat the following information into a JSON dictionary containing lists of strings.\n\n" +
                           "Information:\n{\n" + 
                           flashcardExpandedExplanation + 
                           "\n}\n\n" + 
                           "Use the following format:\n{\n" + 
                           '    "Specific topic": [your text here, your text here, ...],  # Place items from "Specific topic" here, as a list of strings.\n' +
                           '    "General category": [your text here, your text here, ...],  # Place items from "General category" here, as a list of strings.\n' +
                           '    "More general categories": [],  # Place items from "More general categories" here" here, as a list of strings.\n' +
                           '    "Even more general categories": [],  # Place items from "Even more general categories" here, as a list of strings.\n' +
                           '    "Most general category": []  # Place items from "Most general category" here, as a list of strings.\n' +
                           "\n}\n\n" +
                           "For each string in the list, set all plural words to singular form, and capitalize the first character of each word. " +
                           'For example, a specific topic of "car mechanics" should become "Car Mechanic" (plural to singular, and capitalization), ' +
                           '"Flights" should become "Flight" (plural to singular), and "new World Machines" should become "New World Machine.\n\n' + 
                           "Result:\n")
        response_text, used_tokens = gen_response_text_with_backoff(flashcardPrompt, max_tokens = 400)
        total_used_tokens += used_tokens  # print("Used tokens:",used_tokens)
        if verbose:
            print(flashcardPrompt + response_text)

        try:
            dict_of_abstractions = json.loads(response_text)

            # check datatype for everything 
            if not isinstance(dict_of_abstractions, dict):
                jsonConversionFailure = True
                print("   !!!!!!!! JSON conversion failed (not a real dict of abstractions)", " for card index ", card_ind)

            if isinstance(dict_of_abstractions, dict):
                for k, v in dict_of_abstractions.items():
                    if not (isinstance(k, str) and isinstance(v, list)):
                        jsonConversionFailure = True
                        print("   !!!!!!!! JSON conversion failed (abstraction dict doesn't contain all str:list pairs)", " for card index ", card_ind)
                    for _hopefully_a_string in v:
                        if not isinstance(_hopefully_a_string, str):
                            jsonConversionFailure = True
                            print("   !!!!!!!! JSON conversion failed (abstraction dict doesn't contain a list of strings)", " for card index ", card_ind)

        except:
            jsonConversionFailure
            print("   !!!!!!!! JSON conversion failed", " for card index ", card_ind)

        ##########################

        # Save data to a dictionary, then append to flashcard_list_of_dicts

        if not jsonConversionFailure :  # then save 
            combined_dict = dict_of_abstractions.copy()
            combined_dict["Question"] = cards_df_text_descriptions["Question"].values[card_ind]
            combined_dict["Answer"] = cards_df_text_descriptions["Answer"].values[card_ind]
            combined_dict["Subjects"] = subject_list
            combined_dict["Key ideas"] = cards_df_text_descriptions["Key ideas"].values[card_ind]
            # pprint(combined_dict)
            flashcard_list_of_dicts.append(combined_dict)

        print("Total used tokens:", total_used_tokens, " for card index ", card_ind)
        global_tokens_used_for_card_reading += total_used_tokens
        
    cards_df_meta_data = pd.DataFrame(flashcard_list_of_dicts)
    
    return cards_df_meta_data

    
def get_cards_df_abstraction_groups_from_meta_data(cards_df):
    """
    Converts meta data with weird names for abstraction levels into a common format 
    """
    
    new_cards_df = pd.DataFrame({})
    
    # Load in basic info 
    new_cards_df["Question"] = cards_df["Question"].values
    new_cards_df["Answer"] = cards_df["Answer"].values
    new_cards_df["Key ideas"] = cards_df["Key ideas"].values

    # More complicated info
    new_cards_df["Abstraction groups"] = [{} for _ in range(len(cards_df["Question"].values))]
    
    # Add cards 
    for card_ind in range(len(cards_df)):
        abstraction_group_dict = new_cards_df["Abstraction groups"].values[card_ind]
        
        abstraction_group_dict['-1'] = cards_df['Subjects'].values[card_ind]
        abstraction_group_dict['0'] = cards_df['Specific topic'].values[card_ind]
        abstraction_group_dict['1'] = cards_df['General category'].values[card_ind]
        abstraction_group_dict['2'] = cards_df["More general categories"].values[card_ind]
        abstraction_group_dict['3'] = cards_df['Even more general categories'].values[card_ind]
        abstraction_group_dict['4'] = cards_df['Most general category'].values[card_ind]
 
    return new_cards_df

def get_cards_df_abstraction_groups_from_front_and_back_list(flashcardExamples_front, 
                                                   flashcardExamples_back,
                                                  verbose=False):
    # Create text descriptions, then meta data, then formatted abstraction groups 
    cards_df_text_descriptions = get_cards_df_text_descriptions_from_front_and_back(flashcardExamples_front, 
                                                                                    flashcardExamples_back,
                                                                                    verbose=verbose)
    cards_df_meta_data = get_cards_df_meta_data_from_text_descriptions(cards_df_text_descriptions, verbose=verbose)
    cards_df_abstraction_groups = get_cards_df_abstraction_groups_from_meta_data(cards_df_meta_data)
    return cards_df_abstraction_groups


def get_cards_df_abstraction_groups_from_front_and_back_csv(csv_title, verbose=False):

    # read in raw front and back sets of flashcards 
    cards_raw_front_and_back_df = pd.read_csv(csv_title + '.csv')
    flashcardList_front_text = cards_raw_front_and_back_df['front'].values
    flashcardList_back_text = cards_raw_front_and_back_df['back'].values

    cards_df_abstraction_groups = get_cards_df_abstraction_groups_from_front_and_back_list(flashcardList_front_text, 
                                                                                    flashcardList_back_text,
                                                                                    verbose=verbose)
    
    return cards_df_abstraction_groups
