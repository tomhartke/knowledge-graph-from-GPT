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
from knowledge_graph import *

QUESTION_PROMPT = "Question: "
ANSWER_PROMPT = "Answer: "
CONCEPT_LIST_PROMPT = "Extracted key words and concepts: "
RELATED_QUESTION_PROMPT = "Related question: "

def wrap_question_text(question_text):
    return QUESTION_PROMPT + "{" + str(question_text) + '}'

def wrap_answer_text(answer_text):
    return ANSWER_PROMPT + "{" + str(answer_text) + '}'

def wrap_concept_list_text(concept_list):
    concept_list_string = '["' + '", "'.join(concept_list) + '"]'
    concept_string = CONCEPT_LIST_PROMPT + concept_list_string
    return concept_string

def wrap_concept_list_nice_text(concept_list):
    concept_list_string = ', '.join(concept_list) 
    concept_string = CONCEPT_LIST_PROMPT + concept_list_string
    return concept_string
    

def chain_card_example_objects(ordered_list_of_objects, cardID_list, knowledgeGraph):
    possible_objects = ["question", "answer", "concept_list", "concept_list_nice", "abstraction_groups"]
    assert all([(obj in possible_objects) for obj in ordered_list_of_objects]), ("Failed to chain "
              "card examples: some object is not one of the allowed objects")
    
    def get_object_from_kGraph(obj, cardID, knowledgeGraph):
        if obj == "question":
            return wrap_question_text(knowledgeGraph.cards[cardID].question)
        elif obj == "answer":
            return wrap_answer_text(knowledgeGraph.cards[cardID].answer)
        elif obj == "concept_list":
            return wrap_concept_list_text(knowledgeGraph.cards[cardID].concepts.get_concepts_list())
        elif obj == "concept_list_nice":
            return wrap_concept_list_nice_text(knowledgeGraph.cards[cardID].concepts.get_concepts_list())
        elif obj == "abstraction_groups":
            return CONCEPT_LIST_PROMPT  + knowledgeGraph.cards[cardID].concepts.get_abstractions_dict_as_JSON_str()
    
    chain_of_examples = ""
    for cardID in cardID_list:
        for obj in ordered_list_of_objects:
            chain_of_examples += get_object_from_kGraph(obj, cardID, knowledgeGraph) + '\n'
        chain_of_examples += '\n' 
    return chain_of_examples


###################### Question processing ######################

def extract_concepts_in_knowledgeGraph_from_subject_list(subject_list, knowledgeGraph):
    question_concepts_list_in_knowledgeGraph = [concept for concept in subject_list
                                                if concept in knowledgeGraph.nodes.keys()]
    return list(set(question_concepts_list_in_knowledgeGraph))

def get_question_subject_list_from_card_sample(flashcardQuestion, sample_cardIDs, knowledgeGraph,
                                                      verbose=False,
                                              extra_verbose=False):

    flashcardPrompt = (chain_card_example_objects(["question", "concept_list"], sample_cardIDs, knowledgeGraph) +
                       wrap_question_text(flashcardQuestion) +'\n' + CONCEPT_LIST_PROMPT)

    response_text, used_tokens = gen_response_text_with_backoff(flashcardPrompt, max_tokens = 200)
    print("Used tokens:", used_tokens)

    subject_list = json.loads(response_text)
    subject_list_in_knowledgeGraph = extract_concepts_in_knowledgeGraph_from_subject_list(subject_list, knowledgeGraph)

    if extra_verbose:
        print(flashcardPrompt, response_text)
        print('In graph:', subject_list_in_knowledgeGraph)
    elif verbose:
        print(flashcardQuestion, '\nExtracted concept list:', response_text)
        print('In graph:', subject_list_in_knowledgeGraph)
        
    return subject_list_in_knowledgeGraph

def sample_random_cardIDs(knowledgeGraph, num_cards_to_show=20):
    return np.random.choice(range(len(knowledgeGraph.cards.keys())),num_cards_to_show)


def get_related_cardIDs_from_subject_list(subject_list_in_knowledgeGraph, knowledgeGraph,
                                         num_cards_to_show=20):
    # Returns the num_cards_to_show most relevant ones, but in reverse order of relevance (most relevant last)
    
    if subject_list_in_knowledgeGraph is None or len(subject_list_in_knowledgeGraph) == 0:
        random_cardIDs = sample_random_cardIDs(knowledgeGraph, num_cards_to_show=num_cards_to_show)
        return random_cardIDs
    
    # Get embedding vector based on concepts
    question_emb_vec = emb_vec_weighted_union_of_nodes(subject_list_in_knowledgeGraph, knowledgeGraph)
    question_emb_vec_trimmed = trim_embedding_vector(question_emb_vec)

    # Get overlap with all cards directly  
    card_overlaps = {k: emb_vec_inner_product(question_emb_vec_trimmed, card.embedding_vector_trimmed) for k, card in knowledgeGraph.cards.items()}
    card_sorted_keys, card_sorted_overlap_values = get_dict_items_sorted_by_decreasing_value(card_overlaps)

    # Get most closely related card IDs and prompt
    related_cardIDs = list(reversed(card_sorted_keys[0:num_cards_to_show])) # Reversed so most relevant is last
    return related_cardIDs

def get_refined_subject_list_from_question(flashcardQuestion, knowledgeGraph,
                                                num_cards_to_show=20,
                                                      verbose=False,
                                              extra_verbose=False):
    # Get initial embedding based on random sample of cards 
    random_cardIDs = sample_random_cardIDs(knowledgeGraph, num_cards_to_show=num_cards_to_show)
    # Don't show much info from initial query unless asked 
    subject_list_in_knowledgeGraph = get_question_subject_list_from_card_sample(flashcardQuestion, random_cardIDs, knowledgeGraph,
                                                                                               verbose=extra_verbose,
                                                                                extra_verbose=extra_verbose)
    # Refine embeddings
    related_cardIDs = get_related_cardIDs_from_subject_list(subject_list_in_knowledgeGraph, knowledgeGraph, 
                                                            num_cards_to_show=num_cards_to_show)
    subject_list_in_knowledgeGraph = get_question_subject_list_from_card_sample(flashcardQuestion, related_cardIDs, knowledgeGraph,
                                                                                               verbose=verbose,
                                                                                extra_verbose=extra_verbose)
    return subject_list_in_knowledgeGraph

###################### Generating and refining insightful questions ######################

def sort_cardIDs_by_rel_abs(emb_vec_target, input_cardIDs, knowledgeGraph, increasing_abstraction=True):
    # Now sort the related cardIDs by relative abstraction 
    card_rel_abs = {k: get_emb_vec_relative_abstraction_1to2(emb_vec_target, 
                                                                  knowledgeGraph.cards[k].embedding_vector_trimmed,
                                                                 knowledgeGraph) 
                     for k in input_cardIDs}
    card_sorted_keys_by_rel_abs, card_sorted_rel_abs_vals = get_dict_items_sorted_by_decreasing_value(card_rel_abs)
    if not increasing_abstraction:
        card_sorted_keys_by_rel_abs = list(reversed(card_sorted_keys_by_rel_abs))
        card_sorted_rel_abs_vals = list(reversed(card_sorted_rel_abs_vals))
        
    return card_sorted_keys_by_rel_abs, card_sorted_rel_abs_vals 
    
def get_related_cardIDs_to_cards_with_changing_abstraction(input_cardIDs, knowledgeGraph, num_related_to_show=5,
                                increasing_abstraction=True):
    
    related_cardIDs = []
    for _cardID in input_cardIDs:
        emb_vec_cardID = knowledgeGraph.cards[_cardID].embedding_vector_trimmed
        card_overlaps = {k: emb_vec_inner_product(emb_vec_cardID, card.embedding_vector_trimmed) 
                         for k, card in knowledgeGraph.cards.items()}
                        # if get_emb_vec_relative_abstraction_1to2(emb_vec_cardID, card.embedding_vector_trimmed, 
                        #                                          knowledgeGraph) < 0 }
        card_sorted_keys, card_sorted_overlap_values = get_dict_items_sorted_by_decreasing_value(card_overlaps)
        similar_cardIDs = list(card_sorted_keys[:num_related_to_show]) 
        
        # Now sort the related cardIDs by relative abstraction 
        card_sorted_keys_by_rel_abs, _ = sort_cardIDs_by_rel_abs(emb_vec_cardID, similar_cardIDs, knowledgeGraph,
                                                                increasing_abstraction=increasing_abstraction)
        
        related_cardIDs.append(list(card_sorted_keys_by_rel_abs)) 
    return related_cardIDs

def wrap_related_card_examples(related_cardIDs, knowledgeGraph, increasing_abstraction=True):
    increasing_decreasing_text = 'increasing' if increasing_abstraction else 'decreasing'
    detail_change_text = '' if increasing_abstraction else ' and more detail'
    
    example_question_and_related_questions = ""
    example_question_and_related_questions += ("Group of related questions:" 
                                               + detail_change_text + ":\n")
    for _cardID in related_cardIDs:
        example_question_and_related_questions += "Q: " +  '{' + (knowledgeGraph.cards[_cardID].question) + '}\n'
    return example_question_and_related_questions
    
def get_related_question_set_examples(knowledgeGraph, input_cardIDs=None,
                                         num_seed_cards_to_show=3, num_related_cards_to_show=3,
                                     increasing_abstraction=True):
    
    if input_cardIDs == None:
        use_cardIDs = sample_random_cardIDs(knowledgeGraph, num_cards_to_show=num_seed_cards_to_show)
    else:
        use_cardIDs = input_cardIDs 
        
    example_question_and_related_question = ""
    for related_cardIDs in get_related_cardIDs_to_cards_with_changing_abstraction(use_cardIDs, knowledgeGraph, 
                                                        num_related_to_show=num_related_cards_to_show,
                                                       increasing_abstraction=increasing_abstraction):
        new_text = wrap_related_card_examples(related_cardIDs, knowledgeGraph,
                                              increasing_abstraction=increasing_abstraction)
        example_question_and_related_question += new_text +'\n'
    return example_question_and_related_question

def get_suggested_further_questions_from_question_and_subject_list(flashcardQuestion, question_subject_list, knowledgeGraph,
                                              num_seed_cards_to_show=3,
                                              num_related_cards_to_show=5,
                                              num_questions_to_generate=1,
                                              temperature=1.0,
                                                                   increasing_abstraction=True,
                                                                   verbose=False,
                                              extra_verbose=False):
    
    increasing_decreasing_text = 'increasing' if increasing_abstraction else 'decreasing'
    detail_change_text = '' if increasing_abstraction else ' and more detail'
    
    if num_questions_to_generate > num_related_cards_to_show:
        num_questions_to_generate = num_related_cards_to_show
    
    # Get embedding vector based on concepts
    question_emb_vec = emb_vec_weighted_union_of_nodes(question_subject_list, knowledgeGraph)
    question_emb_vec_trimmed = trim_embedding_vector(question_emb_vec)
    
    # Get related card IDs to display along with target question 
    related_cardIDs = get_related_cardIDs_from_subject_list(question_subject_list, knowledgeGraph, 
                                                        num_cards_to_show=num_related_cards_to_show-num_questions_to_generate-1)
    [related_cardIDs_sorted_by_rel_abs, 
     related_cardIDs_sorted_rel_abs] = sort_cardIDs_by_rel_abs(question_emb_vec_trimmed, related_cardIDs, knowledgeGraph, 
                                                               increasing_abstraction=increasing_abstraction)
    related_cardIDs_to_display = related_cardIDs_sorted_by_rel_abs.copy()
    
    
    flashcardPrompt = ("Professor Smith has provided the following groups of questions to the class to review. "
                       "Within each group of questions, successive questions cover topics of " +
                       increasing_decreasing_text + " abstraction" + detail_change_text + ". "
                       "All questions are meant to be sufficiently detailed to be understandable without further context:\n\n" +
                       get_related_question_set_examples(knowledgeGraph, 
                                                          input_cardIDs=None,
                                                          num_seed_cards_to_show=num_seed_cards_to_show, 
                                                          num_related_cards_to_show=num_related_cards_to_show,
                                                        increasing_abstraction=increasing_abstraction) +
                       wrap_related_card_examples(related_cardIDs_to_display, knowledgeGraph, 
                                                  increasing_abstraction=increasing_abstraction) + 
                      "Q: " +  '{' + flashcardQuestion + '}\n' + "Q: {")
                
    response_text, used_tokens = gen_response_text_with_backoff(flashcardPrompt, max_tokens = 400, temperature=temperature)
    print("Used tokens:", used_tokens)

    generated_questions = response_text.strip()
    generated_questions_list = generated_questions.split("Q: ")
    generated_questions_list = [generated_question.strip("}{ \n") for generated_question in generated_questions_list]

    if extra_verbose:
        print(flashcardPrompt, response_text)
    elif verbose:
        print(flashcardQuestion)
        print("\n".join(generated_questions_list))
        # print(flashcardPrompt + response_text)
        # print('Suggested question: ', generated_question)
        
    return generated_questions_list


###################### Detailed answering ######################

def get_answer_from_question_with_subject_list(flashcardQuestion, question_subject_list, knowledgeGraph,
                                          num_cards_to_show=10,
                                               outside_knowledge_allowed=False,
                                                      verbose=False,
                                              extra_verbose=False):
    
    related_cardIDs = get_related_cardIDs_from_subject_list(question_subject_list, knowledgeGraph, 
                                                        num_cards_to_show=num_cards_to_show)
    
    if not outside_knowledge_allowed:
        outside_knowledge_prompt = ("In this answer, we are allowed to ONLY use information contained in the last {} questions and answers. ".format(num_cards_to_show) + 
                           "We cannot rely on any knowledge from outside the last {} questions and answers, ".format(num_cards_to_show) +
                           "and as a result we may not be able to answer the question. ")
    else:
        outside_knowledge_prompt = ("In this answer, we are allowed to use both information in the last {} questions and answers, as well as prior knowledge. ".format(num_cards_to_show) + 
                           "However, when new evidence conflicts with prior knowledge, we should trust the new evidence. ".format(num_cards_to_show) +
                           "We may not be able to answer the question. ")     
    
    expansion_prompt = ("If we are not satisfied with our ability to answer the question fully and completely, then we must say so. "
                        'To indicate this to Professor Smith, we must add an additional statement "Suggested further questions: [your text here]" '
                        'to suggest what else we would need to know to answer the question fully. These suggested further questions should '
                        'be sufficiently detailed to be understood and answered without additional context or explanation. '
                       )


    flashcardPrompt = ("Professor Smith has provided the following {} questions and answers to the class to review:\n".format(num_cards_to_show) +
                       chain_card_example_objects(["question", "answer"], related_cardIDs, knowledgeGraph) +
                       "Professor Smith has asked us to try to answer one final question. " + 
                       expansion_prompt +
                       "In general, the response must be concise, simple, and direct. " + 
                       outside_knowledge_prompt + '\n\n'+
                       QUESTION_PROMPT +
                       wrap_question_text(flashcardQuestion) +'\n' + ANSWER_PROMPT)

    response_text, used_tokens = gen_response_text_with_backoff(flashcardPrompt, max_tokens = 400)
    print("Used tokens:", used_tokens)

    flashcardAnswer = response_text.strip()

    if extra_verbose:
        print(flashcardPrompt, response_text)
    elif verbose:
        print('Question:', flashcardQuestion, '\nAnswer:', flashcardAnswer)
        
    return flashcardAnswer


###################### Enhancing the question only ######################

def enhanced_question_prompt(flashcardQuestion):
    return ("\nA student is currently studying the following question:\n" + (wrap_question_text(flashcardQuestion) + '\n') +'\n' +
            "We have been asked to rephrase this question to better test the students knowledge in as much detail as possible. "
            "The rephrased question must test the exact same information as the original question. "
            "It must be concise, simple, and direct. "
            "Rephrased question: ")

def get_enhanced_question_from_question_and_subject_list(flashcardQuestion, question_subject_list, knowledgeGraph,
                                          num_cards_to_show=10,
                                                      verbose=False,
                                              extra_verbose=False):
    
    related_cardIDs = get_related_cardIDs_from_subject_list(question_subject_list, knowledgeGraph, 
                                                        num_cards_to_show=num_cards_to_show)
    
    flashcardPrompt = ("Professor Smith has provided the following questions to the class to review:\n" +
                    chain_card_example_objects(["question"], related_cardIDs, knowledgeGraph) +
                   enhanced_question_prompt(flashcardQuestion) 
                  )

    response_text, used_tokens = gen_response_text_with_backoff(flashcardPrompt, max_tokens = 400)
    print("Used tokens:", used_tokens)

    generated_question = response_text.strip()

    if extra_verbose:
        print(flashcardPrompt, response_text)
    elif verbose:
        print('Rephrased question: ', generated_question)
        
    return generated_question

###################### Enhancing the whole flashcard ######################

def enhanced_flashcard_prompt(flashcardQuestion, flashcardAnswer):
    return ("\nA student is currently studying the following question and answer pair:\n" + (wrap_question_text(flashcardQuestion) + '\n') +
            # (wrap_concept_list_text(flashcardConceptList) + '\n') + 
            (wrap_answer_text(flashcardAnswer) + '\n') + 
            '\n' +
            "We have been asked to rephrase this question and answer pair to improve its quality. " # to better test the students knowledge in as much detail as possible. "
            "The rephrased question and answer must test the exact same information as the original question. "
            "It must be concise, simple, and direct. "
            "When applicable, it should summarize ideas into numbered lists, or structured notes, to show an organized hierarchy of ideas.\n\n"
            "Rephrased version: ")

def get_enhanced_flashcard_from_question_and_answer_and_subject_list(flashcardQuestion, flashcardAnswer, question_subject_list, knowledgeGraph,
                                          num_cards_to_show=10,
                                                      verbose=False,
                                              extra_verbose=False):
    
    related_cardIDs = get_related_cardIDs_from_subject_list(question_subject_list, knowledgeGraph, 
                                                        num_cards_to_show=num_cards_to_show)
    
    flashcardPrompt = ("Professor Smith has provided the following question and answer pairs to the class to review:\n" +
                    chain_card_example_objects(["question", "answer"], related_cardIDs, knowledgeGraph) +
                   enhanced_flashcard_prompt(flashcardQuestion, flashcardAnswer) 
                  )

    response_text, used_tokens = gen_response_text_with_backoff(flashcardPrompt, max_tokens = 400)
    print("Used tokens:", used_tokens)

    # generated_question = response_text.strip()

    if extra_verbose:
        print(flashcardPrompt, response_text)
    elif verbose:
        print('Response:\n', response_text)
        
    return response_text