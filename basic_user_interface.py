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
from pprint import pprint as pprint

import tkinter as tk
from tkinter import ttk

from basic_utils import *
from knowledge_graph import *
from knowledge_graph_querying import *
from initial_card_processing import *

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

model_chat_engine = "gpt-3.5-turbo" 
    
SYSTEM_MESSAGE = ("You are a helpful professor and polymath scientist. You want to help a fellow researcher learn more about the world. "
                  + "You are clear, concise, and precise in your answers, and you follow instructions carefully.")
    
def _gen_chat_response(prompt='hi'):
    response = openai.ChatCompletion.create(
        model=model_chat_engine,
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ])
    message = response.choices[0]['message']

    return message['content']

def gen_chat_response(prompt='hi'):
    prompt_succeeded = False
    wait_time = 0.1
    while not prompt_succeeded:
        try:
            response = _gen_chat_response(prompt)
            prompt_succeeded = True
        except:
            print('  LM response failed. Server probably overloaded. Retrying after ', wait_time, ' seconds...')
            time.sleep(wait_time)
            wait_time += wait_time*2  # exponential backoff 
    return response

def convert_abstraction_group_to_concept_list(abs_grp):
    """
    Takes abstraction group dictionary, and returns a list of unique concepts as strings 
    """
    concept_list = set()
    [concept_list.update(concepts) for concepts in abs_grp.values()]
    return list(concept_list)


def sample_question_list(question_list):
    """ Take up to 3 questions randomly from a question list and concatenate. used for getting subject lists"""
    return " ".join(np.random.choice(question_list, size=min(3, len(question_list)), replace=False))


def user_triage_list(objects_to_triage):
    """
    Triage a list of objects to decide what to focus on next
    Displays objects to user, and asks whether to keep or not. Returns refined list
    """
    q_to_keep = []
    for _i, q in enumerate(objects_to_triage):
        if isinstance(q, tuple):
            print("Object " + str(_i+1) + ":\n" + '\n'.join(q))
        else:
            print("Object " + str(_i+1) + ": " + str(q))
    answer = input("Choose which to keep (type number to keep): ")
    
    selected_qs = [objects_to_triage[_i] for _i in range(len(objects_to_triage)) if str(_i + 1) in answer]

    return selected_qs

# Extract card representation for existing question/answer pair
# Extract question concept list for a new question

def extract_abstraction_groups(knowledgeGraph, 
                               question, 
                               answer=None, 
                               related_cardIDs = [],  # only show if provided 
                               num_random_cards_to_show=10, 
                               verbose=False):
    """
    A flexible function to extract the abstraction groups (part of the embedding) from a question, or a question and answer.
    It can show random cards, or related cards as examples. 
    """
    
    add_answer_text = "" if answer is None else " and answers" 
    prefix = (f"Professor Smith has provided a set of questions{add_answer_text} to his class to review. "
              + "He has also provided a list of key words and concepts extracted from each one. "
              + f"These words are related to the content of the ideas, not the exact words in the questions{add_answer_text}. "
              + "Moreover, some words convey the very general category of the question (for example, those in the labeled category '4'), "
              + "while others relate to very specific topics (labeled '-1'). From '4' to '-1', the categories become less abstract. "
              + "\nHere are a few examples:\n\n"
             )
    suffix = (f"\nProfessor Smith has not had time to extract the key words for all the questions{add_answer_text}, so he has asked for our help "
              + "extracting the key words and concepts from one more.\n"
              + "Here are his instructions: If we do not know the topic of the question, he has asked us to provide our best guess, "
              + "giving at least one key word in each category. "
              + "Ideally, we must extract a similar number of key words as the other examples. "
              + "Remember, these key words are to help the students understand the topic, and don't have to be perfect.\n\n"
              + "Here is the new question:\n"
              + "Question: {" + question + '}\n'
              + ("" if answer is None else ("Answer: {" + answer + '}\n'))
              + "Extracted key words and concepts (report in the format of a JSON dictionary):"
             )
    
    random_cardIDs = sample_random_cardIDs(knowledgeGraph, num_cards_to_show=num_random_cards_to_show)
    
    if len(related_cardIDs) > 0:
        used_cardIDs = np.append(random_cardIDs, related_cardIDs)
    else:
        used_cardIDs = random_cardIDs

    if answer is None:
        example_cards_and_concept_lists = chain_card_example_objects(["question", "abstraction_groups"], used_cardIDs, knowledgeGraph)
    else:
        example_cards_and_concept_lists = chain_card_example_objects(["question", "answer", "abstraction_groups"], used_cardIDs, knowledgeGraph)
    
    prompt = prefix + example_cards_and_concept_lists + suffix
    
    response_text = gen_chat_response(prompt) 
    if verbose:
        print(prompt + response_text)
    
    try:
        abstraction_groups = json.loads(response_text)
        assert isinstance(abstraction_groups, dict), "Json loading failed"
    except:
        print("Json loading failed")
        return None
    
    return abstraction_groups


def get_card_representation(knowledgeGraph, question, answer, num_random_cards_to_show=5, verbose=False):
    """
    gets card representation that can be directly loaded into a Pandas dataframe
    by calling LLM to extract abstraction_groups
    
    example call:
    new_card_rep = get_card_representation(kGraph, q, a)
    """
    
    
    abstraction_groups = extract_abstraction_groups(knowledgeGraph, question, answer, 
                                                    num_random_cards_to_show=num_random_cards_to_show, 
                                                    verbose=verbose)
    if abstraction_groups is None:
        return None
    return [question, answer, None, abstraction_groups]


def get_question_subject_list_in_graph(knowledgeGraph, 
                                       question, 
                                       related_cardIDs = [],
                                       num_random_cards_to_show=5, verbose=False):
    """
    Takes a question, and shows language model a few examples of extracted concepts from other questions, and then 
    extracts the concepts and subject list for this question that are in the knowledge graph.
    
    If related cards are provided, then it will show those as well. 
    """
    abstraction_groups = extract_abstraction_groups(knowledgeGraph, question, answer=None, 
                                                    related_cardIDs=related_cardIDs,
                                                    num_random_cards_to_show=num_random_cards_to_show, 
                                                    verbose=verbose)
    if abstraction_groups is None or (not isinstance(abstraction_groups, dict)):
        return None
    else:
        concept_list_full = convert_abstraction_group_to_concept_list(abstraction_groups)
        concept_list_in_graph = extract_concepts_in_knowledgeGraph_from_subject_list(concept_list_full, knowledgeGraph)
        if concept_list_in_graph:
            return concept_list_in_graph
        else:
            return None
        
        
def get_refined_question_subject_list_in_graph(knowledgeGraph, 
                                               question, 
                                               num_related_cards_to_show=5,
                                               num_random_cards_to_show=5, 
                                               verbose=False):
    """
    Wrapper to iterate get_question_subject_list_in_graph() twice
    
    Example call:
    concept_list = get_refined_question_subject_list_in_graph(kGraph, q)
    """
    
    # First call 
    concept_list_in_graph = get_question_subject_list_in_graph(knowledgeGraph, question, 
                                                               num_random_cards_to_show=num_random_cards_to_show, 
                                                               verbose=verbose)
    concept_list_in_graph=None
    if concept_list_in_graph is None:
        # we have no idea what this is about. Sample more random card IDs and try again 
        related_cardIDs = sample_random_cardIDs(knowledgeGraph, num_cards_to_show=num_random_cards_to_show)
    else:
        related_cardIDs = get_related_cardIDs_from_subject_list(concept_list_in_graph, knowledgeGraph,
                                         num_cards_to_show=num_related_cards_to_show)

    concept_list_in_graph = get_question_subject_list_in_graph(knowledgeGraph, question, 
                                                               related_cardIDs=related_cardIDs,
                                                               num_random_cards_to_show=num_random_cards_to_show, 
                                                               verbose=verbose)
    return concept_list_in_graph


# Answer a list of questions, generating natural language answers, and saving them in a combined list. 

def get_answers_to_questions(knowledgeGraph, 
                             question_list : List[str],  # the new questions to be answered
                             question_subject_list : List[str] = [],  # the subjects to find relevant things in the knowledge graph to display
                             num_related_cards_to_show=10,
                             verbose=False,
                             extra_verbose=False):
    """
    Answers questions based on style of knowledge graph, and by
    exposing related cards in knowledge graph back to the language model before answering 
    
    """
        
    if question_subject_list == None or len(question_subject_list) == 0:
        # random choice 
        related_cardIDs = sample_random_cardIDs(knowledgeGraph, num_cards_to_show=num_related_cards_to_show)
    else:
        related_cardIDs = get_related_cardIDs_from_subject_list(question_subject_list, knowledgeGraph, 
                                                        num_cards_to_show=num_related_cards_to_show)
    
    example_q_and_a = chain_card_example_objects(["question", "answer"], related_cardIDs, knowledgeGraph)
    
    question_list_wrapped = "".join([QUESTION_PROMPT + "{" + q + "}\n" for q in question_list])
    
    prompt = ("Professor Smith has provided the following question and answer pairs to the class to review as flashcards. "
              + "These are meant to be clear, concise, and self-contained questions and answers, organized into "
              + "simple concepts that can be memorized and retained.\n\n"
              + "Example flashcards:\n\n"
              + example_q_and_a
              + "\n\nWe have been asked to answer an additional set of questions in a similar style "
              + "(level of detail, brevity, structure, assumed knowledge, etc.) to the above examples. "
              + "We are to provide the questions and answers in the exact same format (including brackets),\n"
              + "Question: {restated question}\nAnswer: {our answer}\n\n"              
              + f"Here are the new {len(question_list)} question(s) we must answer:\n"
              + question_list_wrapped
             )
    
    response_text = gen_chat_response(prompt)
    
    new_questions = [card.split("Answer:")[0].strip(" \n").strip("{}") for card in response_text.split("Question:")]
    new_answers = [card.split("Question:")[0].strip(" \n").strip("{}") for card in response_text.split("Answer:")]
    new_question_answer_pairs = [(q, a) for q, a in zip(new_questions, new_answers) if (len(q) > 0 and len(a) > 0)]

    if extra_verbose:
        print(prompt + response_text)
    elif verbose:
        print(new_question_answer_pairs)
        
    return new_question_answer_pairs

# Parse sets of question/answer pairs and load into knowledge graph. 

def load_question_answer_pairs_into_knowledgeGraph(new_question_answer_pairs, cards_df, knowledgeGraph):
    
    def load_question_answer_pairs_into_dataframe(new_question_answer_pairs, cards_df, knowledgeGraph):

        new_cards_df = cards_df[0:0].copy()
        for ind, (q, a) in enumerate(new_question_answer_pairs):
            print(ind)
            new_card_rep = get_card_representation(knowledgeGraph, q, a)
            new_cards_df.loc[len(new_cards_df.index)] = new_card_rep

        return new_cards_df

    def load_dataframe_into_knowledgeGraph(new_cards_df, knowledgeGraph):

        new_card_deck = create_card_deck_from_dataframe_of_abstraction_groups(new_cards_df)
        title_list = knowledgeGraph.add_card_deck(new_card_deck, verbose=True)
        knowledgeGraph.update_all_embeddings(verbose=True)

        return None

    new_cards_df = load_question_answer_pairs_into_dataframe(new_question_answer_pairs, cards_df, knowledgeGraph)
    
    load_dataframe_into_knowledgeGraph(new_cards_df, knowledgeGraph)
    
    return new_cards_df

# Generate list of new questions based on existing style and structure of the knowledge graph. 

def get_suggested_further_questions(knowledgeGraph, 
                                    topic=None,
                                    question_subject_list_in_graph=None,
                                    target_question_list=None,
                                    goal_command = "expand and explore",
                                    num_random_clusters_of_cards_to_show=3,
                                    num_random_related_cards_to_show=10,
                                    num_related_cards_in_graph_to_show=5,
                                    num_questions_to_generate=5,
                                    increasing_abstraction=True,
                                    verbose=False,
                                    extra_verbose=False):
           
    # Random examples
    random_example_clusters_text = get_related_question_set_examples(knowledgeGraph, 
                                                          input_cardIDs=None,
                                                          num_seed_cards_to_show=num_random_clusters_of_cards_to_show, 
                                                          num_related_cards_to_show=num_random_related_cards_to_show,
                                                        increasing_abstraction=increasing_abstraction)
    
    # Related questions in knowledge graph derived from subject_list
    if (question_subject_list_in_graph is not None) and num_related_cards_in_graph_to_show > 0:
        related_cardIDs = get_related_cardIDs_from_subject_list(question_subject_list_in_graph, knowledgeGraph, 
                                                            num_cards_to_show=num_related_cards_in_graph_to_show)
        related_questions_verbose = chain_card_example_objects(["question"], related_cardIDs, knowledgeGraph)
        # Clean up the example questions to be a bit more compact spatially
        related_questions_text_unwrapped = "".join([s +'\n' for s in related_questions_verbose.splitlines() if s.strip()])
        related_questions_text = ("\nHere are examples of a few questions the class has already learned "
                                  + "and carefully reviewed:\n" 
                                  + related_questions_text_unwrapped
                                 )
    else:
        related_questions_text = ""

    # Existing cluster of questions 
    if target_question_list is not None and len(target_question_list) > 0:
        question_list_text = (f"These new questions should relate to the following cluster of ideas:\n"
                              + "Question: {" + "}\nQuestion: {".join(target_question_list)
                              + '}\n\n'
                             )
    else:
        question_list_text = ""
        
    if topic is not None:
        topic_statement_text = f"\nThese questions should be about the topic: \"{topic}\".\n"
        final_request_text = f"{num_questions_to_generate} new questions about \"{topic}\" (goal: {goal_command}):"
    else:
        topic_statement_text = ""
        final_request_text = f"{num_questions_to_generate} new questions (goal: {goal_command}):"
    
    prompt = ("Professor Smith has provided a wide range of questions to his class to review. "
              + "These questions are meant to be sufficiently detailed to be understandable without further context. "
              + "Here are a few examples of clusters of ideas covered in the class:\n\n"
              + random_example_clusters_text
              + "\n----------\n"
              + related_questions_text
              + "\nWe have been asked to provide more questions to the class. " 
              + topic_statement_text
              + question_list_text
              + "These questions should be on subjects that we think the class will find both interesting and useful. "
              + "Moreover, the questions should be concise and only cover one or two ideas each. "
              + 'Use the same format as the above examples (beginning each new question with "Q: {"). \n\n'
              + final_request_text
             )
    
    response_text = gen_chat_response(prompt)
                
    generated_questions = response_text.strip()
    generated_questions_list = generated_questions.split("Q: ")[1:]
    generated_questions_list = [generated_question.strip("}{ \n") for generated_question in generated_questions_list]

    if extra_verbose:
        print(print(prompt + '\n' + response_text))
    elif verbose:
        print("\n".join(generated_questions_list))
        
    return generated_questions_list


class ExplorationData:
    def __init__(self, 
                 topic="anything", 
                 goal_command="explore", 
                 new_question_list=[],  # New questions to answer
                 question_subject_list_in_graph=None,  # Questions we already know
                 question_list_in_graph=[],  # Questions we already know
                 num_rand_clusters=3,
                 num_related_cards=5,
                 num_related_cards_in_graph=0,
                 num_questions_generate=5,
                ):
        self.topic = topic
        self.goal_command = goal_command
        self.new_question_list = new_question_list
        self.question_subject_list_in_graph = question_subject_list_in_graph
        self.question_list_in_graph = question_list_in_graph
        self.num_rand_clusters = num_rand_clusters
        self.num_related_cards = num_related_cards
        self.num_related_cards_in_graph = num_related_cards_in_graph
        self.num_questions_generate = num_questions_generate


class QAEditApp(tk.Toplevel):
    def __init__(self, root, knowledgeGraph, cards_df, question_answer_pairs=[]):
        super().__init__()

        self.question_answer_pairs = question_answer_pairs
        self.checkbox_vars = []
        self.question_entries = []
        self.answer_entries = []

        self.title("Edit QA Pairs")
        self.create_widgets()
        
        self.knowledgeGraph = knowledgeGraph
        self.cards_df = cards_df

    def create_widgets(self):
        ttk.Label(self, text="Editing question answer pairs").grid(column=0, row=0, columnspan=3)

        update_button = ttk.Button(self, text="Update All", command=self.update_all)
        update_button.grid(column=1, row=1)

        toggle_button = ttk.Button(self, text="Toggle All", command=self.toggle_all)
        toggle_button.grid(column=0, row=1)
        
        close_button = ttk.Button(self, text="Save and Reset", command=self.save_and_reset)
        close_button.grid(column=2, row=1)

        for i, (question, answer) in enumerate(self.question_answer_pairs):
            checkbox_var = tk.BooleanVar()
            checkbox_var.set(True)
            checkbox = ttk.Checkbutton(self, variable=checkbox_var)
            checkbox.grid(column=0, row=i+2)
            self.checkbox_vars.append(checkbox_var)

            question_text = tk.Text(self, width=30, height=5, wrap=tk.WORD)
            question_text.insert(tk.END, question)
            question_text.grid(column=1, row=i+2)
            self.question_entries.append(question_text)

            answer_text = tk.Text(self, width=90, height=5, wrap=tk.WORD)
            answer_text.insert(tk.END, answer)
            answer_text.grid(column=2, row=i+2)
            self.answer_entries.append(answer_text)

    def update_all(self):
        updated_pairs = []
        for i, (question_text, answer_text) in enumerate(zip(self.question_entries, self.answer_entries)):
            if self.checkbox_vars[i].get():
                question = question_text.get("1.0", tk.END).strip()
                answer = answer_text.get("1.0", tk.END).strip()
                updated_pairs.append((question, answer))

        self.question_answer_pairs = updated_pairs
        self.checkbox_vars = []
        self.question_entries = []
        self.answer_entries = []
        self.clear_widgets()
        self.create_widgets()
        
    def toggle_all(self):
        if len(self.checkbox_vars) == 0:
            return
        toggle_to = not self.checkbox_vars[0].get()
        for checkbox_var in self.checkbox_vars:
            checkbox_var.set(toggle_to)

    def clear_widgets(self):
        for widget in self.grid_slaves():
            widget.destroy()
            
    def save_and_reset(self, 
                       json_file_title='my_flash_cards_general_cards_df_abstraction_groups'):
        self.update_all()
        
        # Save into knowledge graph 
        if len(self.question_answer_pairs) > 0:
            new_cards_df = load_question_answer_pairs_into_knowledgeGraph(self.question_answer_pairs, self.cards_df, self.knowledgeGraph)
        
            # Updated saved JSON file archive of flashcards to have the new ones
            for index, row in new_cards_df.iterrows():
                self.cards_df = self.cards_df.append(row, ignore_index=True)
            save_cards_df_to_json(self.cards_df, json_file_title)
        
        self.question_answer_pairs = []
        self.checkbox_vars = []
        self.question_entries = []
        self.answer_entries = []
        self.clear_widgets()
        self.create_widgets()
        

    
class ExplorationDataEditor:
    def __init__(self, master, exploration_data, knowledgeGraph, cards_df):
        self.exploration_data = exploration_data
        self.master = master
        self.knowledgeGraph = knowledgeGraph
        self.cards_df = cards_df
        self.master.title("Exploration Data Editor")
        
        self.window_width = 500
        self.master.geometry(str(self.window_width)+"x"+str(self.window_width))
        
        # Master info
        row_now = 0
        tk.Label(self.master, text="Topic:").grid(row=0, column=0)
        self.topic_entry = tk.Entry(self.master)
        self.topic_entry.grid(row=row_now, column=1)
        self.topic_entry.insert(0, self.exploration_data.topic)
        row_now += 1
        tk.Label(self.master, text="Goal command:").grid(row=row_now, column=0)
        self.goal_command_entry = tk.Entry(self.master)
        self.goal_command_entry.grid(row=row_now, column=1)
        self.goal_command_entry.insert(0, self.exploration_data.goal_command)
        row_now += 1
        tk.Label(self.master, text="Num random card clusters:").grid(row=row_now, column=0)
        self.num_rand_clusters_entry = tk.Entry(self.master)
        self.num_rand_clusters_entry.grid(row=row_now, column=1)
        self.num_rand_clusters_entry.insert(0, self.exploration_data.num_rand_clusters)
        row_now += 1
        tk.Label(self.master, text="Num related cards shown with random clusters:").grid(row=row_now, column=0)
        self.num_related_cards_entry = tk.Entry(self.master)
        self.num_related_cards_entry.grid(row=row_now, column=1)
        self.num_related_cards_entry.insert(0, self.exploration_data.num_related_cards)
        row_now += 1
        tk.Label(self.master, text="Num related cards shown from graph:").grid(row=row_now, column=0)
        self.num_related_cards_in_graph_entry = tk.Entry(self.master)
        self.num_related_cards_in_graph_entry.grid(row=row_now, column=1)
        self.num_related_cards_in_graph_entry.insert(0, self.exploration_data.num_related_cards_in_graph)
        row_now += 1
        tk.Label(self.master, text="Num questions to generate per query:").grid(row=row_now, column=0)
        self.num_questions_generate_entry = tk.Entry(self.master)
        self.num_questions_generate_entry.grid(row=row_now, column=1)
        self.num_questions_generate_entry.insert(0, self.exploration_data.num_questions_generate)
        # Save button
        row_now += 1
        self.save_button = tk.Button(self.master, text="Save overall info", command=self.save_data)
        self.save_button.grid(row=row_now, columnspan=2)
        # Query button
        row_now += 1
        self.new_questions_button = tk.Button(self.master, text="Generate new questions", command=self.generate_new_questions)
        self.new_questions_button.grid(row=row_now, columnspan=2)
        # Query button
        row_now += 1
        self.situation_in_graph_button = tk.Button(self.master, text="Gather related questions from graph", command=self.update_related_questions)
        self.situation_in_graph_button.grid(row=row_now, columnspan=2)
        # Answer button
        row_now += 1
        self.answer_button = tk.Button(self.master, text="Answer new questions", command=self.answer_new_questions)
        self.answer_button.grid(row=row_now, columnspan=2)
        # Print button
        row_now += 1
        self.answer_button = tk.Button(self.master, text="Print related questions and answers", command=self.print_related_questions_and_answers)
        self.answer_button.grid(row=row_now, columnspan=2)


        # New question list window
        self.new_question_window = tk.Toplevel(self.master)
        self.new_question_window.geometry(str(self.window_width)+"x"+str(self.window_width))
        self.new_question_window.title("New Question List Editor")
        
        self.update_button = tk.Button(self.new_question_window, text="Update Questions", command=self.update_new_questions)
        self.update_button.grid(row=0, column=1, columnspan=1)
        self.update_button_toggle = tk.Button(self.new_question_window, text="Tog.", command=self.update_new_questions_toggle_all)
        self.update_button_toggle.grid(row=0, column=0, columnspan=1)
        
        tk.Label(self.new_question_window, text="Keep?").grid(row=1, column=0)
        tk.Label(self.new_question_window, text="New questions to answer").grid(row=1, column=1)

        self.question_vars = []
        for i, question in enumerate(self.exploration_data.new_question_list):
            question_var = tk.BooleanVar()
            question_var.set(True)
            self.question_vars.append(question_var)

            question_checkbutton = tk.Checkbutton(self.new_question_window, variable=question_var)
            question_checkbutton.grid(row=i+2, column=0)
            question_label = tk.Label(self.new_question_window, text=question, wraplength=self.window_width-60,
                                     anchor="nw", justify="left")
            question_label.grid(row=i+2, column=1, sticky='w')
        
        # Related question list window
        self.related_question_window = tk.Toplevel(self.master)
        self.related_question_window.geometry(str(self.window_width)+"x"+str(self.window_width))
        self.related_question_window.title("Related Questions from Graph")
        
        # QA pairs window
            
        self.qa_edit_window = QAEditApp(self.master, self.knowledgeGraph, self.cards_df)
        
    def save_data(self):
        self.exploration_data.topic = self.topic_entry.get()
        self.exploration_data.goal_command = self.goal_command_entry.get()
        self.exploration_data.num_rand_clusters = int(self.num_rand_clusters_entry.get())
        self.exploration_data.num_related_cards = int(self.num_related_cards_entry.get())
        self.exploration_data.num_related_cards_in_graph = int(self.num_related_cards_in_graph_entry.get())
        self.exploration_data.num_questions_generate = int(self.num_questions_generate_entry.get())
        
        self.redisplay_new_questions()
        self.redisplay_related_questions()
        
        print("Resaved main info")

    def update_new_questions(self):
        updated_new_question_list = []
        for i, question_var in enumerate(self.question_vars):
            if question_var.get():
                updated_new_question_list.append(self.exploration_data.new_question_list[i])
        self.exploration_data.new_question_list = updated_new_question_list

        self.redisplay_new_questions()
        
    def update_new_questions_toggle_all(self):
        if len(self.question_vars) == 0:
            return
        toggle_to = not self.question_vars[0].get()
        for checkbox_var in self.question_vars:
            checkbox_var.set(toggle_to)

    def redisplay_new_questions(self):
        # Refresh the question list display
        for widget in self.new_question_window.grid_slaves():
            if int(widget.grid_info()["row"]) >= 2:
                widget.grid_forget()

        self.question_vars = []
        for i, question in enumerate(self.exploration_data.new_question_list):
            question_var = tk.BooleanVar()
            question_var.set(True)
            self.question_vars.append(question_var)

            question_checkbutton = tk.Checkbutton(self.new_question_window, variable=question_var)
            question_checkbutton.grid(row=i+2, column=0)
            question_label = tk.Label(self.new_question_window, text=question, wraplength=self.window_width-50,
                                     anchor="nw", justify="left")
            question_label.grid(row=i+2, column=1, sticky='w')

    
    def generate_new_questions(self):
        print("Generating new questions based on random card clusters, related cards in graph, and existing new questions")
        
        self.save_data()
        self.update_new_questions()
        
        generated_new_question_list = get_suggested_further_questions(self.knowledgeGraph, 
                                    topic=self.exploration_data.topic,
                                    question_subject_list_in_graph=self.exploration_data.question_subject_list_in_graph,
                                    target_question_list=self.exploration_data.new_question_list,
                                    goal_command = self.exploration_data.goal_command,
                                    num_random_clusters_of_cards_to_show=self.exploration_data.num_rand_clusters,
                                    num_random_related_cards_to_show=self.exploration_data.num_related_cards,
                                    num_related_cards_in_graph_to_show=self.exploration_data.num_related_cards_in_graph,
                                    num_questions_to_generate=self.exploration_data.num_questions_generate,
                                    increasing_abstraction=True,
                                    verbose=False,
                                    extra_verbose=False)
        
        self.exploration_data.new_question_list = self.exploration_data.new_question_list + generated_new_question_list
        self.redisplay_new_questions()
        
    def update_related_questions(self):
        self.save_data()
        self.update_new_questions()

        display_number=20
        sample_questions = sample_question_list(self.exploration_data.new_question_list)
        question_subject_list = get_refined_question_subject_list_in_graph(self.knowledgeGraph, sample_questions) 
        
        related_cardIDs = get_related_cardIDs_from_subject_list(question_subject_list, self.knowledgeGraph, 
                                                        num_cards_to_show=display_number)
        related_question_list = list(reversed([self.knowledgeGraph.cards[cardID].question for cardID in related_cardIDs]))
            # Reversing puts most important at beginning 
        
        self.exploration_data.question_subject_list_in_graph = question_subject_list
        self.exploration_data.question_list_in_graph = related_question_list
        
        self.redisplay_related_questions()       
            
    def redisplay_related_questions(self):
        for widget in self.related_question_window.grid_slaves():
            widget.grid_forget()
                
        for i, question in enumerate(self.exploration_data.question_list_in_graph):
            if i+1 <= self.exploration_data.num_related_cards_in_graph:
                str_show = str(i+1) + ". " + question
                row_show = i
            else:
                str_show = "xxx - " + str(i+1) + ". " + question
                row_show = i+1
            question_label = tk.Label(self.related_question_window, text=str_show, wraplength=self.window_width,
                                     anchor="nw", justify="left")
            question_label.grid(row=row_show, column=1, sticky='w')
        question_label = tk.Label(self.related_question_window, text='----------------- Below here not displayed -----------------', 
                                  wraplength=self.window_width,
                                     anchor="nw", justify="left")
        question_label.grid(row=self.exploration_data.num_related_cards_in_graph, column=1, sticky='w')

    def answer_new_questions(self):
        print("Answering new questions")
        
        self.save_data()
        self.update_new_questions()
        
        # Generate answers
        question_subject_list = self.exploration_data.question_subject_list_in_graph
        question_list = self.exploration_data.new_question_list
        question_answer_pairs = get_answers_to_questions(self.knowledgeGraph, question_list, question_subject_list)
        
        self.qa_edit_window.question_answer_pairs = question_answer_pairs
        self.qa_edit_window.clear_widgets()
        self.qa_edit_window.create_widgets()
        
    def print_related_questions_and_answers(self):
        question_subject_list = self.exploration_data.question_subject_list_in_graph
        related_cardIDs = get_related_cardIDs_from_subject_list(question_subject_list, self.knowledgeGraph, 
                                                        num_cards_to_show=self.exploration_data.num_related_cards_in_graph)
        related_question_list = list(reversed([self.knowledgeGraph.cards[cardID].question for cardID in related_cardIDs]))
        related_answer_list = list(reversed([self.knowledgeGraph.cards[cardID].answer for cardID in related_cardIDs]))
        q_and_a = "".join(["Question: {" + q + "}\n" + "Answer: {" + a + "}\n" 
                           for q, a in zip(related_question_list, related_answer_list)])
        print(q_and_a)
    
def launch_explorer(knowledgeGraph, cards_df, exploration_data=ExplorationData()):
    
    root = tk.Tk()
    editor = ExplorationDataEditor(root, exploration_data, knowledgeGraph, cards_df)
    root.mainloop()
    
    return exploration_data
