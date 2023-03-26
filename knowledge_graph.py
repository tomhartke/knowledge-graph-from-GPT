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
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100

from basic_utils import *

# A few useful operations on embedding vectors (dictionaries of values

trimming_and_intersection_tolerance_amount = 1e-10

def emb_vec_inner_product(emb_vec1, emb_vec2):
    """
    This is the key metric for the overlap of two embedding vectors 
    """
    # loop over all keys in one vector.     
    # Get product of value in both vectors. If one is non-existent, then set it to 0. 
    # This means we only have to loop over one vector 
    prod_sum = 0.0
    if len(emb_vec1.keys()) < len(emb_vec2.keys()):
        for key, val in emb_vec1.items():
            val_other = emb_vec2.get(key, 0.0)
            if val * val_other > 0 :
                prod_sum += np.sqrt(val * val_other)
    else:
        for key, val in emb_vec2.items():
            val_other = emb_vec1.get(key, 0.0)
            if val * val_other > 0 :
                prod_sum += np.sqrt(val * val_other)
    return prod_sum

def trim_embedding_vector(emb_vec, 
                          embedding_vector_tolerance_fraction=trimming_and_intersection_tolerance_amount):
    """
    Returns a shorter vector, which contains terms which sum to 
        1 - embedding_vector_tolerance_fraction. So most of them.
    """
    
    emb_vec_vals = np.array(list(emb_vec.values()))
    emb_vec_vals.sort() # from low to high 
    emb_vec_cumulative_threshold = np.sum(emb_vec_vals) * embedding_vector_tolerance_fraction
            
    emb_vec_vals_sorted_cumsum = np.cumsum(emb_vec_vals)
    num_below_threshold = len(np.where(emb_vec_vals_sorted_cumsum < emb_vec_cumulative_threshold)[0])
    if num_below_threshold > 0:
        threshold_val = emb_vec_vals[num_below_threshold - 1]  
        # need to be strictly greater than this to be included in the trimmed vector
    else:
        threshold_val = 0.0

    emb_vec_trimmed = {k: v for k, v in emb_vec.items() if v > threshold_val}
    
    return emb_vec_trimmed

def emb_vec_weighted_union_of_nodes(node_title_list, knowledgeGraph):
    """
    Gets an embedding vector which is a weighted sum of embedding vectors for the nodes.
        The intention is to find an embedding which characterizes the unique aspects of this set of nodes.
        That is it emphasizes the less common nodes more.

    Have to pass in knowledge graph as argument because union needs to know about 
        meta data of nodes, in the knowledge graph

    There are two types of weighting used here. 
        1. First, the relative weighting of each node is adjusted since
            we don't want to use concepts associated with many other things. Want to use unique things.
        2. Second, after summing the resulting embedding vector, from all the concepts in the card.
            When a node is highly referenced, we normalize its entry by the total amount of embedding pointing at it
            to get the fraction of all embedding pointing at this concept from unique aspects of the card 
            This once again makes the card embedding sensitive mostly to unique values
    """

    # Gather list of all keys from all the component nodes, to use in the resulting embedding
    node_title_set = set()
    for node_title in node_title_list:
        node = knowledgeGraph.nodes[node_title]
        node_title_set.update(set(node.embedding_vector.keys()))

    # Initialize emb vec to 0 for all keys 
    union_emb_vec_prenorm = {neighbor_concept: 0.0 for neighbor_concept in node_title_set}  

    # loop through embedding vectors of each of the nodes and add weighted value to existing value
    for node_index, node_title in enumerate(node_title_list):
        # note we use the list here, so if duplicates exist, they are intentionally counted more
        node = knowledgeGraph.nodes[node_title]
        effective_node_size_correction = 1.0 / (1.0 + node.sum_of_embeddings_to_node) 
        for neighbor_concept, emb_value in node.embedding_vector.items():
            union_emb_vec_prenorm[neighbor_concept] += emb_value * effective_node_size_correction

    # Now loop through and divide each individual embedding item by the total number of references to it
    union_emb_vec = {k: val_prenorm / (1.0 + knowledgeGraph.nodes[k].sum_of_embeddings_to_node) 
                    for k, val_prenorm in union_emb_vec_prenorm.items()}  

    # Finally, normalize the embedding vector to sum to 1
        # We do this because we want a similarity metric that isn't biased toward sums of many nodes.
    total_emb_vec = np.sum(list(union_emb_vec.values()))
    union_emb_vec = {k: v / total_emb_vec for k, v in union_emb_vec.items()}  
    
    return union_emb_vec


def emb_vec_intersection_with_threshold(emb_vec_list, knowledgeGraph, intersection_threshold_amount=1e-10):
    """
    Takes a set of embedding vectors, and gets the geometric average of their values for each key, 
        It sets values to a lower threshold if they are not present in the vector. 
    
    If a set of vectors have no intersection, it returns a uniform distribution over all keys
        Someday might want to update this to return the background distribution, rather than uniform.
        
    At the end, it normalizes the resulting vector to sum to 1. 
    
    This is like averaging the logit values if we think of the embedding vector as output by a softmax dist.
    
    """

    def get_emb_vec_lower_threshold_value(emb_vec, 
                                          embedding_vector_tolerance_fraction=trimming_and_intersection_tolerance_amount):
        """
        Returns the value in the vector where the cumulative sum of elements larger than this equals
            1 - embedding_vector_tolerance_fraction. So most of them.
        """

        emb_vec_vals = np.array(list(emb_vec.values()))
        emb_vec_vals.sort() # from low to high 
        emb_vec_cumulative_threshold = np.sum(emb_vec_vals) * embedding_vector_tolerance_fraction

        emb_vec_vals_sorted_cumsum = np.cumsum(emb_vec_vals)
        num_below_threshold = len(np.where(emb_vec_vals_sorted_cumsum < emb_vec_cumulative_threshold)[0])
        if num_below_threshold > 0:
            threshold_val = emb_vec_vals[num_below_threshold - 1]  
        else:
            threshold_val = 0.0
            
        # If value happens to be 0, then move up through vector until it's nonzero 
        while threshold_val == 0.0:
            num_below_threshold += 1
            threshold_val = emb_vec_vals[num_below_threshold]  

        return threshold_val

    def get_all_keys_over_emb_vec_list(emb_vec_list):
        all_keys = set(emb_vec_list[0].keys())  # just to get started 
        for emb_vec in emb_vec_list:
            all_keys = all_keys.union(set(emb_vec.keys()))
        return all_keys
    
    def get_geometric_avg_over_emb_vec_list_specific_key(emb_vec_list, lower_threshold_list, k):        
        # Going to take average of logs instead of 1/nth root of product, to encourage numerical stability
        # then exponentiate the average of logs
        
        num_emb_vecs = float(len(emb_vec_list))
        
        avg_of_logs = 0.0 
        for ind_of_vec, emb_vec in enumerate(emb_vec_list):
            # Use the lower threshold value of this vector as the default 
            avg_of_logs += np.log(emb_vec.get(k, lower_threshold_list[ind_of_vec])) / num_emb_vecs
        return np.exp(avg_of_logs)
    
    # Get specific keys 
    all_keys = get_all_keys_over_emb_vec_list(emb_vec_list)
    
    # Get lower threshold value for each embedding vector, to be used as default if key is not contained in vector
    lower_threshold_list = [get_emb_vec_lower_threshold_value(emb_vec,
                                                             embedding_vector_tolerance_fraction=intersection_threshold_amount
                                                             ) for emb_vec in emb_vec_list]
        
    # Take product of values
    emb_vec_intersection = {k: get_geometric_avg_over_emb_vec_list_specific_key(emb_vec_list, lower_threshold_list, 
                                                                                k) for k in all_keys}
    
    # Now renormalize it to sum to 1, if it has nonzero values 
    total_emb_vec = np.sum(list(emb_vec_intersection.values()))
    if total_emb_vec != 0.0:
        emb_vec_intersection = {k: v / total_emb_vec for k, v in emb_vec_intersection.items()}
    
    return emb_vec_intersection


def get_emb_vec_intersection_over_concepts(concept_list, knowledgeGraph):
    
    emb_vec_list = []
    for _ind, concept in enumerate(reversed(concept_list)):
        sig_vec = knowledgeGraph.nodes[concept].significance_vector 
        emb_vec = knowledgeGraph.nodes[concept].embedding_vector 
        emb_vec_list.append(emb_vec)
    
    emb_vec_int = emb_vec_intersection_with_threshold(emb_vec_list, knowledgeGraph)

    return emb_vec_int

def get_emb_vec_relative_abstraction_1to2(emb_vec1, emb_vec2, knowledgeGraph,
                                             heavy_trim_tolerance_fraction=1e-1):

    # Get very trimmed embedding vectors
    trimmed_emb_vec1 = trim_embedding_vector(emb_vec1, 
                              embedding_vector_tolerance_fraction=heavy_trim_tolerance_fraction)
    trimmed_emb_vec2 = trim_embedding_vector(emb_vec2, 
                              embedding_vector_tolerance_fraction=heavy_trim_tolerance_fraction)

    # loop through and get relative abstraction based on node-to-node abstraction
    # weight each cross pair by the product of embedding vector values (then normalized by total)
    rel_abs_with_weights_sum = 0.0 
    weights_sum = 0.0 
    for node1_title, node1_emb in trimmed_emb_vec1.items():
        for node2_title, node2_emb in trimmed_emb_vec2.items():
            rel_abs_dict_1to2 = knowledgeGraph.nodes[node1_title].neighbors_relative_abstraction
            rel_abs = 0.0
            if node2_title in rel_abs_dict_1to2.keys(): 
                # This is the abstraction of 2 relative to 1 
                rel_abs = rel_abs_dict_1to2[node2_title]
            rel_abs_with_weights_sum += rel_abs * node1_emb * node2_emb
            weights_sum += node1_emb * node2_emb

    rel_abs = rel_abs_with_weights_sum / weights_sum

    return rel_abs 

class CardConceptHierarchy:
    """
    A datastructure for listing the major ideas corresponding to a card, both at higher and lower abstraction levels. 
    """
    
    def __init__(self, topic="", topic_description=""):
        # The first key is an integer, the abstraction level. 0 is base. 1 is more abstract, -1 is less abstract 
        # second dictionary is the titles and descriptions of concepts 
        self.abstraction_groups: Dict[int: Dict[str, str]] = {0 : {str(topic): str(topic_description)}}
        
    def set_concept(self, relative_abstraction, title, description):
        self.abstraction_groups.setdefault(relative_abstraction, {})  # creates entry if it does not exist
        self.abstraction_groups[relative_abstraction][str(title)] = str(description)
        
    def get_concepts_list(self):  # Gets unique concepts, ordered by abstraction 
        ordered_concepts_nonunique = [concept for abs_level, concept_dict in reversed(sorted(self.abstraction_groups.items()))
                                 for concept, desc in concept_dict.items() ]
        unique_inds = np.unique(ordered_concepts_nonunique, return_index=True)[1]
        return [ordered_concepts_nonunique[_unique_ind] for _unique_ind in sorted(unique_inds)]
        
    def get_abstractions_dict(self): # returns dictionary containing unique concepts, and abstracton level 
        # If concept shows up at multiple abstraction levels, then average them together. 
        unique_concepts = self.get_concepts_list()
        abs_level_samples = {concept: [] for concept in unique_concepts}
        for abs_level, concept_dict in reversed(sorted(self.abstraction_groups.items())):
            for concept, desc in concept_dict.items():
                abs_level_samples[concept].append(abs_level)
        abstractions_dict = {k: np.average(np.array(v)) for k, v in abs_level_samples.items()}
        return abstractions_dict
    
    def get_abstractions_dict_as_JSON_str(self):

        def list_to_doublequotes(_list):
            return '["' + '", "'.join(_list) + '"]'

        _list = ['"'+str(abs_level)+'" : '+ list_to_doublequotes(list(concept_dict.keys()))  
                                       for abs_level, concept_dict in sorted(self.abstraction_groups.items())]
        return "{" + ', '.join(_list) + "}"
                                                    
class Card:
    """
    Each card is a question answer pair. 
    """
    
    def __init__(self, cardID, topic, question, answer, key_ideas, cardConceptHierarchy):
        self.cardID = cardID  # a unique integer given to each card 
        self.topic: str = topic  # effectively the label of the card
        self.question: str = question
        self.answer: str = answer
        self.key_ideas: str = key_ideas
        self.concepts: CardConceptHierarchy = cardConceptHierarchy   
        # self.retention: float = 0.0  # a number from 0 to 1 giving the estimated correctness if asked now
        # self.history: List[[float, float]] = []  # list of past times, and accuracy at that time, of testing 
        
        self.embedding_vector: Dict[str: float] = {}  
        self.embedding_vector_trimmed: Dict[str: float] = {}  # trimmed to not be as long of a dictionary if values are tiny
        
    def get_abstraction_from_1_to_2(self, concept1, concept2):
        abstractions_dict = self.concepts.get_abstractions_dict()
        return float(abstractions_dict[concept2] - abstractions_dict[concept1])
        
    def display(self, verbose=False):
        if not verbose:
            l1 = "Topic: " + self.topic
            l2 = "Question: " + self.question + ''
            l3 = "Answer: " + self.answer + ''
            print("\n".join([l1,l2,l3]) + "\n")
        if verbose:
            l1 = "Topic: " + self.topic
            l2 = "Question: " + self.question + ''
            l3 = "Answer: " + self.answer + ''
            l4 = "Key ideas:\n" + self.key_ideas
            
            print("\n".join([l1, l2, l3, l4]) + "\n")
            
    def update_embedding_vector(self, knowledgeGraph):
        """
        Embedding vector for card is a weighted sum of embedding vectors for the concepts in the card 
        
        Have to pass in knowledge graph as argument because card needs to know about nodes, in the knowledge graph
        """
        
        # Card concept list
        card_concepts_list = self.concepts.get_concepts_list()

        card_emb_vec = emb_vec_weighted_union_of_nodes(card_concepts_list, knowledgeGraph)
                
        self.embedding_vector = card_emb_vec # update stored value, then return it 
        self.embedding_vector_trimmed = trim_embedding_vector(card_emb_vec)
        
        return card_emb_vec

                
class Node:
    """
    Each node is a concept. 
    
    Main attributes are:
        title : title of node. This is the ID, and the concept of the node. Used for searching
        cards : list of the cardID of cards that touch this node 
        neighbors : a dictionary of neighboring nodes, with the list of cardIDs of cards connecting them 
        neighbors_relative_abstraction : a dictionary of neighboring nodes, with the list of the 
            relative abstraction from node to neighbor (higher means neighbor is more abstract). 
        neighbors_connection_strength : a dictionary of neighboring nodes, with the connection strength from this
            node to that node, from the perspective of this node (number of cards connecting to that node, out of total cards)
    """
    
    def __init__(self, title: str):
        self.title = title
        self.cards: Set[int] = set()  # contains references to card IDs that touch this node 
        self.neighbors: Set[str] = set()
        self.neighbors_card_connections: Dict[str: Set[int]] = {}  # contains references to neighbor nodes, and the set of IDs of cards that connect them 
        self.neighbors_connection_count: Dict[str: int] = {}  
        self.neighbors_connection_strength: Dict[str: float] = {}  
        self.neighbors_reverse_connection_strength: Dict[str: float] = {}  # from the neighbor's perspective
        
        self.neighbors_relative_abstraction: Dict[str: float] = {}  
        self.significance_vector: Dict[str: float] = {}  # vector of probabilities from 0 to 1 that two concepts are associated
        self.sum_of_significances_to_node: float = 0.0  
        self.sum_of_significances_from_node: float = 0.0  
        
        self.raw_embedding_vector: Dict[str: float] = {}  # vector of association strengths from significance and connection
        self.sum_of_raw_embeddings_to_node: float = 0.0  
        
        self.embedding_vector: Dict[str: float] = {}  # vector of network-wise association strengths 
        self.embedding_vector_trimmed: Dict[str: float] = {}  # trimmed to not be as long of a dictionary if values are tiny
        self.sum_of_embeddings_to_node: float = 0.0  # sum of all other node embedding vectors values of this node

    def display_raw_metrics(self, num_connections_required_for_display=1):
        strength_bar_size_display = 20  # width in number of characters 
        rel_abs_bar_size_display = 20
        display_title = ("-------------------------------------------------------------------------\n"+
                        "Relative Abstracton   Connection Strength  Rev. Conn. Strength  Concept\n"+
                        "-------------------------------------------------------------------------")
        sorted_relative_abstraction = dict(reversed(sorted(self.neighbors_relative_abstraction.items(), key=lambda item: item[1])))
        rel_abs_scale = np.max(np.abs(np.array(list(sorted_relative_abstraction.values())))) + 1e-13  # to prevent 0 scale

        print("Node:", self.title)
        print("   {} cards".format(len(self.cards)))
        print("   {} card threshold for display".format(num_connections_required_for_display))
        print(display_title)
        for neighbor_node_title, rel_abs in sorted_relative_abstraction.items():
            connection_strength = self.neighbors_connection_strength[neighbor_node_title]
            reverse_connection_strength = self.neighbors_reverse_connection_strength[neighbor_node_title]
            connection_count = self.neighbors_connection_count[neighbor_node_title]
            rel_abs_display_scale = rel_abs/rel_abs_scale

            if connection_strength >= min(self.neighbors_connection_strength.values())*num_connections_required_for_display:
                strength_bar = get_visual_display_bar_positive(connection_strength, strength_bar_size_display) 
                reverse_strength_bar = get_visual_display_bar_positive(reverse_connection_strength, strength_bar_size_display) 
                rel_abs_bar =  get_visual_display_bar_symmetric(rel_abs_display_scale, rel_abs_bar_size_display)
                
                print(rel_abs_bar, strength_bar, reverse_strength_bar, neighbor_node_title, '   ', connection_count)
                
    def get_neighbor_titles_with_similar_abstraction(self, abstraction_window_plus_minus=0.5):
        n_node_title_list = [n_node_title for n_node_title in self.neighbors
                            if np.abs(self.neighbors_relative_abstraction[n_node_title]) < abstraction_window_plus_minus]
        return n_node_title_list

    def get_sorted_neighbor_titles_by_abstraction(self):
        sorted_neighbor_titles = [str(k) for [k, v] in list(sorted(self.neighbors_relative_abstraction.items(), 
                                                                key=lambda item: item[1]))]
        return np.array(sorted_neighbor_titles)

    def dict_to_array(self, _dict):
        sorted_neighbor_titles = self.get_sorted_neighbor_titles_by_abstraction()
        return np.array([_dict[title] for title in sorted_neighbor_titles])
        
    def _get_predicted_connection_strength(self,  abstraction_window_plus_minus_for_avg=1.501):
        """
        This defines a window around each unique abstraction value, and calculates the average connection strength
            and number of samples in that window

        Returns two dictionaries for all datapoints (neighbors) contiaining the value, and sample count
        """

        def get_avg_connection_strength_in_abstraction_window(node, _low, _high):
            con_samples = [node.neighbors_connection_strength[_title] for _title in node.neighbors
                           if (node.neighbors_relative_abstraction[_title] <= _high and 
                               node.neighbors_relative_abstraction[_title] >= _low)
                          ]
            num_con_samples = len(con_samples)
            avg_con_val = np.average(np.array(con_samples))
            return avg_con_val, num_con_samples

        def get_avg_connection_strength_near_abstraction_value(node, abs_val):
            abs_window = [abs_val - abstraction_window_plus_minus_for_avg, abs_val + abstraction_window_plus_minus_for_avg]
            avg_connection_val, num_samples = get_avg_connection_strength_in_abstraction_window(node, abs_window[0], abs_window[1])
            return [avg_connection_val, num_samples]
        
        def get_avg_connection_strength_vs_unique_abstractions(node, unique_abstraction_vals):
            avg_connection_vals_and_num_samples = np.array([get_avg_connection_strength_near_abstraction_value(node, abs_val) 
                                        for abs_val in unique_abstraction_vals])
            avg_connection_vals = avg_connection_vals_and_num_samples[:,0]
            num_sample_cards = len(node.cards) * avg_connection_vals_and_num_samples[:,1]
            return avg_connection_vals, num_sample_cards

        node = self
        unique_abstraction_vals = sorted(list(set(node.neighbors_relative_abstraction.values())))
        avg_connection_vals, num_sample_cards = get_avg_connection_strength_vs_unique_abstractions(node, unique_abstraction_vals)
            # Note, num_sample_cards is the number of total cards involved in the averaging
            # so it's the number of words averaged together, times the number of cards in the node 
        
        if len(unique_abstraction_vals) > 1:
            # get interpolations
            avg_connection_vals_interp = scipy.interpolate.interp1d(unique_abstraction_vals, avg_connection_vals, kind="linear")
            num_sample_cards_interp = scipy.interpolate.interp1d(unique_abstraction_vals, num_sample_cards, kind="linear")

            # Evaluate for all data points 
            pred_val_dict = {_title: avg_connection_vals_interp(node.neighbors_relative_abstraction[_title]) 
                       for _title in node.neighbors}
            num_samples_dict = {_title: num_sample_cards_interp(node.neighbors_relative_abstraction[_title]) 
                       for _title in node.neighbors}
        else:  # there is just one value
            pred_val_dict = {_title: avg_connection_vals[0] for _title in node.neighbors}
            num_samples_dict = {_title: num_sample_cards[0] for _title in node.neighbors}

        return pred_val_dict, num_samples_dict

    def _calculate_LCB_association_significance(self, con_str, con_str_num_cards, pred_con_str, 
                                   pred_con_str_num_cards, lower_bound_epsilon=0.025):
        # Gets lower confidence bound for whether these concepts are significantly associated
        
        def get_beta_LCB(fraction_observed, n_measurements, lower_bound_epsilon):
            # beta.ppf is the inverse of the cdf distribution
            # it gives the value of the input to the beta distribution with these parameters where
            # the probability of fraction_observed being lower than this is lower_bound_epsilon

            prior_alpha = 1.0
            prior_beta = 1.0

            n_observed = fraction_observed * n_measurements
            lower_bound_value = scipy.stats.beta.ppf(lower_bound_epsilon, prior_alpha + n_observed, 
                                                     prior_beta + n_measurements - n_observed)

            return lower_bound_value

        def get_beta_cdf(test_value, n_measurements, pred_value):
            prior_alpha = 1.0
            prior_beta = 1.0

            pred_n_observed = pred_value * n_measurements
            cdf_probability = scipy.stats.beta.cdf(test_value, prior_alpha + pred_n_observed, 
                                                   prior_beta + n_measurements - pred_n_observed)
            return cdf_probability

        def get_probability_score(connection_strength, n_cards_for_connection, pred_connection_strength, 
                      n_cards_for_prediction, lower_bound_epsilon=0.025):
            lower_bound_value = get_beta_LCB(connection_strength, n_cards_for_connection, lower_bound_epsilon)
            cdf_probability = get_beta_cdf(lower_bound_value, n_cards_for_prediction, pred_connection_strength)
            return cdf_probability

        prob_score = get_probability_score(con_str, con_str_num_cards, pred_con_str, pred_con_str_num_cards, lower_bound_epsilon)
        return prob_score


    def update_significance_vector(self, abstraction_window_plus_minus_for_avg=1.51,
                                       lower_bound_epsilon=0.025):
        """
        Gets a detailed metric for the connection strength from this node to others.
        This metric is based on the lower confidence bound for association strength.
        
        Returns a dictionary with this score
            Does not include self in dictionary. 
        """

        window_pm = abstraction_window_plus_minus_for_avg
        num_cards = len(self.cards)  # number of cards touching this node 
        predicted_con_str, num_sample_cards = self._get_predicted_connection_strength(window_pm)
        con_str = self.neighbors_connection_strength
        
        significance_vector = {_title: self._calculate_LCB_association_significance(con_str[_title], num_cards, 
                                                                         predicted_con_str[_title], 
                                                                         num_sample_cards[_title], 
                                                                         lower_bound_epsilon)
                                      for _title in self.neighbors}
        
        self.significance_vector = significance_vector # update stored value
        self.sum_of_significances_from_node = np.sum(list(self.significance_vector.values()))
        
        return None
    
    def update_raw_embedding_vector(self, abstraction_window_plus_minus_for_avg=1.51,
                                       lower_bound_epsilon=0.025):
        self.update_significance_vector()
        
        # Now use significance vector and number of shared edges to associate concepts 
        self.raw_embedding_vector = {k: v * self.significance_vector[k]
                                 for k, v in self.neighbors_connection_strength.items()}
        
        return self.raw_embedding_vector

class KnowledgeGraph:
    """
    Will contain a dictionary of nodes of the graph, and a dictionary of cards in the graph which connect the nodes.
    
    Initialization process is create card, then add it to graph, then add to nodes, then update nodes based on card properties 
    """
        
    def __init__(
            self,
        lower_bound_epsilon=0.05,  # the significance level for determining if nodes are associated 
    ):
        self.nodes: Dict[str: Node ] = {} 
        self.cards: Dict[int: Card ] = {} 
        self.lower_bound_epsilon = lower_bound_epsilon
        
        self.node_embedding_num_maximum_passes_through_network = 100
            # Sets maximum number of times we'll loop through network when calculating node embedding
        self.node_embedding_update_fraction_condition_for_node_embedding_convergence = 0.01 
            # smaller threshold means slower convergence but more accuracy
        self.node_embedding_update_combination_power = 6  
            # higher power means change node embedding more slowly 
            # This leads to slower convergence, but better averaging over nodes
    
    def _add_card(self, topic, question, answer, key_ideas, cardConceptHierarchy):
        # initialize card
        newCardID = 1 + max(list(self.cards.keys()) + [-1])  # get max value plus one 
        card = Card(newCardID, topic, question, answer, key_ideas, cardConceptHierarchy)
        self.cards[newCardID] = card
        
        self._update_node_parameters_when_adding_card(card)
        
        node_titles_updated = set(card.concepts.get_concepts_list())
        return node_titles_updated
        
    def _update_node_parameters_when_adding_card(self, card):
        """
        # Adds nodes if necessary.
        Updates 
            node.cards set, 
            node.neighbors set of nodes, 
            node.neighbors_card_connections dictionary of cards to each node
            node.neighbors_connection_count dictionary of number of cards connecting to each node
            node.neighbors_connection_strength float of fraction of times mentioned together
            node.neighbors_reverse_connection_strength float of reverse connection (from neighbors perspective)
        """
        
        node_titles_to_update = list(set(card.concepts.get_concepts_list())) # remove duplicates
        for node_title in node_titles_to_update: 
            node = self.nodes.setdefault(node_title, Node(node_title))  # creates node it if not existing yet
            node.cards.add(card.cardID)  # Update card set for this node to include current card
            neighbor_node_titles_to_update = [title for title in node_titles_to_update 
                                              if title != node_title] # gather list of others only
            
            for n_node_title in neighbor_node_titles_to_update:
                node.neighbors.add(n_node_title)
                
                # Update card connection set. Have to add default in case we never connected before
                card_connections = node.neighbors_card_connections.setdefault(n_node_title, set())
                card_connections.add(card.cardID)
                node.neighbors_connection_count[n_node_title] = len(card_connections)
                
            # For connection strength, we have to iterate over all neighbors of the node
            for n_node_title in node.neighbors_card_connections.keys():
                connection_strength = float(node.neighbors_connection_count[n_node_title])/len(node.cards)
                node.neighbors_connection_strength[n_node_title] = connection_strength

        # Finally update the stored reverse connection strengths (it's not a symmetric metric)
        for node_title in node_titles_to_update: 
            node = self.nodes[node_title]
            for n_node_title in node.neighbors:
                n_node = self.nodes[n_node_title]
                n_node_strength_to_node = n_node.neighbors_connection_strength[node_title]
                node.neighbors_reverse_connection_strength[n_node_title] = n_node_strength_to_node
                
    def _get_node_neighbor_rel_abstraction_over_card_list(self, cardIDList, concept1, concept2):
        # This method is in the knowledge graph because it requires information from cards
        relative_abstractions = [self.cards[_cardID].get_abstraction_from_1_to_2(concept1, concept2)
                for _cardID in cardIDList]
        return np.average(np.array(relative_abstractions))
    
    def _recalculate_relative_abstraction(self, node_titles_to_update, verbose=False):
        # Loop through updated nodes and recalculate relative abstraction 
        start_time = time.time()
        for _ind, node_title in enumerate(node_titles_to_update): 
            if verbose:
                if _ind % 200 == 0:
                    print('   Node number: ', _ind, ', Title: "{}" at time '.format(node_title),  
                          np.round(time.time() - start_time,2))
                    
            node = self.nodes[node_title]
            for n_node_title in node.neighbors:
                # Update relative abstraction 
                card_connections = node.neighbors_card_connections[n_node_title]
                rel_abs = self._get_node_neighbor_rel_abstraction_over_card_list(card_connections, 
                                                                   node_title, n_node_title)
                node.neighbors_relative_abstraction[n_node_title] = rel_abs
    
    def add_card_deck(self, card_deck, verbose=False):
        # card_deck is a list of card meta data for input to add_card
        
        # Add basic info of cards 
        node_titles_updated = set()
        for card_data in card_deck:
            node_titles_updated_this_card = self._add_card(*card_data)
            node_titles_updated.update(node_titles_updated_this_card)
        
        # Loop through updated nodes and recalculate relative abstraction 
        print("Recalculating relative abstraction") if verbose else None
        self._recalculate_relative_abstraction(node_titles_updated, verbose=verbose)
        
        # Update raw embedding vector for these nodes
        print("Updating raw embedding vectors") if verbose else None
        start_time = time.time()
        for _ind, node_title in enumerate(node_titles_updated):
            if verbose and _ind % 200 == 0:
                print('   Node number: ', _ind, ', Title: "{}" at time '.format(node_title),  
                          np.round(time.time() - start_time,2))
            self.nodes[node_title].update_raw_embedding_vector(lower_bound_epsilon=self.lower_bound_epsilon)
            
        # Update all counts of significances to each node. This should be fast, so do it over whole graph
        for node in  self.nodes.values():
            node.sum_of_significances_to_node = 0.0 
            node.sum_of_raw_embeddings_to_node = 0.0 
            for k, v in  node.significance_vector.items():
                node.sum_of_significances_to_node += v
            for k, v in  node.raw_embedding_vector.items():
                node.sum_of_raw_embeddings_to_node += v
        
        return node_titles_updated
    
    def update_all_node_embeddings(self,
                                   allow_reusing_existing_node_embedding=True,  # saves computation a ton
                                   verbose=False,
                                  ):
        """
        Get embedding vectors from raw embeddings 
            This effectively measures and builds the network connectivity based on global structure
            Final embedding to neighbor gets increased if neighbors share many mutual neighbors 
        """

        num_maximum_passes_through_network = self.node_embedding_num_maximum_passes_through_network
        update_fraction_condition_for_node_embedding_convergence = self.node_embedding_update_fraction_condition_for_node_embedding_convergence
        update_combination_power = self.node_embedding_update_combination_power

        # Use trimmed raw embedding vectors of other nodes for calculating overlap 
        # This speeds up computation by 10x roughly
        raw_embedding_vectors_trimmed = {node_title: trim_embedding_vector(node.raw_embedding_vector)
                      for node_title, node in self.nodes.items()}

        node_titles_and_sum_of_sigs = self.get_node_titles_and_sum_of_significances_to_node_decreasing_order()

        if verbose:
            print('Updating all node embeddings:')
        start_time = time.time()
        for _ind, (node_title, sum_of_sigs) in enumerate(node_titles_and_sum_of_sigs):
            if verbose:
                if _ind % 200 == 0 or time.time() - prev_time > 1.0:
                    print('   Node number: ', _ind, ', Title: "{}" at time '.format(node_title),  
                          np.round(time.time() - start_time,2))
            prev_time = time.time()

            node = self.nodes[node_title]

            # Get order to loop through neighbors
                # Want to loop through by order of raw embedding in relation to this card, least embedding first. 
            ordering_dict = node.raw_embedding_vector
            sort_inds = np.argsort(list(ordering_dict.values()))
            sorted_neighbor_names = np.array(list(ordering_dict.keys()))[sort_inds] 

            # Initialize embedding vector 
            if len(node.embedding_vector) > 0 and allow_reusing_existing_node_embedding:
                new_emb_vec = node.raw_embedding_vector.copy()  # have to copy because we maybe introduced new elements 
                new_emb_vec.update(node.embedding_vector.copy())  # uses old embedding vector as default values
            else:
                new_emb_vec = node.raw_embedding_vector.copy()  # to be filled 
            new_emb_vec_current_total = np.sum(list(new_emb_vec.values())) 
                # Keep track of vector running sum rather than recalculating inner product (which is slow)

            # Loop through nodes and update embedding
                # need to do multiple passes over all nodes to converge to a new node representation. 
            for pass_ind in range(num_maximum_passes_through_network):  
                total_update_this_pass = 0.0  # keep track how much we have updated, to check convergence
                for n_node_title in sorted_neighbor_names: 
                    n_node_raw_emb_vec_trimmed = raw_embedding_vectors_trimmed[n_node_title]
                    n_node_raw_emb_sum_to_n_node = self.nodes[n_node_title].sum_of_raw_embeddings_to_node 

                    # Calculate normalized overlap (this is like the new target embedding)
                    overlap = min(1, emb_vec_inner_product(new_emb_vec, n_node_raw_emb_vec_trimmed) / 
                                  new_emb_vec_current_total)
                    # Want to rotate slowly to new overlap, so average strongly with previous value
                    overlap_reduced = ((overlap * new_emb_vec[n_node_title] ** (update_combination_power - 1)) 
                                       ** (1.0/ update_combination_power))
                    updated_value = overlap_reduced / (1.0 + n_node_raw_emb_sum_to_n_node) 
                        # The denominator kills major concepts, since they're not very useful for embedding

                    # Update parameters 
                    total_update_this_pass += np.abs(updated_value - new_emb_vec[n_node_title])
                    new_emb_vec_current_total += updated_value - new_emb_vec[n_node_title]
                    new_emb_vec[n_node_title] = updated_value

                # Check for convergence and break if done
                if ((total_update_this_pass / new_emb_vec_current_total) 
                    < update_fraction_condition_for_node_embedding_convergence):
                    break  

            node.embedding_vector = new_emb_vec
            node.embedding_vector_trimmed = trim_embedding_vector(new_emb_vec)
            
        # Update the sum of embeddings to each node
        for node in  self.nodes.values():
            node.sum_of_embeddings_to_node = 0.0
            for k, v in  node.embedding_vector.items():
                node.sum_of_embeddings_to_node += v
        
    def update_all_embeddings(self,
                              allow_reusing_existing_node_embedding=True,  # saves computation a ton
                              verbose=False,
                             ):
        
        # Update all nodes
        self.update_all_node_embeddings(allow_reusing_existing_node_embedding=allow_reusing_existing_node_embedding,
                                        verbose=verbose)
                
        # Update all cards 
        for card in self.cards.values():
            card.update_embedding_vector(self)
    
    def get_node_titles_and_sum_of_significances_to_node_decreasing_order(self):
        node_titles_and_sum_of_significances_to_node = [(node.title, node.sum_of_significances_to_node) for node in self.nodes.values()]    
        sum_of_significances_to_node = [node.sum_of_significances_to_node for node in self.nodes.values()]    
        sum_of_significances_to_node_sorted_inds = np.flip(np.argsort(np.array(sum_of_significances_to_node)))
        sorted_node_titles_and_sum_of_significances_to_node = np.array(node_titles_and_sum_of_significances_to_node)[sum_of_significances_to_node_sorted_inds]
        return sorted_node_titles_and_sum_of_significances_to_node

    def get_dict_of_emb_vec_inner_product_over_nodes(self, emb_vec, useTrimmed=True):
        if useTrimmed:
            overlap_dict = {node_title: emb_vec_inner_product(emb_vec, node.embedding_vector_trimmed)
                           for node_title, node in self.nodes.items()}
        else:
            overlap_dict = {node_title: emb_vec_inner_product(emb_vec, node.embedding_vector)
                           for node_title, node in self.nodes.items()}        
        return overlap_dict

    def get_dict_of_emb_vec_inner_product_over_cards(self, emb_vec, useTrimmed=True):
        if useTrimmed:
            overlap_dict = {cardID: emb_vec_inner_product(emb_vec, card.embedding_vector_trimmed)
                           for cardID, card in self.cards.items()}
        else:
            overlap_dict = {cardID: emb_vec_inner_product(emb_vec, card.embedding_vector)
                           for cardID, card in self.cards.items()}
        return overlap_dict
    
    def display_object_overlaps(self, input_object):
        # input_object should be a node or card

        if isinstance(input_object, Node):
            object_type = 'Node'
            object_title = input_object.title
            emb_vec = input_object.embedding_vector
        elif isinstance(input_object, Card):
            object_type = 'Card'
            object_title = input_object.topic + " ID:" + str(input_object.cardID)
            emb_vec = input_object.embedding_vector
        else:
            raise Exception("Trying to display an object that is not allowed")

        overlap_dict_nodes = self.get_dict_of_emb_vec_inner_product_over_nodes(emb_vec)
        overlap_dict_cards = self.get_dict_of_emb_vec_inner_product_over_cards(emb_vec)

        sorted_node_titles, sorted_node_overlaps = get_dict_items_sorted_by_decreasing_value(overlap_dict_nodes)
        sorted_cardIDs, sorted_card_overlaps = get_dict_items_sorted_by_decreasing_value(overlap_dict_cards)

        def plot_overlaps_vs_names(xvals, yvals, xlabel, title, num_display):
            fig, ax = plt.subplots(1, figsize=(12, 2))
            ax.set_title(title)
            ax.scatter(xvals[0:num_display], yvals[0:num_display], marker = 'o', color = 'blue', s=4, label='')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Overlap')
            plt.xticks(rotation=85)
            plt.ylim([0,1.05*np.max(yvals[0:num_display])])
            plt.show()

        def plot_overlaps_histograms(node_overlaps, card_overlaps, title, num_bins=40):
            fig, ax = plt.subplots(1, 2, figsize=(8, 2))
            ax[0].hist(node_overlaps, bins=num_bins)
            ax[0].set_xlabel('Node overlap')
            ax[0].set_ylabel('Counts')
            ax[0].set_yscale('log')
            ax[1].hist(card_overlaps, bins=num_bins)
            ax[1].set_xlabel('Card overlap')
            ax[1].set_ylabel('Counts')
            ax[1].set_yscale('log')
            ax[0].set_title(title)
            plt.show()

        title = 'Histogram of overlaps for ' + object_type + ' "'+ object_title + '"'
        plot_overlaps_histograms(sorted_node_overlaps, sorted_card_overlaps, title)

        # Visualize the results
        num_display = 60
        xvals = sorted_node_titles
        yvals = sorted_node_overlaps
        xlabel = 'Neighbor node title'
        title = 'Node overlaps for ' + object_type + ' "'+ object_title + '"'
        plot_overlaps_vs_names(xvals, yvals, xlabel, title, num_display)

        # Visualize the results
        num_display = 60
        xvals = [self.cards[cardID].topic + ' ID:' + str(cardID) for cardID in sorted_cardIDs[0:num_display]]
        yvals = sorted_card_overlaps
        xlabel = 'Card topic and ID'
        title = 'Card overlaps for ' + object_type + ' "'+ object_title + '"'
        plot_overlaps_vs_names(xvals, yvals, xlabel, title, num_display)
        
        
def create_card_deck_from_dataframe_of_abstraction_groups(cards_df_abstraction_groups):
    card_deck = []
    
    # Add cards 
    print('Adding {} cards'.format(len(cards_df_abstraction_groups)))
    for card_ind in range(len(cards_df_abstraction_groups)):
        abstraction_groups = cards_df_abstraction_groups['Abstraction groups'].values[card_ind]        
        topic = abstraction_groups['0'][0]  # Extracts abstraction level 0, then first element.
        question = cards_df_abstraction_groups["Question"].values[card_ind]
        answer = cards_df_abstraction_groups["Answer"].values[card_ind]
        key_ideas = cards_df_abstraction_groups["Key ideas"].values[card_ind]
        
        # construct card concept heirarchy
        cardConceptHierarchy = CardConceptHierarchy(topic, "")
        concept_lists = []
        for k, v in abstraction_groups.items():
            if k != '0':
                concept_lists.append([list(v), int(k)])

        # Now make concept hierarchy 
        for concept_list, level in concept_lists:
            for concept in concept_list:
                cardConceptHierarchy.set_concept(level, concept, "")
        
        card_deck.append([topic, question, answer, key_ideas, cardConceptHierarchy])
    
    return card_deck 