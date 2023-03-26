# A knowledge graph from GPT-3
 
## High-level description
This program is meant to create an external memory module for a language model, and ultimately provide 
agent-like capabilities to a language model (long-term goal).
- The combined system would ideally gather information via a text interface, categorize and structure it, then identify gaps in knowledge, 
or inconsistencies.
- The language model can be shown subsets of this information (chosen based on the structure of 
the knowledge graph), and then choose to propose further questions to ask the environment as clarification, 
building knowledge over time.
  
### Goals

This project aims to address a few of the major shortcomings of language models:
* Memory 
  * External memory solves the problem of lack of long-term learning from one or two examples.
* Logic 
  * Language models are generally unable to force structured responses, 
  but recalling logical arguments from memory could help. 
* Interpretability 
  * By observing what memories are accessed, we can understand sources of resulting statements and information flow.
  * Alternatively, by observing how the language model processes and categorizes information, we can understand the 
  inherent structure of the information learned by the raw language model.
* Developing agency
  * Language models lack coherent agency, as language models generate both sides of a conversation.
  * Structuring the language model as a component in a reinforcement learning system, with the goal 
  of categorizing and uncovering information, restores agency.
* Computational resource use for training 
  * Can we continuously improve the entire machine learning system (model and RL wrapper), without 
  continuously retraining the model parameters? Simply by recursively improving the memory (which is re-inserted
  through the prompt).
  * Language models could be trained for specialized sub-component tasks in the resulting global system.
* Bootstrapping capabilities
  * Is a minimal level of reason and analogy sufficient to tackle arbitrarily complex processing of knowledge,
  by breaking down ideas into minimal components, and treating each separately?
  * There are probably opportunities here for bootstrapping and policy improvement of the language model,
    through self-generated examples (as used below in extracting question embeddings, and 
  generating questions from examples of clusters of questions). 
  
### Target uses long term
1. Database generation and parsing + question answering
    * Summarize a research field or class notes or textbook
    * Identify conflicting information and disputes,
   or different explanations for the same topic or idea

2. Educational tool or personal learning tool
    * Construct the agent to serve as a spaced-repetition flashcard assistant.
      * Learn what knowledge the user has, and how quickly they forget it, then periodically reprompt them. 
      * Learn to suggest new information for the user to learn, tailored to their current knowledge and interests. 
      * Do everything with a flexible, natural language interface to pose questions and interpret responses.
    * A structured eucational tool
      * Fix the knowledge in the graph (distilled from experts), then use the knowledge structure and spaced-repetition
      framework to understand a student's learning needs, and interface with them.

3. Hypothesis generation for scientific research 
    * Process entire scientific fields, including papers, textbooks, audio lectures, etc.
    * Come up with novel ideas and research proposals 

### Outline of the program 

The program is designed as a wrapper for a language model in python.
The knowledge graph (stored in python) makes periodic calls to the language model when necessary.

A key feature is that the structure of the knowledge graph is fully-human interpretable.
Even the embeddings of information, and the facts themselves, are in natural language. 
Moreover, all steps of the algorithm are observable to the user (what information is referenced, and why).

Steps:
1. Extraction
   * The language model is shown examples of "flashcards" (minimal question/answer pairs in a field of knowledge) 
   and extracts an embedding of the hierarchy of natural language concepts that the fact relates to 
   (ie. science, biology, cells, DNA...).
2. Embedding
   * The knowledge graph program looks at the combined set of all embeddings for all known information, and 
   constructs a human-interpretable vector embedding of each flashcard. 
      * These vector embeddings are in natural language
        * The "dimensions" of the embedding vector are the names of words or concepts in the graph
          (ie. there is a dimension for "Science" and a dimension for "DNA")
      * The overlap of embedding vectors is high for similar concepts and similar facts, allowing clustering of knowledge. 
      * The embeddings are tailored to the local structure of learned knowledge 
     (they depend on all facts learned, not "the internet" or some other database).
3. Clustering and Structuring
   * Using embedding vectors, facts and concepts can be clustered hierarchically, giving a natural knowledge structure 
   for search and exploration. 
4. Question answering
   * Given a novel question from the environment (ie. a query to the database), the model 
   can extract a tailored natural language embedding (the hierarchy of concepts)
     * We can take advantage of few-shot learning here to extract these concepts, by just showing 
     examples of previous questions + extracted concepts, then prompting with the new question and asking for the 
     extracted concepts following the same style. 
     * This provides opportunities for recursive self-improvement. 
   * This question embedding can be used to identify relevant knowledge in the graph
   * The language model can observe all relevant learned knowledge, 
   and the current question, to answer the current question.
5. Hypothesis generation
   * Once the knowledge in the graph is structured (clustered), the language model can use 
   clusters of existing questions as inspiration (few shot examples) for generating further clusters of questions.
   * By showing the model a few sets of 5 related scientific questions,
   then prompting with 4 new (but related) questions, we can generate conceptually new questions in the style of 
   a scientific researcher. 

Future work is required to really make the program agent-like, by choosing what to do hierarchically,
and where to explore further. See some suggestions below.


## Details of the program: Constructing the knowledge graph

### Initial concept extraction and embedding
1. Concept extraction from question/answer pair:
![Alt text](docs/ConceptExtraction.jpg?raw=true "Optional")
    - See files for the explicit chained prompts used to extract the few-word concepts. 
    - The basic idea is it extracts some information, then reprompts the language model with that
   extracted information as part of the context, thereby progressively refining the concept hierarchy. 
2. Concept embedding (raw embedding then detailed embedding)
![Alt text](docs/ConceptEmbedding.jpg?raw=true "Optional")
   - Step 1: First we extract the raw card-concept connections, which are just a list of how often two concepts appear together, 
   as a fraction of the total time they appear.
     - The connection strength from one concept to another is the fraction of the time the concepts appear together 
     (specifically the fraction of times the first concept appears where the second concept is present).
   - Step 2: Next, we extract a raw concept embedding, which is the statistical significance of the connection between two concepts is measured.
     - This is done in a bit of a complicated way, which is probably not necessary. Feel free to skip on a first read through.
         - For each concept, all of it's neighboring concepts are ranked by their relative abstraction (measured in the card concept hierarchy).
         - Then at each level of abstraction, we find the average connection strength within an abstraction window 
         - This average connection strength defines a beta distribution on the expected observed connection strengths for any two concepts. This allows us to identify outliers.
         - Finally, we find the statistical outliers on the connection strength 
         (those concepts that appear together more often than expected, with a certain threshold).
         - Something like a quasi-"upper confidence bound" probability is calculated to get the embedding connection strength.
           - This says, even if I happen to be very wrong in my expected average connection strength and it's much higher
             (in the worst case scenario, say 5% of the time), what might my average connection strength actually be?
           - Then, even in that worst case scenario, what is the fraction of expected connection strengths which is smaller than my observed connection strength.
           - Thus if a observed connection strength is larger than 90% of all worst case expected connection stengths, it's definitely significant.
         - The ultimate embedding connection strength is then the statistical weight of the upper confidence bound beta distribution below the observed connection strength.
   - Step 3: We gather the final concept embeddings (long-range) based on the raw concept embeddings from step 2. 
     - This is once again somewhat complicated, but probably more necessary.
     It allows us to have a high-quality similarity metric between embeddings (it's just something I hacked together to have the 
     desired properties).
       - In brief, each final concept embedding is designed so that its overlap with its neighbor concepts' raw 
       embeddings is self-consistent, and representative of the local structure of the raw embedding graph, in some sense.
       - The final concept embedding has the following property
         - The final embedding magnitude at each neighboring concept is equal to the fractional projection of the 
         the neighboring concept raw embedding onto the entire final embedding vector. 
         - Geometrically, this means that the final embedding vector is more or less pointing along an 
         "average" direction of its neighbor raw embedding vectors, weighted by their original relevance.  
       - What this does in practice is significantly extend the conceptual reach of the embedding
       - of each concept, since it now includes components that are really "second order" concept connections.
       - To be honest, I don't have this process fully figured out (why it works well), but it does.
3. Card embedding based on concepts
![Alt text](docs/CardEmbedding.jpg?raw=true "Optional")
    - The card embedding is simple. It is a weighted sum of the concept embedding vectors from the concepts identified in the card.
    - There is one added complication: we punish the presence of "common" concepts, because they aren't helpful in identifying
   the structure of knowledge (knowing that most of the concepts are related to "science" isn't very helpful).
        - To do this, concept vectors are summed, but are weighted inversely to their prevalence in the graph.
        - Finally, after summing the concept vectors, individual components are once again divided by their prevalence in the graph
       in order to further favor the presence of unique concepts in the resulting vector. 

### Structuring the knowledge graph
The natural language embeddings of each concept or card allow one to cluster concepts through various 
methods. 

1. Defining the similarity metric
> ![Alt text](docs/SimilarityMetric.jpg?raw=true "Optional")
Here is an example matrix of all card-card similarities in the knowledge graph.
![Alt text](docs/ExampleSimilarityMatrix.jpg?raw=true "Optional") 
Here is an interactive clustering visualization.
![Alt text](docs/ClusteringDemonstration.jpg?raw=true "Optional") 
 

2. Example similarity calculations
    - ![Alt text](docs/ExampleNodeNodeOverlap.jpg?raw=true "Optional")  
    - ![Alt text](docs/ExampleNodeCardOverlap.jpg?raw=true "Optional")  

3. Clustering ideas (question/answer pairs) based on the similarity metric. 
   - Here are some examples of the results of clustering cards. 
     - These examples are 
     phrased as a "family tree" of clusters (raw cards get grouped into small clusters, then larger clusters, and so on. 
     The raw cards are children, first order clusters are parents, second order clusters are called grandparents, etc.)
   ![Alt text](docs/ExampleClusterHierarchies.jpg?raw=true "Optional")
   - The clustering algorithm is just something I hacked together, and isn't very polished (but works reasonably well).
     - See the code for details, but I'll describe the rough idea here. The rough idea is the following:
       - Step 1: find approximate clusters around each card
         - For each card, it finds nearby cards by order of similarity.
         - It proposes using the first k cards as a cluster.
         - The approximate "cluster quality" is calculated. 
           - This is a metric which is high if all the cards have a high similarity to the target card, and if there are more cards.
           - However, it is penalized if any of the cards have a very low similarity to some other card (in particular, 
           it penalizes the cluster based on the minimum similarity measured between cards). 
           - There is a control parameter for the target cluster size, which sets the tradeoff between self similarity and lack of dissimilarity.
         - As the proposed cluster gets larger, it may include more similar cards (and thus have a better cluster metric.)
         However, if any proposed card has a bad similarity to some other card, that quickly kills the cluster quality.
         - Finding the optimal cluster metric, we are left with a cluster of cards which are similar to each other, and none of which are particularly dissimilar to each other.
         - Lastly we clean up the cluster by testing adding or removing a few individual cards (out of order), and seeing 
         if this improves the cluster quality, until this converges. 
         - We return a target cluster around the given card.
       - Step 2: Vary the target cluster size and find the cluster with the highest global self similarity (a new metric).
         - In this step, we find a meta-metric for the cluster quality (again dependent on how self-similar the cluster is, but using the step 1 
         cluster metric as a way to define proposed clusters).
         - We perform step 1 for both the target card, and all of it's identified cluster partners. This is effectively checking
         not only what cluster we identify, but also reflexively at second order what those cards identify.
         - If the proposed cluster partners all identify the same cluster of cards as the target card identifies, that's great. 
         It's a well isolated cluster.
         - On the other hand, if the proposed cluster partners identify their own clusters which are much bigger, or 
         much smaller than the target card's cluster, then the proposed cluster isn't well isolated.
         - The final meta-metric increases with cluster size, and decreases when the reflexive proposed clusters differ a lot from the target cluster. 
       - Step 3: Find the optimal cluster for each card, and repeat to form a cluster hierarchy.
         - At each level of clustering, we take the clusters of cards from the previous level and treat those
         clusters as one big "card" (a combination of all the concepts in all of the cards in the cluster) 
         and combine all their concept embeddings to get an embedding that represents the cluster.
         - We can then cluster those cluster embeddings, and so on.
     - The main point here is that some kind of clustering is readily possible. 
       - Probably my way isn't efficient or optimal quality, but it gets the job done as a proof of concept.

### Querying the knowledge graph
To answer a question about facts in the knowledge graph, we determine the component concepts of a question
and then gather similar cards to this question to re-expose to the language model at answering time.
* Note: we do NOT use the same prompts as the original flashcard concept extraction. 
  * Instead, we leverage the work already done by the model (the examples of concept extraction from questions)
  to serve as examples for rapid concept extraction in the same style.

1. Construct a question embedding
![Alt text](docs/QuestionEmbedding.jpg?raw=true "Optional")
![Alt text](docs/ExampleQuestionEmbeddingExtraction.jpg?raw=true "Optional")
    - This few-shot prompting teaches the model to extract concepts in the correct level of detail
      (going from most abstract to least abstract concepts), and to use the correct format (capitalize first letter, 
   and stick to one or two words usually)
    - The two-stage reprompting also further allows the model to see what concepts are extracted from similar cards,
   which further improves the quality of the question embedding.
     
2. Gather similar cards 
    1. In the simplest case, this can just mean take the top k cards ranked via the similarity metric (ie the top 10 cards or so).
3. Answer the question by re-prompting the language model while showing related information
![Alt text](docs/QuestionAnswering.jpg?raw=true "Optional")
![Alt text](docs/ExampleQuestionAnswering.jpg?raw=true "Optional")
4. Note that, during this final answer, we can choose whether to allow the language model to use outside knowledge, or only information directly  
within the re-prompted cards.

### Question generation 
This is not fully implemented yet, but has a basic demonstration. In the future, this would ideally 
include hypothesis generation for scientific research.

We essentially use the measured knowledge structure to gather clusters of existing questions, on similar topics,
and use them as few-shot examples for generating further questions in a similar style.

![Alt text](docs/ExampleQuestionGeneration.jpg?raw=true "Optional") 

Furthermore, we can change the prompt to display (and ask for) questions with increasing or decreasing level of abstraction.
This is possible because the flashcards' component concepts are originally originally extracted in order of abstraction,
so we can measure this and incorporate it into the structure of the graph.

Long term, this process is promising as a method of recursive self-improvement.
If we can get the model to ask and generate good questions, and if we can identify the best clusters of scientific questions
in the knowledge graph to use as generative examples, the entire combined system should recursively self improve
(think of AlphaGo zero). 

### User interface
March 2023: I added a user interface which can be launched to interactively explore questions, then
generate answers, then save them to the knowledge graph.

Purpose: Use interface to explore related topics and intriguing questions.

General workflow:
- First select the topic, and goal command, then use the "generate new questions" button to generate some number
of new questions the user wants to explore.
- Second look at the generated questions, and choose which ones to keep or not.
- Third, gather related questions in the graph so that LLM knows what you already learned.
- Generate more questions, and repeat triage process, until you have a set of questions you want answered.
- Lastly, answer the new questions, then edit the answers, then save to the knowledge graph and long term storage. 

Example screenshots:

>Generating questions initially, and finding related information in graph:
![Alt text](docs/UserInterface_GeneratingQuestions.jpg?raw=true "Optional") 

>Selecting which questions to keep, or remove, and repeatedly regenerating new questions:
![Alt text](docs/UserInterface_RefiningQuestions.jpg?raw=true "Optional") 

>Finally "Answering new questions" using ChatGPT, then editing and "Save and Reset" to 
load this information into the knowledge graph, and save in long term data.
![Alt text](docs/UserInterface_AnsweringQuestions.jpg?raw=true "Optional") 


## Future extensions

### Agent-like behavior and exploration
The ultimate goal here is to make a self-improving agent that can explore and augment its conceptual environment 
using reinforcement learning and fast algorithms which process the information in the knowledge graph, with only sporadic calls 
to the language model to actually structure the information. 

Ideally the knowledge graph can be parsed via reinforcement learning algorithms to identify gaps in knowledge, and clusters of knowledge,
to then determine what areas to explore further and ask questions about. Then the language model can
be used to actually generate meaningful and conceptually different questions, which can be asked to the environment (ie. the internet) 
to gather more information. 

This will require some notion of "value" for the information in the knowledge graph (to balance pure exploration with 
exploration of relevant concepts). 
It will probably also be necssary to set up self pruning and revisiting of the knowledge graph, when in an idle state (like sleeping).

### Structured and hierarchical question-answering
For improved quality of question answering, it would be good for the language model to choose to
break down a question into a multi-stage question prompt with sub-questions.
These questions and sub-questions and their ultimate answers could be added to the knowledge graph as memory.

### Specializing the language model
It is possible to train the language model to improve itself based on examples 
(for example when extracting the concept hierarchy). 
One could have a dedicated smaller network for this (similar to the dedicated internal policy and value networks in alphaGo zero).

### Set up as a personal learning assistant 
Can we build a smart spaced-repetition memory system for learning with language models?
* Incorporate spaced repetition and probability of memory. 
* Generate complex and organic questions to test concepts within a cluster of cards.
* Learn some parameters for you (your personal rate of forgetting. Your interests and base knowledge).
* Have preset expert knowledge structures to learn, and teach (ie a module of cards on statistics).
* Have a nice way of visualizing your personal knowledge graph by clustering.

Additional functionality:
* The knowledge graph could be set up as a plugin to always ask you to summarize what you read the last 10 minutes.
* It could interface with OpenAI whisper or something analogous to do speech to text and then summarize.

### Long term foreseen issues
* It would probably be helpful to use synonym directory to help collapse nodes (concepts).
* How to handle other datatypes other than text? How to handle very short or very long texts (flashcards) or concepts (things like math, or huge formulas?).



## Final comments, additional references, thoughts 
There are a variety of related concepts floating around on the internet. 
I want to give a rough pointer to them here (not comprehensive).

Embedding vectors and figuring out how to reference memory are common concepts. 
These are a few examples I came across after the fact: 
* OpenAI embeddings
* GPT-index: https://github.com/jerryjliu/gpt_index

There are a variety of apps and startups and research projects trying to do similar things, or parts of this: 
* Startups to build personal knowledge graphs or notes assistance (I believe using language models).
* Tools to summarize articles and information into discrete facts (could be used as a sub-component in future work)
* There are various research directions using re-exposure of facts to a language model to improve quality. 
For example, non-parametric transformers, and various other transformer architectures which save memory. 
* There are attempts to build systems for structured question answering through breaking down into components, such as 
the factored cognition primer - https://primer.ought.org/.

Hinton suggests that using a hierarchy of concepts is crucial: https://arxiv.org/abs/2102.12627 ("How to represent part-whole hierarchies in a neural network").

### Unique(ish) contributions
Just to summarize, a few of the more unique aspects here in my implementation are (to my knowledge, though I 
probably just haven't looked enough):
* Embedding structure - quality and interpretability
    * The use of local knowledge structure to construct the embeddings 
    (instead of a learned embedding from the internet). This may be more accurate/expressive. 
    * The use of natural language in the embeddings (embedding dimensions are "words", not unlabeled axes).

* Policy improvement through experience
  * Question generation based on extracted structure in the local graph. 
  * Question embedding extraction based on past examples.
  * Generally the idea that we might get very far by leveraging language models for subcomponents
  for basic reasoning within a larger system (this idea is present in a lot of current robotics research). 

* The goal of building an agent and independent research assistant, or the goal to build a 
personal assistant in learning and memorization, seems
slightly orthogonal to most efforts.
