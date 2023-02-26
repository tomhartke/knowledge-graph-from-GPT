# A knowledge graph from GPT-3
 
## High-level description
This program is meant to serve as an external memory module for a language model, and ultimately provide 
agent-like capabailities to a language model (long-term goal).
* The language model + knowledge graph can categorize and absorb facts, and quickly update beliefs given new observations. 
* The structure of the knowledge graph is fully-human interpretable. 
  * Even the embeddings of information, and the facts themselves, are in natural language.
* Given a question to the model, it can identify the most related facts, and re-expose them to the model 
when answering the question. 
  * The identical model (model paramters) can thereby learn to change it's behavior with one example.  

Future extensions to increase agency:
* An external reinforcement learning algorithm could identify gaps in knowledge, or inconsistencies, and show these to the language model.
  * This could be fast, without language model queries.
* The language model could then propose further questions to ask the environment as clarification, 
building knowledge over time. 

### Goals:
(To be clarified)

Issues it's addressing:
* Memory 
* Logic 
* Interpretability
* Reinforcement learning with the environment through repeated calls/answers
* Computational resource use for training (knowledge updates are external, no retraining the model)

What it's not addressing:
Hardware of the language model is fundamentally still fixed. 
The model could be further fine tuned to mimic the behavior of the enhanced model + knowledge graph 
(similar to policy improvement in Monte-Carlo tree search).

#### Uses
1. Database generation and parsing + question answering
    * Summarize a research field or class notes or textbook
    * Identify conflicting information and disputes
    * Categorize different explanations for the same topic or idea 
   (ie different quantum mechanics textbooks) and offer them to people

2. Educational and personal learning tool
    * Spaced-repetition flashcard assistant.
    * Can have expert knowledge systems (ie areas of knowledge like statistics) encoded as knowledge graphs.

3. Hypothesis generation for scientific research 
    * Come up with ideas from scratch
    * Read a set of papers and find gaps in knowledge

### Constructing the knowledge graph
1. Concept extraction from question/answer pair:
![Alt text](docs/ConceptExtraction.jpg?raw=true "Optional")
    - For explicit prompts, see files. 
    - The basic idea is it extracts some information, then reprompts the language model with that
   extracted information as part of the context, thereby progressively refining the concept hierarchy. 
2. Concept embedding (raw embedding then detailed embedding)
![Alt text](docs/ConceptEmbedding.jpg?raw=true "Optional")
   - Step 1: The raw card-concept connections are just a list of how often two concepts appear together.
     - The connection strength from one concept to another is the fraction of the time the concepts appear together 
     (specifically the fraction of times the first concept appears where the second concept is present).
   - Step 2: Next, the statistical significance of the connection between two concepts is measured.
     - This is done in a bit of a complicated way, which is probably not necessary. 
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
   - Step 3: Gather long range connection embeddings based on the metric from step 2. 
     - This is once again somewhat complicated, but probably more necessary.
     - (Need to write up the description here still)
3. Card embedding based on concepts
![Alt text](docs/CardEmbedding.jpg?raw=true "Optional")
    - The card embedding is a weighted sum of the concept embedding vectors from the concepts identified in the card.
    - More specifically, concept vectors are summed, but are weighted inversely to their prevalence in the graph.
    - Finally, after summing the concept vectors, individual components are once again divided by their prevalence in the graph
   in order to further favor the presence of unique concepts in the resulting vector. 

### Structuring the knowledge graph
The natural language embeddings of each concept or card allow one to cluster concepts through various 
methods. 

1. Defining the similarity metric
> ![Alt text](docs/SimilarityMetric.jpg?raw=true "Optional")
Here is an example matrix of all card-card similarities in the knowledge graph.
![Alt text](docs/ExampleSimilarityMatrix.jpg?raw=true "Optional") 

2. Example similarity calculations
    - ![Alt text](docs/ExampleNodeNodeOverlap.jpg?raw=true "Optional")  
    - ![Alt text](docs/ExampleNodeCardOverlap.jpg?raw=true "Optional")  

3. Clustering ideas (question/answer pairs) based on the similarity metric. 
   - Here are some example clusters of cards (phrased as a "family tree")
   ![Alt text](docs/ExampleClusterHierarchies.jpg?raw=true "Optional")
   - The clustering algorithm is just something I hacked together, and isn't very polished (but works reasonably well).
     - See code for details. The rough idea is the following:
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
         clusters as one big "card" and combine all their concept embeddings to get an embedding that represents the cluster.
         - We can then cluster those cluster embeddings, and so on.
     - The main point here is that some kind of clustering is readily possible. 

### Querying the knowledge graph
To answer a question about facts in the knowledge graph, we determine the component concepts of a question
and then gather similar cards to this question to re-expose to the language model at answering time.
1. Construct a question embedding
![Alt text](docs/QuestionEmbedding.jpg?raw=true "Optional")
![Alt text](docs/ExampleQuestionEmbeddingExtraction.jpg?raw=true "Optional")
    - This few-shot prompting teaches the model to extract concepts in the correct level of detail
      (going from most abstract to least abstract concepts), and to use the correct format (capitalize first letter, 
   and stick to one or two words usually)
    - The two-stage reprompting allows the model to see what concepts are extracted from similar cards,
   which further improves the quality of the question embedding.
     
2. Gather similar cards 
    1. In the simplest case, this can just be the top k cards ranked via the similarity metric.
3. Answer the question by re-prompting the language model while showing related information
![Alt text](docs/QuestionAnswering.jpg?raw=true "Optional")
![Alt text](docs/ExampleQuestionAnswering.jpg?raw=true "Optional")
4. We can choose whether to allow the language model to use outside knowledge, or only information directly  
within the re-prompted cards.

### Question generation (future: ideally hypothesis generation for scientific research)
Generate questions 
![Alt text](docs/ExampleQuestionGeneration.jpg?raw=true "Optional") 

### Future extensions

#### Agent-like behavior and exploration
The ultimate goal here is to make a self-improving agent that can explore and augment its conceptual environment 
using reinforcement learning and fast algorithms which process the information in the knowledge graph, with only sporadic calls 
to the language model to actually structure the information. 

Ideally the knowledge graph can be parsed via reinforcement learning algorithms to identify gaps in knowledge, and clusters of knowledge,
to then determine what areas to explore further and ask questions about. Then the language model can
be used to actually generate meaningful and conceptually different questions, which can be asked to the environment (ie. the internet) 
to gather more information. 

This will require some notion of "value" for the information in the knowledge graph (to balance pure exploration with 
exploration of relevant concepts). 

Will probably have to set up self pruning and revisiting of the knowledge graph, when in idle state (like sleeping)

#### Structured and hierarchical question-answering
Get it to answer a multi-stage question prompt with sub-questions 

Ideally, I want the language model to only answer questions based on information that is explicitly in the knowledge graph.
I want it to identify when it has sufficient information or not. If not, it should make sub-queries to the knowledge graph to fill in
specific parts of the answer (thereby avoiding the constraint of the concept window size). 

These questions and sub-questions could be added to the knowledge graph as memory.

#### Specializing the language model
Can train language model to improve itself based on examples (for example for extracting the concept hierarchy). 
Could have a dedicated smaller network for this (similar to the dedicated internal policy and value networks in alpha go zero).

#### Set up as a personal flashcard system (smart spaced-repetition memory training with LLMs)
* Incorporate spaced repetition and probability of memory. 
* Generate complex and organic questions to test concepts within a cluster of cards.
* Learn some parameters for you (your personal rate of forgetting. Your interests and base knowledge).
* Have preset expert knowledge structures to learn, and teach (ie a module of cards on statistics).
* Have a nice way of visualizing your personal knowledge graph by clustering.

#### Very long term foreseen issues
* Get synonym directory to help collapse nodes (words).
* How to handle other datatypes other than text?
* How to handle very short or very long text or concepts (things like math, or huge formulas?).
* As a plugin to always ask you to summarize what you read the last 10 minutes.
* Interface it with whisper or something to do speech to text and then summarize.

#### Ideas to emphasize in the description:
- Human interpretability of the embedding vector
- Leveraging the local network structure to have a high quality (useful) embedding (more so than just a globally learned embedding from all text on the internet)
- Bootstrapping and policy improvement through self-generated examples. 
- Link to idea of nearest-neighbors transformer implementations
  - Non-parametric transformer
  - Other ones that use memory of various kinds. 

#### References to add
Reference Glom by Hinton when talking about hierarchy. 

Contrastive divergence embedding thing needs to be clearer.  

Reference to chain of thought reasoning paper, and realization that I can't get it to provide the concept
hierarchy before it explains what's going on (has to explain first, then concept extraction).

Reference OpenAI embeddings, and GPT-index. 