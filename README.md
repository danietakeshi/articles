# Sentence Transformers - My (not so) Secret Weapon in the ML Zoomcamp Q&A Challenge

![[Natural Language Processing (NLP).png]]

Just like me, you might be diving into the Machine Learning Zoomcamp, or maybe you're just eager to unwrap the mysteries of Natural Language Processing and its practical uses. Join me on this adventure where I'll show you how to use Sentence Transformers to craft a model that picks the best answer for a given question.

I embarked on this quest motivated by the 'DTC Zoomcamp Q&A Challenge'. I'll admit, when I first opened my notebook, I was as clueless as a penguin in a disco, even with the provided Baseline.

After some digging and video watching, I stumbled upon [an article](https://huggingface.co/blog/how-to-train-sentence-transformers) that illustrates how to train and refine a sentence transformers model, and the article states that:

>1. Pre-trained Transformers require heavy computation to perform semantic search tasks. For example, finding the most similar pair in a collection of 10,000 sentences [requires about 50 million inference computations (~65 hours) with BERT](https://arxiv.org/abs/1908.10084). In contrast, a BERT Sentence Transformers model reduces the time to about 5 seconds.
>
>2. Once trained, Transformers create poor sentence representations out of the box. A BERT model with its token embeddings averaged to create a sentence embedding [performs worse than the GloVe embeddings](https://arxiv.org/abs/1908.10084) developed in 2014.

Reading this, I thought, "Eureka! I've struck the ~~NVIDIA GPU~~ goldmine!"

But before we roll up our sleeves, let's brush up on some key concepts, starting with:

## What is a Sentence Transformer?

Plucked straight from the [SentenceTransformers](https://www.sbert.net/index.html) documentation:

>  SentenceTransformers is a Python framework for state-of-the-art sentence, text and image embeddings. The initial work is described in our paper Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.

The documentation also describes how it is possible to compute embeddings for over 100 languages and the use of cosine similarity to identify sentences with similar meanings.

In essence, it's an embedding creation model **optimized for finding related sentences** – extremely handy for our task, coupled with the ease of training and refining the model as per our whims.

## But What are Embeddings?

A computer doesn’t read and understand sentences or texts as we do. So, we need to transform phrases and words into numerical representations. The more dimensions, the richer the semantic information stored in this vector.

This enables us to calculate the distances between vectors, with closer ones being more related.

This all sounded promising, **so I decided to test it out to see if it could beat the proposed Baseline**.

## Training the Model

When embedding a phrase or word, we aim for a fixed-size vector. First, we feed our sentence into a transformer like BERT, which produces an embedding for each text token. To get a fixed-size vector, we need a pooling layer to transform the BERT output, typically by averaging the contextualized embeddings.

The sentence-transformers framework has a class named `InputExample` for storing training data. For our case with a set of questions and answers without a similarity label, we can store the data as pairs (Question and Correct Answer) or triplets (Question, Correct Answer, and Other Candidate Answers).

The training structure influences the Loss Function used for model refinement.

For a Question-Answer InputExample, we use the MultipleNegativesRankingLoss. But for triplets, we need the TripletLoss function.

Enough talk, let's code!
![](https://y.yarn.co/5109a04e-a6d2-479f-90ca-2f230bdcb408_text.gif)

You can reference the notebook I shared on Kaggle (##Insert Link).

First, install the dependencies with `!pip install -U sentence-transformers`.

Then, load the provided tables:

```python
test_answers_df = pd.read_csv("/kaggle/input/dtc-zoomcamp-qa-challenge/test_answers.csv")
test_questions_df = pd.read_csv("/kaggle/input/dtc-zoomcamp-qa-challenge/test_questions.csv")
train_answers_df = pd.read_csv("/kaggle/input/dtc-zoomcamp-qa-challenge/train_answers.csv")
train_questions_df = pd.read_csv("/kaggle/input/dtc-zoomcamp-qa-challenge/train_questions.csv")
```

Next, preprocess the training data into Triplets (question, positive answer, and negative answers), allowing us to train our model in two different ways.

```python
train_questions_df_triplets = pd.DataFrame()

for question_id, question, course, year, candidate_answers, answer_id in train_questions_df.values:
    answers_list = candidate_answers.split(',')
    negative_ids = [x for x in answers_list if x != str(answer_id)]
    negative_list, positive_list = [],[]
    
    for neg_id in negative_ids:
        negative_list.append(train_answers_df[train_answers_df.answer_id == int(neg_id)]['answer'].values[0])
        
    positive_list.append(train_answers_df[train_answers_df.answer_id == int(answer_id)]['answer'].values[0])
    
    # Adding a single new row
    new_row = {
        'question_id': question_id, 
        'question': question, 
        'positive': positive_list[0],
        'negative': negative_list,
    }
    
    train_questions_df_triplets = train_questions_df_triplets._append(new_row, ignore_index=True)
```

The selected Sentence Transformer was the `all-MiniLM-L6-v2` but there are plenty of options on HuggingFace that can be used as a base.

```python
model_id = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_id)
```

The MultipleNegativesRankingLoss needs a pair of related sentences, in this case, question and answer, the provided notebook uses this method to fine tune the model.

```python
train_examples = []

for question_id, question, positive, negatives in train_questions_df_triplets.values:
    train_examples.append(InputExample(texts=[question, positive]))
    
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

train_loss = losses.MultipleNegativesRankingLoss(model=model)
```

We could use the TripletLoss loss function with the question as an Anchor, the selected answer as a positive and the other candidate answers as negatives, but you can try this approach on yourself. I have already tried but didn't add the results to the provided notebook (it is up to you to figure it out if I did it to help you or not).

As simple as that we have all set up to train our model to predict the correct answers, all that we need to do its use the `model.fit` method passing our DataLoader and our training loss.

In addition we set up a number of epochs of 10 and a warm up step of 10% of the training data.

```python
num_epochs = 10
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=warmup_steps)
```

This process can take a while, so why not grab a cup of coffee...
![Me Hopping for a Good Model](data:image/jpeg;base64,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)

Now that we have trained our model we can test it on our training data to check the accuracy of the sentence transformer.

Preparing the questions dataframe:

```python
train_questions_df_predict = pd.DataFrame()

for question_id, question, course, year, candidate_answers, answer_id in train_questions_df.values:
    answers_list = candidate_answers.split(',')
    candidate_answers_list = []

    for ans_id in answers_list:
        candidate_answers_list.append(train_answers_df[train_answers_df.answer_id == int(ans_id)]['answer'].values[0])

    positive_list.append(train_answers_df[train_answers_df.answer_id == int(answer_id)]['answer'].values[0])

    # Adding a single new row
    new_row = {
        'question_id': question_id, 
        'question': question, 
        'candidate_answers_id': answers_list,
        'candidate_answers': candidate_answers_list,
    }

    train_questions_df_predict = train_questions_df_predict._append(new_row, ignore_index=True)
```

After that we can use the cosine similarity to get the closest answer for our embeddings:

```python
train_predictions = []
similarity_scores_df = []

for question, candidate_answers, candidate_answers_id in zip(
    train_questions_df_predict['question'], 
    train_questions_df_predict['candidate_answers'],
    train_questions_df_predict['candidate_answers_id']
):
    list_aswers = candidate_answers
    embeddings = model.encode([question] + candidate_answers)
    similarity_scores = cosine_similarity([embeddings[0]], embeddings[1:])
    train_predictions.append(int(candidate_answers_id[np.argmax(similarity_scores)]))
    similarity_scores_df.append(similarity_scores)

train_questions_df['predictions'] = train_predictions
```

Finally, we can use this to calculate the models accuracy:
```python
accuracy = (train_questions_df['predictions'] == train_questions_df['answer_id']).mean()
print(f'Accuracy: {accuracy:.4f}')
```

Surprisingly, **we got an accuracy of 100%**! That seems too good to be true, possibly indicating overfitting.

But the best course of action now is to test it on the test dataset and submit our answer to the competition to see if the train-test gap is small.

```python
test_questions_df = test_questions_df.drop_duplicates(subset='question_id')
test_questions_df.shape

test_questions_df_predict = pd.DataFrame()

for question_id, question, course, year, candidate_answers in test_questions_df.values:
    answers_list = candidate_answers.split(',')
    candidate_answers_list = []

    for ans_id in answers_list:
        candidate_answers_list.append(test_answers_df[test_answers_df.answer_id == int(ans_id)]['answer'].values[0])

    # Adding a single new row
    new_row = {
        'question_id': question_id, 
        'question': question, 
        'candidate_answers_id': answers_list,
        'candidate_answers': candidate_answers_list,
    }

    test_questions_df_predict = test_questions_df_predict._append(new_row, ignore_index=True)

test_predictions = []
similarity_scores_df = []

for question, candidate_answers, candidate_answers_id in zip(
    test_questions_df_predict['question'], 
    test_questions_df_predict['candidate_answers'],
    test_questions_df_predict['candidate_answers_id']
):
    list_aswers = candidate_answers
    embeddings = model.encode([question] + candidate_answers)
    similarity_scores = cosine_similarity([embeddings[0]], embeddings[1:])
    test_predictions.append(int(candidate_answers_id[np.argmax(similarity_scores)]))
    similarity_scores_df.append(similarity_scores)

test_questions_df['predicted_answer_id'] = test_predictions
```

Finally, we submit our results:

```python
test_questions_df[['question_id', 'predicted_answer_id']].to_csv('submission.csv', index=False)

submission = pd.read_csv("/kaggle/working/submission.csv")
submission.head()
```

Our submission scored 94.696%, which is pretty good for a first model, but there's room for improvement with different datasets, transformers, training methods, and adding image embeddings.

The HuggingFace article shows a way to fine tune the created model with some datasets available on their website (if you have some trouble doing this I can help you out - but I did not have better results adding more data).

I hope you found this guide a fun intro to Natural Language Processing specially using Sentence Transformers. If you have better approaches, let's enhance this model further!

Participating in the ML Zoomcamp Q&A Challenge sparked ideas for my Capstone Project 2, but that's a story for another day.

So long, and thanks for all the tokens!