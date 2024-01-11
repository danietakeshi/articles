# Sentence Transformers - My (not so) Secret Weapon in the ML Zoomcamp Q&A Challenge

!([Natural_Language_Processing.png](https://github.com/danietakeshi/articles/edit/main/README.md#:~:text=Natural_Language_Processing.png,41))

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

You can reference the notebook I shared on Kaggle (https://www.kaggle.com/code/dtakeshi/dtc-zoomcamp-q-a-sentence-transformers).

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

![Me Hopping for a Good Model](https://www.troublefreepool.com/media/mr-bean-waiting-gif.190931/full)

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
