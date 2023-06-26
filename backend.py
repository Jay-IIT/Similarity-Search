#authored by : Jayavelu Balaji
# Import the necessary modules
import torch
from transformers import AutoTokenizer, TFAutoModel
from transformers import logging
import pandas as pd
from datasets import Dataset, concatenate_datasets
import ssl
import sys
import os
import flask

app = flask.Flask(__name__)
os.environ['CURL_CA_BUNDLE'] = ''
 

# Load the pre-trained BERT model
model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModel.from_pretrained(model_name, from_pt=True)

embeddings_dataset = None
text_embeddings = []


def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]


def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="tf"
    )
    encoded_input = {k: v for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)


def concatenate_text(data):
    return {
        "text": data["description"]
        + " \n "
        + data["difficulty_level"] + " " + data["instructor_bio"]
        + " \n "
        + data["language"]
    }


def make_embeddings(data_texts):
    global text_embeddings
    for data_text in data_texts:
        data_inputs = tokenizer.encode(data_text, return_tensors='pt')
        with torch.no_grad():
            text_embeddings.append(model(data_inputs)[0][0][0])


ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


def append_to_embeddings(dataset):
    global embeddings_dataset
    embeddings_dataset = dataset


def add_embedding(feedback):
    try:
        train_data = pd.DataFrame.from_dict([feedback])
        train_dataset = Dataset.from_pandas(train_data)
        train_dataset = train_dataset.map(concatenate_text)
        embeddings_dataset_local = train_dataset.map(
            lambda x: {"embeddings": get_embeddings(x["text"]).numpy()[0]}
        )
        embeddings_dataset_local = concatenate_datasets(
            [embeddings_dataset, embeddings_dataset_local])
        embeddings_dataset_local.add_faiss_index(column="embeddings")
        append_to_embeddings(embeddings_dataset_local)
    except Exception as e:
        print(e)


def train():
    train_data = pd.read_csv("peloton_nlp.csv")
    # Fill the missing values with the mode for each column
    train_data   = train_data.fillna(train_data.mode().iloc[0])
    # For quick training to get the fullest potential eed to train with whole data
    train_dataset = Dataset.from_pandas(train_data[:50])
    train_dataset = train_dataset.map(concatenate_text)
    embeddings_dataset_local = train_dataset.map(
        lambda x: {"embeddings": get_embeddings(x["text"]).numpy()[0]}
    )
    embeddings_dataset_local.add_faiss_index(column="embeddings")
    append_to_embeddings(embeddings_dataset_local)


def post_request():
    while True:
        question = input("PLEASE ENTER YOUR PREFERENCE (Enter 'exit' to quit)\n")
        if question == 'exit':
            break
        
        question_embedding = get_embeddings([question]).numpy()
        scores, samples = embeddings_dataset.get_nearest_examples(
            "embeddings", question_embedding, k=3
        )
        samples_df = pd.DataFrame.from_dict(samples)
        samples_df["scores"] = scores
       # samples_df.sort_values("scores", ascending=True, inplace=True)
        print("   RECOMMENDATIONS ")
        for _, row in samples_df.iterrows(): 
             print("###############################")
             print(row.class_type_ids)
             print(row.description)
             print(row.difficulty_level)
             print(row.instructor_bio)
             print("###############################")


@app.route("/recommendations", methods=["POST"])
def get_recommendations():
    request = flask.request
    question = request.json["question"]

    question_embedding = get_embeddings([question]).numpy()
    scores, samples = embeddings_dataset.get_nearest_examples(
            "embeddings", question_embedding, k=3
        )
    samples_df = pd.DataFrame.from_dict(samples)
    samples_df["scores"] = scores 
    recommendations = []
    for _, row in samples_df.iterrows():
        recommendation = {
            "class_type_ids": row["class_type_ids"],
            "language": row["language"],
            "description": row["description"],
            "difficulty_level": row["difficulty_level"],
            "instructor_bio": row["instructor_bio"],
            "fitness_discipline" : row["fitness_discipline"]
         }
        recommendations.append(recommendation)

    return flask.jsonify(recommendations)

train()
#post_request() 
if __name__ == "__main__":
    app.run()

#Evaluated NLP model to run multiples times to see if recommendations are proper
#manual model evaluation
 


 
