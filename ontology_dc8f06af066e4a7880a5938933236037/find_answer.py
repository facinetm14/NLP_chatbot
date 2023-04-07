import json
import torch
from transformers import BertTokenizer, BertForQuestionAnswering

# Load the SQuAD dataset
with open('./data/train-v2.0.json') as f:
    squad_data = json.load(f)['data']

# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

def get_answer(question, context):
    # Tokenize the input
    input_ids = tokenizer.encode(question, context)

    # Format the input
    sep_index = input_ids.index(tokenizer.sep_token_id)
    num_tokens = len(input_ids)
    segment_ids = [0] * (sep_index + 1) + [1] * (num_tokens - sep_index - 1)
    start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))

    # Decode the output
    if isinstance(start_scores, torch.Tensor):
        start_index = torch.argmax(start_scores[0])
        end_index = torch.argmax(end_scores[0])
        answer_tokens = input_ids[start_index:end_index+1]
        answer = tokenizer.decode(answer_tokens)
        return answer
    else:
        return None

def findAnswer(question):
    # Iterate over the data points in the dataset
    for article in squad_data:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            answer = get_answer(question, context)
            if answer:
                return answer
    return "Sorry, I don't know the answer to this question."
