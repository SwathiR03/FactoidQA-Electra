from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Load multiple ELECTRA models
model_names = [
    "bhadresh-savani/electra-base-squad2",
]

models = [AutoModelForQuestionAnswering.from_pretrained(model_name) for model_name in model_names]
tokenizers = [AutoTokenizer.from_pretrained(model_name) for model_name in model_names]

# Allow the user to input the context and question
text = input("Enter the context: ")
question = input("Enter the question: ")

# Initialize lists to store predicted answers from each model
predicted_answers = []

# Get predictions from each model
for model, tokenizer in zip(models, tokenizers):
    inputs = tokenizer(question, text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    predicted_answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
    predicted_answers.append(predicted_answer)

# Aggregate the predictions using a simple voting mechanism. 
aggregated_answer = max(set(predicted_answers), key=predicted_answers.count)

print("Aggregated Answer:", aggregated_answer)
