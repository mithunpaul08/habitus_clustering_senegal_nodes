from transformers import pipeline
from transformers import AutoModelWithLMHead, AutoTokenizer

#for mlm
# unmasker = pipeline('fill-mask', model='bert-base-uncased')
# ret=unmasker("Hello I'm a [MASK] model.")
# print(ret)



##for psentiment analysis
# classifier = pipeline('sentiment-analysis')
# result = classifier(["We are very happy to show you the 🤗 Transformers library.",
#             "We hope you don't hate it."])
# print(result)


##for summarization
# summarizer = pipeline("summarization")
# article="Natural language inference (NLI) is a stepping stone towards building machines that can reason over text, by understanding big ideas from knowing smaller component parts that were extracted from text. NLI is at the core of many critical natural language processing applications such as detecting undiagnosed medical conditions from electronic medical records (EMR). Deep learning (DL) methods have improved inference in many such applications through robust learned representations. However, these models suffer in terms of explainability due to the hidden states in the learned networks, which are hard to interpret. Explainability, if possible at all, is incidental. This lack of explainable inference prevents these advances from being applied in important domains such as scientific discovery or medical applications"
# print(summarizer(article, max_length=130, min_length=30, do_sample=False))
#

#text generation
# article="fact verification is a"
# text_generator = pipeline("text-generation")
# print(text_generator(article, max_length=50, do_sample=False))

# model = AutoModelWithLMHead.from_pretrained("xlnet-base-cased")
# tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
#  # Padding text helps XLNet with short prompts - proposed by Aman Rusia in https://github.com/rusiaaman/XLNet-gen#methodology
# PADDING_TEXT = """Natural language inference (NLI) is a stepping stone towards building machines that can reason over text, by understanding big ideas from knowing smaller component parts that were extracted from text. NLI is at the core of many critical natural language processing applications such as detecting undiagnosed medical conditions from electronic medical records (EMR). Deep learning (DL) methods have improved inference in many such applications through robust learned representations. However, these models suffer in terms of explainability due to the hidden states in the learned networks, which are hard to interpret. Explainability, if possible at all, is incidental. This lack of explainable inference prevents these advances from being applied in important domains such as scientific discovery or medical applications"""
# prompt = "Inference is the technique of comparing "
# inputs = tokenizer.encode(PADDING_TEXT + prompt, add_special_tokens=False, return_tensors="pt")
# prompt_length = len(tokenizer.decode(inputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
# outputs = model.generate(inputs, max_length=250, do_sample=True, top_p=0.95, top_k=60)
# generated = prompt + tokenizer.decode(outputs[0])[prompt_length:]
# print(generated)



##for NLI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

if __name__ == '__main__':
    max_length = 256

    premise = "Two women are embracing while holding to go packages."
    hypothesis = "The men are fighting outside a deli."

    hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/electra-large-discriminator-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli"

    tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
    model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name)

    tokenized_input_seq_pair = tokenizer.encode_plus(premise, hypothesis,
                                                     max_length=max_length,
                                                     return_token_type_ids=True, truncation=True)

    input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
    # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
    token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
    attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)

    outputs = model(input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=None)
    # Note:
    # "id2label": {
    #     "0": "entailment",
    #     "1": "neutral",
    #     "2": "contradiction"
    # },

    predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()  # batch_size only one

    print("Premise:", premise)
    print("Hypothesis:", hypothesis)
    print("Entailment:", predicted_probability[0])
    print("Neutral:", predicted_probability[1])
    print("Contradiction:", predicted_probability[2])