from utils import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, logging, os
from datetime import date
import nltk

logger = logging.getLogger(__name__)
today = date.today()
today_date = today.strftime("%b-%d-%Y")
log_file_name = "log_rte_pathways.log"
full_path = os.path.join("./logs", log_file_name)
FORMAT = '%(message)s'

logging.basicConfig(
    format=FORMAT,
    level=logging.INFO,
    filename=full_path,
    filemode='w'
)


hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
#hg_model_hub_name = "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
#hg_model_hub_name = "ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli"
# hg_model_hub_name = "ynie/electra-large-discriminator-snli_mnli_fever_anli_R1_R2_R3-nli"
#hg_model_hub_name ="typeform/distilbert-base-uncased-mnli"
#hg_model_hub_name ="joeddav/xlm-roberta-large-xnli"

def get_entailment(premise, hypothesis,tokenizer,model):
    max_length = 256


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


    predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()  # batch_size only one

    #     logger.info("***********")
    # logger.info(f"Premise:{premise}")
    # logger.info(f"Hypothesis:{hypothesis}")
    # logger.info(f"predicted_probability:{predicted_probability}")

    if(predicted_probability.index(max(predicted_probability)))==0:
        if predicted_probability[0]>0.9:
            logger.info("***********")
            logger.info("found entailment with")
            logger.info(f"Premise:{premise}")
            logger.info(f"Hypothesis:{hypothesis}")
            logger.info(f"predicted_probability:{predicted_probability}")


def cleanup(data_pdfs):
    all_data=[]
    for x in tqdm(data_pdfs, total=len(data_pdfs),desc="cleanup"):
        x=x.replace("\n","")
        x = x.replace("-", "")
        x = x.replace("˜", "")
        x = x.replace("˚", "")
        x = x.replace("~", "")
        x = x.replace("™", "")
        x = x.replace("˝", "")

        x=x.lower()
        all_data.append(x)
    return all_data

def get_sents(data_pdfs):
    sent_tokenizer = nltk.tokenize.PunktSentenceTokenizer()
    all_data=[]
    for x in tqdm(data_pdfs, total=len(data_pdfs),desc="converting to sentences"):
        result = (sent_tokenizer.tokenize(x))
        for y in result:
            all_data.append(y)
    return all_data

if __name__ == "__main__":
    # small number of files for testing
    data_pdfs = get_data_pdf_files("data/temp/")
    #data_pdfs = get_data_pdf_files("data/habitus_rice_growing_senegal/")
    data_pdfs_sents=get_sents(data_pdfs)
    data_pdfs_sents=cleanup(data_pdfs_sents)
    data_human_desc = get_data("data/human_description.txt")
    data_human_desc_sent=[]
    for l in data_human_desc:
        data_human_desc_sent.append(l)
    tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
    model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name)
    #
    # premise="A soccer game with multiple males playing."
    # hyp="	Some men are playing a sport."
    # get_entailment(premise.lower(), hyp.lower(), tokenizer, model)
    #
    # import sys
    # sys.exit()

    for premise in tqdm(data_human_desc_sent,total=len(data_human_desc_sent),desc="premises:"):
        for hyp in tqdm(data_pdfs_sents,total=len(data_pdfs_sents),desc="hypothesis:"):
            get_entailment(premise.lower(), hyp.lower(), tokenizer, model)

