from transformers import pipeline


#for mlm
# unmasker = pipeline('fill-mask', model='bert-base-uncased')
# ret=unmasker("Hello I'm a [MASK] model.")
# print(ret)



##for psentiment analysis
classifier = pipeline('sentiment-analysis')
result = classifier(["We are very happy to show you the 🤗 Transformers library.",
            "We hope you don't hate it."])
print(result)