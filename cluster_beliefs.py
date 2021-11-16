import numpy as np

beliefs=[
    "work.Like working.So wried wwent xalisae xtremecaffeine yildirim z/28 zipout zulchzulu"
"loans are useful.",
    "Trump",
    "Paul makes good beer",
    "Dogfish Head makes better beer",
    "Paul makes the best beer",
    "there will be more virulent mutations",
    "premises of Social Darwinism",
"Troeg’s might know a thing or two about beer",
"Rudy Giuliani broke the law",
"anyone would hate The Princess Bride"
]

def load_glove_model(glove_file):
    """
    :param glove_file: embeddings_path: path of glove file.
    :return: glove model
    """
    embeddings_dict = {}
    with open(glove_file, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict

embeddings_dict = load_glove_model("glove_bottom10.txt")
print(embeddings_dict.keys())
word_avgemb={}
for belief in beliefs:
    belif_split=belief.split(" ")
    for each_word in belif_split:
        if each_word in embeddings_dict:
            word_avgemb[each_word]=np.average(embeddings_dict[each_word])

print(word_avgemb)




