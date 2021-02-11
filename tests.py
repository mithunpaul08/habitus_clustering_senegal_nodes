from glove_read_get_embed import *
from run_hac_senegal import split_concept_get_combined_embedding

read_from_glove()
cos1=return_pairwise_cosine_similarity('long','longer')
cos2=return_pairwise_cosine_similarity('shorter','longer')

print(f"cos1={cos1}")
print(f"cos2={cos2}")
assert cos1[0][0] > cos2[0][0]

emb_word1=split_concept_get_combined_embedding('climate information village')
emb_word2=split_concept_get_combined_embedding('climate information provision ')

emb_word1=emb_word1.reshape(1,-1)
emb_word2=emb_word2.reshape(1,-1)

cos3=cosine_similarity(emb_word1, emb_word2)
print(f"cos3={cos3}")

assert cos3[0][0]>0.8