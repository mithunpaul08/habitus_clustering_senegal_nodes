from sklearn.metrics.pairwise import cosine_similarity
from glove_read_get_embed import read_from_glove
from run_hac_senegal import return_pairwise_cosine_similarity,split_concept_get_average_embedding
from unittest import TestCase

class test_q4(TestCase):
    read_from_glove()

    def test_splitting_of_concepts(self):

        emb_word1 = split_concept_get_average_embedding('climate information village')
        emb_word2 = split_concept_get_average_embedding('climate information provision ')

        emb_word1 = emb_word1.reshape(1, -1)
        emb_word2 = emb_word2.reshape(1, -1)

        cos3 = cosine_similarity(emb_word1, emb_word2)
        assert cos3[0][0] > 0.8

        avg_emb = split_concept_get_average_embedding("climate information village")
        assert avg_emb[0][0] < -2.3e-01
        assert avg_emb[0][0] > -2.4e-01
        assert avg_emb[0][3] < -5.8e-02
        assert avg_emb[0][3] > -5.9e-02

    def test_glove_is_good(self):
        cos1=return_pairwise_cosine_similarity('long','longer')
        cos2=return_pairwise_cosine_similarity('shorter','longer')

        print(f"cos1={cos1}")
        print(f"cos2={cos2}")
        assert cos1[0][0] > cos2[0][0]



