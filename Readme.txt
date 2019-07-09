This Code is implementation of our paper titled as "Link Prediction based on Interlayer Similarity (LIPS)"

If you use this code please cite our article as:

"Najari Shaghayegh, Salehi Mostafa, Ranjbar Vahid, and Jalili Mahdi. "Link prediction in multiplex networks based on interlayer similarity." Physica A: Statistical Mechanics and its Applications (2019): 120978."

In this code we're following these major steps for link prediction in a two layers network by LPIS method:

1. Reading our networks and preparing our test and train networks.
2. Calculating probabilities of link existence in each of networks by using only intralayer features such as adamic-adar, jaccard, and so on.(this functioned as: prob_in_net)
3. improving this probabilities through LPIS method by using interlayer similarities such as ASN , AASN and so on, which are determined in our paper (this functioned as: prob_with_sim_AASN)
4. For training the best alpha and threshold we can train our model.
5. Finally we test our model.
