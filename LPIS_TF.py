import networkx as nx
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import precision_score,recall_score, f1_score, accuracy_score
import pylab as pl
import matplotlib.pyplot as plt

import time
import copy

def create_matrix(edges,n):
    matrix = np.zeros((n, n))
    for (i,j) in edges:
        matrix[i,j]=1
    return matrix

def create_matrix_main_test(t_edges,matrix,n):
    G_matrix = np.zeros((n,n))
    out_edges=[]
    for (i,j)in t_edges:
        if matrix[i,j]==1:
            G_matrix[i,j]=1
    out_edges=t_edges
    return (G_matrix,out_edges)

def create_matrix_main_train(t_edges,matrix,n):
    G_matrix = np.zeros((n,n))
    num_edges=0
    out_edges=[]
    #print (t_edges)
    for (i,j) in t_edges:
        if matrix[i,j]==1:
            G_matrix[i,j]=1
            out_edges.append((i, j))
            num_edges+=1
        elif num_edges>0:
            # if random.random()>1/2:
            out_edges.append((i,j))
            num_edges-=1
    return (G_matrix,out_edges)


def similarity_Features(G_sim1,G_matrix,edges):
    Features=[]
    truth=[]
    sim_matrix=np.zeros((n,n))
    for (i,j) in edges:
        Features.append(G_sim1[i,j])
        truth.append(G_matrix[i,j])
    collect_features=np.zeros((len(Features),1))
    collect_features[:,0]=Features
    return (collect_features,truth)

def bar_plot(AUC_M,AUC_S,s):
    n_groups = 2
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, AUC_M, bar_width,
                     alpha=opacity,
                     color='y',
                     label='Multiplex')

    rects2 = plt.bar(index + bar_width, AUC_S, bar_width,
                     alpha=opacity,
                     color='r',
                     label='SingleLayer')

    plt.xlabel('networks')
    plt.ylabel('AUC')
    plt.title('alpha is:'+str(s))
    plt.xticks(index + bar_width, ('Twitter', 'Foursquare'))
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.show()
def alpha_bar_plot(AUC_M,AUC_S,s):
    n_groups = 3
    # create plot
    fig, ax = plt.subplots()
    index =np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, AUC_M, bar_width,
                     alpha=opacity,
                     color='y',
                     label='Multiplex')

    rects2 = plt.bar(index + bar_width, AUC_S, bar_width,
                     alpha=opacity,
                     color='r',
                     label='SingleLayer')

    plt.xlabel('alpha')
    plt.ylabel('AUC')
    plt.title(s)
    plt.xticks(index + bar_width, ('0.2', '0.5','0.8'))
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.show()

def plot(item1,item2,item3,s):
    pl.title(s)
    pl.xlabel('Threshold')
    pl.ylabel('AUC')
    pl.gca().set_ylim([0, 1])
    pl.gca().set_xlim([0, 1])

    pl.plot(item1, item2, 'r',)
    pl.plot(item1,item3,'g')
    #pl.plot(item1, item4, 'b')
    plt.plot(item1, item2, color="red", linewidth=5, linestyle="-", label="with_out sim")
    plt.plot(item1, item3, color="green", linewidth=5, linestyle="-", label="with sim")
    #plt.plot(item1, item4, color="blue", linewidth=2.5, linestyle="-", label="Article")

    plt.legend(loc='lower right')
    pl.show()

def plot3D(best_m_list,best_acc_list,alpha_list):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(best_m_list, best_acc_list, alpha_list, c='r', marker='o')

    ax.set_xlabel('alpha')
    ax.set_ylabel('m Label')
    ax.set_zlabel('acc Label')

    plt.show()
def  ASN(N_A,N_B,test_edges):
    K_A = 0
    K_B = 0
    K_C = 0
    # az 0 shoru mishe
    sim_array = np.zeros((3, len(nodes) + 1))
    for i in nodes:
        for j in nodes:
            if i < j:
                if N_A[i, j] == 1:
                    K_A += 1
                    sim_array[0, i] += 1
                if N_B[i, j] == 1:
                    K_B += 1
                    sim_array[1, i] += 1
                    if N_A[i, j] == 1:
                        K_C += 1
                        sim_array[2, i] += 1
    s_array = []
    for (i, j) in test_edges:
        s_array.append((sim_array[2, i] + sim_array[2, j] + 1) / (
            1 + sim_array[0, i] + sim_array[0, j] + sim_array[1, i] + sim_array[1, j] - sim_array[2, i] - sim_array[
                2, j]))
    return ((K_C + 1) / (K_A + K_B - K_C + 1), s_array)

def AASN(N_A,N_B,test_edges):
    ''''''
    K_A = 0
    K_B = 0
    K_C = 0
    # az 0 shoru mishe
    sim_array = np.zeros((3, len(nodes) + 1))
    for i in nodes:
        for j in nodes:
            if i < j:
                if N_A[i, j] == 1:
                    K_A += 1
                    sim_array[0, i] += 1
                if N_B[i, j] == 1:
                    K_B += 1
                    sim_array[1, i] += 1
                    if N_A[i, j] == 1:
                        K_C += 1
                        sim_array[2, i] += 1

    s1_array = [((sim_array[2, i] + sim_array[2, j] + 1) / (1 + sim_array[0, i] + sim_array[0, j])) for (i, j) in
                test_edges]
    #s2_array = [((sim_array[2, i] + sim_array[2, j] + 1) / (1 + sim_array[1, i] + sim_array[1, j])) for (i, j) in test_edges]
    return (0, s1_array)
#Here we tried to calcuate link probability by using only classic intralayer features such as jaccard, adamicadar and so on
def prob_in_net(n,m,graphs_test,train_edges_0, test_edges_0 ,matrixes, M_test):

    feature_train = []
    feature_test = []
    feature_train_1 = []
    feature_test_1 = []
    pred_prob = []
    pred = []
    roc_auc=[]
    roc_avg=0
    truth_test=[]
    for i in range(k_net):
        #jaccard_coefficient
        feature_train.append(nx.resource_allocation_index(graphs_test[i], train_edges_0))
        feature_test.append(nx.resource_allocation_index(graphs_test[i], test_edges_0))

        A2=M_test[i]*M_test[i]
        A3=A2*M_test[i]
        Lp_matrix=A2+(0.001*A3)
        (features_train, truth_train) = similarity_Features(Lp_matrix, matrixes[i], train_edges_0)
        (features_test, truth) = similarity_Features(Lp_matrix, matrixes[i], test_edges_0)
        truth_test.append(truth)

        LR =LogisticRegression(class_weight='balanced')
        LR.fit(features_train, truth_train)
        pred.append(LR.predict(features_test))
        x=LR.predict_proba(features_test)[:, 1]
        pred_prob.append(x)
        fpr, tpr, thrshold = metrics.roc_curve(truth, pred[i])
        roc = metrics.auc(fpr, tpr)
        roc_auc.append(roc)
        print('Net',i,roc)
        roc_avg += roc
    roc_auc.append(roc_avg / k_net)
    return (roc_auc,matrix,pred_prob,truth_test)

def similarity_of_networks(G1,G2,M1,M2):
    s=0
    for (i,j) in train_edges:
        if M1[i,j]==1 and M2[i,j]==1:
            s+=1
    return (s/len(G1.edges()),s/len((G2.edges())))
#Here we calculate the predicted probabilities by using LPIS methosd(Link Prediction accounting Interlayer Similarities)
def prob_with_sim_ASN(predictions_prob,test_edges1 ,truth, best_alpha_list, threshold,matrix,graphs_test,k_net,nodes):
    sim_predictins = []
    all_predictins = []
    roc_auc = []
    roc_avg = 0
    for net_a in range(k_net):
        alpha = best_alpha_list[net_a]
        total_sim_array = np.zeros(len(predictions_prob[net_a]))
        p_sim = np.zeros(len(predictions_prob[net_a]))
        p_total =(1-alpha)*predictions_prob[net_a]
        p_sim_total=np.zeros(len(predictions_prob[net_a]))
        for net_b in range(k_net):

            if net_a != net_b:
                (s, sim_array) = ASN(matrix[net_a], matrix[net_b], test_edges)
                for k in range(len(truth[net_b])):
                    if truth[net_b][k] == 1:
                        p_sim[k] += sim_array[k] * (predictions_prob[net_b][k])
                    else:
                        p_sim[k] += (1-sim_array[k]) *(1-predictions_prob[net_b][k])
                p_sim_total+=p_sim
        m_p = min(p_sim_total)
        if m_p < 0:
            print(m_p)
            m_p *= (-1)
            p_sim_total+=m_p
        p_total+=alpha*p_sim_total/max(p_sim_total)
        predictins = [1 if p_total[x] > thresholdshold else 0 for x in range(len(predictions_prob[net_a]))]
        all_predictins.append(p_total)
        fpr, tpr, threshol = metrics.roc_curve(truth[net_a], predictins)
        sim_predictins.append(p_sim)
        roc = metrics.auc(fpr, tpr)
        roc_auc.append(roc)
        roc_avg += roc
    roc_auc.append(roc_avg / k_net)
    return (roc_auc, matrix, all_predictins)

def get_features(scores,matrix):
    F=[]
    truth=[]
    for (i, j, k) in scores:
        F.append(k)
        truth.append(matrix[i,j])
    return (F,truth)

    for t in range(len(features[0])):
        for f in features:
            Features[t]-=((f[t]+1)/(S_total[t]+1))*(np.log((f[t]+1)/(S_total[t]+1)))
    return Features
def calculate_sim_matrix(matrix, common_graph, graphs_test, train_edges):
 #ASN
    sim_matrix = np.zeros((k_net, k_net))
    for net_a in range(k_net):
        for net_b in range(k_net):
            if net_a < net_b:
                sim_matrix[net_a, net_b] =AASN(np.array(matrix[net_a]), np.array(matrix[net_b]))
                # print('sim: ',sim_matrix[net_a, net_b] )
                sim_matrix[net_b, net_a] = sim_matrix[net_a, net_b]
    return sim_matrix

def get_data(filename,s):
    data = [line.strip().split(s) for line in open(filename).readlines()] # random.shuffle(data)
    return data

def thresholding(i,pred_prob,truth,threshold1):
    roc_auc=[]
    roc_avg=0
    pred=np.empty_like(truth)
    pred[i] = [1 if pred_prob[i][x] > threshold1 else 0 for x in range(len(pred[i]))]
    fpr, tpr, threshold = metrics.roc_curve(truth[i], pred[i])
    roc = metrics.auc(fpr, tpr)
    roc_auc.append(roc)
    return (roc_auc)

p1_lis=[]
p2_list=[]
m_list=[]
#m_list=[]
acc_list1=[]
acc_list2=[]
best_m_list=[]
best_acc_list=[]
alpha_list=[]
p=0.3
best_of_bes_acc=0
best_of_bes_m=0
#best_alpha=0
print('read data',time.ctime())
foursquar = get_data('F_edges.txt',' ')
f_edges = []
for edge in foursquar:
    if int(edge[0]) in range(1565) and int(edge[1]) in range(1565):
        f_edges.append((int(edge[0]), int(edge[1])))

twitter=get_data('T_edges.txt',' ')
t_edges_d=[]
for edge in twitter:
    t_edges_d.append((int(edge[0]), int(edge[1])))
t_edges=[]
for (i,j) in t_edges_d:
    if (j,i) in t_edges_d:
        t_edges.append((i,j))
print('read data',time.ctime())
threshold=0
n=1565
m=n*(n-1)/2
k_net=2
alpha=0.2

graphs = []
nodes=list(range(1,1565))
G_T=nx.Graph()
G_T.add_nodes_from(nodes)
G_T.add_edges_from(t_edges)
#G_T = nx.barabasi_albert_graph(n, m)
graphs.append(G_T)
G_F=nx.Graph()
G_F.add_nodes_from(nodes)
G_F.add_edges_from(f_edges)
#G_F =G_T#nx.barabasi_albert_graph(n, m)
graphs.append(G_F)
G_T=0
G_F=0
best_acc1=0
best_acc2=0
best_acc3=0
threshold_list=[]
total_edges=[]
for i in nodes:
	for j in nodes:
		total_edges.append((i,j))
print('totaledges',len(total_edges))
print(len(f_edges))
print(len(t_edges))
T=(len(total_edges))
#T=round(T)
T=round(T*0.2)
T1=round(T*0.8)
train_edges=total_edges[:T1]
test_edges=total_edges[T1:T]

train_edges,test_edges=train_test_split(total_edges, test_size=0.2)
test_edges1,test_edges2=train_test_split(total_edges, test_size=0.5)
graphs_test=[]
print ('test_edge',time.ctime())
matrix_F=create_matrix(graphs[1].edges(),n)
matrix_T=create_matrix(graphs[0].edges(),n)

matrix=[]
matrix.append(matrix_T)
matrix.append(matrix_F)

print(property(G_F))
G_F_test=copy.deepcopy(graphs[1])
G_T_test=copy.deepcopy(graphs[0])

for (i,j) in test_edges:
    if matrix[1][i,j]==1:
        G_F_test.remove_edge(*(i,j))
    if matrix[0][i,j]==1:
        G_T_test.remove_edge(*(i,j))
matrix_F_test=create_matrix(G_F_test.edges(),n)
matrix_T_test=create_matrix(G_T_test.edges(),n)
M_test=[]
M_test.append(matrix_T_test)
M_test.append(matrix_F_test)

graphs_test.append(G_T_test)
graphs_test.append(G_F_test)

print ('test_edge',time.ctime())
#Train
(accuracy,x, predictions_prob, truth) = prob_in_net(n, m,graphs_test,train_edges, test_edges1,matrix,M_test)

#Here we tried to find the best alpha and threshold for getting the best output
max1=0
max2=0
best_alpha_list=np.zeros(k_net)
while alpha<1:
    (acc2, sim, preds) = prob_with_sim_AASN(predictions_prob,test_edges1 ,truth, best_alpha_list, threy,matrix,graphs_train,k_net,nodes)
    acc_list2_T = [thresholding(0,preds,truth,th)[0] for th in threy_list]
    m1=max(acc_list2_T)
    if m1>max1:
        max1=m1
        th1=acc_list2_T.index(m1)
        th1=threy_list[th1]
        best_alpha_list[0]=alpha
    acc_list2_F= [thresholding(1,preds,truth,th)[0] for th in threy_list]
    m2=max(acc_list2_F)
    if m2>max2:
        max2=m2
        th2=acc_list2_F.index(m2)
        th2=threy_list[th2]
        best_alpha_list[1]=alpha
    alpha+=0.3
 # Test
(acc2, sim, preds) = prob_with_sim_AASN(predictions_prob,test_edges2 ,truth, best_alpha_list, threshold,matrix,graphs_test,k_net,nodes)
acc_list2_T = thresholding(0,preds,truth,th1)[0]
acc_list2_F= thresholding(1,preds,truth,th2)[0]