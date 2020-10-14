import pandas as pd
import numpy as np
import pickle, os
import networkx as nx


def baseline_random_sampling(positive_train, positive_test, reverse_train, reverse_test, negative_data_full, n_sample):
    train, val, test = [], [], []
    probability_neg = []
    for (u,v,weight) in negative_data_full: probability_neg.append(1)
    draw_neg = np.random.choice([i for i, neg in enumerate(negative_data_full)], n_sample, p=[val/sum(probability_neg) for val in probability_neg])
    negatives = [negative_data_full[i] for i in draw_neg]
    
    probability_pos = []
    for (u,v,weight) in positive_train: probability_pos.append(1)
    draw_pos = np.random.choice([i for i, pos in enumerate(positive_train)], n_sample, p=[val/sum(probability_pos) for val in probability_pos])
    positives = [positive_train[i] for i in draw_pos]
    reverses = [reverse_train[i] for i in draw_pos]
    
    n_draw = { i:0 for i in range(len(negative_data_full))}
    for i in draw_neg: n_draw[i]+=1
    df_neg = pd.DataFrame({'uv':negative_data_full, 'draw':[n_draw[i] for i in range(len(negative_data_full))] }) 
    n_draw = { i:0 for i in range(len(positive_train))}
    for i in draw_pos: n_draw[i]+=1
    df_pos = pd.DataFrame({'uv':positive_train, 'draw':[n_draw[i] for i in range(len(positive_train))] })

    train = negatives + positives
    val = negatives[0:int(len(negatives)*0.25)] + positives[0:int(len(positives)*0.25)]
    test = reverse_test + positive_test
    return train, val, test, df_neg, df_pos, draw_neg, draw_pos   
  
    
def baseline_reverse_sampling(positive_train, positive_test, reverse_train, reverse_test, n_sample):
    train, val, test = [], [], []
    probability_pos = []
    for (u,v,weight) in positive_train: probability_pos.append(1)
    draw_pos = np.random.choice([i for i, pos in enumerate(positive_train)], n_sample, p=[val/sum(probability_pos) for val in probability_pos])
    positives = [positive_train[i] for i in draw_pos]
    reverses = [reverse_train[i] for i in draw_pos]
    
    n_draw = { i:0 for i in range(len(reverse_train))}
    for i in draw_pos: n_draw[i]+=1
    df_neg = pd.DataFrame({'uv':reverse_train, 'draw':[n_draw[i] for i in range(len(reverse_train))] }) 
    n_draw = { i:0 for i in range(len(positive_train))}
    for i in draw_pos: n_draw[i]+=1
    df_pos = pd.DataFrame({'uv':positive_train, 'draw':[n_draw[i] for i in range(len(positive_train))] })

    train = reverses + positives
    val = reverses[0:int(len(reverses)*0.25)] + positives[0:int(len(positives)*0.25)]
    test = reverse_test + positive_test
    return train, val, test, df_neg, df_pos, draw_pos, draw_pos


def local_neighborhood_sampling_fast(pos_attribute, neg_attribute, positive_train, positive_test, reverse_train, reverse_test, negative_data_full, n_sample, strategy): 
    (inu, outu, inv, outv, l3, l4, l5, l6, g3, g4, g5, g6, ancu, ancv, auv, descu, descv, cuv ) = neg_attribute
    probability_neg = []
    if strategy == "l1": probability_neg= inu
    elif strategy == "l2": probability_neg = outv
    elif strategy == "l3": probability_neg = l3
    elif strategy == "l4": probability_neg = l4   
    elif strategy == "l5": probability_neg = l5
    elif strategy == "l6": probability_neg = l6

    draw_neg = np.random.choice([i for i, neg in enumerate(negative_data_full)], n_sample, p=[val/sum(probability_neg) for val in probability_neg])
    negatives = [negative_data_full[i] for i in draw_neg]
    
    probability_pos = []
    for (u,v,weight) in positive_train: probability_pos.append(1)
    draw_pos = np.random.choice([i for i, pos in enumerate(positive_train)], n_sample, p=[val/sum(probability_pos) for val in probability_pos])
    positives = [positive_train[i] for i in draw_pos]
    reserves = [reverse_train[i] for i in draw_pos]
    
    n_draw = { i:0 for i in range(len(negative_data_full))}
    for i in draw_neg: n_draw[i]+=1
    df_neg = pd.DataFrame({'uv':negative_data_full, 'draw':[n_draw[i] for i in range(len(negative_data_full))], 'puv':[val/sum(probability_neg) for val in probability_neg], 'ancu':ancu, 'ancv':ancv, 'auv':auv, 'ancu+descv':g3, 'ancu/descu':g4, 'descv/ancv':g5, '(ancu/descu)+(descv/ancv)':g6, 'descu':descu, 'descv':descv, 'cuv':cuv, 'in_u':inu, 'out_u':outu, 'in_v':inv, 'out_v':outv, 'in_u+out_v':l3, '(in_u/out_u)':l4, '(out_v/in_v)':l5, '(in_u/out_u)+(out_v/in_v)':l6}) 
    
    (inu, outu, inv, outv, l3, l4, l5, l6, g3, g4, g5, g6, ancu, ancv, auv, descu, descv, cuv ) = pos_attribute
    n_draw = { i:0 for i in range(len(positive_train))}
    for i in draw_pos: n_draw[i]+=1
    df_pos = pd.DataFrame({'uv':positive_train, 'draw':[n_draw[i] for i in range(len(positive_train))], 'puv':[val/sum(probability_pos) for val in probability_pos], 'ancu':ancu, 'ancv':ancv, 'auv':auv, 'ancu+descv':g3, 'ancu/descu':g4, 'descv/ancv':g5, '(ancu/descu)+(descv/ancv)':g6, 'descu':descu, 'descv':descv, 'cuv':cuv, 'in_u':inu, 'out_u':outu, 'in_v':inv, 'out_v':outv, 'in_u+out_v':l3, '(in_u/out_u)':l4, '(out_v/in_v)':l5, '(in_u/out_u)+(out_v/in_v)':l6}) 

    train = negatives + positives
    val = negatives[0:int(len(negatives)*0.25)] + positives[0:int(len(positives)*0.25)]
    test = reverse_test + positive_test
    return train, val, test, df_neg, df_pos, draw_neg, draw_pos


def global_neighborhood_sampling_fast(pos_attribute, neg_attribute, positive_train, positive_test, reverse_train, reverse_test, negative_data_full, n_sample, strategy): 
    train, val, test = [], [], []

    (inu, outu, inv, outv, l3, l4, l5, l6, g3, g4, g5, g6, ancu, ancv, auv, descu, descv, cuv ) = neg_attribute
    probability_neg = []
    if strategy == "g1": probability_neg= ancu
    elif strategy == "g2": probability_neg = descv
    elif strategy == "g3": probability_neg = g3
    elif strategy == "g4": probability_neg = g4   
    elif strategy == "g5": probability_neg = g5
    elif strategy == "g6": probability_neg = g6

    draw_neg = np.random.choice([i for i, neg in enumerate(negative_data_full)], n_sample, p=[val/sum(probability_neg) for val in probability_neg])
    negatives = [negative_data_full[i] for i in draw_neg]
    
    probability_pos = []
    for (u,v,weight) in positive_train: probability_pos.append(1)
    draw_pos = np.random.choice([i for i, pos in enumerate(positive_train)], n_sample, p=[val/sum(probability_pos) for val in probability_pos])
    positives = [positive_train[i] for i in draw_pos]
    reserves = [reverse_train[i] for i in draw_pos]
    
    n_draw = { i:0 for i in range(len(negative_data_full))}
    for i in draw_neg: n_draw[i]+=1
    df_neg = pd.DataFrame({'uv':negative_data_full, 'draw':[n_draw[i] for i in range(len(negative_data_full))], 'puv':[val/sum(probability_neg) for val in probability_neg], 'ancu':ancu, 'ancv':ancv, 'auv':auv, 'ancu+descv':g3, 'ancu/descu':g4, 'descv/ancv':g5, '(ancu/descu)+(descv/ancv)':g6, 'descu':descu, 'descv':descv, 'cuv':cuv, 'in_u':inu, 'out_u':outu, 'in_v':inv, 'out_v':outv, 'in_u+out_v':l3, '(in_u/out_u)':l4, '(out_v/in_v)':l5, '(in_u/out_u)+(out_v/in_v)':l6}) 
    
    (inu, outu, inv, outv, l3, l4, l5, l6, g3, g4, g5, g6, ancu, ancv, auv, descu, descv, cuv ) = pos_attribute
    n_draw = { i:0 for i in range(len(positive_train))}
    for i in draw_pos: n_draw[i]+=1
    df_pos = pd.DataFrame({'uv':positive_train, 'draw':[n_draw[i] for i in range(len(positive_train))], 'puv':[val/sum(probability_pos) for val in probability_pos], 'ancu':ancu, 'ancv':ancv, 'auv':auv, 'ancu+descv':g3, 'ancu/descu':g4, 'descv/ancv':g5, '(ancu/descu)+(descv/ancv)':g6, 'descu':descu, 'descv':descv, 'cuv':cuv, 'in_u':inu, 'out_u':outu, 'in_v':inv, 'out_v':outv, 'in_u+out_v':l3, '(in_u/out_u)':l4, '(out_v/in_v)':l5, '(in_u/out_u)+(out_v/in_v)':l6}) 

    train = negatives + positives
    val = negatives[0:int(len(negatives)*0.25)] + positives[0:int(len(positives)*0.25)]
    test = reverse_test + positive_test
    return train, val, test, df_neg, df_pos, draw_neg, draw_pos


def descendant_relation_check(w,negative_train_full,desc):
    relation_w = [ (idx,u,v,label) for idx, (u,v,label) in enumerate(negative_train_full) if u in desc[w] and v in desc[w]]
    valid_relation_idx = []
    for (idx,u,v,label) in relation_w:
        if not v in desc[u] and not v == u: valid_relation_idx.append(idx)
    return valid_relation_idx 

def descendant_sampling_fast(neighborhood, pos_attribute, neg_attribute, positive_train, positive_test, reverse_train, reverse_test, negative_data_full, n_sample, strategy):
    train, val, test = [], [], []
    (inu, outu, inv, outv, l3, l4, l5, l6, g3, g4, g5, g6, ancu, ancv, auv, descu, descv, cuv ) = neg_attribute

    probability_neg = []
    if strategy == 'd0': probability_neg = [1] * len(inu)
    elif strategy == "d1": probability_neg= inu
    elif strategy == "d2": probability_neg = outv
    elif strategy == "d3": probability_neg = l3
    elif strategy == "d4": probability_neg = l4   
    elif strategy == "d5": probability_neg = l5
    elif strategy == "d6": probability_neg = l6 
    elif strategy == "dg1": probability_neg= ancu
    elif strategy == "dg2": probability_neg = descv
    elif strategy == "dg3": probability_neg = g3
    elif strategy == "dg4": probability_neg = g4   
    elif strategy == "dg5": probability_neg = g5
    elif strategy == "dg6": probability_neg = g6
    
    node2desc = list(zip(*neighborhood))[5]
    desc = { w:list(desc_w) for w, desc_w in enumerate(node2desc)}

    negatives = []
    draw_neg = []
    n_draw = { i:0 for i in range(len(negative_data_full))}
    invalid_nodes = 0
    while len(negatives) < n_sample:
        for w, desc_w in desc.items(): 
            if len(negatives) == n_sample: break
            if len(desc_w) < 2: invalid_nodes+=1; continue
            subset = descendant_relation_check(w,negative_data_full,desc)
            subprob = [probability_neg[r_idx] for r_idx in subset]   
            if sum(subprob) == 0: subprob; continue
            multi_draw = np.random.multinomial(n_sample, [val/sum(subprob) for val in subprob], size=1)


            for i, cnt in enumerate(multi_draw[0]):
                if cnt == 0: continue
                r_idx = subset[i]
                negatives += [negative_data_full[r_idx]]*cnt
                draw_neg += [r_idx]*cnt
                n_draw[r_idx]+=1  

    
    probability_pos = []
    for (u,v,weight) in positive_train: probability_pos.append(1)
    draw_pos = np.random.choice([i for i, pos in enumerate(positive_train)], n_sample, p=[val/sum(probability_pos) for val in probability_pos])
    positives = [positive_train[i] for i in draw_pos]
    reserves = [reverse_train[i] for i in draw_pos]
#     print("\t", strategy, "len(positives), len(reserves), len(negatives):", len(positives), len(reserves), len(negatives))
    
    df_neg = pd.DataFrame({'uv':negative_data_full, 'draw':[n_draw[i] for i in range(len(negative_data_full))], 'puv':[val/sum(probability_neg) for val in probability_neg], 'ancu':ancu, 'ancv':ancv, 'auv':auv, 'ancu+descv':g3, 'ancu/descu':g4, 'descv/ancv':g5, '(ancu/descu)+(descv/ancv)':g6, 'descu':descu, 'descv':descv, 'cuv':cuv, 'in_u':inu, 'out_u':outu, 'in_v':inv, 'out_v':outv, 'in_u+out_v':l3, '(in_u/out_u)':l4, '(out_v/in_v)':l5, '(in_u/out_u)+(out_v/in_v)':l6}) 
    (inu, outu, inv, outv, l3, l4, l5, l6, g3, g4, g5, g6, ancu, ancv, auv, descu, descv, cuv ) = pos_attribute
    n_draw = { i:0 for i in range(len(positive_train))}
    for i in draw_pos: n_draw[i]+=1
    df_pos = pd.DataFrame({'uv':positive_train, 'draw':[n_draw[i] for i in range(len(positive_train))], 'puv':[val/sum(probability_pos) for val in probability_pos], 'ancu':ancu, 'ancv':ancv, 'auv':auv, 'ancu+descv':g3, 'ancu/descu':g4, 'descv/ancv':g5, '(ancu/descu)+(descv/ancv)':g6, 'descu':descu, 'descv':descv, 'cuv':cuv, 'in_u':inu, 'out_u':outu, 'in_v':inv, 'out_v':outv, 'in_u+out_v':l3, '(in_u/out_u)':l4, '(out_v/in_v)':l5, '(in_u/out_u)+(out_v/in_v)':l6}) 

    train = negatives + positives
    val = negatives[0:int(len(negatives)*0.25)] + positives[0:int(len(positives)*0.25)]
    test = reverse_test + positive_test    
    return train, val, test, df_neg, df_pos, np.array(draw_neg), draw_pos 

