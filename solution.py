import difflib

import pandas as pd
import numpy as np
from os.path import join

# 1. read data

ltable = pd.read_csv(join('data', "ltable.csv"))
rtable = pd.read_csv(join('data', "rtable.csv"))
train = pd.read_csv(join('data', "train.csv"))

# 2. blocking
def pairs2LR(ltable, rtable, candset):
    ltable.index = ltable.id
    rtable.index = rtable.id
    pairs = np.array(candset)
    tpls_l = ltable.loc[pairs[:, 0], :]
    tpls_r = rtable.loc[pairs[:, 1], :]
    tpls_l.columns = [col + "_l" for col in tpls_l.columns]
    tpls_r.columns = [col + "_r" for col in tpls_r.columns]
    tpls_l.reset_index(inplace=True, drop=True)
    tpls_r.reset_index(inplace=True, drop=True)
    LR = pd.concat([tpls_l, tpls_r], axis=1)
    return LR

import csv
import pprint
# pprint.pprint(ltable)
def csv_dict_list(name):
    a_csv_file = open(name, "r")
    dict_reader = csv.DictReader(a_csv_file)
    dict_list = []
    for line in dict_reader:
        dict_list.append(line)
    return dict_list

ldict = csv_dict_list("ltable.csv")
rdict = csv_dict_list("rtable.csv")
training = csv_dict_list("train.csv")
# pprint.pprint(training)
result = []
for i in range(len(ldict)):
    for j in range(len(rdict)):
        a = [(int)(ldict[i]["id"]), (int)(rdict[j]["id"])]
        if ldict[i]["brand"] == rdict[j]["brand"]:
            # if contains_digit = any(map(str.isdigit, ldict[i]["modelno"])):
            if ldict[i]["modelno"] == rdict[j]["modelno"]:
                result.append(a)
            elif ldict[i]["modelno"] != "" and rdict[j]["modelno"] != "":

                if ldict[i]["modelno"] in rdict[j]["title"] or rdict[j]["modelno"] in ldict[i]["title"]:
                    result.append(a)
                    break
                else:
                    r = rdict[j]["title"].split()
                    l = ldict[i]["title"].split()
                    for item in r:
                        if difflib.SequenceMatcher(None, ldict[i]["modelno"], item).ratio() >= 0.8:
                            result.append(a)
                            break
                    for item in l:
                        if difflib.SequenceMatcher(None, rdict[j]["modelno"], item).ratio() >= 0.8:
                            result.append(a)
                            break
            elif any(map(str.isdigit, ldict[i]["modelno"])):
                if difflib.SequenceMatcher(None, ldict[i]["modelno"], rdict[j]["modelno"]).ratio() >= 0.85:
                    result.append(a)
            elif difflib.SequenceMatcher(None, ldict[i]["modelno"], rdict[j]["modelno"]).ratio() >= 0.8:
                result.append(a)
# matched = []
# for i in range(len(train)):
#     if training[i]["label"] == "1":
#         matched.append([(int)(training[i]["ltable_id"]), (int)(training[i]["rtable_id"])])
# # print(matched)
# # candset = []
# for item in matched:
#     if item not in result:
#         aa.append(item)
#
# # #
# # for item in result:
# #     if item not in matched:
# #         candset.append(item)
#
# print(len(aa))

# candset_df.to_csv("candset_df.csv", index=False)

# blocking to reduce the number of pairs to be compared

print("number of pairs originally", ltable.shape[0] * rtable.shape[0])
print("number of pairs after blocking", len(result))
candset_df = pairs2LR(ltable, rtable, result)
candset_df.to_csv("candset_df.csv", index=False)
# 3. Feature engineering
import Levenshtein as lev

def jaccard_similarity(row, attr):
    x = set(row[attr + "_l"].lower().split())
    y = set(row[attr + "_r"].lower().split())
    return len(x.intersection(y)) / max(len(x), len(y))


def levenshtein_distance(row, attr):
    x = row[attr + "_l"].lower()
    y = row[attr + "_r"].lower()
    return lev.distance(x, y)

def feature_engineering(LR):
    LR = LR.astype(str)
    attrs = ["title", "category", "price"]
    features = []
    for attr in attrs:
        j_sim = LR.apply(jaccard_similarity, attr=attr, axis=1)
        l_dist = LR.apply(levenshtein_distance, attr=attr, axis=1)
        features.append(j_sim)
        features.append(l_dist)
    features = np.array(features).T
    return features
candset_features = feature_engineering(candset_df)

# also perform feature engineering to the training set
training_pairs = list(map(tuple, train[["ltable_id", "rtable_id"]].values))
training_df = pairs2LR(ltable, rtable, training_pairs)
training_features = feature_engineering(training_df)
training_label = train.label.values

# 4. Model training and prediction
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(class_weight="balanced", random_state=0)
rf.fit(training_features, training_label)
y_pred = rf.predict(candset_features)

# 5. output

matching_pairs = candset_df.loc[y_pred == 1, ["id_l", "id_r"]]
matching_pairs = list(map(tuple, matching_pairs.values))

matching_pairs_in_training = training_df.loc[training_label == 1, ["id_l", "id_r"]]
matching_pairs_in_training = set(list(map(tuple, matching_pairs_in_training.values)))

pred_pairs = [pair for pair in matching_pairs if
              pair not in matching_pairs_in_training]  # remove the matching pairs already in training
pred_pairs = np.array(pred_pairs)
print(len(pred_pairs))
pred_df = pd.DataFrame(pred_pairs, columns=["ltable_id", "rtable_id"])
pred_df.to_csv("output1.csv", index=False)
