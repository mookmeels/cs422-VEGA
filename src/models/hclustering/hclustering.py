#dependencies: numpy, polars, sklearn, sys
"""
Hclustering(unstandardized 0.59)   validation seed 422   
top-5 36.6%  top-1 24.1%  MRR 0.304  per-label top-5 37.7% (thin 38.2%, rest 37.2%)  
coverage 71.8%  family top-1 42.3%, in top 3 63.0%  [1351 images, 503 labels]

Hclustering(standardized 0.16)     validation seed 423   
top-5 44.4%  top-1 26.5%  MRR 0.348  per-label top-5 43.1% (thin 40.6%, rest 46.1%)  
coverage 80.3%  family top-1 47.2%, in top 3 69.4%  [1351 images, 503 labels]
"""


import sys, numpy as np

import polars as pl
from polars import read_csv, selectors, col # basic polars 

from sklearn.metrics.pairwise import cosine_similarity # finds distance between two points


sys.path.insert(0, "/srv/ml/vega/code/shared")
from vega_data import load

d = load()

# update to classes
classesIndices = d["classes"]

label_to_idx = {label: idx for idx, label in enumerate(classesIndices)}
### Hierarchical Clustering function

STYLE_COL = "class"
ID_COL = "id"

"""
The Hierarchical Clustering Method

knowns: array of training data
unknowns: array of data to categorize
cut: the distance threshold

returns an array of all the unknown samples' score to all the classes
"""
def find_average_linkage(knowns: pl.DataFrame, unknowns: pl.DataFrame, cut = 0.16):
    clusters = {}

    # convert df to vectors
    known_vectors = knowns.select(pl.exclude(STYLE_COL, ID_COL)).to_numpy()

    unknown_vectors = unknowns.select(pl.exclude(STYLE_COL, ID_COL)).to_numpy()

    

    for style in knowns[STYLE_COL].unique().to_list():
        memberIndices = np.where(knowns[STYLE_COL].to_numpy() == style)[0]
        clusters[style] = {
            "known_members": memberIndices,
            "members": [],
            "class": style,
            "classId": label_to_idx[style]
            } # creates points in cluster

    for unknownI in range(len(unknown_vectors)):
        clusters[f"unknown_{unknownI}"] = {
            "known_members": [],
            "members": [unknownI],
            "class": None
            } # every unknown is their own cluster
    
    
    # predictions for unknown classes
    predicted_unknowns = [
        [0.0] * len(classesIndices)
        for _ in range(len(unknown_vectors))
    ]

    # create similarity matrix
    known_unknown = cosine_similarity(known_vectors, unknown_vectors)

    unknown_unknown = cosine_similarity(unknown_vectors)

    # initialize cluster sizes
    cluster_sizes = {}

    for name, cluster in clusters.items():

        cluster_sizes[name] = (
            len(cluster["known_members"]) +
            len(cluster["members"])
        )

    #initialize similarilities
    cluster_similarities = {}

    for style, known_cluster in clusters.items():
        if known_cluster["class"] is None:
            continue
        for unknown_name, unknown_cluster in clusters.items():
            if unknown_cluster["class"] is not None:
                continue

            known_indices = known_cluster["known_members"]
            unknown_indices = unknown_cluster["members"] 

            # calculate all similarities
            similarity = known_unknown[
                np.ix_(known_indices, unknown_indices)
            ].mean()

            # record similarities
            cluster_similarities[tuple(sorted((style, unknown_name)))] = similarity


    # unknown_unknown
    unknown_names = [
        name for name, cluster in clusters.items()
        if cluster["class"] is None
    ]

    # pair all unknowns
    for i, a in enumerate(unknown_names):

        for b in unknown_names[i + 1:]:

            ia = clusters[a]["members"][0] 
            ib = clusters[b]["members"][0] 

            similarity = unknown_unknown[ia, ib]

            cluster_similarities[
                tuple(sorted((a, b)))
            ] = similarity

    """
    checks if the two clusters cna merge

    cluster_a: cluster one
    cluster_b: cluster two

    returns whether it can merge or not
    """
    def can_merge(cluster_a, cluster_b):

        class_a = cluster_a["class"]
        class_b = cluster_b["class"]

        # Two different known classes cannot merge
        if (
            class_a is not None
            and class_b is not None
            and class_a != class_b
        ):
            return False

        return True

    # main merging process
    while True:
        cluster_names = list(clusters.keys())

        best_similarity = float("-inf")
        best_pair = None

        # searches for highest similarity
        for (a, b), similarity in cluster_similarities.items():

            # Don't waste time with two knowns
            if (not can_merge(clusters[a], clusters[b])):
                continue

            # replace if lowest distance
            if similarity > best_similarity:
                best_similarity = similarity
                best_pair = (a, b)

        # if no more unknowns / no matching unknowns
        if best_pair is None or best_similarity < cut:
            break

    
        a, b = best_pair

        # Merge clusters
        class_a = clusters[a]["class"]

        new_similarities = {}

        # shortcuts
        replaceByB = (class_a is None)
        removedParent = a if replaceByB else b

        allChanged = {a, b}

        # multiclass predictions
        if(class_a is not None or clusters[b]["class"] is not None):
            if(class_a is None):
                # b is the known class
                for (a2, b2), similarity in cluster_similarities.items():
                    #check other similarities
                    if a in (a2, b2) and similarity >= cut and (clusters[a2]["class"] is not None or clusters[b2]["class"] is not None):
                        # another possibility
                        
                        if clusters[a2]["class"] is not None:
                            classId = clusters[a2]["classId"]
                            allChanged.add(a2)
                        else: 
                            classId = clusters[b2]["classId"]
                            allChanged.add(b2)
                        for member in clusters[a]["members"]:
                            predicted_unknowns[member][classId] = similarity
                        

            else:
                # a
                for (a2, b2), similarity in cluster_similarities.items():
                    #check other similarities
                    if b in (a2, b2) and similarity >= cut and (clusters[a2]["class"] is not None or clusters[b2]["class"] is not None):
                        # another possibility
                        if clusters[a2]["class"] is not None:
                            classId = clusters[a2]["classId"]
                            allChanged.add(a2)
                        else: 
                            classId = clusters[b2]["classId"]
                            allChanged.add(b2)
                        for member in clusters[b]["members"]:
                            predicted_unknowns[member][classId] = similarity


        #update similarities between this merged cluster and others
        for dest in allChanged:

            # avoid checking merging cluster 
            if dest == removedParent:
                continue

            #dest now encompasses all known clusters that have been changed

            for otherUnknown in clusters:
                if clusters[otherUnknown]["class"] is not None and clusters[dest]["class"] is not None:
                    continue

                #avoid merging, nor its own unknown class
                if(otherUnknown in (removedParent, dest)):
                    continue

                

                #otherUnknown cluster now encompasses all other unknown clusters
                key_a = tuple(sorted((dest, otherUnknown)))
                key_b = tuple(sorted((removedParent, otherUnknown)))

                sim_a = cluster_similarities[key_a]
                size_a = cluster_sizes[dest]

                sim_b = cluster_similarities[key_b]
                size_b = cluster_sizes[removedParent]
                # new similarity calculation: weights of both averages together
                new_similarity = (size_a * sim_a + size_b * sim_b) / (size_a + size_b)


                # declare new similarities array
                new_similarities[key_a] = new_similarity



        # merge step
        if(replaceByB):
            # remove old similiarities (add similarities that are not replaced)
            cluster_similarities = {
                key: value
                for key, value in cluster_similarities.items()
                if set(key).isdisjoint(allChanged)
            }
            # merge
            #when both are unknown
            if clusters[b]["class"] is None:
                
                clusters[b]["members"] += clusters[a]["members"] # same for here
                cluster_sizes[b] += cluster_sizes[a]
            else:
            #when theres a known
                for styleClass in allChanged:
                    #not inputing into unknown
                    if(clusters[styleClass]["class"] is None):
                        continue
                    for addedMember in clusters[a]["members"]:
                        clusters[styleClass]["members"] += [addedMember] # could add confidence here

                    # update size
                    cluster_sizes[styleClass] += cluster_sizes[a]
            # release cluster
            #print(a, b)
            #print(clusters[a]["class"], clusters[b]["class"])
            
            del cluster_sizes[a]
            del clusters[a]
        else:
            cluster_similarities = {
                key: value
                for key, value in cluster_similarities.items()
                if set(key).isdisjoint(allChanged)
            }
            
            #if b is known
            for styleClass in allChanged:
                #not inputing into unknown
                if(clusters[styleClass]["class"] is None):
                    continue
                for addedMember in clusters[b]["members"]:
                    clusters[styleClass]["members"] += [addedMember] # could add confidence here

                # update size
                cluster_sizes[styleClass] += cluster_sizes[b]
            # release cluster
            #print(b, a)
            
            del cluster_sizes[b]
            del clusters[b]

        
        cluster_similarities.update(new_similarities)

        
    return predicted_unknowns

### training section

# Training DataFrame (id to label)
"""
Turns training data into a polar dataframe of the associated classes, will duplicate those with multiple classes
to multiple samples representing each of the classes

split: The data to convert to polars dataframe

returns polars dataframe
"""
def make_long_df(split):
    y = split["y"]
    df = pl.DataFrame({
        "id": split["ids"],
        **{
            label: y[:, i]
            for i, label in enumerate(d["classes"])
        }
    })

    return (
        df
        .unpivot(
            on=d["classes"],
            index="id",
            variable_name=STYLE_COL,
            value_name="weight"
        )
        .filter(pl.col("weight") > 0)
        .drop("weight")
    )
# create all class rows
train_df = make_long_df(d["fit"])


#format the eval df to polars
eval_df = pl.DataFrame({
    "id": d["eval"]["ids"],
    **{f"emb_{i}": d["eval"]["x"][:, i]
        for i in range(d["eval"]["x"].shape[1])
    }
})

"""
Joins the data associated with the classes based on id so that the duplicated samples have the same data except for class

df: dataframe
split: the data that shows the embedding and id

returns the completed dataframe
"""
def add_embeddings(df, split):
    embeddings = pl.DataFrame(
        split["x"],
        schema=[f"emb_{i}" for i in range(split["x"].shape[1])],
        orient="row"
    ).with_columns(
        pl.Series(ID_COL, split["ids"])
    )

    return df.join(embeddings, on="id")
# add embeddings to each of the classes
train_df = add_embeddings(train_df, d["fit"])

# standardizing


means = train_df.mean()
sds = train_df.std()

for colName in train_df.columns:
    if colName not in (STYLE_COL, ID_COL):
        train_df = train_df.with_columns(((col(colName) - means.get_column(colName)[0]) / sds.get_column(colName)[0]).alias(colName))
        eval_df = eval_df.with_columns(((col(colName) - means.get_column(colName)[0]) / sds.get_column(colName)[0]).alias(colName))
# eval_df = eval_df.head(250)


predictions = find_average_linkage(train_df, eval_df)

scores = np.asarray(predictions, dtype=np.float32)

print(scores.shape)
print(len(d["eval"]["ids"]), len(d["classes"]))
assert scores.shape == (len(d["eval"]["ids"]), len(d["classes"]))


# Save in correct format
np.savez(
    "my-preds.npz",
    ids=d["eval"]["ids"],
    scores=scores,
    classes=np.asarray(d["classes"])
)
