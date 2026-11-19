#dependencies: numpy, polars, sklearn

import numpy as np

import polars as pl
from polars import read_csv, selectors, col # basic polars 

from sklearn.metrics.pairwise import cosine_similarity # finds distance between two points

from sklearn import model_selection #stratification k-fold
from sklearn.metrics import accuracy_score # accurary



### Hierarchical Clustering function


"""
knowns: array of training data
unknowns: array of data to categorize
cut: the distance threshold
"""
def find_average_linkage(knowns: pl.DataFrame, unknowns: pl.DataFrame, cut = 0.3):
    clusters = {}

    # convert df to vectors
    known_vectors = knowns.select(pl.exclude(STYLE_COL)).to_numpy()

    unknown_vectors = unknowns.select(pl.exclude(STYLE_COL)).to_numpy()
    

    for style in knowns[STYLE_COL].unique().to_list():
        memberIndices = np.where(knowns[STYLE_COL].to_numpy() == style)[0]
        clusters[style] = {
            "known_members": memberIndices,
            "members": [],
            "class": style
            } # creates points in cluster

    for unknownI in range(len(unknown_vectors)):
        clusters[f"unknown_{unknownI}"] = {
            "known_members": [],
            "members": [unknownI],
            "class": None
            } # every unknown is their own cluster
    
    
    # create similarity matrix

    known_unknown = cosine_similarity(known_vectors, unknown_vectors)

    unknown_unknown = cosine_similarity(unknown_vectors)


    cluster_sizes = {}

    for name, cluster in clusters.items():

        cluster_sizes[name] = (
            len(cluster["known_members"]) +
            len(cluster["members"])
        )

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
        class_b = clusters[b]["class"]

        size_a = cluster_sizes[a]
        size_b = cluster_sizes[b]

        new_similarities = {}

        # shortcuts
        replaceByB = (class_a is None)
        newParent = b if replaceByB else a
        newParentCluster = clusters[newParent]


        #update similarities between this merged cluster and others
        for other in clusters:

            if other in (a, b):
                continue

            # skips another known cluster
            if not can_merge(newParentCluster, clusters[other]):
                continue

            key_a = tuple(sorted((a, other)))
            key_b = tuple(sorted((b, other)))

            sim_a = cluster_similarities[key_a]
            sim_b = cluster_similarities[key_b]

            # new similarity calculation: weights of both averages together
            new_similarity = (size_a * sim_a + size_b * sim_b) / (size_a + size_b)


            # declare new similarities array
            new_similarities[tuple(sorted((newParent, other)))] = new_similarity



        # merge step
        if(replaceByB):
            # remove old similiarities
            cluster_similarities = {
                key: value
                for key, value in cluster_similarities.items()
                if a not in key and b not in key
            }
            # merge
            clusters[class_b]["members"] += clusters[a]["members"]

            # update size
            cluster_sizes[class_b] = size_a + size_b
            # release cluster
            del cluster_sizes[class_a]
            del clusters[a]
        else:
            cluster_similarities = {
                key: value
                for key, value in cluster_similarities.items()
                if b not in key and a not in key
            }
            clusters[class_a]["members"] += clusters[b]["members"] 
            cluster_sizes[class_a] = size_a + size_b
            del cluster_sizes[class_b]
            del clusters[b]

        
        cluster_similarities.update(new_similarities)

        
    return clusters

### training section

# say df is a dataframe of all the info

dataPath = ".../" 

STYLE_COL = "style"

df = read_csv(dataPath, separator=",", has_header=False) # contains image vectors

### Split Data into training, verification, test /w startification

# split into training + test
ss = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
X = df.select(pl.exclude(STYLE_COL)).to_numpy() 
Y = df.select(pl.col(STYLE_COL)).to_numpy()

for train_index, test_index in ss.split(X, Y):
    completeTrainDF = df.gather(train_index)
    skf = model_selection.StratifiedKFold(n_splits=5)
    X2 = completeTrainDF.select(pl.exclude(STYLE_COL)).to_numpy()
    Y2 = completeTrainDF.select(pl.col(STYLE_COL)).to_numpy()

### verification section

    #K Fold
    for fold, (train_index, ver_index) in enumerate(skf.split(X2, Y2)):
        trainDF = completeTrainDF.gather(train_index)
        verifyDF = completeTrainDF.gather(ver_index)

        ### Standarization

        #standardizating features (can be removed if needed)
        means = trainDF.mean()
        sds = trainDF.std()
        for colName in trainDF.columns:
            if colName != STYLE_COL:
                trainDF = trainDF.with_columns(((pl.col(colName) - means.get_column(colName)[0]) / sds.get_column(colName)[0]).alias(colName))
                verifyDF = verifyDF.with_columns(((pl.col(colName) - means.get_column(colName)[0]) / sds.get_column(colName)[0]).alias(colName))

        foundClusters = find_average_linkage(trainDF, verifyDF)

        #check all clusters
        trueClass = verifyDF.select(pl.col(STYLE_COL)).to_numpy()
        predictedClass = ["Unknown"] * verifyDF.height
        for style, cluster in foundClusters:
            #check unknown members (cluster[members] is an array of indices)
            predictedClass[cluster["members"]] = style
        
        accuracy = accuracy_score(trueClass, predictedClass)

        




### testing


"""
    testDF = df.gather(test_index)

    predictedClass = ["Unknown"] * testDF.height
    trueClass = testDF.select(pl.col(STYLE_COL)).to_numpy()
    #standardizating features (can be removed if needed)
    means = completeTrainDF.mean()
    sds = completeTrainDF.std()
    for colName in completeTrainDF.columns:
        if colName != STYLE_COL:
            completeTrainDF = completeTrainDF.with_columns(((pl.col(colName) - means.get_column(colName)[0]) / sds.get_column(colName)[0]).alias(colName))
            testDF = testDF.with_columns(((pl.col(colName) - means.get_column(colName)[0]) / sds.get_column(colName)[0]).alias(colName))

    foundClusters = find_average_linkage(completeTrainDF, testDF)
    for style, cluster in foundClusters:
        #check unknown members (cluster[members] is an array of indices)
        predictedClass[cluster["members"]] = style
    
    accuracy = accuracy_score(trueClass, predictedClass)
"""