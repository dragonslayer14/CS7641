# Observations

## K-means

- elbow seems to be 2-3 on dataset 1
    - went with 3
- elbow seems to be 3-5 on dataset 2
    - raw numbers seem like 5 is where the drops are not as drastic
- need to use something to "describe plots"

## EM

- bic plots seem to put dataset 1 at about 3 components
    - extended it will choose 18, but it's not too much lower to justify 6x
- plot for dataset 2 puts it around 9-12 components
    - 12 isn't unreasonable
    - extend to 50 and 12 is still picked, so a good fit and not terribly large

## PCA

- data set 1 gets to about 90% with 6 features and 95 at 8
    - 90 is probably good enough, plus it's a nice round halving
    - 8 seems to get to 95% and 6% recon error as opposed to 13
- data set 2 does about the same, but 95 at 7
    - 14% recon at 6, 10 at 7, about 5 at 8

- kmeans seems to have similar values, 3 for 1 and 5 for 2
- em is a little more varied, up to 20, 14 for 1 and 12 for 2
    - up to 50, 2 stays at 12, so that is a good level, but 1 goes to 50 
    - up to 100, 2 stays at 12, but 1 goes to 100 with negative bic
        - 1's behavior is likely due to getting close to 1 generator per point, which is not great
            - 14 is probably a fine place

## ICA

- dataset 1, max kurtosis at 12 components, max components 2,5,8,10
- dataset 2, max kurtosis at 10 components, max components 2,5,3,8

- kmeans, seems to elbow slightly around 10 for 1 and 13 for 2
- em, spherical 5 components for 1 and diagonal 9 for 2 
    - only negative values, but it is only the relative scores that matter, not absolutes

# RCA

- both perform pretty poorly on recon error all through, cutting to about 20% error
    - 10 for 1, 9 for 2
- kmeans 3 on 1 and 4 on 2
- em full 10 on 2 and 20 on 1 (up to 50)
- from lecture: m is likely not as small as m from PCA or ICA, since it is not targeted

## LDA

- should perform poorly on dataset 1, since it only have 3 classes
    - should be better on 2 since there are technically 7-ish
- not using the same variance cutoff graph as pca since the number of classes < features, so there is already reduction

## ANN

scores/times
ANN (kmeans) score: 0.4390150057714506
ANN (kmeans) time: 0.79
ANN (em) score: 0.49595998460946517
ANN (em) time: 1.42
ANN (pca) score: 0.5263562908811081
ANN (pca) time: 0.82
ANN (ica) score: 0.4367064255482878
ANN (ica) time: 0.69
ANN (rca) score: 0.5236629472874182
ANN (rca) time: 0.64
ANN (lda) score: 0.5390534821085032
ANN (lda) time: 0.57

## scores

K-means
1: 0.7286612758310872
2: 0.5991071001848128

EM
1: 0.9502246181491465
2: 0.6341011945224208

PCA
1: 0.13368620486872015
2: 0.14808237812267594

+ Kmeans
1: 0.9141060197663972
2: 0.6095320327051647
+ EM
1: 0.732973944294699
2: 0.6416253678448894

ICA
1: 0.007344952143856595
pulling components [4, 1, 0, 3]
2: 0.0031265702751792374
pulling components [2, 8, 3, 9]

+kmeans 
1: 0.6165318957771788
2: 0.6353606654050892
+EM
1: 0.3786163522012579
2: 0.5808616250176262

RCA
1: 0.20496830048707013
2: 0.14205137932004377

+kmeans
1: 0.9141060197663972
2: 0.5814688463965585

+em
1: 0.7132075471698113
2: 0.6354058251910766


LDA
1: 1.0
2: 0.5469471523858389

kmeans
1: 1.0
2: 0.6341904608049265

em
1: 0.8880503144654088
2: 0.6223479323203586

ANN

kmeans
0.4390150057714506
em
0.4974990380915737
PCA
0.5244324740284725
ICA
ANN (ica) score: 0.5298191612158523
ANN (ica) time: 1.00
RCA
0.5267410542516352
LDA
0.5467487495190458