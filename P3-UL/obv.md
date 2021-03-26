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
1:
2:

+kmeans
1:
2:

+em
1:
2:

LDA
1:
2:

kmeans
1:
2:

em
1:
2:

ANN

kmeans
1:
2:

em
1:
2: