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
