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