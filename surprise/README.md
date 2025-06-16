# TIL-AI 2025 Surprise Challenge

**Contents**
- [TIL-AI 2025 Surprise Challenge](#til-ai-2025-surprise-challenge)
  - [Overview](#overview)
  - [Submission procedure](#submission-procedure)
  - [Challenge specifications](#challenge-specifications)
    - [Input format](#input-format)
    - [Output format](#output-format)
    - [Scoring](#scoring)
      - [In short](#in-short)
      - [In detail](#in-detail)


## Overview

Your task is to develop a system for reassembly of shredded documents. Given slices of a document cut along the vertical axis, your system is to output the permutation of input slices that correctly reassembles the document.

All strips will be of the same width and height. You are guaranteed that all strips are the right way up.


## Submission procedure

You've done the other challenges, so you should know the drill:

```bash
# Build your container
docker build -t TEAM_ID-surprise:TAG .

# Run your container
docker run -p 5005:5005 --gpus all -d TEAM_ID-surprise:TAG

# Test your container
pip install -r requirements-dev.txt
python3 test.py

# Submit
./submit-surprise.sh TEAM_ID-surprise:TAG
```


## Challenge specifications

### Input format

The input is sent via a POST request to the `/surprise` route on port 5005. It is a JSON document structured as such:

```json
{
  "instances": [
    {
      "key": 0,
      "slices": ["base64_encoded_slice", ...]
    },
    ...
  ]
}
```

Each object in the `instances` list corresponds to a single shredded document. Each string in the `slices` list contains the base64-encoded bytes of a single slice of that document represented as a JPEG image.

The length of the `instances` list is variable.


### Output format

Your route handler function must return a `dict` with this structure:

```json
{
    "predictions": [
        [p11, p12, ..., p1s],
        [p21, p22, ..., p2s],
        ...
    ]
}
```

where:

- $s$ is the number of slices in each document, and 
- $[p_{i1}, p_{i2}, \dots, p_{is}]$ is the predicted permutation of input slices that correctly reassembles the $i$-th document, leftmost slice first.

For instance, if you're given a single document in three slices, and you think that:

- the first slice should go in the middle
- the second slice should go on the left
- the third slice should go on the right

then you would return:

```json
{
    "predictions": [
        [1, 0, 2]
    ]
}
```

The $k$-th element of `predictions` must be the prediction corresponding to the $k$-th element of `instances` for all $1 \le k \le n$, where $n$ is the number of input instances. The length of `predictions` must equal that of `instances`.


### Scoring

#### In short

- Predictions that contain long runs of correctly assembled slices will score better.
- Predictions that contain several short runs will score worse. 
- If your prediction is not a permutation of $[0, 1, \dots, s-1]$, your score will be zero.

#### In detail

First we check that your prediction is a permutation of $[0, 1, \dots, s-1]$. If it is, we convert it into a list of run-lengths of correctly assembled sub-segments. For instance, if the ground-truth is $[1, 3, 0, 2]$ but you predicted $[3, 0, 1, 2]$, then your list of run-lengths is $[2, 1, 1]$.

Let $\\{r_i\\}_{i=1}^k$ be the list of run-lengths, and let $p_i = r_i/s$ for $1 \le i \le k$. Interpreting $\{p_i\}$ as a probability distribution, we calculate its base-*s* entropy by:

$$
H = -\sum_{i=1}^n{p_i \log_s p_i}
$$

Your score is then given by $1 - H$.

The highest-scoring run-length list is $[s]$, which scores 1. This corresponds to a perfectly reassembled document.

The lowest-scoring run-length list is $[1, 1, \dots, 1]$ ($s$ 1's), which scores 0. This corresponds to an attempt where no pair of strips was correctly reassembled.

