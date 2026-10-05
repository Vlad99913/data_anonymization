# Benchmark on `benchmark.csv` (10 docs, 51 entities)

## Overall

| model_name         | evaluation_mode   |   precision |   recall |    f1 | same_docs   |   ms_per_doc |
|:-------------------|:------------------|------------:|---------:|------:|:------------|-------------:|
| spacy-lg           | strict            |       0.891 |    0.804 | 0.845 | 2/10        |          3.4 |
| spacy-lg           | exact             |       0.913 |    0.824 | 0.866 | 2/10        |          3.4 |
| spacy-lg           | partial           |       0.957 |    0.863 | 0.907 | 2/10        |          3.4 |
| spacy-lg + rules   | strict            |       0.922 |    0.922 | 0.922 | 4/10        |          4.2 |
| spacy-lg + rules   | exact             |       0.922 |    0.922 | 0.922 | 4/10        |          4.2 |
| spacy-lg + rules   | partial           |       0.961 |    0.961 | 0.961 | 4/10        |          4.2 |
| spacy-trf          | strict            |       0.889 |    0.784 | 0.833 | 3/10        |         20.8 |
| spacy-trf          | exact             |       0.911 |    0.804 | 0.854 | 3/10        |         20.8 |
| spacy-trf          | partial           |       0.978 |    0.863 | 0.917 | 3/10        |         20.8 |
| spacy-trf + rules  | strict            |       0.92  |    0.902 | 0.911 | 5/10        |         20.6 |
| spacy-trf + rules  | exact             |       0.92  |    0.902 | 0.911 | 5/10        |         20.6 |
| spacy-trf + rules  | partial           |       0.98  |    0.961 | 0.97  | 5/10        |         20.6 |
| gliner-pii         | strict            |       0.75  |    0.647 | 0.695 | 1/10        |         25.3 |
| gliner-pii         | exact             |       0.75  |    0.647 | 0.695 | 1/10        |         25.3 |
| gliner-pii         | partial           |       0.932 |    0.804 | 0.863 | 1/10        |         25.3 |
| gliner-pii + rules | strict            |       0.756 |    0.667 | 0.708 | 1/10        |         10.7 |
| gliner-pii + rules | exact             |       0.756 |    0.667 | 0.708 | 1/10        |         10.7 |
| gliner-pii + rules | partial           |       0.933 |    0.824 | 0.875 | 1/10        |         10.7 |

## Strict-match F1 per label

| model_name         |   PERSON |   ORG |   JOB |   EMAIL_ADDRESS |   LOCATION |   AMOUNT |   DATE_TIME |   UNIVERSITY |   PHONE_NUMBER |   URL |
|:-------------------|---------:|------:|------:|----------------:|-----------:|---------:|------------:|-------------:|---------------:|------:|
| spacy-lg           |        1 |  0.78 |  0    |               0 |       0.92 |      0.5 |         1   |            0 |              0 |     0 |
| spacy-lg + rules   |        1 |  0.82 |  1    |               1 |       0.92 |      0.5 |         1   |            1 |              1 |     1 |
| spacy-trf          |        1 |  0.82 |  0    |               0 |       0.87 |      0.5 |         1   |            0 |              0 |     0 |
| spacy-trf + rules  |        1 |  0.88 |  1    |               1 |       0.87 |      0.5 |         1   |            1 |              1 |     1 |
| gliner-pii         |        1 |  0.63 |  0.67 |               1 |       0.44 |      1   |         0.8 |            1 |              1 |     1 |
| gliner-pii + rules |        1 |  0.63 |  1    |               1 |       0.44 |      1   |         0.8 |            1 |              1 |     1 |

## Exact-match F1 per label

| model_name         |   PERSON |   ORG |   JOB |   EMAIL_ADDRESS |   LOCATION |   AMOUNT |   DATE_TIME |   UNIVERSITY |   PHONE_NUMBER |   URL |
|:-------------------|---------:|------:|------:|----------------:|-----------:|---------:|------------:|-------------:|---------------:|------:|
| spacy-lg           |        1 |  0.82 |  0    |               0 |       0.92 |      0.5 |         1   |            1 |              0 |     0 |
| spacy-lg + rules   |        1 |  0.82 |  1    |               1 |       0.92 |      0.5 |         1   |            1 |              1 |     1 |
| spacy-trf          |        1 |  0.88 |  0    |               0 |       0.87 |      0.5 |         1   |            1 |              0 |     0 |
| spacy-trf + rules  |        1 |  0.88 |  1    |               1 |       0.87 |      0.5 |         1   |            1 |              1 |     1 |
| gliner-pii         |        1 |  0.63 |  0.67 |               1 |       0.44 |      1   |         0.8 |            1 |              1 |     1 |
| gliner-pii + rules |        1 |  0.63 |  1    |               1 |       0.44 |      1   |         0.8 |            1 |              1 |     1 |

## Partial-match F1 per label

| model_name         |   PERSON |   ORG |   JOB |   EMAIL_ADDRESS |   LOCATION |   AMOUNT |   DATE_TIME |   UNIVERSITY |   PHONE_NUMBER |   URL |
|:-------------------|---------:|------:|------:|----------------:|-----------:|---------:|------------:|-------------:|---------------:|------:|
| spacy-lg           |        1 |  0.94 |  0    |               0 |       0.92 |        1 |        1    |            1 |              0 |     0 |
| spacy-lg + rules   |        1 |  0.94 |  1    |               1 |       0.92 |        1 |        1    |            1 |              1 |     1 |
| spacy-trf          |        1 |  1    |  0    |               0 |       0.92 |        1 |        1    |            1 |              0 |     0 |
| spacy-trf + rules  |        1 |  1    |  1    |               1 |       0.92 |        1 |        1    |            1 |              1 |     1 |
| gliner-pii         |        1 |  0.84 |  0.67 |               1 |       0.75 |        1 |        0.93 |            1 |              1 |     1 |
| gliner-pii + rules |        1 |  0.84 |  1    |               1 |       0.75 |        1 |        0.93 |            1 |              1 |     1 |

