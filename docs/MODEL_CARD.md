# ReviewSense model card

## Purpose and scope

Classify short English review text as Positive or Negative and make a small text-classification workflow accessible through a local application. The model is a benchmark prototype, not an automated measure of customer satisfaction or a guarantee of a review writer's intended meaning.

The application also supports Not scored when no learned vocabulary features match. That is an inference-support condition, not a learned neutral label. Other out-of-distribution reviews can still match vocabulary and receive misleading scores.

## Source and attribution

Data source: [Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences), Dimitrios Kotzias (2015), UCI Machine Learning Repository, DOI [10.24432/C57604](https://doi.org/10.24432/C57604). The source is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Raw download provenance and content hashes are retained by the download workflow.

The benchmark contains clearly positive or negative sentences from Amazon product reviews, Yelp restaurant reviews, and IMDb movie reviews. It is not a random sample of all current reviews, and neutral sentiment is outside its training label space.

## Data preparation and boundaries

1. Read source records using literal LF record boundaries. Embedded Unicode next-line characters in source reviews must not create extra records.
2. Normalize Unicode with NFKC, case-fold, and collapse whitespace through the shared text function. Reject structurally invalid reviews with no alphabetic content.
3. Detect repeated normalized review identities before creating partitions.
4. Remove every row in a conflicting-label text group and keep a single representative of non-conflicting duplicate text.
5. Make an 80/20 training/test split stratified by review source and label.
6. Record review identities for both partitions in `artifacts/splits.json`.

The recorded preparation starts with 3,000 rows and removes 2 invalid records and 20 duplicate records. No conflicting-label records were found. The 2,978 retained rows are divided into 2,382 training and 596 test rows.

Deduplication is deterministic cleaning rather than learning a model from the test features. It prevents exact text identity from appearing on both sides. It does not guarantee the absence of paraphrases, common authors, or temporal overlap. The dataset does not support a verified future-time performance claim.

## Preprocessing and model selection

The scikit-learn Pipeline contains TF-IDF followed by a classifier. The vectorizer represents unigrams and bigrams. Stop-word removal is not used, preserving words such as "not" rather than automatically discarding negation. The vocabulary and IDF weights are learned inside each training fold and then inside the final training-only fit.

A prior-only dummy model provides a reference. Logistic regression candidates use C values 0.1, 1, and 4; smaller C means stronger regularization under the same penalty. Three-fold training cross-validation stratified by source and label selects the largest mean positive-class F1. The final classifier is fitted on the training partition, with a fixed threshold of 0.5. The test results are reported after selection and are not used for another tuning round.

These are modest, intentional choices for an explainable baseline. This experiment does not compare transformer language models, perform exhaustive tuning, or establish that logistic regression is universally best.

## Reported evaluation

`artifacts/report.json` is the exact numerical record for the saved run. It includes cleaning counts, partition sizes, model comparison scores, overall metrics, confusion counts, domain metrics, and misclassified examples. The application reads this record.

The selected candidate is logistic regression with C=4. Its mean training-CV positive F1 is 0.8082, with fold standard deviation 0.0046. On the 596 test reviews, accuracy is 0.8121, positive precision 0.8039, positive recall 0.8255, positive F1 0.8146, macro F1 0.8120, and ROC-AUC 0.8961.

| Actual / Predicted | Negative | Positive |
| --- | ---: | ---: |
| Negative | 238 | 60 |
| Positive | 52 | 246 |

These classifier metrics include every test row, including one with no learned vocabulary features. The application instead returns Not scored for such a row. Its displayed supported-review summary must not be described as having the identical denominator or policy as the all-row classifier benchmark.

The prior-only dummy's CV positive F1 is 0 because its positive training prior is slightly below 0.5 and its threshold decision predicts Negative throughout. This is not random guessing and does not mean zero accuracy. Accuracy and macro F1 are useful additional context when interpreting that reference.

The confusion matrix uses rows as actual labels and columns as predictions, in order Negative (0), Positive (1). Positive-class precision measures the proportion of positive predictions with a positive label; recall measures the fraction of all positive examples recovered. F1 combines those two measures. ROC-AUC characterizes discrimination across thresholds. Accuracy measures correctness across both classes. Any macro-F1 value averages the two class-level F1 values rather than reporting only the positive class.

The per-domain metrics indicate how this same fitted model behaves on held-out examples from each review source. They do not constitute evaluation on a wholly unseen domain. A next experiment could hold out a complete source, with a new model-selection/evaluation plan specified in advance.

## Interpreting predictions and terms

The positive-class score is the fitted model's predicted probability. It has not undergone a separate calibration experiment, so the interface does not call it guaranteed confidence. A binary model must choose one of two labels even for mixed or neutral language, unless the special no-vocabulary-match rule declines scoring.

Top coefficient terms are global learned associations. They are not a causal explanation and do not alone explain a specific review. Correlated features and bigram context affect how logistic regression combines weights. A familiar word can occur in a negated, sarcastic, or quoted context and still mislead the model.

## Limitations

- Curated, balanced binary sentences differ from real feedback distributions.
- Long reviews can contain conflicting opinions; this model assigns one overall label.
- Sarcasm, implicit meaning, spelling variation, slang, and language changes can break simple lexical patterns.
- Exact deduplication does not remove every semantic duplicate or establish author independence.
- A random split does not verify future-time or production performance.
- No operational user study, retention outcome, production throughput, fairness study, or confidence guarantee is claimed.

## Artifact trust and reproduction

Run the download script, install the locked environment, and execute `python -m reviewsense.train`. Source hashes, fixed random settings, partition identities, and the report improve reproducibility. Serialized model bytes may differ across platforms or reruns; review predictions and metrics with sensible numerical tolerance.

The scoring service checks model bytes against the report before loading. This detects mismatch but is not a signature. joblib deserialization is safe only for trusted artifacts; do not accept arbitrary user-uploaded model files.

Future improvements should begin with a representative labelled evaluation set, a separate treatment of neutral/mixed sentiment, and error analysis before adding a more complex model. Once the current test results are known, avoid repeatedly tuning against them and calling them unseen evaluation.
