# Understanding ReviewSense

This guide explains the smallest complete path from a review sentence to a usable sentiment application. Read one section, open the referenced module, and trace the data with a short example.

## 1. Start with the prediction problem

The input is a review string. The training target is a binary sentiment label, Negative (0) or Positive (1). The output is a label plus the model's positive-class score. This is supervised classification because training examples include known labels.

The application adds a practical workflow: one-review analysis, batch CSV analysis, summaries, and safe downloads. Those features make the estimator accessible; they do not change the benchmark's validity limits.

## 2. Trace cleaning and validation

Open `reviewsense/text.py`. Unicode normalization makes compatible character forms more consistent, case-folding reduces case variation, and whitespace normalization handles extra spaces and line breaks. The shared function is used for both training and inference.

Negation is intentionally retained. "Good" and "not good" may express different sentiment, so deleting words from a generic stop-word list can remove useful information. The model still has limitations: a bigram can help represent local negation, but it does not create full language understanding.

Validation asks whether a record is structurally acceptable. It does not prove that the model understands it. Alphabetic nonsense may pass structural validation and then have no matching vocabulary features; that separate condition returns Not scored.

## 3. Understand TF-IDF

Open the vectorizer configuration in `reviewsense/train.py`.

- A unigram is one token, such as "helpful".
- A bigram is a pair, such as "not helpful".
- Term frequency reflects occurrence within a review.
- Inverse document frequency reduces the relative weight of terms common across the training collection.

The resulting vector is a numerical representation of text. Its vocabulary and IDF weights are learned parameters. Fitting them on all data before cross-validation would let held-out examples influence preprocessing. The full Pipeline is therefore passed into cross-validation.

## 4. Understand logistic regression

Logistic regression combines feature values with learned coefficients, then maps its decision score to a positive-class probability. The threshold converts that probability into a class. Here the policy is fixed at 0.5.

A positive coefficient is associated with the positive class in the fitted model, holding its mathematical combination with other features in mind. A negative coefficient contributes in the opposite direction. These are predictive weights, not causal effects or complete explanations of a person's opinion.

Regularization discourages excessively large weights. With other settings fixed, a smaller C means stronger regularization. The experiment compares three C settings rather than assuming a more complicated model is necessary.

## 5. Follow the experiment boundary

Find the split and cross-validation code in `reviewsense/train.py`.

The training partition supports model fitting and three-fold comparison. The test partition is reserved for reporting the selected model. Source-and-label stratification helps maintain representation of each benchmark domain/class combination. Content-based identifiers prevent exact normalized text from occurring in both partitions.

The dummy model is useful because a score without a baseline is hard to interpret. Choosing by training CV F1 and then inspecting held-out errors gives a more useful account than presenting accuracy alone.

## 6. Read the metrics

Open `artifacts/report.json` and the model-report page.

- **Precision:** of the reviews predicted positive, how many are actually labelled positive?
- **Recall:** of all positive-labelled reviews, how many did the model identify?
- **F1:** harmonic mean of positive precision and recall.
- **Accuracy:** correct labels across both classes.
- **ROC-AUC:** ranking discrimination across score thresholds.

Use the confusion matrix to calculate precision and recall yourself. Then read misclassified examples and identify cases involving mixed sentiment, unusual wording, or insufficient training support. Error examples are observations, not proof that one guessed explanation caused a failure.

## 7. Follow inference through the service

Open `reviewsense/service.py`. The service loads the complete trusted Pipeline rather than reimplementing TF-IDF in application code. A checksum compares model bytes with the saved report. This catches mismatched files, but it cannot authenticate an artifact if someone replaced both the model and report.

The vectorizer can detect a review that produces no known vocabulary features. That review is Not scored. This is not a neutral prediction: it is a signal that this model has no matching text representation. A review with some familiar tokens can still be outside the intended data distribution.

## 8. Follow CSV input and output

Open `reviewsense/io.py`. The parser validates encoding, headers, record count, text values, and identifiers before scoring. Invalid batches are rejected as a whole so the displayed totals are not silently based on only part of a file.

The export preserves review order and adds the prediction fields. It also protects cells that spreadsheet applications might otherwise interpret as formulas. This deliberately changes the exported representation of formula-like text by adding an apostrophe; the model uses validated text rather than executing any content.

## 9. Read the Streamlit application

Open `app.py`. Follow the model loader, navigation, input controls, prediction action, and download button. Streamlit reruns the script when widgets change, so application state and stale results need deliberate handling.

Do not cache arbitrary user review contents globally. The shared model can be cached as a resource, while uploaded/predicted content belongs to the current session. The current application is a local prototype; hosted privacy and access rules require additional design.

## 10. Check behavior with tests

Open `tests/test_reviewsense.py` and run pytest after training. Start with normalization, prediction parity, exact duplicate/split checks, unknown vocabulary, invalid CSV, and safe export.

A useful learning exercise is to change one validation limit in a temporary copy, make a test fail for the old expectation, then explain the new behavior. Another is to trace a single review's TF-IDF features and see how the classifier combines them.

## Suggested next experiments

1. Build an independent labelled sample of realistic short feedback, with permission to use it.
2. Decide how neutral and mixed sentiment should be represented rather than treating every score near 0.5 as a true neutral label.
3. Compare a unigram-only pipeline with unigrams and bigrams using training-only validation.
4. Evaluate a completely held-out source under a new predeclared protocol.
5. Measure runtime and memory before making performance claims.

Keep an experiment log. State the data, metric, decision rule, and evaluation boundary before looking at results. Once you tune based on a test set, it no longer serves as a clean final test for that tuning process.
