# Dataset attribution

Kotzias, D. (2015). *Sentiment Labelled Sentences* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C57604.

Source: https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences

License: Creative Commons Attribution 4.0 International (CC BY 4.0). Terms: https://creativecommons.org/licenses/by/4.0/

The associated paper is *From Group to Individual Labels Using Deep Features*, Dimitrios Kotzias, Misha Denil, Nando de Freitas, and Padhraic Smyth, KDD 2015.

ReviewSense parses the three source files into CSV, normalizes Unicode/case/whitespace, removes rows unsupported by the text-input contract, removes normalized exact duplicates, and discards all rows in any conflicting-label text group. The changed dataset is `data/reviews.csv`; source hashes are recorded in `data/source.json`. The authors and UCI do not endorse this application.
