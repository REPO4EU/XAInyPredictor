## v0.0.1 (2025-07-15): First version of the app

This is the first version of the app, with two tabs: `Inspect input file` and `Run analysis`.

Pull request link: https://github.com/REPO4EU/ShinyRAI/pull/1


## v1.0.0 (2026-02-03): App revamp

### Back-end
* Included functions to process and normalize data (`src/ShinyRAI/modules/data_processing.py`) which allow to normalize new input data introduced by the user and use it with our model.
* Included option to update the RAI model by reading the files `src/ShinyRAI/data/formula.pkl` and `src/ShinyRAI/data/current_best_formula.txt`. Before, the RAI model was hardcoded within the code of the app. Now, the model is read from these files. They are not input files that the user needs to provide, but "hidden" input files that are meant to be updated by the developers. Thus, it's a way to facilitate the update of the model for the developers. The function to read the RAI model is in `src/ShinyRAI/modules/rai.py` --> `read_delta_rai_formula`.

### Front-end
* Improved the overall design of the app.
* Created new page "Data Input" that facilitates the user to upload input data from two different sources:
  * By manual entry, through a form.
  * By uploading an input file.
* Created new page "Data Exploration" that allows the user to observe the distribution of the features from the input data, the distribution statistics (mean, median, std dev, min, max, count, missing), focus on a specific patient and even visualize the distribution of the training data.
* Improved the "Run analysis" page, now re-named as "Prediction":
  * The Prediction results table now focuses only on the Probability and the predicted RAI-R class. the rows predicted as RAI-R positive are in red, whereas the RAI-R negative are in green. The patient selected is highlighted in yellow.
  * We only show one plot (either the radar plot or the curves plot) depending on the selection of the user through a "select" button.
  * The user can modify the features to view through a select widget.
  * The curves plot height is modified dynamically depending on the number of features selected.
  * Included a text box where the user can specify the Min. % of false negatives, that is, the minimum % of false negative patients that we want the model to allow. Based on this, the probability threshold used to classify the patients in RAI-R positive / negative is dynamically modified.
* Included help options throughout the whole app, plus a general information button that explains the overall functioning of the app.

Pull request link: https://github.com/REPO4EU/ShinyRAI/pull/2
