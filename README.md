# ml-interview
Writing a doc for ML tips and techniques as a glance


# Machine Learning Concepts Overview

## Supervised Learning

*Goal: Learn a mapping from inputs (features) to outputs (labels) based on labeled examples.*

---

### 1. Linear Regression

* **Explanation:** Models the relationship between a dependent variable (continuous) and one or more independent variables by fitting a linear equation ($y = Wx + b$).
* **Tricks & Treats:** Sensitive to outliers; Feature scaling (Standardization/Normalization) often improves performance.
* **Caveats/Questions:** Assumes linearity, independence of errors, homoscedasticity. Is the relationship truly linear?
* **Python (House Pricing):**
    ```python
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    # Assume X_house (features like size, rooms) and y_price (labels) are loaded
    # X_house = np.array([[1500, 3], [2000, 4], ...])
    # y_price = np.array([300000, 450000, ...])

    X_train, X_test, y_train, y_test = train_test_split(X_house, y_price, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    print(f"Linear Regression MSE: ${mse:.2f}")
    # R-squared: model.score(X_test, y_test)
    ```
* **Eval/Best Practices:** Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R-squared ($R^2$). Use cross-validation. Check residuals.
* **Libraries:** `scikit-learn`, `statsmodels`.
* **GPU Opt:** Yes, via `cuml.linear_model.LinearRegression` (RAPIDS cuML) for significant speedups.
* **Math/Subtleties:** Solved via Ordinary Least Squares (OLS) minimizing $\sum(y_i - \hat{y}_i)^2$ or Gradient Descent.
* **SOTA Improvement:** Foundational. Improvements often via feature engineering, regularization, or using more complex models when linearity doesn't hold.

---

### 2. Logistic Regression

* **Explanation:** Used for binary classification problems. Models the probability of a class using the logistic (sigmoid) function: $P(y=1|x) = \frac{1}{1 + e^{-(Wx + b)}}$.
* **Tricks & Treats:** Output is probability, needs a threshold (e.g., 0.5) for class assignment. Good baseline for classification.
* **Caveats/Questions:** Assumes linear separability (in the feature space or transformed space). Can be sensitive to outliers influencing the decision boundary.
* **Python (Simple Classification):**
    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    # Assume X_clf (features) and y_clf (binary labels 0/1) are loaded
    # X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, ...)

    model = LogisticRegression(solver='liblinear') # Different solvers available
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Logistic Regression Accuracy: {accuracy:.2f}")
    # Probabilities: model.predict_proba(X_test)
    ```
* **Eval/Best Practices:** Accuracy, Precision, Recall, F1-Score, ROC AUC, Confusion Matrix. Regularization (L1/L2) is common.
* **Libraries:** `scikit-learn`.
* **GPU Opt:** Yes, via `cuml.linear_model.LogisticRegression`.
* **Math/Subtleties:** Solved via Maximum Likelihood Estimation (MLE), often using optimization algorithms like Gradient Descent. Uses Log Loss (Cross-Entropy) as the loss function.
* **SOTA Improvement:** Foundational classifier. Performance gains often from feature engineering or moving to more complex models for non-linear boundaries.

---

### 3. Naive Bayes Classifier

* **Explanation:** Probabilistic classifier based on Bayes' Theorem ($P(A|B) = \frac{P(B|A)P(A)}{P(B)}$) with a "naive" assumption of conditional independence between features given the class.
* **Tricks & Treats:** Works surprisingly well on high-dimensional data (e.g., text classification, spam filtering) despite the naive assumption often being violated. Very fast to train.
* **Caveats/Questions:** The independence assumption is strong. Performance degrades if features are highly correlated. Zero-frequency problem (if a feature value never occurs with a class in training) needs smoothing (e.g., Laplace smoothing).
* **Python (Text Classification - Conceptual):**
    ```python
    from sklearn.naive_bayes import MultinomialNB # Common for text counts
    from sklearn.feature_extraction.text import CountVectorizer
    # Assume text_data (list of strings) and y_labels (categories) are loaded
    # vectorizer = CountVectorizer()
    # X_counts = vectorizer.fit_transform(text_data)
    # X_train, X_test, y_train, y_test = train_test_split(X_counts, y_labels, ...)

    model = MultinomialNB()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    # Evaluate using accuracy_score, classification_report etc.
    ```
* **Eval/Best Practices:** Accuracy, Precision, Recall, F1, ROC AUC. Choose variant based on data (GaussianNB for continuous, MultinomialNB/BernoulliNB for discrete/text).
* **Libraries:** `scikit-learn`.
* **GPU Opt:** Yes, cuML supports Naive Bayes (`cuml.naive_bayes.MultinomialNB`).
* **Math/Subtleties:** Calculates posterior probability $P(\text{Class}| \text{Features})$ using prior $P(\text{Class})$ and likelihood $P(\text{Features}|\text{Class})$. Assumes $P(\text{feat}_1, \text{feat}_2 | \text{Class}) = P(\text{feat}_1 | \text{Class}) P(\text{feat}_2 | \text{Class})$.
* **SOTA Improvement:** Provides a simple, fast baseline, especially for text. Less competitive now against deep learning for complex tasks but still useful.

---

### 4. K-Nearest Neighbors (KNN)

* **Explanation:** Instance-based or "lazy" learning algorithm. Classifies a new point based on the majority class among its 'k' closest neighbors in the feature space. Can also be used for regression (average of neighbors' values).
* **Tricks & Treats:** Simple concept. No explicit training phase. Performance highly depends on the choice of 'k' and the distance metric (e.g., Euclidean, Manhattan). Feature scaling is crucial.
* **Caveats/Questions:** Computationally expensive during prediction (needs to calculate distances to all training points). Sensitive to irrelevant features ("curse of dimensionality"). Needs careful choice of 'k'.
* **Python (Classification):**
    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    # Assume X_clf, y_clf loaded & split
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=5) # Choose k
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    # Evaluate
    ```
* **Eval/Best Practices:** Accuracy, etc. for classification; MSE, etc. for regression. Use cross-validation to find the optimal 'k'. Use appropriate distance metric.
* **Libraries:** `scikit-learn`.
* **GPU Opt:** Yes, `cuml.neighbors.KNeighborsClassifier` and `KNeighborsRegressor` offer massive speedups, especially for large datasets, leveraging FAISS for efficient search.
* **Math/Subtleties:** Core is distance calculation. Various weighting schemes exist (e.g., distance-weighted KNN).
* **SOTA Improvement:** Simple baseline. Advanced versions (weighted KNN, optimized search structures like KD-trees, Ball trees, LSH) improve efficiency and performance. GPU acceleration makes it viable for larger datasets.

---

### 5. Decision Trees

* **Explanation:** Tree-like structure where internal nodes represent tests on features, branches represent outcomes of tests, and leaf nodes represent class labels (classification) or continuous values (regression).
* **Tricks & Treats:** Easy to understand and interpret. Handles both numerical and categorical data. Non-parametric.
* **Caveats/Questions:** Prone to overfitting, especially if deep. Can be unstable (small data changes lead to different trees). Optimal tree finding is NP-hard; greedy approaches (like CART using Gini impurity or Information Gain) are used.
* **Python (House Pricing):**
    ```python
    from sklearn.tree import DecisionTreeRegressor # or DecisionTreeClassifier
    # Assume X_house, y_price loaded & split

    model = DecisionTreeRegressor(max_depth=5, random_state=42) # Control complexity
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    # Evaluate using MSE, R2, etc.
    ```
* **Eval/Best Practices:** Standard classification/regression metrics. Pruning (pre or post) or setting `max_depth`, `min_samples_leaf` is crucial to prevent overfitting. Visualize the tree.
* **Libraries:** `scikit-learn`.
* **GPU Opt:** Not typically GPU-accelerated directly in scikit-learn, but ensembles like Random Forest/GBTs built on trees *are* often accelerated. cuML accelerates inference for tree models via Forest Inference Library (FIL).
* **Math/Subtleties:** Splitting criteria: Gini Impurity, Information Gain (Entropy) for classification; Variance Reduction (MSE) for regression.
* **SOTA Improvement:** Single trees are rarely SOTA but are building blocks for powerful ensembles (Random Forests, GBTs). Research exists on finding optimal (non-greedy) trees.

---

### 6. Bagging (Random Forests)

* **Explanation:** Ensemble method. Trains multiple Decision Trees independently on different bootstrap samples (random subsets with replacement) of the training data. Often also uses random subsets of features for splitting nodes. Predictions are averaged (regression) or majority-voted (classification).
* **Tricks & Treats:** Reduces variance compared to single trees, less prone to overfitting. Generally robust and performs well with default parameters. Handles missing data reasonably well.
* **Caveats/Questions:** Less interpretable than a single tree. Can be computationally intensive and require more memory.
* **Python (House Pricing):**
    ```python
    from sklearn.ensemble import RandomForestRegressor # or RandomForestClassifier
    # Assume X_house, y_price loaded & split

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # 100 trees, use all CPU cores
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    # Evaluate
    ```
* **Eval/Best Practices:** Standard metrics. Feature importance scores are a useful output. Tune `n_estimators`, tree depth parameters (`max_depth`, `min_samples_leaf`), and `max_features`.
* **Libraries:** `scikit-learn`.
* **GPU Opt:** Yes, `cuml.ensemble.RandomForestRegressor` / `Classifier` provides significant acceleration.
* **Math/Subtleties:** Core idea is Bootstrap Aggregating (Bagging) + feature randomness. Reduces correlation between trees.
* **SOTA Improvement:** Very strong general-purpose algorithm, often near SOTA for tabular data, though sometimes edged out by Gradient Boosting.

---

### 7. Boosting (Gradient Boosted Trees - GBTs)

* **Explanation:** Ensemble method where trees are built sequentially. Each new tree tries to correct the errors made by the previous ones (typically by fitting to the residual errors or gradients).
* **Tricks & Treats:** Often achieves state-of-the-art performance on tabular data. Can overfit if too many trees are used. Requires careful tuning of hyperparameters (learning rate, tree depth, number of trees).
* **Caveats/Questions:** Training is sequential, can be slower than Random Forests (though implementations are highly optimized). More sensitive to hyperparameters than Random Forests.
* **Python (House Pricing using XGBoost):**
    ```python
    import xgboost as xgb
    # Assume X_house, y_price loaded & split

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    # Evaluate
    ```
* **Eval/Best Practices:** Standard metrics. Use early stopping based on a validation set to prevent overfitting. Tune `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, and regularization parameters.
* **Libraries:** `scikit-learn` (GradientBoosting), `xgboost`, `lightgbm`, `catboost` (more advanced, often faster/better).
* **GPU Opt:** Yes, XGBoost, LightGBM, and CatBoost have excellent built-in GPU support.
* **Math/Subtleties:** Based on gradient descent in function space. Different loss functions can be used. Regularization is built-in in advanced libraries.
* **SOTA Improvement:** Often the leading approach for structured/tabular data competitions (Kaggle). Libraries like LightGBM introduced histogram-based splitting and efficient sampling (GOSS, EFB) for speed.

---

### 8. Support Vector Machines (SVM)

* **Explanation:** Finds an optimal hyperplane that best separates data points of different classes in a high-dimensional space. The "optimal" hyperplane is the one with the maximum margin (distance) between itself and the nearest points (support vectors) of each class.
* **Tricks & Treats:** Effective in high-dimensional spaces. Memory efficient (uses subset of points - support vectors). Can use different kernels for non-linear separation (see Kernel Methods).
* **Caveats/Questions:** Can be computationally expensive for large datasets, especially with non-linear kernels. Performance depends heavily on the choice of kernel and regularization parameter (C). Does not directly provide probability estimates (requires calibration).
* **Python (Classification):**
    ```python
    from sklearn.svm import SVC # Support Vector Classifier (or SVR for Regression)
    from sklearn.preprocessing import StandardScaler
    # Assume X_clf, y_clf loaded & split, scaled (important for SVM!)

    model = SVC(kernel='rbf', C=1.0, gamma='scale') # RBF kernel is common
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    # Evaluate
    ```
* **Eval/Best Practices:** Standard classification/regression metrics. Feature scaling is crucial. Tune `C` (regularization parameter - balances margin width and misclassifications) and kernel parameters (e.g., `gamma` for RBF) using cross-validation/grid search.
* **Libraries:** `scikit-learn`, `LIBSVM` (underlying library).
* **GPU Opt:** Yes, `cuml.svm.SVC` / `SVR` provides GPU acceleration.
* **Math/Subtleties:** Solves a constrained quadratic optimization problem. The concept of margin maximization and support vectors is key. Uses the "kernel trick" for non-linearity.
* **SOTA Improvement:** Was SOTA for many classification tasks before deep learning. Still very powerful, especially for smaller datasets or when interpretability of support vectors is useful. Effective for wide datasets (many features).

---

### 9. Kernel Methods

* **Explanation:** A class of algorithms, most famously used with SVMs. The "kernel trick" allows algorithms operating on dot products (like SVM) to implicitly map data into higher-dimensional spaces and find non-linear relationships without explicitly calculating the coordinates in that space.
* **Tricks & Treats:** Enables linear algorithms to solve non-linear problems. Common kernels: Linear ($x^T z$), Polynomial ($(x^T z + c)^d$), Radial Basis Function (RBF) / Gaussian ($exp(-\gamma ||x-z||^2)$).
* **Caveats/Questions:** Choice of kernel and its parameters is crucial and problem-dependent. Can be computationally expensive ($O(n^2)$ or worse) for large n. Defining custom kernels requires ensuring the kernel function satisfies Mercer's condition (is a valid positive semi-definite kernel).
* **Python:** (See SVM example - implemented via `kernel` parameter).
* **Eval/Best Practices:** Use cross-validation to select the best kernel and its hyperparameters (`d`, `gamma`, `c`). RBF is a good default starting point.
* **Libraries:** `scikit-learn` (integrated into SVM, Kernel PCA, etc.).
* **GPU Opt:** Yes, algorithms using kernels (like SVM, Kernel Ridge) can be accelerated on GPU via cuML.
* **Math/Subtleties:** The kernel function $K(x, z)$ computes the dot product $\phi(x)^T \phi(z)$ in some (potentially infinite-dimensional) feature space $\phi$.
* **SOTA Improvement:** Revolutionized SVMs and other algorithms by efficiently handling non-linearity. Foundation for Gaussian Processes.

---

### 10. Neural Networks (Briefly)

* **Explanation:** Models inspired by the brain, consisting of interconnected nodes (neurons) organized in layers. Learn complex patterns by adjusting connection weights during training (using backpropagation). Basic form: Feedforward Neural Network (FNN).
* **Tricks & Treats:** Can model highly complex non-linear relationships. Basis for Deep Learning.
* **Caveats/Questions:** Require large amounts of data. Computationally expensive to train. Prone to overfitting. Less interpretable ("black box"). Sensitive to hyperparameter tuning (architecture, learning rate, activation functions).
* **Python (Conceptual using Keras):**
    ```python
    # import tensorflow as tf
    # from tensorflow import keras
    # model = keras.Sequential([
    #     keras.layers.Dense(128, activation='relu', input_shape=(num_features,)),
    #     keras.layers.Dense(64, activation='relu'),
    #     keras.layers.Dense(1) # Output layer (e.g., for regression)
    # ])
    # model.compile(optimizer='adam', loss='mse')
    # model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    ```
* **Eval/Best Practices:** Appropriate loss functions (MSE for regression, Cross-Entropy for classification) and metrics. Regularization (Dropout, L1/L2), batch normalization, careful choice of optimizers (Adam, SGD) and activation functions (ReLU, sigmoid, tanh) are essential.
* **Libraries:** `TensorFlow`/`Keras`, `PyTorch`.
* **GPU Opt:** **Crucial.** Deep learning heavily relies on GPUs for training large models. Libraries have seamless GPU integration.
* **Math/Subtleties:** Backpropagation algorithm (chain rule for derivatives), activation functions, gradient descent variants, loss functions.
* **SOTA Improvement:** Deep Learning (using deep NNs like CNNs, RNNs, Transformers) represents SOTA in many domains like vision, NLP, speech. (More in Deep Learning section).

---

### 11. Stochastic Gradient Descent (SGD)

* **Explanation:** Optimization algorithm used to train many ML models (Linear/Logistic Regression, SVMs, NNs). Updates model parameters iteratively using the gradient of the loss function calculated on a single training example ('stochastic') or a small 'mini-batch' at a time, rather than the whole dataset (Batch Gradient Descent).
* **Tricks & Treats:** Much faster per iteration than Batch GD on large datasets. Can escape shallow local minima. Introduces noise which can help generalization.
* **Caveats/Questions:** Noisy updates can lead to slower convergence or oscillations around the minimum. Requires careful tuning of the learning rate (often needs a decaying schedule). Sensitive to feature scaling.
* **Python (Using scikit-learn's SGDClassifier/Regressor):**
    ```python
    from sklearn.linear_model import SGDRegressor # or SGDClassifier
    from sklearn.preprocessing import StandardScaler
    # Assume X, y loaded & split, scaled (important for SGD!)

    model = SGDRegressor(loss='squared_error', penalty='l2', alpha=0.0001, max_iter=1000, tol=1e-3, random_state=42)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    # Evaluate
    ```
* **Eval/Best Practices:** Tune learning rate, regularization (`penalty`, `alpha`), number of iterations (`max_iter`). Monitor loss on training/validation sets. Use mini-batches for stability. Consider adaptive learning rate methods (Adam, Adagrad, RMSprop) which are often used in NNs.
* **Libraries:** `scikit-learn` (for linear models/SVMs), `TensorFlow`/`Keras`, `PyTorch` (optimizers for NNs).
* **GPU Opt:** While the core SGD logic is simple, it's the backbone of NN training which is heavily GPU-accelerated. cuML also provides SGD for linear models.
* **Math/Subtleties:** Parameter update rule: $W = W - \eta \nabla L(W; x_i, y_i)$, where $\eta$ is learning rate, $\nabla L$ is gradient w.r.t. parameters $W$ for sample $(x_i, y_i)$.
* **SOTA Improvement:** Essential enabling algorithm for large-scale machine learning and deep learning. Advanced variants (Adam, etc.) are SOTA optimizers for NNs.

---

### 12. Sequence Modeling (Briefly)

* **Explanation:** Deals with data where order matters (e.g., time series, text, DNA). Aims to predict future elements or classify entire sequences.
* **Tricks & Treats:** Captures temporal dependencies. Foundational for NLP, speech recognition, financial forecasting.
* **Caveats/Questions:** Handling long-range dependencies can be challenging. Requires specialized architectures (RNNs, LSTMs, Transformers).
* **Python:** (See Sequential Models / Deep Learning sections for RNN/LSTM examples).
* **Eval/Best Practices:** Metrics depend on task (e.g., perplexity for language models, accuracy/F1 for sequence classification, MSE for forecasting).
* **Libraries:** `TensorFlow`/`Keras`, `PyTorch`.
* **GPU Opt:** Crucial for training complex sequence models like LSTMs and Transformers.
* **Math/Subtleties:** Concepts of hidden states, recurrence relations, attention mechanisms.
* **SOTA Improvement:** Dominated by Recurrent Neural Networks (RNNs), LSTMs, and now primarily Transformers (see Deep Learning / NLP).

---

### 13. Bayesian Linear Regression

* **Explanation:** A Bayesian approach to linear regression. Instead of finding single point estimates for weights ($W, b$), it infers a probability distribution over them. Predictions are also distributions, naturally incorporating uncertainty.
* **Tricks & Treats:** Provides uncertainty estimates for predictions. Can incorporate prior knowledge. Less prone to overfitting with appropriate priors (which act as regularization).
* **Caveats/Questions:** Can be computationally more expensive than standard linear regression, often requiring sampling methods (MCMC) or variational inference. Choosing priors can be subjective.
* **Python (Conceptual using `sklearn`):**
    ```python
    from sklearn.linear_model import BayesianRidge # Implements Bayesian Regression with specific priors
    # Assume X_house, y_price loaded & split

    model = BayesianRidge()
    model.fit(X_train, y_train)
    # predictions = model.predict(X_test) # Point estimate (mean of posterior)
    # To get uncertainty, typically need libraries like PyMC or Stan
    # y_pred, y_std = model.predict(X_test, return_std=True) # Some models offer this
    ```
* **Eval/Best Practices:** Evaluate point predictions (mean/median) using standard regression metrics (MSE, R²). Assess calibration of uncertainty estimates.
* **Libraries:** `scikit-learn` (BayesianRidge, ARDRegression), `statsmodels`, `PyMC`, `Stan`, `Pyro`, `TensorFlow Probability`.
* **GPU Opt:** Sampling (MCMC) can sometimes be parallelized, and libraries like `NumPyro` or `TensorFlow Probability` leverage GPUs for faster inference, especially with variational methods.
* **Math/Subtleties:** Uses Bayes' Theorem: $P(\text{weights}|\text{Data}) \propto P(\text{Data}|\text{weights}) P(\text{weights})$. Involves defining priors for weights and likelihood for data. Posterior distribution is key output.
* **SOTA Improvement:** Valuable when uncertainty quantification is important (e.g., science, finance, medical). Gaussian Processes offer a related non-parametric approach.

---

### 14. Gaussian Processes (GP)

* **Explanation:** Non-parametric Bayesian approach for supervised learning (mainly regression, but also classification). Defines a distribution over functions. Assumes that the values of the function at any finite set of points have a joint Gaussian distribution.
* **Tricks & Treats:** Provides well-calibrated uncertainty estimates. Flexible (non-parametric). Works well on small datasets. Kernels allow incorporating prior knowledge about the function (e.g., smoothness, periodicity).
* **Caveats/Questions:** Computationally expensive: standard implementation scales as $O(n^3)$ for training, $O(n^2)$ for prediction. Requires careful kernel selection and hyperparameter tuning.
* **Python (Conceptual using `sklearn`):**
    ```python
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    # Assume X_house (1D for simplicity here), y_price loaded & split

    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, random_state=42)
    model.fit(X_train, y_train)
    y_pred, sigma = model.predict(X_test, return_std=True) # Get mean prediction and std dev
    # Evaluate mean prediction, assess uncertainty intervals (e.g., y_pred +/- 1.96*sigma)
    ```
* **Eval/Best Practices:** Evaluate mean prediction (MSE, R²). Assess uncertainty calibration (e.g., coverage of prediction intervals). Optimize kernel hyperparameters via marginal likelihood maximization.
* **Libraries:** `scikit-learn`, `GPy`, `GPflow`, `PyMC`, `Stan`.
* **GPU Opt:** Possible, especially for approximate inference methods (Variational Inference, sparse GPs) implemented in libraries like `GPflow` (using TensorFlow backend) or `GPyTorch` (using PyTorch backend).
* **Math/Subtleties:** Defined by a mean function (often zero) and a covariance function (kernel). Predictions are based on conditioning the joint Gaussian distribution on observed data.
* **SOTA Improvement:** Powerful for regression with uncertainty, active learning, Bayesian optimization. Scalability is improved via sparse approximations (using inducing points).

---

### 15. Concepts of Overfitting and Underfitting

* **Overfitting:** Model learns the training data too well, including noise and random fluctuations. Performs well on training data but poorly on unseen (test/validation) data. High variance, low bias.
    * *Cause:* Model too complex for the data amount/noise (e.g., deep decision tree, high-degree polynomial regression, too many NN parameters).
    * *Detection:* Large gap between training performance and validation/test performance.
    * *Mitigation:* Get more data, simplify model, regularization, cross-validation, early stopping, dropout (NNs).
* **Underfitting:** Model is too simple to capture the underlying structure of the data. Performs poorly on both training and test data. High bias, low variance.
    * *Cause:* Model not complex enough (e.g., linear model for non-linear data). Insufficient training.
    * *Detection:* Poor performance on both training and validation/test sets.
    * *Mitigation:* Use a more complex model, add features/feature engineering, train longer (if applicable).
* **The Goal:** Find a balance (Bias-Variance Tradeoff) where the model generalizes well to new data.

---

### 16. Regularization

* **Explanation:** Techniques used to prevent overfitting by adding a penalty term to the model's loss function, discouraging overly complex models (typically by penalizing large parameter/weight values).
* **Types:**
    * **L1 Regularization (Lasso):** Adds penalty proportional to the *absolute value* of weights ($\lambda \sum |w_j|$). Encourages sparsity (some weights become exactly zero), performing implicit feature selection.
    * **L2 Regularization (Ridge):** Adds penalty proportional to the *squared value* of weights ($\lambda \sum w_j^2$). Shrinks weights towards zero but rarely makes them exactly zero. Handles multicollinearity well.
    * **Elastic Net:** Combination of L1 and L2 penalties ($\lambda_1 \sum |w_j| + \lambda_2 \sum w_j^2$). Combines benefits of both.
* **Tricks & Treats:** $\lambda$ (or `alpha` in scikit-learn) is the regularization strength hyperparameter, tuned via cross-validation. Feature scaling is usually required.
* **Caveats/Questions:** How to choose $\lambda$? Which type (L1/L2/Elastic Net) is best?
* **Python:** (Integrated into many `scikit-learn` models like `Ridge`, `Lasso`, `ElasticNet`, `LogisticRegression(penalty='l1'/'l2')`, `SGDClassifier`/`Regressor`).
* **Eval/Best Practices:** Use cross-validation to find optimal `alpha`/`lambda`. Elastic Net is often a good compromise if unsure between L1 and L2. Dropout is a common regularization technique for Neural Networks.
* **Libraries:** `scikit-learn`, `TensorFlow`/`Keras`, `PyTorch`.
* **GPU Opt:** The penalty calculation itself is minor, but models using regularization (Linear/Logistic/SVM/NNs) benefit from GPU acceleration for the main training computations.
* **Math/Subtleties:** Modifies the optimization objective: $\text{NewLoss} = \text{OriginalLoss} + \text{PenaltyTerm}$. From a Bayesian perspective, L2 corresponds to a Gaussian prior on weights, L1 to a Laplacian prior.

---

### 17. Evaluation Metrics

* **Explanation:** Quantify model performance. Choice depends on the task (regression vs. classification) and business goal.
* **Classification Metrics:**
    * **Accuracy:** (TP+TN)/(TP+TN+FP+FN). Simple, but misleading for imbalanced datasets.
    * **Precision:** TP/(TP+FP). Of those predicted positive, how many actually are? (Minimizes False Positives).
    * **Recall (Sensitivity, True Positive Rate):** TP/(TP+FN). Of all actual positives, how many were found? (Minimizes False Negatives).
    * **F1-Score:** $2 * \frac{Precision * Recall}{Precision + Recall}$. Harmonic mean, balances Precision and Recall.
    * **ROC AUC:** Area Under the Receiver Operating Characteristic curve (plots TPR vs. FPR at various thresholds). Measures ability to distinguish between classes across thresholds. Value > 0.5 is better than random; 1.0 is perfect.
    * **Confusion Matrix:** Table showing TP, TN, FP, FN counts.
    * **Log Loss (Cross-Entropy):** Measures performance of a classifier outputting probabilities.
* **Regression Metrics:**
    * **Mean Absolute Error (MAE):** $\frac{1}{n} \sum |y_i - \hat{y}_i|$. Average absolute difference. Robust to outliers. Interpretable in original units.
    * **Mean Squared Error (MSE):** $\frac{1}{n} \sum (y_i - \hat{y}_i)^2$. Average squared difference. Penalizes large errors more. Not in original units.
    * **Root Mean Squared Error (RMSE):** $\sqrt{MSE}$. In original units, easier to interpret than MSE. Still sensitive to large errors.
    * **R-squared ($R^2$, Coefficient of Determination):** $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$. Proportion of variance in the dependent variable predictable from the independent variables. Ranges from $-\infty$ to 1 (higher is better, 0 means no better than mean).
    * **Adjusted $R^2$:** $R^2$ adjusted for the number of predictors. Penalizes adding useless features.
* **Best Practices:** Use multiple metrics. Consider data imbalance (use Precision, Recall, F1, AUC for classification). Use cross-validation for robust estimates. Split data into Train/Validation/Test sets.

## Unsupervised Learning

*Goal: Discover patterns, structures, or representations in unlabeled data.*

---

### 1. Clustering Algorithms (General)

* **Explanation:** The task of grouping a set of objects such that objects in the same group (cluster) are more similar to each other than to those in other groups. Similarity is often based on distance or density in the feature space.
* **Types:** Include centroid-based (K-Means), density-based (DBSCAN), distribution-based (GMM), and hierarchical clustering.
* **Tricks & Treats:** Feature scaling is often crucial. The definition of "similarity" (distance metric) impacts results. Visualizing high-dimensional clusters is challenging (often requires dimensionality reduction).
* **Caveats/Questions:** How to determine the optimal number of clusters? How to evaluate clustering quality without labels? How to handle different cluster shapes and densities?
* **Eval/Best Practices:** See "Evaluation Metrics for Clustering" below. Domain knowledge often helps interpret clusters.
* **Libraries:** `scikit-learn`, `scipy.cluster`.
* **GPU Opt:** Possible for specific algorithms (see below).
* **SOTA Improvement:** Active research area, especially for large-scale, high-dimensional, or streaming data. Algorithms handling complex shapes, varying densities, and providing interpretability are key focus areas. Methods combining clustering with deep learning (deep clustering) are also advancing.

---

### 2. K-Means Clustering

* **Explanation:** A centroid-based partitioning algorithm. Aims to partition *n* observations into *k* clusters where each observation belongs to the cluster with the nearest mean (cluster centroid).
* **Algorithm:**
    1. Initialize *k* centroids (randomly or using methods like k-means++).
    2. **Assignment Step (E-step):** Assign each data point to the nearest centroid.
    3. **Update Step (M-step):** Recalculate the centroid position as the mean of all points assigned to that cluster.
    4. Repeat steps 2-3 until centroids stabilize or max iterations are reached.
* **Tricks & Treats:** Simple and fast for large datasets. k-means++ initialization helps avoid poor local optima.
* **Caveats/Questions:** Must specify *k* beforehand. Sensitive to initial centroid placement (run multiple times with different seeds). Assumes clusters are spherical, equally sized, and have similar density. Can be sensitive to outliers.
* **Python (Simple Example):**
    ```python
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    # Assume X_data (unlabeled features) is loaded
    # X_data = np.random.rand(100, 2) # Example

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) # Specify k=3, run 10 times
    kmeans.fit(X_scaled)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    print("Cluster labels assigned:", labels[:10])
    # print("Centroids:", centroids)
    ```
* **Eval/Best Practices:** Use the Elbow method (plotting inertia/sum of squared distances vs. *k*) or Silhouette score to help choose *k*. Run multiple initializations (`n_init`). Feature scaling is important.
* **Libraries:** `scikit-learn`, `scipy`.
* **GPU Opt:** Yes, `cuml.cluster.KMeans` provides significant speedups.
* **Math/Subtleties:** Minimizes within-cluster variance (Inertia or Sum of Squared Errors - SSE). Can be seen as an Expectation-Maximization algorithm with hard assignments.
* **SOTA Improvement:** Foundational. Improvements include better initialization (k-means++), mini-batch k-means for very large data, and kernel k-means for non-spherical shapes (though less common now).

---

### 3. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

* **Explanation:** A density-based clustering algorithm. Groups together points that are closely packed, marking outliers (noise points) that lie alone in low-density regions. Can find arbitrarily shaped clusters.
* **Algorithm:** Defines clusters based on core points, border points, and noise points.
    * **Core Point:** A point with at least `min_samples` neighbors within distance `eps`.
    * **Border Point:** Reachable from a core point but has fewer than `min_samples` neighbors.
    * **Noise Point:** Neither a core nor a border point.
* **Tricks & Treats:** Does not require specifying the number of clusters beforehand. Robust to outliers. Can find non-spherical clusters.
* **Caveats/Questions:** Sensitive to parameters `eps` (neighborhood distance) and `min_samples`. Struggles with clusters of varying densities. Distance metric choice is important. Finding optimal `eps` can be tricky (k-distance plot analysis is common).
* **Python:**
    ```python
    from sklearn.cluster import DBSCAN
    # Assume X_scaled is loaded

    # Parameters need tuning, e.g., via k-distance plot analysis or grid search
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan.fit(X_scaled)

    labels = dbscan.labels_ # Cluster labels, -1 indicates noise points
    print("DBSCAN labels (noise=-1):", labels[:10])
    # Number of clusters = len(set(labels)) - (1 if -1 in labels else 0)
    ```
* **Eval/Best Practices:** Tune `eps` and `min_samples`. Silhouette score can be used (ignoring noise points), but density-based metrics might be more appropriate if available.
* **Libraries:** `scikit-learn`.
* **GPU Opt:** Yes, `cuml.cluster.DBSCAN` offers significant acceleration.
* **Math/Subtleties:** Concepts of density reachability and density connectivity are central.
* **SOTA Improvement:** Core algorithm is widely used. Variants like OPTICS (orders points to extract density-based structure) and HDBSCAN* (hierarchical DBSCAN) address parameter sensitivity and varying densities. Specialized versions (e.g., T-DBSCAN for spatio-temporal data) exist.

---

### 4. Gaussian Mixture Models (GMM) & Expectation Maximization (EM)

* **Explanation:** A probabilistic model assuming data is generated from a mixture of several Gaussian distributions, each representing a cluster. Provides "soft" clustering (probabilities of belonging to each cluster).
* **EM Algorithm:** Used to fit GMMs. Iterative process:
    1. **Initialize:** Guess initial parameters (means $\mu_k$, covariances $\Sigma_k$, mixing coefficients $\pi_k$) for *k* Gaussians.
    2. **Expectation (E-step):** Calculate the probability (responsibility) that each data point belongs to each Gaussian component, given current parameters.
    3. **Maximization (M-step):** Update the parameters ($\mu_k, \Sigma_k, \pi_k$) to maximize the likelihood of the data given the responsibilities calculated in the E-step.
    4. Repeat steps 2-3 until convergence (log-likelihood stabilizes).
* **Tricks & Treats:** More flexible than K-Means (handles ellipsoidal clusters due to covariance). Provides cluster probabilities.
* **Caveats/Questions:** Must specify the number of components (*k*). Assumes data follows Gaussian distributions. Can be computationally intensive. Sensitive to initialization. Covariance type (spherical, tied, diag, full) is an important choice.
* **Python:**
    ```python
    from sklearn.mixture import GaussianMixture
    # Assume X_scaled is loaded

    gmm = GaussianMixture(n_components=3, random_state=42, covariance_type='full') # Specify k=3
    gmm.fit(X_scaled)

    labels = gmm.predict(X_scaled) # Hard assignment based on max probability
    probabilities = gmm.predict_proba(X_scaled) # Soft assignments
    print("GMM labels:", labels[:10])
    # print("GMM probabilities (first point):", probabilities[0])
    ```
* **Eval/Best Practices:** Use metrics like Bayesian Information Criterion (BIC) or Akaike Information Criterion (AIC) to help select *k* and `covariance_type`. Check convergence.
* **Libraries:** `scikit-learn`.
* **GPU Opt:** Less direct than K-Means/DBSCAN in scikit-learn. Probabilistic programming libraries (`Pyro`, `NumPyro`, `TensorFlow Probability`) built on GPU backends can accelerate EM or variational inference for GMMs.
* **Math/Subtleties:** Maximizes data log-likelihood. Latent variables indicate component membership. BIC/AIC balance model fit and complexity.
* **SOTA Improvement:** Standard technique for density estimation and soft clustering. Variational inference offers an alternative to EM for large datasets or more complex Bayesian GMMs.

---

### 5. Anomaly Detection (Outlier Detection)

* **Explanation:** Identifying data points, events, or observations that deviate significantly from the majority of the data ("normal" behavior). Can be unsupervised (no pre-labeled anomalies).
* **Methods:**
    * **Statistical:** Based on distribution (e.g., points outside 3 standard deviations for Gaussian data, EWMA for time series).
    * **Distance-Based:** Outliers are far from neighbors (e.g., KNN-based outlier factor).
    * **Clustering-Based:** Points not assigned to any cluster (like DBSCAN noise) or belonging to very small clusters.
    * **Dedicated Algorithms:** Isolation Forest (isolates anomalies using random trees), One-Class SVM (learns a boundary around normal data).
    * **Deep Learning:** Autoencoders (poor reconstruction error for anomalies), GANs, specialized networks.
* **Tricks & Treats:** Choice of method depends heavily on data type (tabular, time series, image) and expected anomaly type. Defining "normal" is key.
* **Caveats/Questions:** Need to define what constitutes an anomaly. Performance highly sensitive to thresholds or model parameters. Handling high-dimensional data can be challenging ("curse of dimensionality"). Often deals with highly imbalanced data (few anomalies).
* **Python (Isolation Forest Example):**
    ```python
    from sklearn.ensemble import IsolationForest
    # Assume X_scaled is loaded

    iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    # 'contamination' is expected proportion of outliers, can be float e.g. 0.01
    predictions = iso_forest.fit_predict(X_scaled) # Returns 1 for inliers, -1 for outliers

    print("Isolation Forest predictions (outlier=-1):", predictions[:10])
    # anomaly_scores = iso_forest.decision_function(X_scaled) # Lower score = more anomalous
    ```
* **Eval/Best Practices:** If some labeled anomalies exist (semi-supervised), use Precision, Recall, F1 on the anomaly class. Without labels, evaluation is often qualitative or relies on assumptions about anomaly scores. Domain expertise is crucial.
* **Libraries:** `scikit-learn` (IsolationForest, LocalOutlierFactor, OneClassSVM), `PyOD`, specialized libraries (`Anomalib`, `DeepOD`, `Orion`).
* **GPU Opt:** Some deep learning methods leverage GPUs heavily. Some components of other methods (like distance calculations) might be accelerated.
* **SOTA Improvement:** Deep learning approaches are SOTA for complex data like images/video. Research focuses on robustness, handling concept drift (changing normality), explainability, and few-shot/zero-shot anomaly detection.

---

### 6. Markov Methods (Briefly in Unsupervised Context)

* **Explanation:** Markov models (especially Hidden Markov Models - HMMs) can be used in an unsupervised way to discover underlying "hidden" states or recurring patterns in sequential data without explicit labels for those states.
* **Example Use Case:** Identifying different phases of machine operation based on sensor data, segmenting customer behavior sequences, or finding patterns in biological sequences.
* **Tricks & Treats:** Automatically detects characteristic, recurring patterns (states) from time series or sequences.
* **Caveats/Questions:** Requires specifying the number of hidden states. Assumes the Markov property holds (current state depends only on the previous state). Training (e.g., using Baum-Welch, an EM algorithm variant) can be complex.
* **Eval/Best Practices:** Often qualitative evaluation or based on downstream task performance. BIC/AIC can help select the number of states.
* **Libraries:** `hmmlearn`, `pomegranate`.
* **SOTA Improvement:** Foundational for sequence analysis. While powerful, often complemented or replaced by RNNs/LSTMs/Transformers for complex sequence modeling tasks (see Sequential Models / Deep Learning).

---

### 7. Self-Organizing Maps (SOM)

* **Explanation:** A type of Artificial Neural Network used for unsupervised learning, primarily for dimensionality reduction and clustering. Maps high-dimensional input data onto a lower-dimensional (typically 2D) grid of neurons, preserving topological relationships.
* **Algorithm:** Uses competitive learning. Input data points are presented, and the neuron with the closest weight vector (Best Matching Unit - BMU) is found. The BMU and its neighbors on the grid update their weights to become closer to the input point.
* **Tricks & Treats:** Excellent for visualization and exploratory data analysis. Creates a topology-preserving map.
* **Caveats/Questions:** Requires specifying grid size and topology. Training can be iterative and sensitive to learning rate and neighborhood function decay. Doesn't produce explicit cluster boundaries like K-Means.
* **Python (Using MiniSom):**
    ```python
    # pip install MiniSom
    from minisom import MiniSom
    # Assume X_scaled is loaded

    map_width = 10
    map_height = 10
    som = MiniSom(map_width, map_height, X_scaled.shape[1], sigma=1.0, learning_rate=0.5, random_seed=42)
    som.train_random(X_scaled, 100) # Train for 100 iterations

    # To get cluster assignments, can find BMU for each point
    winner_coords = np.array([som.winner(x) for x in X_scaled]).T
    # Further analysis/visualization needed (e.g., U-Matrix, mapping points to grid)
    ```
* **Eval/Best Practices:** Quantization error (average distance between data points and their BMU weights) and topographic error (proportion of points where the first and second BMUs are not adjacent) can measure map quality. Visualization (U-Matrix) is key.
* **Libraries:** `MiniSom`, `SuSi` (supports unsupervised/supervised SOMs).
* **GPU Opt:** Less common for standard SOMs, though some research explores parallelization.
* **Math/Subtleties:** Neighborhood function (often Gaussian) shrinks over time. Learning rate decays. Competitive learning updates weights: $\Delta w_{ij} = \eta(t) h_{ij}(t) (x - w_{ij})$.
* **SOTA Improvement:** Classic technique for visualization and topological mapping. Less common as a primary clustering method compared to K-Means/DBSCAN/GMM now, but valuable for specific exploratory tasks.

---

### 8. Deep Belief Nets (DBN) - Historical Context

* **Explanation:** Generative graphical models composed of multiple layers of latent variables ("beliefs"). Each layer is typically a Restricted Boltzmann Machine (RBM). Historically significant for unsupervised pre-training of deep neural networks.
* **Algorithm:** Trained greedily, one layer (RBM) at a time using Contrastive Divergence. The output activations of one trained RBM serve as input to train the next.
* **Tricks & Treats:** Can learn hierarchical representations of features. Provided a way to initialize deep networks before better methods (ReLU, better optimizers, batch norm) became widespread.
* **Caveats/Questions:** Training is complex and slow. Largely superseded by other deep learning techniques (VAEs, GANs, Transformers, better initialization strategies) for most tasks.
* **Python:** (Less common now, libraries like `scikit-learn` removed it. Requires older `tensorflow` versions or specialized libraries).
* **Eval/Best Practices:** Often evaluated by the performance of a supervised model fine-tuned after DBN pre-training, or by reconstruction error.
* **Libraries:** Historically in `scikit-learn`, older `TensorFlow`, specialized deep learning libraries.
* **GPU Opt:** RBM training (Contrastive Divergence) benefits from GPU acceleration.
* **Math/Subtleties:** Based on RBMs (energy-based models). Contrastive Divergence approximates the gradient of the log-likelihood.
* **SOTA Improvement:** Historically important for enabling deeper networks. Current usage is niche; concepts influenced modern deep generative models.

---

### 9. Evaluation Metrics for Clustering Problems

* **Explanation:** Assessing the quality of clusters formed by an algorithm. Metrics can be internal (based only on data and cluster assignments) or external (requiring ground truth labels, rarely available in pure unsupervised settings).
* **Internal Metrics (No Ground Truth Needed):**
    * **Silhouette Score:** Measures how similar a point is to its own cluster compared to other clusters. Ranges from -1 (bad) to +1 (dense, well-separated clusters). Average score over all points is often used. Higher is better.
    * **Davies-Bouldin Index (DBI):** Measures the average similarity ratio of each cluster with its most similar cluster. Based on cluster compactness and separation. Lower is better (minimum 0).
    * **Calinski-Harabasz Index (Variance Ratio Criterion):** Ratio of between-cluster dispersion to within-cluster dispersion. Higher score indicates denser, well-separated clusters. Higher is better.
* **External Metrics (Require Ground Truth Labels):**
    * **Adjusted Rand Index (ARI):** Measures similarity between true and predicted clusterings, adjusted for chance. Ranges from -1 to +1 (1 is perfect match, 0 is random).
    * **Mutual Information (MI) based scores:** (e.g., Normalized Mutual Information - NMI, Adjusted Mutual Information - AMI). Measure agreement between two assignments, ignoring permutations. Adjusted versions account for chance. Higher is better (0 to 1).
    * **Homogeneity, Completeness, V-measure:** Homogeneity = each cluster contains only members of a single class. Completeness = all members of a given class are assigned to the same cluster. V-measure = harmonic mean of homogeneity and completeness. (0 to 1).
* **Visual Methods:**
    * **Elbow Method:** Plots within-cluster sum of squares (Inertia for K-Means) vs. number of clusters (*k*). Look for an "elbow" point where the rate of decrease sharply changes (suggests optimal *k*).
    * **Silhouette Plots:** Visualizes the Silhouette score for each point within each cluster. Helps assess cluster density and separation, and identify potential misclassifications.
* **Python:**
    ```python
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    # Assume X_scaled and resulting cluster 'labels' are available

    # Use if you DON'T have ground truth
    sil_score = silhouette_score(X_scaled, labels)
    db_score = davies_bouldin_score(X_scaled, labels)
    ch_score = calinski_harabasz_score(X_scaled, labels)
    print(f"Silhouette Score: {sil_score:.3f}") # Higher better
    print(f"Davies-Bouldin Score: {db_score:.3f}") # Lower better
    print(f"Calinski-Harabasz Score: {ch_score:.3f}") # Higher better

    # from sklearn.metrics import adjusted_rand_score
    # Assume y_true (ground truth labels) are available
    # ari_score = adjusted_rand_score(y_true, labels)
    # print(f"Adjusted Rand Index: {ari_score:.3f}") # Higher better
    ```
* **Best Practices:** Use multiple internal metrics when ground truth is unavailable. Use visualization methods like Elbow and Silhouette plots. External metrics are definitive if ground truth exists. The "best" number of clusters often depends on the context and downstream application.

