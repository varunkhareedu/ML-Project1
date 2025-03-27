# Project 1 

Team Members:

* Naga Sunith Appasani
* Varun Khareedu
* Venkata Gandhi Varma Thotakura
* Parsha Saradhi Bobburi


Your objective is to implement the LASSO regularized regression model using the Homotopy Method. You can read about this method in [this](https://people.eecs.berkeley.edu/~elghaoui/Pubs/hom_lasso_NIPS08.pdf) paper and the references therein. You are required to write a README for your project. Please describe how to run the code in your project *in your README*. Including some usage examples would be an excellent idea. You may use Numpy/Scipy, but you may not use built-in models from, e.g. SciKit Learn. This implementation must be done from first principles. You may use SciKit Learn as a source of test data.

You should create a virtual environment and install the packages in the requirements.txt in your virtual environment. You can read more about virtual environments [here](https://docs.python.org/3/library/venv.html). Once you've installed PyTest, you can run the `pytest` CLI command *from the tests* directory. I would encourage you to add your own tests as you go to ensure that your model is working as a LASSO model should (Hint: What should happen when you feed it highly collinear data?)

In order to turn your project in: Create a fork of this repository, fill out the relevant model classes with the correct logic. Please write a number of tests that ensure that your LASSO model is working correctly. It should produce a sparse solution in cases where there is collinear training data. You may check small test sets into GitHub, but if you have a larger one (over, say 20MB), please let us know and we will find an alternative solution. In order for us to consider your project, you *must* open a pull request on this repo. This is how we consider your project is "turned in" and thus we will use the datetime of your pull request as your submission time. If you fail to do this, we will grade your project as if it is late, and your grade will reflect this as detailed on the course syllabus. 

You may include Jupyter notebooks as visualizations or to help explain what your model does, but you will be graded on whether your model is correctly implemented in the model class files and whether we feel your test coverage is adequate. We may award bonus points for compelling improvements/explanations above and beyond the assignment.

Running Tests

To run the test suite and validate your LASSO Homotopy model:

1. Setup Environment

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

2. Update Dataset Path in Test File

Open LassoHomotopy/tests/test_LassoHomotopy.py and update this line:

with open("../../data.csv", "r") as file:

Replace it with:

with open("LassoHomotopy/tests/small_test.csv", "r") as file:

Or use collinear_data.csv depending on the test scenario.

3. Run the Tests

From the root directory:

pytest LassoHomotopy/tests -s

The -s flag shows printed outputs (like coefficient values and sparsity checks).

Put your README here. Answer the following questions.

* **What does the model you have implemented do and when should it be used?**

  The version we carried out is a LASSO (Least Absolute Shrinkage and Selection Operator) regression the usage of the Homotopy approach, that's a path-following set of rules. LASSO is a shape of linear regression that provides L1 regularization to the loss function. This regularization time period encourages sparsity withinside the version coefficients, efficiently acting characteristic choice via way of means of using a few coefficients to precisely 0. The Homotopy approach solves the LASSO hassle via way of means of steadily adjusting the regularization parameter λ and tracing how the solution (i.e., the coefficients) evolves as λ decreases. This approach is green and interpretable.
This version have to be used whilst you count on that simplest a small subset of functions are sincerely critical for predicting your goal variable. It`s best in high-dimensional datasets, or whilst you need a version it is each correct and clean to interpret. Common use instances encompass scientific diagnosis, economic modeling, advertising analytics, and textual content classification, wherein deciding on a small variety of critical variables is crucial.

* **How did you test your model to determine if it is working reasonably correctly?**

  To make certain the version works correctly, we evolved a complete take a look at suite the usage of pytest. First, We created a take a look at the usage of artificial records with acknowledged coefficients to verify the version returns correct predictions and accurately sparse coefficients.We additionally as compared the output of my version to the effects from Scikit-learn's Lasso version beneathneath comparable settings. This allowed me to validate that the discovered coefficients have been numerically just like a well-installed library. Further,We examined the version's conduct on collinear records, wherein  functions are flawlessly correlated. In this kind of scenario, LASSO have to hold simplest one of the correlated functions — and my implementation behaved as expected, losing one and maintaining the other. Another take a look at evaluated how growing the regularization parameter (lambda_val) will increase the version's sparsity — lowering the variety of non-0 coefficients. These checks together demonstrated correctness, sparsity, and robustness, and helped construct self belief withinside the version`s conduct throughout quite a few real-global scenarios.

* **What parameters have you exposed to users of your implementation in order to tune performance?**

  The maximum critical parameter uncovered to customers is lambda_val, which controls the energy of L1 regularization. A better lambda penalizes coefficients greater heavily, main to a sparser version with fewer non-0 coefficients. Conversely, a decrease lambda lets in the version to in shape the records greater closely, doubtlessly together with greater functions. Tuning lambda_val lets in customers to stability among version simplicity and prediction accuracy.Other parameters encompass max_iter, which defines what number of steps the Homotopy set of rules have to take earlier than stopping. This is beneficial for stopping endless loops or overly lengthy education times. Additionally, tol (tolerance) controls the edge for convergence — it determines how small a coefficient should be to be taken into consideration 0 and whilst to forestall iterating. Together, those parameters provide customers manipulate over the sparsity, efficiency, and precision of the version. In practice, lambda_val is the maximum generally tuned parameter, regularly decided on thru strategies like cross-validation to discover the great trade-off among generalization and interpretability.

* **Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?**

  Yes, certain types of inputs can challenge the current implementation. Specifically, highly collinear features or very high-dimensional data (p >> n) can lead to numerical instability in the matrix inversion steps used in the Homotopy algorithm. Although the current version uses the pseudo-inverse to avoid crashing, it may still become unstable or imprecise for extremely ill-conditioned matrices. Additionally, the model assumes that all input features are numeric and continuous — it won't work properly on raw categorical or missing data without preprocessing. Another limitation is that the current implementation performs batch updates, which may not scale efficiently for very large datasets. In contrast, some production-ready libraries use coordinate descent or stochastic methods that scale better. Given more time, We could improve stability by integrating regularized matrix inversion, better numerical conditioning, or switching to Cholesky decomposition. For scalability, implementing a coordinate descent version or adding mini-batch support would be beneficial. These are not fundamental limitations of LASSO itself, but rather areas where engineering improvements could make the model more robust and practical.

