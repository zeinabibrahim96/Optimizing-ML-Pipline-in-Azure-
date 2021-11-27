# Optimizing-ML-Pipline-in-Azure-
i had  the opportunity to create and optimize an ML pipeline, with  provided a custom-coded model—a standard Scikit-learn Logistic Regression—the hyperparameters of which to optimize using HyperDrive, Also use AutoML to build and optimize a model on the same dataset, so that you can compare the results of the two methods

![Diagram](https://user-images.githubusercontent.com/59172649/143721229-830f94d2-e4b1-473d-95a9-2c086826db9f.JPG)
# Scikit-learn Pipeline
Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.

# Parameter sampler

I used RandomParameterSampling as  the parameter sampler :

RandomParameterSampling is one of the choices available for the sampler and I chose it because it is the faster and supports early termination of low-performance runs. If budget is not an issue, we could use GridParameterSampling to exhaustively search over the search space or BayesianParameterSampling to explore the hyperparameter space.

# Early stopping policy

An early stopping policy is used to automatically terminate poorly performing runs thus improving computational efficiency. I chose the BanditPolicy which I specified as follows:

policy = BanditPolicy(slack_factor = 0.2, evaluation_interval=1, delay_evaluation=5)

evaluation_interval: This is optional and represents the frequency for applying the policy. Each time the training script logs the primary metric counts as one interval.

slack_factor: The amount of slack allowed with respect to the best performing training run. This factor specifies the slack as a ratio.

Any run that doesn't fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated. This means that with this policy, the best performing runs will execute until they finish and this is the reason I chose it.
# AutoML
The best performing model was a StackEnsembleClassifier with one of its metaleaners as a logistic regression model with a max iteration value of 100 and C of 10. The metalearner used cross-validation version. AutoML also produced useful classification metrics that we didn't define and has a feature to explain the best model.
# Pipeline comparison
The best model was a StackEnsemble Model with an accuracy of 0.9168 that was obtained using AutoML vs Hyperdrive's logistic regression of max iterations 82 and C of 1 that had an accuracy of 0.91162. Since ensembling using many models to make a decision, as compared to Hyperdrive's one logistic regression, it was able to achieve a higher score.
# Future Improvement
Advanced Feature engineering. deleting and creating new features and using other methods such as target encoding could help improve the results. The resulting feature set will be able to map well to the target variable resulting into better models.

To improve the AutoML score, we could increase the amount of time so that many more algorithms and approaches can be tried. To improve HyperDrive, increase the range of hyperparameters that are to be tested which allows for thorough tuning.
