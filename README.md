# Optimizing-ML-Pipline-in-Azure-
i had  the opportunity to create and optimize an ML pipeline, with  provided a custom-coded model—a standard Scikit-learn Logistic Regression—the hyperparameters of which to optimize using HyperDrive, Also use AutoML to build and optimize a model on the same dataset, so that you can compare the results of the two methods

![Diagram](https://user-images.githubusercontent.com/59172649/143721229-830f94d2-e4b1-473d-95a9-2c086826db9f.JPG)
# Scikit-learn Pipeline
Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.
# Scikit-learn

A Logistic Regression model was first created and trained using Scikit-learn in the train.py. The steps taken in the python script were as follows:

    Import the banking dataset using Azure TabularDataset Factory

    Data is then cleaned and transformed using a cleaning function

    Processed data is then split into a training and testing set

    Scikit-learn was used to train an initial Logistic Regression model while specifying the value of two hyper parameters, C and max_iter. C represents the inverse of the regularization strength, while max_iter represents the maximum number of iterations taken for the model to converge. These two parameters were initially passed in the python script so they can be optimised later using Hyperdrive.
    Once the data has been prepared it is split into a training and test set. A test set size of 22% of total entries was selected as a compromise between ensuring adequate representation in the test data and providing sufficient data for model training.

The classification method used here is logistic regression. Logistic regression uses a fitted logistic function and a threshold. The parameters available within the training script are C (which indicates the regularization strength i.e. preference for sparser models) and maximum number of iterations.

The trained model is then saved
# Hyper Drive

The initial model trained is then optimised using Hyperdrive. Hyperdrive is a method of implementing automatic hyperparameter tuning. Hyperparameter tuning is typically computationally expensive and manual, therefore, by using Hyperdrive we can automate this process and run experiments in parallel to efficiently optimize hyperparameters.

# The steps taken to implement Hyperdrive were as follows:

    Configuration of the Azure cloud resources

    Configuring the Hyperdrive

    Running the Hyperdrive

    Retrieving the model with the parameters that gave the best model

Elaborating more on the second step in configuring the Hyperdrive, there are two extremely beneficial parameters that are included in the configuration; RandomParameterSampling and BanditPolicy.


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
i defined the following configuration for the AutoML Run :

![image](https://user-images.githubusercontent.com/59172649/144405310-535c8f58-150f-4c30-8d94-884083a941d9.png)

experiment_timeout_minutes=15

This is an exit criterion and is used to define how long, in minutes, the experiment should continue to run. To help avoid experiment time out failures, I used the minimum of 15 minutes.

task='classification'

This defines the experiment type which in this case is classification.

primary_metric='accuracy'

I chose accuracy as the primary metric.

n_cross_validations=2

This parameter sets how many cross validations to perform, based on the same number of folds (number of subsets). As one cross-validation could result in overfit, in my code I chose 2 folds for cross-validation; thus the metrics are calculated with the average of the 2 validation metrics.
The Voting Ensemble model selected used a slight amount of l1 regularization, meaning that some penalty was placed the number of non-zero model coefficients. Additionally, the voting method was soft voting (as compared to hard), where all models' class probabilities are averaged and the highest probablility selected to make a prediction. Although the learning rate scheduling for gradient descent is specified as 'invscaling' (i.e. inverse scaling), the scaling factor power is 0 indicating that the learning rate is constant in this case.

The best performing model was a VotingEnsemble with one of its metaleaners as a logistic regression model with a max iteration value of 200 and C of 100. The metalearner used cross-validation version. AutoML also produced useful classification metrics that we didn't define and has a feature to explain the best model.
# Pipeline comparison
The difference in accuracy between the two models is not so big although the HyperDrive model performed better in terms of accuracy.If we were given more time to run the AutoML, the resulting model would certainly be much more better. And the best thing is that AutoML would make all the necessary calculations, trainings, validations, etc. without the need for us to do anything. 
The best model was a VotingEnsemble Model with an accuracy of 0.9159 that was obtained using AutoML vs Hyperdrive's logistic regression of max iterations 200 and C of 100 that had an accuracy of 0.9176. Since ensembling using many models to make a decision, as compared to Hyperdrive's one logistic regression, it was able to achieve a higher score.
# Future Improvement
Advanced Feature engineering. deleting and creating new features and using other methods such as target encoding could help improve the results. The resulting feature set will be able to map well to the target variable resulting into better models.

To improve the AutoML score, we could increase the amount of time so that many more algorithms and approaches can be tried. To improve HyperDrive, increase the range of hyperparameters that are to be tested which allows for thorough tuning.
