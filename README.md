# Wine Quality Classification with Support Vector Machines and Neural Networks 

## Summary
This project revolves around quickly deploying support vector machines and neural networks in R to classify the quality of various red and white Portuguese "Vinho Verde" wines on a scale of 1-10 using only the physicochemical properties of the wines. The <a href="https://archive.ics.uci.edu/ml/datasets/Wine+Quality" target="_blank">Wine Dataset</a> includes 6,497 observations with 12 variables in total. In order to train the models, the data is split into three datasets using random sampling methods: Training, Validation, and Testing. Because there are very few observations regarding quality three and nine wines, they are removed from the dataset to avoid introducing unneccessary noise in the data. 

## Support Vector Machine Results
The <a href="https://github.com/mlombera94/Wine-Classification/blob/master/SVM_algorithm.md" target="_blank">first machine learning model</a> attempted involved using support vector machines. Prior to running SVM models on the data, the data must be normalized between 0 and 1. To begin, a simple linear SVM model was applied as a baseline model. Because the data consists of multiple variables, it is clear the data cannot be linearly seperable. Therefore, in order to improve upon the model, different kernel functions should be applied to project the data into higher dimensions. The most common kernel applied for data that is not linearly seperable is the radial kernel (RBF). Using the tune.svm function from the package <a href="https://www.rdocumentation.org/packages/e1071/versions/1.7-2" target="_blank">e1071</a>, we can tune the hyperparameters of statistical methods using a grid search over the supplied parameter ranges to obtain the most optimized models. For example, when working with the polynomial kernel, there are five hyperparameters that can be tuned: cost, coef0, degree, gamma and epsilon. It is recommended to work with at most three hyperparameters at once and no more than four potential choices per hyperparameter as this may take hours to run or could possibly run indefinitely. Using the tune.svm function with various kernels, the best model resulted from the RBF kernel with cost set to 1, gamma set to 0.5 and epsilon set to 0.2 with an accuracy rate of 61.2%. This classification rate however is substantially low. Using the confusion matrix visualization, it becomes clear the model struggles to classify any quality of wine other than 6. This could be due to the fact that 44% of the observations were quality six wines resulting in bias within the model. Applying the model to the test dataset, the model achieves a classification rate of 60.4%

## Neural Network Results 
The <a href="https://github.com/mlombera94/Wine-Classification/blob/master/Neural_Networks.md" target="_blank">second machine learning model</a> deployed to classify the quality of the wine was neural networks. Similar to support vector machines, the data must be normalized within a standard range of 0 to 1 prior to feeding the data to the model. Once the data has been normalized and split into seperate datasets, a simple  neural network model can be applied. The first model attempted is a simple multilayer feedforward network with a single hidden layer is trained. This results in classification accuracy of 52%. To improve upon the baseline model, hidden layers can be added and/or different activation functions can be applied as well to the model. By default, the <a href="https://www.rdocumentation.org/packages/neuralnet/versions/1.44.2" target="_blank">neural net</a> package uses the a single hidden layer and a logistic activation function. An important note, adding too many hidden layers to the model may result in bias when training the model on the training dataset. This in turn leads to poor performance on unseen data (validation and test datasets). After attempting various number of hidden layers and different activation functions, the model that performed the best resulted from using a logistic activation function (default function) and five hidden layers. The classification accuracy from this model was 55.9%. Applying the model to the test dataset, the model achieves a classification rate of 52.3%. 

## Conclusion
It is evident that both models performed poorly once applied to the test dataset. Although the best support vector machine outperformed the best neural network model by 8.4%, it is clear there is room for improvement on either the model side or the data side. However, based on prior experience working with classification datasets, the problem appears to stem with the data either being too noisy, non-informative, or a combination of both. It is more likely that the physiochemical properties of the various wines provide no insight into the quality of the wine. Another problem that arises with the dataset is there is no clear scientific methodology for classifying and rating the qualities of the wines. This information is crucial as there is no way to verify the reasons behind each rating and whether the reasons remain consistent with each wine. Moving forward with this project would require more information on how the rating system works and possibly introducing further data such as the what grapes were used, the age of the wine, and how the wine was distilled and stored. 


