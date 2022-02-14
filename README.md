# Description of project

## Program
We can display how program works by showcasing simple steps inside of it which can be translated into functions in main:
1. Loading annotations - **GetAnnotationData**
2. Displaying objects quantity - **CheckQuantity**
3. Printing loaded training annotations - **PrintAnnotations**
4. Loading images data - **LoadData**
5. Balancing data - **BalanceData**
6. Either loading voc.npy data file or learning with bag of views usage - **LearnBoVW**
7. Extracting descriptors and adding them to training data - **ExtractFeatures**
8. Training on our training dataset and then returning rainforest - **Train**
9. Printing test annotations - **PrintAnnotations**
10. Loading testing dataset - **LoadData**
11. Balancing testing dataset - **BalanceData**
12. Extracting descriptors and adding them to testing dataset - **ExtractFeatures**
13. Predict objects using rainforest created with dataset on testing dataset - **Predict**
14. Evaluating results of prediction - **Predict**

That's how we can shortly describe how code works. What is worth mentioning is that first approach was made with only reading images. What followed was that results ware clearly having low accuraccy(75%). After seeing that during loading images process every object inside of dataset was extracted from image so that accuraccy would increase. That indeed happened and source of this issue was probably very little amount of actual crosswalk signs inside of data. After adding object extraction to training dataset accuracy increased to almost 90%.