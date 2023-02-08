<h1 align='center'>Melbourne Housing Prices Regression</h1>

![alt text](https://github.com/vanquies/HousingPrices/blob/master/Images/Melbourne.png)

## Project Introduction

Over the last decades situation in the real estate sector has changed significantly. Properties prices are higher than ever before. 
Especially in well developed cities which offer numerous jobs for educated people. Urbanization is progressing all over the world. 
Inhibitants of rural areas migrate to surrounding cities in search of better jobs. Such process disturbs supply demand balance and causes real estate prices to rise.
<br>

Looking for your own property in such a large market can be overwhelming and cause problems in estimating whether the price from sales announcement 
is adequate to property parameters. The model created in this project addresses this problem. Algorithm estimates approximate value of property price 
then user can compare result with the price included in the sale offer. This model is useful for individual clients looking for own place to live. 
Also could be part of decision support system for real estate agents and companies. 

## Table of Contents

1. [ Data Source ](#Data_Source)
2. [ Files Structure ](#Files_Structure)
3. [ Technologies ](#Technologies)    
4. [ Notebooks Structure ](#Notebooks_Structure)
5. [ Project Summary ](#Project_Summary)
   * [ 1. Exploratiory Data Analysis ](#EDA)
       * [ Data understanding ](#Data_Understanding)
       * [ Preliminary cleaning ](#Preliminary_Cleaning)
       * [ Further data exploration ](#Further_Data_Exploration)
       * [ Data visualization ](#Data_Visualization)
   * [ 2. Data Preprocessing and Basic Models ](#DPandBM) 
       * [ Data preprocessing ](#Data_Preprocessing)
       * [ Basic models ](#Basic_Models)
   * [ 3. Random Forest Regression ](#RFR)
   * [ 4. Support Vector Regression ](#SVR)
   * [ 5. XGBoost Regression ](#XGB)
6. [ Conclusion ](#Conclusion)
 

## Data Source
<details>  
  <a name="Data_Source"></a>
  <summary>Show/Hide</summary>
  
  <br>

  Whole project is based on [dataset](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot) named Melourne Housing Snapshot from kaggle.com. 
  This csv file is a snapshot of a [dataset created by Tony Pino](https://www.kaggle.com/datasets/anthonypino/melbourne-housing-market). Data was scraped from 
  publicly available results posted every week from Domain.com.au and cleaned. The dataset includes Address, Type of Real estate, Suburb, Method of Selling, 
  Rooms, Price, Real Estate Agent, Date of Sale and distance from C.B.D (central business district). 

  ### Features description:

  Rooms: Number of rooms

  Price: Price in dollars

  Method: S - property sold; SP - property sold prior; PI - property passed in; PN - sold prior not disclosed; SN - sold not disclosed; NB - no bid; VB - vendor bid; W   - withdrawn prior to auction; SA - sold after auction; SS - sold after auction price not disclosed. N/A - price or highest bid not available.

  Type: br - bedroom(s); h - house,cottage,villa, semi,terrace; u - unit, duplex; t - townhouse; dev site - development site; o res - other residential.

  SellerG: Real Estate Agent

  Date: Date sold

  Distance: Distance from CBD

  Regionname: General Region (West, North West, North, North east …etc)

  Propertycount: Number of properties that exist in the suburb.

  Bedroom2 : Scraped # of Bedrooms (from different source)

  Bathroom: Number of Bathrooms

  Car: Number of carspots

  Landsize: Land Size

  BuildingArea: Building Size

  CouncilArea: Governing council for the area
</details>

## Files Structure
<details>
  <a name="Files_Structure"></a>
  <summary>Show/Hide</summary>
  <br>

  * <strong>[ Data ](https://github.com/vanquies/HousingPrices/tree/master/Data)</strong>: folder containing all data files
      * <strong>HousingCleaned.csv</strong>: data after cleaning
      * <strong>PreparedFeatures.csv</strong>: dataframe of variables (columns names needed in feature importance check)
      * <strong>X_test.csv</strong>: features of test data
      * <strong>X_train.csv</strong>: features of training data
      * <strong>melb_data.csv</strong>: raw imported data from kaggle
      * <strong>y_test.csv</strong>: labels of test data
      * <strong>y_train.csv</strong>: labels of training data
  * <strong>[ Images ](https://github.com/vanquies/HousingPrices/tree/master/Images)</strong>: folder with images used for README
  * <strong>[ Models ](https://github.com/vanquies/HousingPrices/tree/master/Models)</strong>: folder containing saved models
  * <strong>[ Workspace ](https://github.com/vanquies/HousingPrices/tree/master/Workspace)</strong>: folder containing all notebooks created in this project
      * <strong>01_ExploratoryDataAnalysis.ipynb</strong>
      * <strong>02_DataPreprocessing&BasicModels.ipynb</strong>
      * <strong>03_RandomForrestRegression.ipynb</strong>
      * <strong>04_SupportVectorRegression.ipynb</strong>
      * <strong>05_XGBoostRegression.ipynb</strong>
      * <strong>Functions.ipynb</strong>
</details>

## Technologies
<details>
  <a name="Technologies"></a>
  <summary>Show/Hide</summary>
  <br>

  * <strong>Matplotlib</strong>
  * <strong>Numpy</strong>
  * <strong>Pandas</strong>
  * <strong>Python</strong>
  * <strong>Scikit-Learn</strong>
  * <strong>Seaborn</strong>
  * <strong>XGBoost</strong>
</details>

## Notebooks Structure:
<details>
  <a name="Notebooks_Structure"></a>
  <summary>Show/Hide</summary>
  <br>

  1. [Exploratory Data Analysis](https://nbviewer.org/github/vanquies/HousingPrices/blob/master/Workspace/01_ExploratoryDataAnalysis.ipynb)
     * 1.1 Imports
     * 1.2 Data understanding
     * 1.3 Preliminary cleaning
     * 1.4 Further data exploration
     * 1.5 Data visualization

  2. [Data Preprocessing and Basic Models](https://nbviewer.org/github/vanquies/HousingPrices/blob/master/Workspace/02_DataPreprocessing%26BasicModels.ipynb)
     * 2.1 Imports
     * 2.2 Data preprocessing
     * 2.3 Basic models training

  3. [Random Forest Regression](https://nbviewer.org/github/vanquies/HousingPrices/blob/master/Workspace/03_RandomForestRegression.ipynb)
     * 3.1 Imports
     * 3.2 Learning process visualization
     * 3.3 Hyperparameters tuning
     * 3.4 Tuned model evaluation

  4. [Support Vector Regression](https://nbviewer.org/github/vanquies/HousingPrices/blob/master/Workspace/04_SupportVectorRegression.ipynb)
     * 4.1 Imports
     * 4.2 Learning process visualization
     * 4.3 Hyperparameters tuning
     * 4.4 Tuned model evaluation

  5. [XGBoost Regression](https://nbviewer.org/github/vanquies/HousingPrices/blob/master/Workspace/05_XGBoostRegression.ipynb)
      * 5.1 Imports
      * 5.2 Ploting learning curve, model scalability and performance
      * 5.3 Hyperparameter tuning
      * 5.4 Tuned model evaluation
      * 5.5 Back to real prices
</details>  
   
<a name="Project_Summary"></a>
## Project Summary


<a name="EDA"></a>
### Exploratory Data Analysis
<details open>
  <summary>Show/Hide</summary>
  <br>

  <a name="Data_Understanding"></a>
  #### Data understanding

  In first contact with data it is vital to understand data in general. For this purpose after reading csv file there are used methods like shape (shape of data 
  frame), 
  info (mainly informations about data types of indivifual variables) and columns (list of columns names). Set consists of 13580 rows and 21 variables describing 
  each row (variables description is included in Data Source section). Set takes only 7.4MB memory, most of features are float64 type, 7 is object and one for 
  int64 and datetime64.
  <br>

  <a name="Preliminary_Cleaning"></a>
  #### Preliminary cleaning

  At this stage it was decided to change columns names to Pascal Cases for better clarity and uniformity. Also columns like Address, Postcode and SellerG 
  were drop because of their lack of usability.
  <br>

  <a name="Further_Data_Exploration"></a>
  #### Further data exploration

  Next step is focused on descriptive statistical aspect of the set. First data frame of this notebook section contains descriptive statistics like mean, count, 
  standard deviation, min/max and quantiles of numerical features. Is very important casue gives initial insight in variables nature and their order of magnitude. 
  Second frame is similar but informations are about object variables. In third are about date. Below these dataframes cells outputs returns informations about 
  number of unique values for every object type variable and most frequent values. 
  <br>

  <a name="Data_Visualization"></a>
  #### Data visualization

  When we have got basic knowledge about the nature of the variables we can start visualization. For the beginning helpful is pandas profiling library. 
  Here we can check distribution shape of features and basic informations abount them in compact viewing method. <br>
  
  ![alt text](https://github.com/vanquies/HousingPrices/blob/master/Images/01_1.png)

  Correlation matrix shows high correlation between number of rooms and bedrooms. Moderate correlation with bathrooms and price. What is expected we can observe 
  interdependence between land size and building area. <br>
  
  <strong>Timeline with frequency of sales</strong> 
  ![alt text](https://github.com/vanquies/HousingPrices/blob/master/Images/01_2.png)

  Next picture represents timeline and number of sold properties in partitioned periods. Record sales occurred in August 2017. 
  The period examined is too short to unequivocally state whether there is a phenomenon of yearly seasonality in the demand on 
  the real estate market. Nonetheless histogram shows significant drop in winter season 2017 what may herald such phenomenon. <br>
  
  <strong>Scatterplot with co-ordinates of properties sold</strong> 
  ![alt text](https://github.com/vanquies/HousingPrices/blob/master/Images/01_3.png)

  Here we can see scatterplot of properties co-ordinates forming regions separated with hue color. Five regions are metropolitan and three (victoria) are suburbs.
  Most valueable housings on average are place in southern and eastern metropolitan region. <br>
  
  <strong>Scatterplot of distance and price dependence</strong> 
  ![alt text](https://github.com/vanquies/HousingPrices/blob/master/Images/01_4.png)

  This scatterplot illustrates dependence between distance from city centre and price. The vast majority of properties sold with price exceeding $2mln are closer 
  to C.B.D. than 20 miles. Building type u (flat, semi-detatched house) on average cost much lower than h (houses, villas) and t (townhouses). <br> 
</details>

<a name="DPandBM"></a>
### Data Preprocessing and Basic Models
<details open>
  <summary>Show/Hide</summary>
  <br>

  <a name="Data_Preprocessing"></a>
  #### Data preprocessing

  For efficient training and better results data need to be processed. At first in this notebook there were filled NaN values. Where it was possible missing 
  data were filled based on other features indicating value of empty column. This method was used in case of council area variable. Other features containing 
  missing data were imputated with typical methods like backfill, forwardfill or statistical median. <br>

  Second step in data preparation was elimination of outliers. It was decided that outliers of five features above will be eliminated manualy (because of 
  narrow range of values). On three other features with significant outliers was used IQR method. Some of variables has outlying values which are indelible, 
  for example longitude and latitude. <br>
  
  <strong>Graphs showing distriburion of seasons</strong> 
  ![alt text](https://github.com/vanquies/HousingPrices/blob/master/Images/02_2.png)
  
  It was decided that new variable which splits observations into for year season would be somehow informative for model. Surely more than individual dates 
  so SaleDate was replaced with Seasons. In second and third quarter of the year there were about two times more signed sale agreements than in autumn-winter 
  season. In violinplot it can be noticed that mean of season 4 is slightly higher than other seasons and can be cause by thicker tail of this violin. <br>
  
  <strong>Skewness of variables</strong> <br>
  ![alt text](https://github.com/vanquies/HousingPrices/blob/master/Images/02_1.png)
  
  Some of variables are visibly skewed (distribution plots from pandas profiling) and frame above shows numerical representation of this skew. 
  Distance and PropertyCount are above 1, so these two was transformed. Also for better fit to data Price dependent variable was transformed 
  with logarithm. <br> 
  
  Next step of preprocessing is encoding of categorical features to numerical representation so algorithm can process the data. There are five 
  variables with object data types. Method, Type and RegionName have less than 10 unique values so these can be encoded with get_dummies method. 
  Suburb and CouncilArea are too diverse in number of unique values for this method, so binary encoding was used. <br>
  
  Before train-test split it is time for check the multicollinearity between variables. Variance Inflation Factor method was used to measure occurance. 
  VIF interpretation:
  - VIF < 1 (no multicollinearity)
  - 1 < VIF < 10 (moderate multicollinearity, no additional action needed)
  - ViF > 10 (multicollinearity among two or more variables, one of them should be eliminated) <br>
  
  Third situation occured in one case in discussed, rooms and bedrooms number specifically. Under these circumstances Rooms variable was eliminated. <br>
  
  The only steps remaining in data preprocessing are train-test split and feature scalling. In this project data was scaled with robust scaler because 
  that one ignores outliers and as it was said before some of variables have significant outliers e.g. longitude and latitude. At this stage data is ready 
  for modelling, preprocessed and split to train/test features/labels.
  
  <a name="Basic_Models"></a>
  #### Basic models
  
  <strong>Basic models training results</strong> <br>
  ![alt text](https://github.com/vanquies/HousingPrices/blob/master/Images/02_3.png)
  
  For modelling stage under consideration were taken five regression algorithms. Linear regression, ransac, random forest regresion, support vector regression 
  and XGBoost regression. Training with default hyperparameters brought results shown on picture above. at first can be seen that linear regression and ransac 
  are not siutable for this kind of task. Linear regression is to straight-forward method for building model with such number of features used in modelling. 
  For now the most promising results gives XGBoost regressor with RMSE = 0.17 and R^2 0.864. Not much worse did random forest regression and support vector 
  regression. With this results further modelling and tuning will be focused on these three algorithms.
</details>

<a name="RFR"></a>
### Random Forest Regression
<details open>
  <summary>Show/Hide</summary>
  <br>
  
  At the beginning of this and other modelling notebooks there are graphs showing how the algoritm behaves in the learning process. Graphs are generated 
  with extensive functions stored in Functions.ipynb and imported to notebooks where were used. First function generates three graphs: learning curve 
  (how score of training and cross-validation set is increasing when size of set is enlarged), scalability of the model (how training set enlarging influences 
  on time algorithm needs to learn) and performance of the model (it is combination of previous graphs, shows dependence between score increase with increasing
  training time). <br>
  
  ![alt text](https://github.com/vanquies/HousingPrices/blob/master/Images/03_1.png)
  
  In this case Random Forrest Regressor reaches generalization plateau at R2 = 0.85 and 7500 samples. Training and cross-validation curves are almost parallel. 
  Theres no reason for increasing training dataset because it would not improve model fit. Model scales neutrally and training takes time proportionally to 
  rising number of expamples. Third graph shows that model performance improves in whole fitting time. <br>
  
  For efficient random forest hyperparameters tuning were used functions RandomizedSearchCV and GridSearchCV from library sklearn.model_selection. These popular
  tools can reduce time needed for searching optimal model parameters. Also part of these functions is training with cross-validation method, which reduces risk 
  of model overfitting. <br>
  
  Despite multiple attemps hyperparameter tuning hasn't bring better results on test set than basic model. R squared and rmse are similar to not tuned one. 
  Default random forest hyperparameters are not providing many limitations on trees size in the forest so it is hard to increase performance of basic model. 
  Despite little worse results trained model better generalizes data and is less overfitting. <br>
  
  <strong>Feature importances in random forest regression model</strong> 
  ![alt text](https://github.com/vanquies/HousingPrices/blob/master/Images/03_2.png)
  
  For algorithm there were many informative features. Highest influence on price estimation has building type_u (twin house, flat), propably these types are 
  cheaper than houses. At second place is building age and third bedrooms number, landsize and latitude ex aequo. Least useful variables for random forest 
  were 7 suburb related features and sale method. <br>
  
  Next part of notebook is residuals analysis used to determine the validity of constructed model. Methods used in studying this topic were: residuals 
  distribution plot, residuals q-q plot, heteroskedasticity plot and scale location plot.
</details>

<a name="SVR"></a>
### Support Vector Regression
<details open>
  <summary>Show/Hide</summary>
  <br>
  
  Support vector regression hyperparameter tuning process was very similar to random forest tuning. In contrast to random forrest, training score on learning 
  curve is very high when using few samples and decreses along with increasing samples number. Training and validation curves are convergent along the x-axis. 
  Model needs more examples to reach score similar to random forrest regressor performance. Model scales quadratically which significantly increases the time 
  needed for training. <br> 
  
  ![alt text](https://github.com/vanquies/HousingPrices/blob/master/Images/04_1.png)
  
  Validation curve is generated with second function saved in Functions.ipynb and its role is to present model fit to the data with specified value of 
  tested parameter. Graph above shows that optimal C parameter value is located around 1. It is helpful for selecting proper value range in further 
  GridSearch cross-validation and gives insight in individual parameters nature. <br> 
  
  ![alt text](https://github.com/vanquies/HousingPrices/blob/master/Images/04_2.png)
  
  Second graph shows similar curves. This time tested is value of gamma parameter. Changes in score are subtle until the value reaches 0.1. When gamma 
  is bigger score falls down. In this case optimal value is located somewhere around 0.1. <br> 
  
  These parameters are have largest influence in support vector machine performance, other are negligible. Tuning process brought following results: 
  RMSE = 0.175 and R^2 = 0.856. And unfortunately again it is performance bit worse than basic model. In tuning process many different configurations
  were tried. Cross-validation indicated that results are better when C is equal 0.7 (default = 1). This had no reflection when tested on test set. 
  Default gamma setting is 'scale' and apparently it is the best option, none of the manual settings came close to the result of the gamma calculated 
  with 'scale' formula. Both random forest regression and support vector regression tuning haven't brought new level of model performance. Maybe 
  default settings are optimal for purpose of this project or what is more likely I'll able to imporve their results when I gain more knowledge and 
  familiarity with these algorithms.
</details>

<a name="XGB"></a>
### XGBoost Regression
<details open>
  <summary>Show/Hide</summary>
  <br>
  
  Last and the most fruitful tuning notebook in this project is about XGBoost regression. Even though the model had the best performance among all other 
  models with default hyperparameters, there were no complications in improving the score. As with other algorithms notebooks starts with learning curve 
  visualization. <br> 
  
  ![alt text](https://github.com/vanquies/HousingPrices/blob/master/Images/05_6.png)
  
  Like in svr case training curve and cross-validation curve are convergent along the x-axis but gap between curves is larger. Training score is much higher, 
  which may mean that XGBoost tends more to overfit but in cross-validation reaches plateau at score 0.85 R^2 with 7500 training examples. XGB has the best 
  scalability among all three models. Samples collection enlargement does not imply much fit time increase. What is more examples number above 6000 decreases 
  training time. <br>
  
  Seven hyperparameters were selected for model tuning:
  - Eta (Eta is step size shrinkage used in update to prevents overfitting) <br>
  - Colsample by tree (Colsamples by tree represents the fraction of columns to be randomly sampled for each tree. It might improve overfitting) <br>
  - Learning rate (The learning rate determines size of models parameters adjustment step at each iteration of model training. Too big can cause overpassing 
  optimal parameter value) <br>
  - Alpha (L1 regularization term on weights. Increasing makes model more conservative, default value is 0) <br>
  - Max depth (Max depth determines maximum depth per tree. A deeper tree usually increases performance but also may cause overfitting) <br>
  - Subsample (Subsample determines size of randomly selected samples sets for each tree training) <br>
  - N-estimators (The number of trees in trained ensemble. Value greater than 0, default equals 100) <br>
  
  Thing that differentiates this notebook from the previous is fact that every parameter was tuned separately one by one with validation curve and GridSearchCV. <br> 
  
  <strong>XGBoost tuned results</strong> <br>
  ![alt text](https://github.com/vanquies/HousingPrices/blob/master/Images/05_1.png)
  
  This approach allowed model to improve performance over the base model on test data significantly (previously: RMSE=0.17, R^2=0.864). Undoubtedly this model 
  works best among the algorithms proposed here and is the most powerful in predicting adequate housing prices. So now when it is known which model makes 
  closest price assigns, naturally it's necessary to show real values, differences and errors making by model. Due to logarithmic transformations actual form 
  is unreadable for humans. But first lets look and feature importances.<br>
  
  <strong>XGB feature importances</strong> <br>
  ![alt text](https://github.com/vanquies/HousingPrices/blob/master/Images/05_2.png)
  
  Unlike random forrest two variables dominates in importance for the model. Type u reffers flats and semidetached buildings and that may have significant 
  influence on price. The second most important is South-eastern region. Presumably it is the most desirable part of the city. It is only in third place 
  variable describing properties size aspects. <br>
  
  <strong>Statistical approach to real scale result</strong> <br>
  ![alt text](https://github.com/vanquies/HousingPrices/blob/master/Images/05_3.png)
  
  On average model was off by $122,832 and is about 12.5% of average price in test set. Value deviates from the mean by an average of $126,956. 
  More than 50% of predictions errors were below $82,000. Nonetheless maximal error made by model is huge and amounts to $1.1M. <br> 
  
  ![alt text](https://github.com/vanquies/HousingPrices/blob/master/Images/05_4.png)
  
  Boxplot represents distribution of absolute errors. Values higher than $600,000 are in marginal 1% of observations. <br>
  
  ![alt text](https://github.com/vanquies/HousingPrices/blob/master/Images/05_5.png)
  
  Shape similar to cone is caused by expotential transformation of both true and predicted values. The higher the price is, the wider the spread. 
</details>

  
## Conclusion
  <a name="Conclusion"></a>
  
  According to the results in the previous section, the use of machine learning in predicting prices in the housing market has proven to be a valuable tool. 
  The combination of feature selection and engineering, data preprocessing, model tuning, and cross-validation techniques have allowed to fine-tune models 
  and achieve a high precision of price prediction. For a person who does not have much to do with the real estate market such algorithm can be valuable element 
  in determining the profitability of the purchase and can successfully point out over and under-priced properties which can occur real bargain. Also model 
  could be useful in decision-making support system for companies which offer properties for sale (determining value of property) and companies dealing in real estate. 
  Data-driven approach in this kind of investments allows to reduce unnecessary losses and brings profit maximization.
  


















