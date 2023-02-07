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
4. [ Project Summary ](#Project_Summary)
   * [ 1. Exploratiory Data Analysis ](#EDA)
       * [ Data understanding ](#Data_Understanding)
       * [ Preliminary cleaning ](#Preliminary_Cleaning)
       * [ Further data exploration ](#Further_Data_Exploration)
       * [ Data visualization ](#Data_Visualization)
   * [ 2. Data Preprocessing and Basic Models ](#DPandBM) 
   * [ 3. Random Forest Regression ](#RFR)
   * [ 4. Support Vector Regression ](#SVR)
       * [  ](#)
   * [ 5. XGBoost Regression ](#XGB)

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

Method: S - property sold; SP - property sold prior; PI - property passed in; PN - sold prior not disclosed; SN - sold not disclosed; NB - no bid; VB - vendor bid; W - withdrawn prior to auction; SA - sold after auction; SS - sold after auction price not disclosed. N/A - price or highest bid not available.

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

In first contact with data it is vital to understand data in general. For this purpose after reading csv file there are used methods like shape (shape of data frame), 
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

![alt text](https://github.com/vanquies/HousingPrices/blob/master/Images/01_2.png)

Next picture represents timeline and number of sold properties in partitioned periods. Record sales occurred in August 2017. 
The period examined is too short to unequivocally state whether there is a phenomenon of yearly seasonality in the demand on 
the real estate market. Nonetheless histogram shows significant drop in winter season 2017 what may herald such phenomenon. <br>

![alt text](https://github.com/vanquies/HousingPrices/blob/master/Images/01_3.png)

Here we can see scatterplot of properties co-ordinates forming regions separated with hue color. Five regions are metropolitan and three (victoria) are suburbs.
Most valueable housings on average are place in southern and eastern metropolitan region. <br>

![alt text](https://github.com/vanquies/HousingPrices/blob/master/Images/01_4.png)

This scatterplot illustrates dependence between distance from city centre and price. The vast majority of properties sold with price exceeding $2mln are closer 
to C.B.D. than 20 miles. Building type u (flat, semi-detatched house) on average cost much lower than h (houses, villas) and t (townhouses). <br> 
</details>

<a name="DPandBM"></a>
### Data Preprocessing and Basic Models
<details open>
<summary>Show/Hide</summary>
<br>

</details>

<a name="RFR"></a>
### Random Forest Regression
<details open>
<summary>Show/Hide</summary>
<br>

</details>

</details>

<a name="SVR"></a>
### Support Vector Regression
<details open>
<summary>Show/Hide</summary>
<br>

</details>

</details>

<a name="XGB"></a>
### XGBoost Regression
<details open>
<summary>Show/Hide</summary>
<br>

</details>





















