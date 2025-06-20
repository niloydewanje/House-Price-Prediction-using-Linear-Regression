# 🏡 California Housing Price Prediction using Linear Regression

This project is a **basic implementation of a linear regression model** using the **California Housing dataset**. It demonstrates how machine learning can be used to predict housing prices based on various features of the dataset. The goal is to understand the relationship between different housing and geographical factors and the median house value in various districts of California.

---

## 📂 Dataset Information

The dataset used is the **California Housing dataset**, available through `scikit-learn`. It contains data collected from the 1990 California census and includes the following features:

* `MedInc`: Median income in block group
* `HouseAge`: Median house age in block group
* `AveRooms`: Average number of rooms per household
* `AveBedrms`: Average number of bedrooms per household
* `Population`: Block group population
* `AveOccup`: Average house occupancy
* `Latitude`: Block group latitude
* `Longitude`: Block group longitude

The **target variable** is:

* `MedHouseVal`: Median house value for California districts (in \$100,000s)

---

## ⚙️ Project Workflow

### 1. **Data Loading and Preprocessing**

* The dataset is loaded using `fetch_california_housing()` from `scikit-learn`.
* It is stored in a pandas DataFrame for easier manipulation and visualization.
* The target column (`MedHouseVal`) is separated from the features.

### 2. **Train-Test Split**

* The dataset is split into training and testing sets using `train_test_split()`.
* 80% of the data is used for training, and 20% is reserved for testing.
* A fixed `random_state` ensures reproducibility.

### 3. **Model Training**

* A **Linear Regression model** is instantiated from `sklearn.linear_model`.
* The model is trained on the training data (`X_train`, `y_train`) using `.fit()`.

### 4. **Prediction and Evaluation**

* Predictions are made on the test set (`X_test`) using `.predict()`.
* Model performance is evaluated using:

  * **Mean Squared Error (MSE)**: Measures average squared difference between actual and predicted values.
  * **R² Score**: Indicates how well the model explains variance in the target variable.

### 5. **Visualization**

* A scatter plot compares actual vs. predicted median house values.
* Helps visualize how close the model’s predictions are to the real values.

---

## 📊 Model Performance

After training, the model achieved the following results:

* **Mean Squared Error (MSE)**: *\[e.g. 0.5243]*
* **R² Score**: *\[e.g. 0.6063]*

> *Note: These values will vary slightly depending on the machine and environment.*

---

## 📌 Requirements

Make sure the following libraries are installed:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/california-housing-lr.git
   cd california-housing-lr
   ```

2. Run the script:

   ```bash
   python california_housing_lr.py
   ```

---

## 📁 File Structure

```
├── california_housing_lr.py   # Main script file
├── README.md                  # Project documentation
```

---

## 🤖 Future Improvements

* Try more advanced models like Random Forest or Gradient Boosting for improved accuracy.
* Use feature scaling or normalization techniques.
* Add cross-validation to prevent overfitting.
* Visualize feature importance.

---

## 🧠 Learnings

This project demonstrates:

* How to load and process real-world datasets.
* The use of linear regression for regression problems.
* How to evaluate regression models using MSE and R² score.
* The importance of visualizing model performance.

---

Let me know if you want this formatted as a Markdown file or want it shortened into a compact summary.
