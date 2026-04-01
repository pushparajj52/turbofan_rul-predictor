# Turbofan Engine — Remaining Useful Life (RUL) Predictor

A machine learning project built on the NASA CMAPSS FD001 dataset to predict how many cycles a turbofan engine has left before it fails. The model is deployed as a web application where anyone can enter sensor readings and get a prediction instantly.

---

## Why I Built This

Predictive maintenance is one of the most practical applications of machine learning in the real world. Instead of replacing engine parts on a fixed schedule (which wastes money) or waiting for failure (which is dangerous), a model that predicts *when* something will fail gives engineers time to act. That is what this project does predict the Remaining Useful Life of a turbofan engine using its sensor readings.

---

## Dataset

**NASA CMAPSS FD001** — Turbofan Engine Degradation Simulation

- 100 engines in training set, 100 in test set
- Each engine runs from healthy state until failure
- 21 sensor readings per cycle + 3 operating condition columns
- Single operating condition, single fault mode (FD001)

The dataset comes with 3 files:
```
train_FD001.txt   → sensor readings during full engine life until failure
test_FD001.txt    → sensor readings up to some point before failure
RUL_FD001.txt     → true RUL values for each test engine
```

---

## What I Learned

### Data Understanding
- How to read and load space separated NASA text files into pandas
- What engine_id, cycle, operating conditions and sensor columns mean
- Why each engine has a different number of cycles they all fail at different times

### RUL Label Creation
- RUL is not given in the training data you have to compute it yourself
- Formula: `RUL = max_cycle_of_engine - current_cycle`
- Used `groupby("engine_id")` to find the max cycle per engine separately
- Used `merge` to bring max cycle back into every row so subtraction works
- Applied piecewise linear capping at 125 cycles early healthy cycles all look the same to the model so there is no point in distinguishing RUL=250 from RUL=200

### Sensor Selection
- Not all 21 sensors are useful some are completely constant across all cycles
- Used standard deviation to drop constant sensors (std < 0.01)
- Used Pearson correlation with RUL to find sensors that actually change with degradation
- Used Random Forest feature importance to confirm which sensors matter most
- Used PCA to visually verify that sensors together capture the degradation trajectory
- Final selected sensors: `sensor_11, sensor_9, sensor_4, sensor_12, sensor_7, sensor_14, sensor_15, sensor_21`

### Preprocessing
- Applied MinMaxScaler fitted only on training data
- Used `scaler.fit_transform()` on train and `scaler.transform()` on test never fit on test data to avoid data leakage
- Grouped test data by engine and took the last row of each engine for prediction the most recent reading is what you use to predict remaining life

### Model Training
- Trained XGBoost Regressor with 500 trees, learning rate 0.05, max depth 5
- Used `subsample=0.8` and `colsample_bytree=0.8` to prevent overfitting
- Evaluated on test set using RMSE and R2 score

### Deployment
- Saved model, scaler and feature list using joblib
- Built a Flask API with two routes one to serve the webpage, one to handle predictions
- Built a simple HTML/CSS/JavaScript frontend with input boxes for each sensor
- Flask receives sensor values from browser, scales them, predicts RUL and sends result back

---

## Pipeline I Understood

```
Raw Data (train/test/RUL txt files)
            ↓
Load into pandas DataFrame
            ↓
Compute RUL labels
  max_cycle per engine → merge → subtract → clip at 125
            ↓
Sensor Selection
  drop constant sensors (std filter)
  correlation with RUL
  Random Forest importance
  PCA visualization
            ↓
Preprocessing
  MinMaxScaler fitted on train
  transform test with same scaler
            ↓
Model Training
  XGBoost Regressor
  500 trees, lr=0.05, depth=5
            ↓
Evaluation
  RMSE + R2 Score on test set
  R2 = 0.81
            ↓
Save Model
  model.pkl / scaler.pkl / features.pkl
            ↓
Deployment
  Flask API (app.py)
  HTML Frontend (index.html)
  Run → open browser → enter values → get RUL
```

---

## Key Concepts I Understood

**groupby** — splits data into groups by engine_id so operations run per engine, not on the entire dataset

**merge** — joins two dataframes on a common column, like SQL join. Used to bring max_cycle back into every row

**reset_index** — after groupby the grouped column becomes an index not a column. reset_index makes it a regular column again so merge can use it

**fit vs transform** — fit learns the scaling parameters from training data. transform applies those parameters. Never fit on test data

**RUL capping** — engines in early healthy state all look the same. Capping prevents the model from trying to learn meaningless differences between RUL=250 and RUL=200

**PCA** — compresses many sensor dimensions into 2 so you can plot and visually check if sensors capture degradation. Not used for training, only for sanity checking

**Flask** — Python web framework that acts as a bridge between the browser and the model. Browser cannot talk to Python directly — Flask translates between them

**joblib** — saves and loads trained Python objects like models and scalers so you do not have to retrain every time

---

## Project Structure

```
RUL_Project/
├── app.py                  ← Flask API
├── model.pkl               ← trained XGBoost model
├── scaler.pkl              ← fitted MinMaxScaler
├── features.pkl            ← list of selected sensor names
├── README.md               ← this file
└── templates/
    └── index.html          ← web frontend
```

---

## How to Run

**Install dependencies:**
```bash
pip install flask xgboost scikit-learn joblib numpy pandas
```

**Run the app:**
```bash
python app.py
```

**Open browser:**
```
http://127.0.0.1:5000
```

Enter sensor values and click **Predict RUL** to get the prediction.

---

## Results

| Model | RMSE | R2 Score |
|---|---|---|
| XGBoost | — | 0.81 |

---

## Status Output

| RUL | Status |
|---|---|
| > 60 cycles | HEALTHY ✅ |
| 30 – 60 cycles | WARNING 🔶 |
| < 30 cycles | CRITICAL ⚠️ |

---

## Tech Stack

- Python
- pandas, numpy
- scikit-learn
- XGBoost
- Flask
- HTML / CSS / JavaScript
- joblib

---

*Built from scratch — data loading, label engineering, sensor selection, preprocessing, model training and web deployment.*# turbofan-rul-predictor
