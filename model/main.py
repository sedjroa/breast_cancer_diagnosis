import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle as pk

def data_cleaner():
    data = pd.read_csv("data/data.csv", sep=",")
    data.drop(["id","Unnamed: 32"], inplace=True, axis=1)

    return data

def create_model(data):

    # Data transform
    X = data.drop("diagnosis", axis=1)
    y = data["diagnosis"]
    y = y.map({"M":1, "B":0})

    # Data scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Data split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model creation
    lr = LogisticRegression()
    model = lr.fit(X_train, y_train)

    # Model Evaluation
    y_pred = model.predict(X_test)
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")

    return model, scaler
    
    
def main():
    data = data_cleaner()
    model, scaler = create_model(data)
    

if __name__ == '__main__':
    main()