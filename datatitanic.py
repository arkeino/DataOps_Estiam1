import pandas as pd
import json

def request_data(url):
    data = pd.read_csv(url)
    return data

def extract_model(data):
    passenger_model = data[["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]].to_dict(orient="records")
    return passenger_model

def transform(data):
    data = data.dropna()
    data.loc[:, 'Age'] = data['Age'].astype(int)
    return data

def load(data, filename):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

if __name__ == "__main__":
    data = request_data('titanic.csv')
    passenger_model = extract_model(data)
    cleaned_data = transform(data)
    load(passenger_model, 'titanic_model.json')

    women_under_18_survived = cleaned_data[(cleaned_data["Sex"] == "female") & (cleaned_data["Age"] < 18) & (cleaned_data["Survived"] == 1)]
    total_women_under_18_survived = len(women_under_18_survived)

    survived_women_under_18_by_class = women_under_18_survived["Pclass"].value_counts().sort_index()

    survival_by_embarked_port = cleaned_data.groupby("Embarked")["Survived"].value_counts().unstack().fillna(0)

    age_gender_distribution = cleaned_data.groupby(["Sex", pd.cut(cleaned_data["Age"], bins=[0, 18, 30, 50, 100])]).size().unstack().fillna(0)

    with open('titanic_report.txt', 'w') as report_file:
        report_file.write("Analyse des données du Titanic\n\n")
        report_file.write("1. Nombre de femmes de moins de 18 ans ayant survécu : {}\n".format(total_women_under_18_survived))
        report_file.write("2. Répartition par classe des femmes de moins de 18 ans qui ont survécu :\n{}\n".format(survived_women_under_18_by_class))
        report_file.write("3. Répartition des morts et des survivants en fonction du port de départ :\n{}\n".format(survival_by_embarked_port))
        report_file.write("4. Répartition par sexe et par âge des passagers du navire :\n{}\n".format(age_gender_distribution))

    print("Pipeline DataOps terminé avec succès ! Les données ont été nettoyées, les analyses ont été effectuées, et le rapport a été généré sous le nom 'titanic_report.txt'.")
