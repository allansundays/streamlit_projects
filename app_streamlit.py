# Importando bibliotecas
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Titulo
st.write("""
Prevendo diabetes \n
App que utiliza Machine Learning para prever possível diabetes dos pacientes\n
fonte: PIMA - INDIA
""")

# dataset
df = pd.read_csv("C:/Users/allan/repositorios/streamlit/diabetes/diabetes.csv")

# Cabeçalho
st.subheader('Informações dos dados')

# nome do usuário
user_input = st.sidebar.text_input("Digite seu nome:")

st.write("Paciente: ", user_input)

# dados de entrada
X = df.drop(['Outcome'], axis=1)
Y = df["Outcome"]

# Separa dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=42)

# dados dos usuários com a função


def get_user_data():
    pregnancies = st.sidebar.slider("Gravidez", 0, 15, 1)
    glucouse = st.sidebar.slider("Glicose", 0, 200, 110)
    blod_pressure = st.sidebar.slider("Pressao Sanqguinia", 0, 122, 72)
    skin_tchicknee = st.sidebar.slider("Espessura da pele", 0, 99, 20)
    insulin = st.sidebar.slider("Insulina", 0, 900, 30)
    bmi = st.sidebar.slider("Índice de massa corporal", 0.0, 70.0, 15.0)
    dpf = st.sidebar.slider("Histórico familiar de diabetes", 0.0, 3.0, 0.0)
    age = st.sidebar.slider("Idade", 15, 100, 21)

    user_data = {"Gravidez": pregnancies,
                 "Glicose": glucouse,
                 "Pressao Sanguinia": blod_pressure,
                 "Espessura da pele": skin_tchicknee,
                 "Insulina": insulin,
                 "Índice de massa corporal": bmi,
                 "Histórico familiar de diabetes": dpf,
                 "Idade": age
                 }
    features = pd.DataFrame(user_data, index=[0])

    return features


user_input_variables = get_user_data()

# Grafico
graf = st.bar_chart(user_input_variables)

st.subheader("Dados do usuário")
st.write(user_input_variables)
# Modelo
dtc = DecisionTreeClassifier(criterion='entropy', max_depth=3)
dtc.fit(X_train, y_train)

# Acurácia do modelo
st.subheader("Acurácia do modelo")
st.write(accuracy_score(y_test, dtc.predict(X_test))*100)

# Previsao
prediction = dtc.predict(user_input_variables)

st.subheader('Previsão: ')
st.write(prediction)
