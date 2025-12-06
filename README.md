# EcoAgent
**Disciplina:** Introdução à Inteligência Artificial<br>
**Semestre:** 2025.2<br>
**Professor:** André Luis Fonseca Faustino<br>
**Turma:** T04

## Integrantes do Grupo
- Ana Luiza Medeiros da Silva (20230048610)
- Carolina Medeiros Coutinho (20220041222)
- Luiz Gustavo de Souza Rego (20220038835)

## Descrição do Projeto
O **EcoAgent** é um sistema de Inteligência Artificial desenvolvido para estimar o consumo energético de um ambiente com base em variáveis ambientais, estruturais e operacionais — como temperatura, umidade, ocupação, iluminação, energia renovável e dia da semana.  

O projeto combina **modelos de Machine Learning** com uma **interface interativa em Streamlit**, permitindo que o usuário insira os dados e visualize instantaneamente a predição e os fatores que influenciaram o resultado.  

As tecnologias utilizadas incluem **Python, Pandas, Numpy, Seaborn, Scikit-Learn, Streamlit, Matplotlib**

## Guia de Instalação e Execução
### 1. Instalação das Dependências
Certifique-se de ter o Python 3.x instalado. Clone o repositório e instale as bibliotecas listadas no requirements.txt:
```
# Clone o repositório
git clone [https://github.com/carolinamcoutinho/EcoAgent.git](https://github.com/carolinamcoutinho/EcoAgent.git)

# Entre na pasta do projeto
cd EcoAgent

# Instale as dependências
pip install -r requirements.txt
```
### 2. Como Executar
Execute o Streamlit com:
```
streamlit run src/app.py
```
Se necessário, especifique a porta ou url de acesso, ex: http://localhost:8501

## Estrutura dos Arquivos
```
EcoAgent/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── energy_dataset.csv          # Dataset original
│
├── models/
│   ├── ecoagent_linear_model.pkl   # Modelo final salvo
│   └── ecoagent_scaler.pkl         # Scaler usado no treinamento
│
├── notebooks/
│   └── EcoAgent_Modeling.ipynb     # Notebook com EDA, testes e modelagem
│
├── src/
│   └── app.py                      # Interface Streamlit
│
└── assets/
```

## Resultados e Demonstração
[Adicione prints da aplicação em execução ou gráficos com os resultados do modelo/agente. Se for uma aplicação Web, coloque um print da interface.]

## Referências
[Link para o Dataset original]
[Artigo, Documentação ou Tutorial utilizado como base]
