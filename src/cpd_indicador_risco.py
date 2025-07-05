import pandas as pd
import json

# Caminho para o CSV
caminho_csv = "data/input.csv"

# Nome da coluna de risco múltiplo (4.1)
coluna_risco_multiplo = "4.1 - Dentro das áreas de riscos quais são as áreas mais comuns nos projetos em que você trabalha?"

# Perguntas selecionadas pela rede neural
perguntas_selecionadas = {
    "Gestão": [
        "1.2 - Qual o nível médio de complexidade dos projetos nos quais você trabalha?",
        "1.4 - Qual o tamanho médio da(s) equipe(s) em que você trabalha?"
    ],
    "Cultural": [
        "1.6 - Qual a forma de organização da sua equipe?",
        "3.4 - Qual o nível de conflito interno no time?"
    ],
    "Econômico": [
        "2.6 - Qual o nível de dependência externa (Terceirização)?",
        "2.11 - Qual a disponibilidade dos recursos adequados ao projeto?"
    ],
    "Requisitos": [
        "2.2 - Qual o nível de estabilidade dos requisitos?",
        "2.3 - Qual o nível de complexidade dos requisitos?"
    ],
    "Aplicação": [
        "2.9 - Qual o nível de reuso dos componentes?",
        "2.10 - Qual o nível de usabilidade da interface?"
    ]
}

# Riscos correspondentes de 4.1 por grupo
riscos_por_grupo = {
    "Gestão": ["Riscos de gestão empresarial"],
    "Cultural": ["Riscos culturais"],
    "Econômico": ["Riscos econômicos"],
    "Requisitos": ["Riscos sobre requisitos tecnológicos"],
    "Aplicação": ["Riscos de aplicação tecnológica"]
}

# Carregar dados
df = pd.read_csv(caminho_csv)

# Converter respostas múltiplas de 4.1 em listas
df[coluna_risco_multiplo] = df[coluna_risco_multiplo].astype(str).str.split(",")

# Explodir múltiplos riscos em linhas
df_exploded = df.explode(coluna_risco_multiplo)
df_exploded[coluna_risco_multiplo] = df_exploded[coluna_risco_multiplo].str.strip()

# Resultado: CPDs P(I | R = x)
cpds_P_I_dado_R = {}

for grupo, perguntas in perguntas_selecionadas.items():
    riscos_grupo = riscos_por_grupo[grupo]
    df_grupo = df_exploded[df_exploded[coluna_risco_multiplo].isin(riscos_grupo)]
    cpds_grupo = {}

    for pergunta in perguntas:
        cpd = df_grupo[pergunta].value_counts(normalize=True).sort_index()
        for i in range(1, 6):
            if i not in cpd:
                cpd[i] = 0.0
        cpd = cpd.sort_index()
        cpds_grupo[pergunta] = {str(k): round(v, 6) for k, v in cpd.items()}

    cpds_P_I_dado_R[grupo] = cpds_grupo

# Salvar como JSON
with open("cpds_P_I_dado_R.json", "w", encoding="utf-8") as f:
    json.dump(cpds_P_I_dado_R, f, ensure_ascii=False, indent=4)

print("Arquivo 'cpds_P_I_dado_R.json' salvo com sucesso.")
