import json

# P(R)
P_R = {
    "Gestão": 11 / 22,
    "Cultural": 2 / 22,
    "Econômico": 9 / 22,
    "Requisitos": 11 / 22,
    "Aplicação": 12 / 22
}

# P(I) dos especialistas
P_I = {
    "Gestão": {
        "1.2": [0.0, 0.0, 0.5909, 0.4091, 0.0],
        "1.4": [0.5, 0.3182, 0.0909, 0.0455, 0.0455]
    },
    "Cultural": {
        "1.6": [0.4091, 0.0909, 0.0455, 0.3182, 0.1364],
        "3.4": [0.3636, 0.3182, 0.2273, 0.0909, 0.0]
    },
    "Econômico": {
        "2.6": [0.4545, 0.3182, 0.0909, 0.0455, 0.0909],
        "2.11": [0.0, 0.1364, 0.2727, 0.4545, 0.1364]
    },
    "Requisitos": {
        "2.2": [0.0455, 0.1364, 0.4545, 0.3182, 0.0455],
        "2.3": [0.0, 0.0455, 0.6364, 0.2727, 0.0455]
    },
    "Aplicação": {
        "2.9": [0.0, 0.0909, 0.2273, 0.4545, 0.2273],
        "2.10": [0.0, 0.0909, 0.3182, 0.4545, 0.1364]
    }
}

# P(I | R)
P_I_dado_R = {
    "Gestão": {
        "1.2": [0.0, 0.0, 0.363636, 0.636364, 0.0],
        "1.4": [0.454545, 0.272727, 0.090909, 0.090909, 0.090909]
    },
    "Cultural": {
        "1.6": [0.5, 0.0, 0.0, 0.0, 0.5],
        "3.4": [0.0, 1.0, 0.0, 0.0, 0.0]
    },
    "Econômico": {
        "2.6": [0.333333, 0.444444, 0.111111, 0.0, 0.111111],
        "2.11": [0.0, 0.111111, 0.444444, 0.333333, 0.111111]
    },
    "Requisitos": {
        "2.2": [0.090909, 0.181818, 0.454545, 0.181818, 0.090909],
        "2.3": [0.0, 0.0, 0.636364, 0.363636, 0.0]
    },
    "Aplicação": {
        "2.9": [0.0, 0.083333, 0.333333, 0.333333, 0.25],
        "2.10": [0.0, 0.0, 0.333333, 0.5, 0.166667]
    }
}

# Resultado final
prob_R_given_I = {}

for grupo in P_I:
    prob_R_given_I[grupo] = {}
    p_r = P_R[grupo]

    for pergunta, p_i_list in P_I[grupo].items():
        prob_list = []
        for i in range(5):
            p_i = p_i_list[i]
            p_i_given_r = P_I_dado_R[grupo][pergunta][i]

            if p_i > 0:
                p_r_given_i = (p_i_given_r * p_r) / p_i
            else:
                p_r_given_i = 0.0  # evitar divisão por zero

            prob_list.append(round(p_r_given_i, 6))

        prob_R_given_I[grupo][pergunta] = prob_list

# Salvar JSON
with open("P_R_given_I_final.json", "w", encoding="utf-8") as f:
    json.dump(prob_R_given_I, f, indent=4, ensure_ascii=False)

print("✅ JSON salvo como 'P_R_given_I_final.json'")
