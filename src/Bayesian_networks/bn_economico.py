from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import json

# Distribuições marginais dos indicadores
dependencia_dist = [0.4545, 0.3182, 0.0909, 0.0455, 0.0909]
disponibilidade_dist = [0.0, 0.1364, 0.2727, 0.4545, 0.1364]

# P(Risco | indicador)
p_risco_dado_dependencia = [0.30003, 0.571395, 0.50005, 0.0, 0.50005]
p_risco_dado_disponibilidade = [0.0, 0.333244, 0.666733, 0.30003, 0.333244]

# Criação da estrutura da rede
model = DiscreteBayesianNetwork([
    ("dependencia", "riscoEconomico"),
    ("disponibilidade", "riscoEconomico")
])

# CPDs dos indicadores
cpd_dependencia = TabularCPD("dependencia", 5, [[p] for p in dependencia_dist])
cpd_disponibilidade = TabularCPD("disponibilidade", 5, [[p] for p in disponibilidade_dist])

# CPD do risco (binário: 0 = não ocorre, 1 = ocorre)
cpd_risco_values = []
for i in range(5):
    for j in range(5):
        prob = (p_risco_dado_dependencia[i] + p_risco_dado_disponibilidade[j]) / 2
        c0 = round(1 - prob, 6)
        c1 = round(prob, 6)
        cpd_risco_values.append([c0, c1])

cpd_risco_matrix = [
    [pair[0] for pair in cpd_risco_values],
    [pair[1] for pair in cpd_risco_values]
]

cpd_risco = TabularCPD(
    variable="riscoEconomico",
    variable_card=2,
    values=cpd_risco_matrix,
    evidence=["dependencia", "disponibilidade"],
    evidence_card=[5, 5]
)

model.add_cpds(cpd_dependencia, cpd_disponibilidade, cpd_risco)
assert model.check_model()

# Salva em JSON
rede_info = {
    "edges": list(model.edges()),
    "cpds": {
        "dependencia": dependencia_dist,
        "disponibilidade": disponibilidade_dist,
        "riscoEconomico (0=Não, 1=Sim)": cpd_risco_matrix
    }
}

with open("results/json/rede_economico.json", "w", encoding="utf-8") as f:
    json.dump(rede_info, f, indent=4, ensure_ascii=False)

print("✅ Rede bayesiana salva como 'rede_economico.json'")
