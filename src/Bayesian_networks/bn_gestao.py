from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import json

# Distribuições marginais dos indicadores
complexidade_dist = [0.0, 0.0, 0.5909, 0.4091, 0.0]
tamanhoEquipe_dist = [0.5, 0.3182, 0.0909, 0.0455, 0.0455]

# P(Risco | indicador)
p_risco_dado_complexidade = [0.0, 0.0, 0.307697, 0.777761, 0.0]
p_risco_dado_tamanhoEquipe = [0.454545, 0.428547, 0.50005, 0.999, 0.999]

# Criação da estrutura da rede
model = DiscreteBayesianNetwork([
    ("complexidade", "riscoGestao"),
    ("tamanhoEquipe", "riscoGestao")
])

# CPDs dos indicadores
cpd_complexidade = TabularCPD("complexidade", 5, [[p] for p in complexidade_dist])
cpd_tamanhoEquipe = TabularCPD("tamanhoEquipe", 5, [[p] for p in tamanhoEquipe_dist])

# CPD do risco (binário: 0 = não ocorre, 1 = ocorre)
cpd_risco_values = []
for i in range(5):
    for j in range(5):
        prob = (p_risco_dado_complexidade[i] + p_risco_dado_tamanhoEquipe[j]) / 2
        c0 = round(1 - prob, 6)
        c1 = round(prob, 6)
        cpd_risco_values.append([c0, c1])

cpd_risco_matrix = [
    [pair[0] for pair in cpd_risco_values],
    [pair[1] for pair in cpd_risco_values]
]

cpd_risco = TabularCPD(
    variable="riscoGestao",
    variable_card=2,
    values=cpd_risco_matrix,
    evidence=["complexidade", "tamanhoEquipe"],
    evidence_card=[5, 5]
)

model.add_cpds(cpd_complexidade, cpd_tamanhoEquipe, cpd_risco)

assert model.check_model()

# Salva estrutura e CPDs em JSON
rede_info = {
    "edges": list(model.edges()),
    "cpds": {
        "complexidade": complexidade_dist,
        "tamanhoEquipe": tamanhoEquipe_dist,
        "riscoGestao (0=Não, 1=Sim)": cpd_risco_matrix
    }
}

with open("results/json/rede_gestao.json", "w", encoding="utf-8") as f:
    json.dump(rede_info, f, indent=4, ensure_ascii=False)

print("✅ Rede bayesiana salva como 'rede_gestao.json'")
