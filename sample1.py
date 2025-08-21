from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
model = DiscreteBayesianNetwork([
    ('Burglary', 'Alarm'), ('Earthquake', 'Alarm'),
    ('Alarm', 'JohnCalls'), ('Alarm', 'MaryCalls')
])
cpd_burglary = TabularCPD('Burglary', 2, [[0.999], [0.001]])
cpd_earthquake = TabularCPD('Earthquake', 2, [[0.998], [0.002]])
cpd_alarm = TabularCPD('Alarm', 2,
                       [[0.95, 0.06, 0.71, 0.999],
                        [0.05, 0.94, 0.29, 0.001]],
                       ['Burglary', 'Earthquake'], [2, 2])
cpd_john = TabularCPD('JohnCalls', 2,
                      [[0.95, 0.10], [0.05, 0.90]],
                      ['Alarm'], [2])
cpd_mary = TabularCPD('MaryCalls', 2,
                      [[0.99, 0.30], [0.01, 0.70]],
                      ['Alarm'], [2])
model.add_cpds(cpd_burglary, cpd_earthquake, cpd_alarm, cpd_john, cpd_mary)
assert model.check_model()
infer = VariableElimination(model)
def get_yes_no(prompt):
    while (ans := input(prompt + " (yes/no): ").lower()) not in ('yes', 'no'):
        print("Please answer 'yes' or 'no'")
    return 1 if ans == 'yes' else 0
evidence = {
    'JohnCalls': get_yes_no("Did John call?"),
    'MaryCalls': get_yes_no("Did Mary call?")
}
result = infer.query(['Burglary'], evidence=evidence)
print("\nProbability of Burglary:")
print(result)
