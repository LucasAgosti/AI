import numpy as np

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
saidasDesejadas = np.array([0, 1, 1, 1])
weights = np.array([0.0, 0.0])
constLearningRate = 0.1

def stepFunction(soma):
    if(soma >= 1):
        return 1
    return 0

def calculaSaida(registro):
    somatorio = registro.dot(weights)
    return stepFunction(somatorio)

def train():
    erroTotal = 1

    while(erroTotal != 0):
        erroTotal = 0
        for i in range(len(saidasDesejadas)):
            saidaCalculada = calculaSaida(np.array(inputs[i]))
            erro = saidasDesejadas[i] - saidaCalculada
            erroTotal += erro

            for j in range(len(weights)):
                weights[j] = weights[j] + (constLearningRate * inputs[i][j] * erro)
                print('Peso atualizado ' + str(weights[j]))
        print('\nTotal de erros: ' + str(erroTotal))
    print('Training succefull.')

train()

print(calculaSaida(inputs[0]))
print(calculaSaida(inputs[1]))
print(calculaSaida(inputs[2]))
print(calculaSaida(inputs[3]))
