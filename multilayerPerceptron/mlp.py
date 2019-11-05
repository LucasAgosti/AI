import numpy as np

inputs = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])
outputs = np.array([[0], [1], [1], [0]])

#Pesos das entradas até a camada oculta
weights0 = np.array([[-0.424, -0.740, -0.961],
                     [0.358, -0.577, -0.469]])

#Pesos da camada oculta até a camada de saída
weightsOutput = np.array([[-0.017], [-0.893], [0.148]])
epochs = 1


def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))


for i in range(epochs):
    inputLayer = inputs
    # Realiza o produto escalar entre a camada de entrada e a camada oculta
    somaHiddenLayer = np.dot(inputLayer, weights0)
    #print('Valor da função soma da camada oculta:\n', somaHiddenLayer)
    ativacaoHiddenLayer = sigmoid(somaHiddenLayer)
    #print('\nValor dos pesos da camada oculta:\n', ativacaoHiddenLayer)

    somaOutputLayer = np.dot(ativacaoHiddenLayer, weightsOutput)
    ativacaoOutputLayer = sigmoid(somaOutputLayer)
    print(ativacaoOutputLayer)

    erroOutputLayer = outputs - ativacaoOutputLayer
    mediaAbsoluta = np.mean(abs(erroOutputLayer))
    print('Erro na epoca ' + str(epochs) + ': ', mediaAbsoluta)
