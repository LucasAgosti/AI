import numpy as np
import matplotlib.pyplot as plt

CONST_LEARNINGRATE = 0.05
treinamento = np.array([[1, 0, 0],
               [1, 0, 1],
               [1, 1, 0],
               [1, 1, 1]])

saidaDesejada = np.array([0, 0, 0, 1])

bias = 1
pesos = np.array([bias,0.2,0.1])
#pesos = np.random.uniform((1, 2), (-1, 1))
#print(len(treinamento))

teste = 1

#print(pesos)

def train(flag,bias):
    while flag == 1:
        countAcerto = 0
        for x in range(4):
            np.insert(treinamento[x],0,bias)
            soma = np.dot(treinamento[x], pesos)

            #print(soma)
            saidaAtual = ativacao(soma)
            e = (saidaDesejada[x] - saidaAtual)

            if(e == 0):
                countAcerto += 1
                #print("Resultado Condizente")
                #print(count)

            else:
                #print("Deu bode")

                #print(pesos[0], saidaDesejada[x], saidaAtual, treinamento[x][0] , bias)
                bias += (CONST_LEARNINGRATE * e)
                pesos[0] += (CONST_LEARNINGRATE * e * treinamento[x][0]) + bias
                pesos[1] += (CONST_LEARNINGRATE * e * treinamento[x][1]) + bias
                #print(pesos[1], saidaDesejada[x], saidaAtual, treinamento[x][1] , bias)

            if countAcerto == 4:
                flag = 0
        #print("\n")

def ativacao(soma):

    if soma >= 0:
        return 1
    else:
        return 0


train(teste,bias)

print("Digite as duas entradas da porta AND")

x1 = float(input())
x2 = float(input())

A = [1, x1, x2]
print("O resultado e =", ativacao(np.dot(A, pesos)))



plt.scatter(treinamento[0: ,1,],treinamento[0: ,2],c= saidaDesejada)
plt.title("X x Y")
plt.xlabel('x')
plt.ylabel('y')
plt.plot(bias)
plt.show()
