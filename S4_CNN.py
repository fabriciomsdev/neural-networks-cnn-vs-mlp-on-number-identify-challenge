# -*- coding: utf-8 -*-

#==============================================================================
# INTELIGÊNCIA ARTIFICIAL APLICADA
# REDES NEURAIS - SEMANA 4
# REDE NEURAL CONVOLUCIONAL (CNN)
# ALUNO: Fabricio Magalhães
#==============================================================================
#------------------------------------------------------------------------------
# IMPORTAÇÃO DE BIBLIOTECAS
from dataclasses import asdict, dataclass
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from PIL import Image
from itertools import product
from enum import Enum
import pandas as pd


SHOULD_TEST_ALL_MODELS = os.environ.get('TEST_ALL_MODELS', False)

class OptmizatorType(Enum):
    adam='adam'
    sgd='sgd'
    rmsprop='rmsprop'
    adadelta='adadelta'
    adagrad='adagrad'
    adamax='adamax'
    nadam='nadam'

class LosesVerifyFunctionTypesForCNN(Enum):
    categorical_crossentropy = 'categorical_crossentropy'
    binary_crossentropy = 'binary_crossentropy'
    mean_squared_error = 'mean_squared_error'
    mean_absolute_error = 'mean_absolute_error'
    categorical_hinge = 'categorical_hinge'
    logcosh = 'logcosh'

class ActivationFunctionTypesForCNN(Enum):
    relu = 'relu'
    sigmoid = 'sigmoid'
    tanh = 'tanh'
    softmax = 'softmax'

class CNNMetricsType(Enum):
    accuracy = 'accuracy'
    mean_squared_error = 'mean_squared_error'
    mean_absolute_error = 'mean_absolute_error'

@dataclass
class CNNPerformanceTestParams:
    numbers_of_neurons_on_first_convolutional_layer: int = 9
    number_of_neurons_on_second_convolutional_layer: int = 9
    number_of_neurons_on_hidden_layer: int = 64
    number_of_neurons_on_exit_layer: int = 10
    activation_function_of_first_convolutional_layer: ActivationFunctionTypesForCNN = ActivationFunctionTypesForCNN.sigmoid.value
    activation_function_of_second_convolutional_layer: ActivationFunctionTypesForCNN = ActivationFunctionTypesForCNN.tanh.value
    activation_function_of_first_hidden_layer: ActivationFunctionTypesForCNN = ActivationFunctionTypesForCNN.tanh.value
    activation_function_of_exit_layer: ActivationFunctionTypesForCNN = ActivationFunctionTypesForCNN.softmax.value
    lose_verify_function: LosesVerifyFunctionTypesForCNN = LosesVerifyFunctionTypesForCNN.categorical_crossentropy.value
    optimizator_type: OptmizatorType = OptmizatorType.adam.value
    metric: CNNMetricsType = CNNMetricsType.accuracy.value
    training_quota: float = 0.90
    epochs: int = 26
    amostras: int = 2048
    use_hard_test_data_to_validate_model: bool = True
    images_folder_path: str = './RN/'
    show_results: bool = False

@dataclass
class ProccessResult:
    result_metric: int = 0
    result_loss: int = 0

def add_none_on_list(list_to_add, qty = 7):
    if (len(list_to_add) < qty):
        for i in range(qty - len(list_to_add)):
            list_to_add.append(None)

    return list_to_add


posible_activation_function_of_first_convolutional_layer = add_none_on_list([
    ActivationFunctionTypesForCNN.relu.value,
    ActivationFunctionTypesForCNN.tanh.value,
])

posible_activation_function_of_second_convolutional_layer = add_none_on_list([
    ActivationFunctionTypesForCNN.sigmoid.value,
    ActivationFunctionTypesForCNN.tanh.value,
])

posible_activation_function_of_first_hidden_layer = add_none_on_list([
    ActivationFunctionTypesForCNN.relu.value,
    ActivationFunctionTypesForCNN.sigmoid.value,
    ActivationFunctionTypesForCNN.tanh.value,
])
posible_activation_function_of_exit_layer = add_none_on_list([
    ActivationFunctionTypesForCNN.softmax.value,
])
posible_lose_verify_function = add_none_on_list([
    LosesVerifyFunctionTypesForCNN.categorical_crossentropy.value,
    LosesVerifyFunctionTypesForCNN.categorical_hinge.value,
])
posible_optimizator_type = add_none_on_list([
    OptmizatorType.adam.value,
    OptmizatorType.rmsprop.value,
    OptmizatorType.sgd.value,
    OptmizatorType.nadam.value,
])
posible_metric = add_none_on_list([
    CNNMetricsType.accuracy.value,
])


possible_combinations_dicts = {
    'posible_activation_function_of_first_convolutional_layer': posible_activation_function_of_first_convolutional_layer,
    'posible_activation_function_of_second_convolutional_layer': posible_activation_function_of_second_convolutional_layer,
    'posible_activation_function_of_first_hidden_layer': posible_activation_function_of_first_hidden_layer,
    'posible_activation_function_of_exit_layer': posible_activation_function_of_exit_layer,
    'posible_lose_verify_function': posible_lose_verify_function,
    'posible_optimizator_type': posible_optimizator_type,
    'posible_metric': posible_metric,
}

possible_combinations_dfs = pd.DataFrame(possible_combinations_dicts)

msgd_result = np.meshgrid(
    [   
        possible_combinations_dfs[attr]
        for attr in list(possible_combinations_dicts.keys())
    ]
)

uniques = [possible_combinations_dfs[i].unique().tolist() for i in possible_combinations_dfs.columns ]
combinations =  pd.DataFrame(product(*uniques), columns = possible_combinations_dfs.columns)

combinations = combinations.dropna()

possible_combinations = combinations.to_dict('records')

print('Total de combinações possíveis: ' + str(len(possible_combinations)))

params = CNNPerformanceTestParams()


class CNNModel:
    def __init__(self, params):
        self.params = params

    def run(self, params: CNNPerformanceTestParams = None):
        if params is None:
            params = self.params

        #------------------------------------------------------------------------------
        # DEFINIÇÃO DO PERCENTUAL DE DADOS PARA TREINAMENTO
        percentual_treinamento = params.training_quota # 0.7 significa 70% para treinamento e 30% para teste

        #------------------------------------------------------------------------------
        # Carregar dados MNIST
        (x_train_full, y_train_full), (x_test_full, y_test_full) = mnist.load_data()

        #------------------------------------------------------------------------------
        # Determinar o tamanho do conjunto de treinamento com base na proporção especificada
        train_size = int(len(x_train_full) * percentual_treinamento)

        #------------------------------------------------------------------------------
        # Dividir os dados em conjuntos de treinamento e teste
        x_train = x_train_full[:train_size]
        y_train = y_train_full[:train_size]
        x_test = x_train_full[train_size:]
        y_test = y_train_full[train_size:]

        #------------------------------------------------------------------------------
        # Pré-processamento dos dados
        x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)

        #------------------------------------------------------------------------------
        # DEFINIÇÃO DAS FUNÇÕES DE ATIVAÇÃO
            # relu: Rectified Linear Unit, retorna valor positivo e zero caso contrário.
            # sigmoid: Função logística, retorna valores entre 0 e 1.
            # tanh: Tangente hiperbólica, que retorna valores entre -1 e 1.
            # softmax: Normaliza as saídas em uma distribuição de probabilidade.

        activation_function_of_first_convolutional_layer = params.activation_function_of_first_convolutional_layer # função de ativação da primeira camada de convolução
        activation_function_of_second_convolutional_layer = params.activation_function_of_second_convolutional_layer # função de ativação da segunda camada de convolução
        activation_function_of_first_hidden_layer = params.activation_function_of_first_hidden_layer # função de ativação da camada densa oculta
        activation_function_of_exit_layer = params.activation_function_of_exit_layer # função de ativação da camada densa de saída

        #------------------------------------------------------------------------------
        # DEFINIÇÃO DA ESTRUTURA DA REDE

        # DEFINA O TAMANHO MxN DAS MATRIZES DE CONVOLUÇÃO
        m_C1 = 9 # 9 significa uma matriz 9x9 para a primeira camada de convolução
        m_C2 = 9 # 9 significa uma matriz 9x9 para a segunda camada de convolução

        # DEFINA O NÚMERO DE NEURÔNIOS DAS CAMADAS DA REDE CNN
        number_of_neurons_on_first_convolutional_layer = params.numbers_of_neurons_on_first_convolutional_layer # número de neurônios da primeira camada de convolução
        number_of_neurons_on_second_convulutional_layer = params.number_of_neurons_on_second_convolutional_layer # número de neurônios da segunda camada de convolução
        number_of_neurons_on_hidden_layer = params.number_of_neurons_on_hidden_layer # número de neurônios da camada densa oculta

        #------------------------------------------------------------------------------
        # CONSTRUÇÃO DO MODELO CNN
        model = Sequential()

        model.add(Conv2D(number_of_neurons_on_first_convolutional_layer, (m_C1, m_C1), activation = activation_function_of_first_convolutional_layer, input_shape=(28, 28, 1)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(number_of_neurons_on_second_convulutional_layer, (m_C2, m_C2), activation = activation_function_of_second_convolutional_layer))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(number_of_neurons_on_hidden_layer, activation = activation_function_of_first_hidden_layer))
        model.add(Dense(10, activation = activation_function_of_exit_layer))

        #------------------------------------------------------------------------------
        # DEFINIÇÃO DO OTIMIZADOR
        # adam
        # sgd
        # rmsprop
        # adadelta
        # adagrad
        # adamax
        # nadam

        otimizador = params.optimizator_type

        #------------------------------------------------------------------------------
        # DEFINIÇÃO DA FUNÇÃO DE PERDA
        # categorical_crossentropy
        # binary_crossentropy
        # mean_squared_error
        # mean_absolute_error
        # categorical_hinge
        # logcosh

        funcao_perda = params.lose_verify_function

        #------------------------------------------------------------------------------
        # DEFINIÇÃO DA MÉTRICA DE DESEMPENHO
        # accuracy
        # mean_squared_error
        # mean_absolute_error

        metrica = params.metric

        #------------------------------------------------------------------------------
        # Compilar o modelo
        model.compile(optimizer = otimizador,
                    loss = funcao_perda,
                    metrics = [metrica])

        #------------------------------------------------------------------------------
        # DEFINIÇÃO DO NÚMERO DE ÉPOCAS E AMOSTRAS DE TREINAMENTO
        epocas = params.epochs
        amostras = params.amostras

        #------------------------------------------------------------------------------
        # Treinar o modelo
        history = model.fit(x_train, y_train, \
                            epochs = epocas, \
                            batch_size = amostras, \
                            validation_data = (x_test, y_test))

        #------------------------------------------------------------------------------
        # Exibir resultados na aba "Plots"
        plt.plot(history.history[metrica], label='Métrica de Treinamento')
        plt.plot(history.history['val_' + metrica], label='Métrica de Validação')
        plt.plot(history.history['loss'], label='Função de Perda de Treinamento')
        plt.plot(history.history['val_loss'], label='Função de Perda de Validação')
        plt.xlabel('Épocas')
        plt.ylabel('Métrica / Função de Perda')
        plt.legend()

        if params.show_results:
            plt.show()

        #==============================================================================
        # Carregar e pré-processar as imagens de teste

        if params.show_results:
            if params.use_hard_test_data_to_validate_model:
                archive_suffix = 'b.png'
            else:
                archive_suffix = '.png'

            imagem0 = 'zero0' + archive_suffix
            imagem1 = 'um1' + archive_suffix
            imagem2 = 'dois2' + archive_suffix
            imagem3 = 'tres3' + archive_suffix
            imagem4 = 'quatro4' + archive_suffix
            imagem5 = 'cinco5' + archive_suffix
            imagem6 = 'seis6' + archive_suffix
            imagem7 = 'sete7' + archive_suffix
            imagem8 = 'oito8' + archive_suffix
            imagem9 = 'nove9' + archive_suffix

            image_paths = {
                params.images_folder_path + imagem0 :0,
                params.images_folder_path + imagem1 :1,
                params.images_folder_path + imagem2 :2,
                params.images_folder_path + imagem3 :3,
                params.images_folder_path + imagem4 :4,
                params.images_folder_path + imagem5 :5,
                params.images_folder_path + imagem6 :6,
                params.images_folder_path + imagem7 :7,
                params.images_folder_path + imagem8 :8,
                params.images_folder_path + imagem9 :9,
            }

            class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

            fig, axs = plt.subplots(2, 5, figsize=(8, 5))
            axs = axs.flatten()

            for i, (image_path, real_label) in enumerate(image_paths.items()):
                # Carregar a imagem
                image = Image.open(image_path).convert('L')  # Converter para escala de cinza
                image = image.resize((28, 28))  # Redimensionar para 28x28 pixels
                image = np.array(image)  # Converter para matriz numpy
                image = image.reshape(1, 28, 28, 1)  # Adicionar dimensão de lote e canal

                # Normalizar a imagem
                image = image.astype('float32')
                image /= 255.0

                # Fazer a previsão da classe
                prediction = model.predict(image)
                predicted_class = np.argmax(prediction)

                # Exibir a imagem e as informações sobre o número real e previsto
                axs[i].imshow(np.squeeze(image), cmap='gray')
                axs[i].axis('off')
                axs[i].set_title(f'Real: {class_names[real_label]}\nPrevisto: {class_names[predicted_class]}')

            plt.tight_layout()
            plt.show()

        #==============================================================================
        # AVALIACAO DO MODELO (PERDA E METRICA DE DESEMPENHO)

        print('\n\n\n' + '=' * 70)
        print('CALCULANDO FUNCAO DE PERDA E METRICA DE DESEMPENHO...\n')
        loss, metric = model.evaluate(x_test, y_test)

        print('\n' + '=' * 70)
        print('*** DESEMPENHO DO MODELO APOS O TREINAMENTO ***\n')


        print("Funcao de Perda utilizada: " + funcao_perda)
        print("Valor obtido: " + f" = {loss:.4f}" + '\n')

        print('-' * 70 + '\n')

        print("Metrica de Desempenho utilizada: " + metrica)
        print("Valor obtido: " + f" = {metric:.4f}\n")

        print('=' * 70)

        #==============================================================================

        return ProccessResult(
            result_metric=metric,
            result_loss=loss
        )
    

if SHOULD_TEST_ALL_MODELS:
    results = []

    for conbination in possible_combinations:
        current_params = CNNPerformanceTestParams(
            activation_function_of_first_convolutional_layer=conbination.get('posible_activation_function_of_first_convolutional_layer'),
            activation_function_of_second_convolutional_layer=conbination.get('posible_activation_function_of_second_convolutional_layer'),
            activation_function_of_first_hidden_layer=conbination.get('posible_activation_function_of_first_hidden_layer'),
            activation_function_of_exit_layer=conbination.get('posible_activation_function_of_exit_layer'),
            lose_verify_function=conbination.get('posible_lose_verify_function'),
            optimizator_type=conbination.get('posible_optimizator_type'),
            metric=conbination.get('posible_metric'),
        )

        result = CNNModel(current_params).run()

        data = asdict(current_params)
        data.update(asdict(result))

        results.append(data)

    results_df = pd.DataFrame.from_dict(results)
    results_df = results_df.sort_values('result_metric', ascending=False)

    current_datetime_as_int = int(pd.Timestamp.now().timestamp())
    archive_name = './results/result-cnn-' + str(current_datetime_as_int) + '.csv'
    results_df.to_csv(archive_name)

second_winner_model = CNNPerformanceTestParams(
    numbers_of_neurons_on_first_convolutional_layer=9,
    number_of_neurons_on_second_convolutional_layer=9,
    number_of_neurons_on_hidden_layer=64,
    number_of_neurons_on_exit_layer=10,
    activation_function_of_first_convolutional_layer='relu',
    activation_function_of_second_convolutional_layer='tanh',
    activation_function_of_first_hidden_layer='tanh',
    activation_function_of_exit_layer='softmax',
    lose_verify_function='categorical_crossentropy',
    optimizator_type='adam',
    metric='accuracy',
    training_quota=0.9,
    epochs=26,
    use_hard_test_data_to_validate_model=True,
    show_results=True
)

model = CNNModel(second_winner_model).run()

second_winner_model = CNNPerformanceTestParams(
    numbers_of_neurons_on_first_convolutional_layer=9,
    number_of_neurons_on_second_convolutional_layer=9,
    number_of_neurons_on_hidden_layer=64,
    number_of_neurons_on_exit_layer=10,
    activation_function_of_first_convolutional_layer='relu',
    activation_function_of_second_convolutional_layer='tanh',
    activation_function_of_first_hidden_layer='tanh',
    activation_function_of_exit_layer='softmax',
    lose_verify_function='categorical_crossentropy',
    optimizator_type='nadam',
    metric='accuracy',
    training_quota=0.9,
    epochs=26,
    use_hard_test_data_to_validate_model=True,
    show_results=True
)

model = CNNModel(second_winner_model).run()

