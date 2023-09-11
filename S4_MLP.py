# -*- coding: utf-8 -*-

#==============================================================================
# INTELIGENCIA ARTIFICIAL APLICADA
# REDES NEURAIS - SEMANA 4
# REDE NEURAL MLP (MULTI-LAYER PERCEPTRON)
# ALUNO: FABRICIO MAGALHÃES SENA 
#==============================================================================
#------------------------------------------------------------------------------
# IMPORTACAO DE BIBLIOTECAS
from dataclasses import dataclass, asdict
import multiprocessing
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from PIL import Image
from itertools import product
from enum import Enum
import pandas as pd

SHOULD_TEST_ALL_MODELS = os.environ.get('TEST_ALL_MODELS', False)

class OptmizatorType(Enum):
    sgd = 'sgd'
    adam = 'adam'
    rmsprop = 'rmsprop'
    adagrad = 'adagrad'

class LosesVerifyFunctionTypesForMLP(Enum):
    mean_squared_error = 'mean_squared_error'
    binary_crossentropy = 'binary_crossentropy'
    categorical_crossentropy = 'categorical_crossentropy'
    mean_absolute_error = 'mean_absolute_error'

class ActivationFunctionTypesForMLP(Enum):
    relu = 'relu'
    sigmoid = 'sigmoid'
    tanh = 'tanh'
    softmax = 'softmax'

class MLPMetricsType(Enum):
    accuracy = 'accuracy'
    mean_squared_error = 'mean_squared_error'
    mean_absolute_error = 'mean_absolute_error'

possible_classes_length = 10
size_of_images = 784
neurons_on_first_layer = (possible_classes_length + size_of_images)

@dataclass
class MLPPerformanceTestParams:
    numbers_of_neurons_on_first_layer: int = neurons_on_first_layer
    number_of_neurons_on_second_layer: int = neurons_on_first_layer
    number_of_neurons_on_third_layer: int = neurons_on_first_layer
    number_of_neurons_on_exit_layer: int = 10
    activation_function_of_first_layer: ActivationFunctionTypesForMLP = ActivationFunctionTypesForMLP.sigmoid.value
    activation_function_of_second_layer: ActivationFunctionTypesForMLP = ActivationFunctionTypesForMLP.tanh.value
    activation_function_of_exit_layer: ActivationFunctionTypesForMLP = ActivationFunctionTypesForMLP.softmax.value
    lose_verify_function: LosesVerifyFunctionTypesForMLP = LosesVerifyFunctionTypesForMLP.categorical_crossentropy.value
    optimizator_type: OptmizatorType = OptmizatorType.adam.value
    metric: MLPMetricsType = MLPMetricsType.accuracy.value
    test_quota: float = 0.10
    epochs: int = 16
    amostras: int = 2048
    use_hard_test_data_to_validate_model: bool = False
    images_folder_path: str = './RN/'
    show_results: bool = False


@dataclass
class ProccessResult:
    result_metric: int = 0
    result_loss: int = 0
    metric_name: str = 'accuracy'


def add_none_on_list(list_to_add, qty = 7):
    if (len(list_to_add) < qty):
        for i in range(qty - len(list_to_add)):
            list_to_add.append(None)

    return list_to_add

possible_activation_function_of_first_layer = add_none_on_list([
    ActivationFunctionTypesForMLP.relu.value,
    ActivationFunctionTypesForMLP.sigmoid.value,
    ActivationFunctionTypesForMLP.tanh.value,
])

possible_activation_function_of_second_layer = add_none_on_list([
    ActivationFunctionTypesForMLP.relu.value,
    ActivationFunctionTypesForMLP.sigmoid.value,
    ActivationFunctionTypesForMLP.tanh.value,
])

possible_activation_function_of_exit_layer = add_none_on_list([
    ActivationFunctionTypesForMLP.softmax.value,
])

possible_lose_verify_function = add_none_on_list([
    LosesVerifyFunctionTypesForMLP.binary_crossentropy.value,
    LosesVerifyFunctionTypesForMLP.categorical_crossentropy.value,
])

possible_optimizator_type = add_none_on_list([
    OptmizatorType.sgd.value,
    OptmizatorType.adam.value,
    OptmizatorType.rmsprop.value,
    OptmizatorType.adagrad.value,
])

possible_metric = add_none_on_list([
    MLPMetricsType.accuracy.value,
    MLPMetricsType.mean_squared_error.value,
    MLPMetricsType.mean_absolute_error.value,
])


possible_combinations_dicts = {
    'possible_activation_function_of_first_layer' : possible_activation_function_of_first_layer,
    'possible_activation_function_of_exit_layer' : possible_activation_function_of_exit_layer,
    'possible_lose_verify_function' : possible_lose_verify_function,
    'possible_optimizator_type' : possible_optimizator_type,
}
possible_metric_df = {
    'possible_metric' : possible_metric,
}

possible_combinations_dfs = pd.DataFrame(possible_combinations_dicts)
possible_combinations_dfs = pd.concat([possible_combinations_dfs, pd.DataFrame(possible_metric_df)], axis=1)

possible_combinations_dicts.update(possible_metric_df)

msgd_result = np.meshgrid(
    [   
        possible_combinations_dfs[attr]
        for attr in list(possible_combinations_dicts.keys())
    ]
)

# combinations = pd.DataFrame(np.column_stack([
#     r.ravel()
#     for r in msgd_result
# ]), columns=list(possible_combinations_dicts.keys()))

uniques = [possible_combinations_dfs[i].unique().tolist() for i in possible_combinations_dfs.columns ]
combinations =  pd.DataFrame(product(*uniques), columns = possible_combinations_dfs.columns)

combinations = combinations.dropna()

possible_combinations = combinations.to_dict('records')

class MLPModel():
    def __init__(self, params = MLPPerformanceTestParams()) -> None:
        self.params = params

    def run(self, params = None):
        if not params:
            params = self.params
        #------------------------------------------------------------------------------
        # CRIACAO DO MODELO MLP
        model = Sequential()

        #------------------------------------------------------------------------------
        # DEFINICAO DA QUANTIDADE DE NEURONIOS DAS CAMADAS
        numbers_of_neurons_on_first_layer = params.numbers_of_neurons_on_first_layer  # Quantidade de neuronios da Camada Oculta 1
        number_of_neurons_on_second_layer = params.number_of_neurons_on_second_layer  # Quantidade de neuronios da Camada Oculta 2
        number_of_neurons_on_third_layer = params.number_of_neurons_on_third_layer  # Quantidade de neuronios da Camada Oculta 3
        number_of_neurons_on_exit_layer =  params.number_of_neurons_on_exit_layer  # Quantidade de neuronios da Camada de Sai­da

        #------------------------------------------------------------------------------
        # DEFINICAO DAS FUNCOES DE ATIVACAO
            # relu    -> Rectified Linear Unit (Unidade Linear Retificada)
            # sigmoid -> sigmoid(x) = 1 / (1 + exp(-x))
            # tanh    -> Tangente hiperbolica
            # softmax -> Utilizada na camada de sai­da
            
        # Selecione a Funcao de Ativacao da Camada Oculta 1
        activation_function_of_first_layer = params.activation_function_of_first_layer

        # Selecione a Funcao de Ativacao da Camada Oculta 2
        activation_function_of_second_layer = params.activation_function_of_second_layer

        # Selecione a Funcao de Ativacao da Camada de Sai­da
        activation_function_of_exit_layer = params.activation_function_of_exit_layer

        #------------------------------------------------------------------------------
        # ADICAO DE CAMADAS A REDE NEURAL

        # Primeira Camada Oculta:
        # Com 784 neuronios na camada de entrada -> 28x28 pixels (tamanho das imagens)
        model.add(Dense(units=numbers_of_neurons_on_first_layer, activation=activation_function_of_first_layer, input_dim=size_of_images))

        # Segunda Camada Oculta (desabilitada por padrao)
        model.add(Dense(units=number_of_neurons_on_second_layer, activation=activation_function_of_second_layer, input_dim=size_of_images))

        # # Terceira camada oculta
        model.add(Dense(units=number_of_neurons_on_third_layer, activation=activation_function_of_second_layer, input_dim=size_of_images))

        # Camada de Sai­da
        model.add(Dense(units=number_of_neurons_on_exit_layer, activation=activation_function_of_exit_layer))

        #==============================================================================
        #------------------------------------------------------------------------------
        # DEFINICAO DA FUNCAO DE PERDA
            # mean_squared_error       -> Erro quadratico medio.
            # binary_crossentropy      -> Entropia cruzada binaria.
            # categorical_crossentropy -> Entropia cruzada categorica.
            # mean_absolute_error      -> Erro absoluto medio.
            
        # INSIRA ABAIXO A FUNCAO DE PERDA DE SUA ESCOLHA
        lose_verify_function = params.lose_verify_function

        #------------------------------------------------------------------------------
        # DEFINICAO DO OTIMIZADOR
            # sgd     -> Descida de gradiente estocastico (SGD).
            # adam    -> SGD com adaptacao de taxa de aprendizado.
            # rmsprop -> Baseado em Root Mean Square Propagation.
            # adagrad -> Adapta a taxa de aprendizado para cada parametro.
            
        # INSIRA ABAIXO O OTIMIZADOR DE SUA ESCOLHA
        otimizador = params.optimizator_type

        #------------------------------------------------------------------------------
        # DEFINICAO DA METRICA DE DESEMPENHO
            # accuracy            -> Acuracia.
            # mean_squared_error  -> Erro quadratico medio.
            # mean_absolute_error -> Erro absoluto medio.

        #------------------------------------------------------------------------------
        # COMPILACAO DO MODELO

        model.compile(loss=lose_verify_function, optimizer=otimizador, metrics=[params.metric])

        #==============================================================================
        #------------------------------------------------------------------------------
        # CARREGAMENTO DOS DADOS DE TREINAMENTO E TESTE

        (x_train, y_train), (x_test, y_test) =\
            tensorflow.keras.datasets.mnist.load_data()

        #------------------------------------------------------------------------------
        # PRE-PROCESSAMENTO DOS DADOS

        # Redimensionar as imagens para um vetor unidimensional
        x_train = x_train.reshape(-1, 784)
        x_test = x_test.reshape(-1, 784)

        # Converter para tipo float32
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # Normalizar os valores dos pixels para o intervalo [0, 1]
        x_train /= 255.0
        x_test /= 255.0 

        #------------------------------------------------------------------------------
        # TRANSFORMACAO DOS ROTULOS EM CODIFICACAO ONE-HOT

        y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes=number_of_neurons_on_exit_layer)
        y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes=number_of_neurons_on_exit_layer)

        #------------------------------------------------------------------------------
        # DEFINICAO DO PERCENTUAL DO CONJUNTO DE DADOS DE TESTE

        dados_teste = params.test_quota # 0.3 significa 30%

        #------------------------------------------------------------------------------
        # DIVISAO DOS DADOS EM CONJUNTOS DE TREINAMENTO E TESTE

        x_train, x_test, y_train, y_test =\
            train_test_split(x_train, y_train, \
            test_size=dados_teste, random_state=42)

        #------------------------------------------------------------------------------
        # DEFINICAO DO NUMERO DE EPOCAS E NUMERO DE AMOSTRAS

        epocas = params.epochs
        amostras = params.amostras

        #------------------------------------------------------------------------------
        # TREINAMENTO DA REDE NEURAL MLP

        print('\n' + '=' * 70)
        print('INICIANDO O TREINAMENTO DO MODELO... \n')

        model.fit(x_train, y_train, epochs = epocas, batch_size = amostras)

        #==============================================================================
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
                image = image.reshape(1, 784)  # Redimensionar para (1, 784)

                # Normalizar a imagem
                image = image.astype('float32')
                image /= 255.0

                # Fazer a previsão da classe
                prediction = model.predict(image)
                predicted_class = np.argmax(prediction)

                # Exibir a imagem e as informações sobre o número real e previsto
                axs[i].imshow(np.squeeze(image.reshape(28, 28)), cmap='gray')
                axs[i].axis('off')
                axs[i].set_title(f'Real: {class_names[real_label]}\nPrevisto: {class_names[predicted_class]}')

            
            plt.tight_layout()
            plt.show()

        #==============================================================================
        # AVALIACAO DO MODELO (PERDA E METRICA DE DESEMPENHO)

        print('\n' + '=' * 70)
        print('CALCULANDO FUNCAO DE PERDA E MÉTRICA DE DESEMPENHO...\n')
        loss, metric = model.evaluate(x_test, y_test)

        # Exibir a Perda e a PrecisÃ£o
        print('\n' + '=' * 70)
        print('*** DESEMPENHO DO MODELO APÓS O TREINAMENTO ***\n')

        print("Funcao de Perda utilizada: " + lose_verify_function)
        print("Valor obtido: " + f" = {loss:.4f}" + '\n')

        print('-' * 70 + '\n')

        print("Metrica de Desempenho utilizada: " + params.metric)
        print("Valor obtido: " + f" = {metric:.4f} \n")

        print('=' * 70)

        #==============================================================================

        return ProccessResult(
            result_metric=metric, 
            result_loss=loss,
            metric_name=params.metric
        )



results = list()
iterator = 0

def test_model_param_combination(param_combination):
    print("Iteration: " + str(iterator))
    print('---' * 10)
    print(param_combination)

    current_params = MLPPerformanceTestParams(
        activation_function_of_first_layer=param_combination.get('possible_activation_function_of_first_layer'),
        activation_function_of_second_layer=param_combination.get('possible_activation_function_of_first_layer'),
        activation_function_of_exit_layer=param_combination.get('possible_activation_function_of_exit_layer'),
        lose_verify_function=param_combination.get('possible_lose_verify_function'),
        optimizator_type=param_combination.get('possible_optimizator_type'),
        metric=param_combination.get('possible_metric'),
    )

    result = MLPModel(current_params).run()

    execution_summary = asdict(current_params)
    execution_summary.update(asdict(result))

    return execution_summary


print('Total of combinations: ' + str(len(possible_combinations)))


if SHOULD_TEST_ALL_MODELS:
    for param_combination in possible_combinations:
        results.append(test_model_param_combination(param_combination))
        iterator += 1

    results_df = pd.DataFrame.from_dict(list(results))
    current_datetime_as_int = int(pd.Timestamp.now().timestamp())
    archive_name = './results/result-mlp-' + str(current_datetime_as_int) + '.csv'
    results_df.to_csv(archive_name)

first_winner_model = MLPPerformanceTestParams(
    activation_function_of_first_layer=ActivationFunctionTypesForMLP.relu.value,
    activation_function_of_second_layer=ActivationFunctionTypesForMLP.sigmoid.value,
    activation_function_of_exit_layer=ActivationFunctionTypesForMLP.softmax.value,
    lose_verify_function=LosesVerifyFunctionTypesForMLP.categorical_crossentropy.value,
    optimizator_type=OptmizatorType.rmsprop.value,
    metric=MLPMetricsType.accuracy.value,
    use_hard_test_data_to_validate_model=True,
    show_results=True,
    epochs=32,
    test_quota=0.3
)

result = MLPModel(first_winner_model).run()

second_winner_model = MLPPerformanceTestParams(
    activation_function_of_first_layer=ActivationFunctionTypesForMLP.relu.value,
    activation_function_of_second_layer=ActivationFunctionTypesForMLP.sigmoid.value,
    activation_function_of_exit_layer=ActivationFunctionTypesForMLP.softmax.value,
    lose_verify_function=LosesVerifyFunctionTypesForMLP.categorical_crossentropy.value,
    optimizator_type=OptmizatorType.rmsprop.value,
    metric=MLPMetricsType.accuracy.value,
    use_hard_test_data_to_validate_model=True,
    show_results=True,
    epochs=32,
    test_quota=0.3
)

result = MLPModel(second_winner_model).run()