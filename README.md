# Classificação de Imagens

Este repositório contem todo o código utilizado no projeto de bloco B do curso de pós graduação da Infnet em Big Data.

Todo código aqui contido possui todos os jupyter notebooks e modulo python para o treinamento dos modelos.

## Notebooks

Os notebooks são usados com função de fazer o pré-processamento das imagens para o formato dos modelos de redes pré-treinadas e a avaliação dos modelos gerando seu f1-score e matriz de confusão.

* notebooks/preprocessing.ipynb VGG16
* notebooks/preprocessing-vgg19.ipynb VGG19
* notebooks/preprocessing-resnet50.ipynb ResNet50
* notebooks/preprocessing-inceptionv3.ipynb InceptionV3
* notebooks/preprocessing-NASNetLarge.ipynb NasNetLarge
* notebooks/preprocessing-simple.ipynb Treinamento de camadas de feature extraction
* notebooks/evaluate.ipynb Cálculas métricas de performance

## Códigos .py

O código desenvolvido para o processo de treinamento e validação pode ser reutilizado em qualquer experimento, apenas ajustando um script em shell com parâmetros que constroem e alimentam o modelo. O principal objetivo do algoritmo é viabilizar e agilizar a implantação de um processo de treinamento de Redes Neurais Convolucionais em uma instância de máquina virtual do Google Cloud Platform, utilizando a biblioteca Keras, e acessando os dados de um storage dentro da mesma plataforma.

* train-on-cloud.sh Script em shell que define todos os parâmetros necessários para iniciar o treinamento
* train.py As principais tarefas são iniciadas e as dependências para os demais arquivos definidas. O algoritmo é iniciado pelo train-on-cloud.sh e é responsável por: inicializar parâmetros, consturir rede neural, iniciar treinamento e persistir modelo em hdf5
* parameter.py Dependência do train.py, contróis classe de parâmetros e inicializa os parâmetros definidos no train-on-cloud
* dataset.py Dependêndia do train.py,importa imagens estruturadas em tfrecords do GCP storage

## Como usar train-on-cloud.sh

Os seguintes parâmetros devem ser modificados/atualizados/conferidos antes de inicar um novo treinamento:

REGION= #região em que a instância de processamento está lcoalizada no GCP (string)

MODEL_NAME= #nome arbitrário do modelo (string)

EPOCHS= #quantidade de épocas de treinamento (int)

BATCH_SIZE= #tamanho do batch de treinamento (int)

LAYERS= #camadas completamente conectadas a serem construídas no modelo (json)
  exemplo: "[{\"class_name\":\"Dense\",\"params\":{\"units\":256,\"activation\":\"sigmoid\"}},{\"class_name\":\"Dropout\",\"params\":{\"rate\":0.5}}]"

BASE_PATH= #endereço de localização do dataset em tfrecords

BUCKET_NAME= #nome do bucket no GCP
