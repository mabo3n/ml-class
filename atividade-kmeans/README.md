# Atividade kmeans marcel

Esta pasta contém um script `kmeans.py` que aceita no stdin um arquivo .csv com colunas numéricas, e calcula e compara o número ideal de clusters para os dados de acordo com o algoritmo KMeans. As técnicas comparadas são wcss (within-cluster sum of squares) e distância euclidiana.

O script salva uma imagem no diretório onde foi executado com gráficos demonstando os resultados.

> Nota: a última coluna do arquivo .csv é desconsiderada.
> Nota: o número máximos de clusters considerados é 100

# Como executar

Navegar pela linha de comando até a pasta do script e executá-lo enviando um arquivo por standard input. Exemplo com o arquivo _data/fertility\_diagnosis.csv_ (Linux shell):

``` shell
python kmeans.py < data/fertility_diagnosis.csv
```

Caso Python 3 ou alguma biblioteca requerida não esteja instalada, é possível executar o script por um container. 

Para construir a imagem:

``` shell
docker build . -t atividade-kmeans-marcel
```

Para executar o script em um container (Linux shell):

``` shell
docker run --rm -iv $(pwd):/app/ atividade-kmeans-marcel  < data/iris.csv
```

Para o dataset _fertility\_diagnosis.csv_, o número ideal de cluster estimado foi 19 com a técnica wcss e 22 com distância euclidiana.
