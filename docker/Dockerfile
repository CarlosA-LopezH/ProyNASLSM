# Imagen Ubuntu
FROM ubuntu:18.04

LABEL mainteiner=@naperez

# Evita que el frontend de APT requiera interacción
ENV DEBIAN_FRONTEND=noninteractive

# Aqui se deben dependencias necesarias para el proyecto
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    software-properties-common \
    && apt-get clean

RUN apt-get install software-properties-common \
    && apt-get update

RUN add-apt-repository ppa:nest-simulator/nest \
    && apt-get update \
    && apt-get install nest

# Establece 'python3' como el comando por defecto para 'python'
RUN ln -s /usr/bin/python3 /usr/bin/python

# Actualiza pip a la última versión
RUN pip3 install --upgrade pip

# Librerías requeridas de Python Proy-Detección-Nitrógeno
RUN pip3 install \
    numpy \
    deap \
    scikit-learn \
    scikit-posthocs \
    scipy \
    matplotlib \
    seaborn


ENTRYPOINT [ "/bin/bash"]