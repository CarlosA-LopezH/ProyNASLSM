# Imagen Ubuntu
FROM ubuntu:22.04

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

RUN apt-get install software-properties-common
RUN apt-get update

RUN add-apt-repository ppa:nest-simulator/nest
RUN apt-get update

# Establece 'python3' como el comando por defecto para 'python'
RUN ln -s /usr/bin/python3 /usr/bin/python

# Actualiza pip a la última versión
RUN pip3 install --upgrade pip

RUN apt-get install nest -y

# Librerías requeridas de Python
RUN pip3 install \
    numpy==1.21.5 \
    deap \
    scikit-learn \
    scikit-posthocs \
    scipy \
    matplotlib \
    seaborn

RUN /usr/bin/nest_vars.sh

ENTRYPOINT ["/bin/bash", "-c", "source /usr/bin/nest_vars.sh && /bin/bash"]