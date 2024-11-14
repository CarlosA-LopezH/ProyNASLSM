# NEST image
FROM nest/nest-simulator:3.8

LABEL mainteiner=@naperez

# Evita que el frontend de APT requiera interacción
ENV DEBIAN_FRONTEND=noninteractive

# Aqui se deben dependencias necesarias para el proyecto
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    && apt-get clean

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