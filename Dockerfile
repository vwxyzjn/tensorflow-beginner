FROM ufoym/deepo:all-py36-jupyter-cpu

# Install gym related libraries
RUN pip install gym==0.10.9
RUN pip install gym[atari]
RUN apt-get update && apt-get install -y python-opengl

# Install gym render related libraries
# RUN apt-get install -y xvfb
# RUN pip install pyvirtualdisplay==

# Install other dependencies
RUN pip install tensorflow_hub==0.1.1

WORKDIR /root/

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
