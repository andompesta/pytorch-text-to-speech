FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime
MAINTAINER sandro cavallari "scavallari@paypal.com"

# update apt
RUN apt update
RUN apt install nano

# install ssh server
RUN apt install -y openssh-server

# setup ssh server
RUN mkdir /var/run/sshd
RUN echo 'root:password' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd


# init conda
RUN conda init bash
RUN /bin/bash -c "source /root/.bashrc"


# install requirments
ADD ./requirements.txt /workspace
RUN pip install -r /workspace/requirements.txt

# expose ports
EXPOSE 22
EXPOSE 6006

# add environmental variables
ENV NLTK_DATA="/workspace/libraries/nltk_data"
ENV SPACY="/workspace/libraries/spacy"
ENV PT_HUB="/workspace/libraries/hub"
ENV LASES="/workspace/libraries/laser"

RUN echo "export NLTK_DATA=/workspace/libraries/nltk_data" >> /etc/profile
RUN echo "export SPACY=/workspace/libraries/spacy" >> /etc/profile
RUN echo "export PT_HUB=/workspace/libraries/hub" >> /etc/profile
RUN echo "export LASES=/workspace/libraries/laser" >> /etc/profile


CMD ["/usr/sbin/sshd", "-D"]