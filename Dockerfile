FROM ubuntu:xenial
MAINTAINER Bas Nijholt <bas@nijho.lt>

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

RUN apt-get update -q && apt-get install -qy \
    # texlive-full \
    python-pygments gnuplot \
    make git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /environments
COPY environment.yml /environments/

RUN conda-env create -f /environments/environment.yml

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]