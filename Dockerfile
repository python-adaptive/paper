FROM ubuntu:xenial
MAINTAINER Bas Nijholt <bas@nijho.lt>

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y \
    # for miniconda
    wget bzip2 ca-certificates curl git \
    # for TeX
    texlive-full python-pygments gnuplot make \
    # for gcc
    build-essential \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

RUN mkdir /environments
COPY environment.yml /environments/

RUN conda-env create -f /environments/environment.yml && \
    # ensure that we activate the environment in any shell (Gitlab CI uses 'sh')
    echo "conda activate revtex-markdown-paper" >> ~/.bashrc

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]
