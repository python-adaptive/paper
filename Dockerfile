FROM continuumio/miniconda
MAINTAINER Bas Nijholt <bas@nijho.lt>

RUN apt-get -o Acquire::Check-Valid-Until=false update -q && apt-get install -qy \
#    texlive-full \
#    python-pygments gnuplot \
	build-essential \
    make git \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /environments
COPY environment.yml /environments/

RUN conda-env create -f /environments/environment.yml
