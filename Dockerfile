FROM continuumio/miniconda
MAINTAINER Bas Nijholt <bas@nijho.lt>

RUN apt-get -o Acquire::Check-Valid-Until=false update -q && apt-get install -qy \
    texlive-full \
    python-pygments gnuplot \
    make git \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml .

RUN conda-env create -f environment.yml
