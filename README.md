# Adaptive

This is the companion paper to the [adaptive](https://adaptive.readthedocs.io/en/latest/) Python library.

See the latest draft [here](https://gitlab.kwant-project.org/qt/adaptive-paper/builds/artifacts/master/file/paper.pdf?job=make).

### Building the paper

The simplest way to build the paper is with Docker, to make sure that all the necessary dependencies are installed.
First build the Docker image:

```
docker build -t adaptive-paper .
```

Then run `make` inside a docker container using the image you just built:

```
docker run -it --rm -v $(pwd):/work -w /work adaptive-paper make
```

### Update the bibliography

```bash
yaml2bib \
  --bib_fname "paper.bib" \
  --dois_yaml "paper.yaml" \
  --replacements_yaml "replacements.yaml" \
  --static_bib "not_on_crossref.bib" \
  --email "bas@nijho.lt"
```
