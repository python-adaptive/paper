# Adapted from https://gitlab.kwant-project.org/qt/basnijholt/thesis-bas-nijholt/blob/master/create_bib_file.py

import functools
import glob
import os
from concurrent.futures import ThreadPoolExecutor

import requests
import yaml


def edit_raw_bibtex_entry(key, bib_entry):
    bib_type, *_ = bib_entry.split("{")
    _, *rest = bib_entry.split(",")
    rest = ",".join(rest)
    # Now only modify `rest` because we don't want to touch the key.

    # XXX: I am not sure whether these substitutions are needed.
    # the problem seemed to be the utf-8 `requests.get` encoding.
    to_replace = [("ö", r"\"{o}"), ("ü", r"\"{u}"), ("ë", r"\"{e}"), ("ï", r"\"{i}")]

    for old, new in to_replace:
        rest = rest.replace(old.upper(), new.upper())
        rest = rest.replace(old.lower(), new.lower())

    to_replace = [
        (r"a{\r}", r"\r{a}"),  # "Nyga{\r}rd" -> "Nyg\r{a}rd", bug in doi.org
        ("Josephson", "{J}osephson"),
        ("Majorana", "{M}ajorana"),
        ("Andreev", "{A}ndreev"),
        ("Kramers", "{K}ramers"),
        ("Kitaev", "{K}itaev"),
        (
            r"metastable0and$\uppi$states",
            r"metastable $0$ and $\pi$ states",
        ),  # fix for 10.1103/physrevb.63.214512
        (
            r"Land{\'{e}}{gFactors}",
            r"Land{\'{e}} {$g$} Factors",
        ),  # fix for PhysRevLett.96.026804
    ]

    journals = [
        ("Advanced Materials", "Adv. Mater."),
        ("Annals of Physics", "Ann. Phys."),
        ("Applied Physics Letters", "Appl. Phys. Lett."),
        ("JETP Lett", "JETP Lett."),
        ("Journal de Physique", "J. Phys."),
        ("Journal of Computational Physics", "J. Comput. Phys."),
        ("Journal of Experimental and Theoretical Physics", "J. Exp. Theor. Phys."),
        ("Journal of Low Temperature Physics", "J. Low Temp. Phys."),
        (
            "Journal of Physics A: Mathematical and Theoretical",
            "J. Phys. A: Math. Theor.",
        ),
        ("Journal of Physics: Condensed Matter", "J. Phys.: Condens. Matter"),
        ("Nano Letters", "Nano Lett."),
        ("Nature Communications", "Nat. Commun."),
        ("Nature Materials", "Nat. Mater."),
        ("Nature Nanotechnology", "Nat. Nanotechnol."),
        ("Nature Physics", "Nat. Phys."),
        ("New Journal of Physics", "New J. Phys."),
        ("Physical Review B", "Phys. Rev. B"),
        ("Physical Review Letters", "Phys. Rev. Lett."),
        ("Physical Review X", "Phys. Rev. X"),
        ("Physical Review", "Phys. Rev."),  # should be before the above subs
        ("Physics-Uspekhi", "Phys. Usp."),
        ("Reports on Progress in Physics", "Rep. Prog. Phys."),
        ("Review of Scientific Instruments", "Rev. Sci. Instrum."),
        ("Reviews of Modern Physics", "Rev. Mod. Phys."),
        ("Science Advances", "Sci. Adv."),
        ("Scientific Reports", "Sci. Rep."),
        ("Semiconductor Science and Technology", "Semicond. Sci. Technol."),
        (
            "Annual Review of Condensed Matter Physics",
            "Annu. Rev. Condens. Matter Phys.",
        ),
        ("{EPL} (Europhysics Letters)", "{EPL}"),
        ("Nature Reviews Materials", "Nat. Rev. Mater."),
        ("Physics Letters", "Phys. Lett."),
        ("The European Physical Journal B", "Eur. Phys. J. B"),
        ("{SIAM} Journal on Numerical Analysis", "{SIAM} J. Numer. Anal."),
        ("{AIP} Conference Proceedings", "{AIP} Conf. Proc."),
    ]

    for old, new in to_replace + journals:
        rest = rest.replace(old, new)

    result = bib_type + "{" + key + "," + rest

    print(result, "\n")
    return result


@functools.lru_cache()
def doi2bib(doi):
    """Return a bibTeX string of metadata for a given DOI."""
    url = "http://dx.doi.org/" + doi
    headers = {"accept": "application/x-bibtex"}
    r = requests.get(url, headers=headers)
    r.encoding = "utf-8"
    return r.text


fname = "paper.yaml"
print("Reading: ", fname)

with open(fname) as f:
    dois = yaml.safe_load(f)
dois = dict(sorted(dois.items()))


with ThreadPoolExecutor() as ex:
    futs = ex.map(doi2bib, list(dois.values()))
    bibs = list(futs)


entries = [edit_raw_bibtex_entry(key, bib) for key, bib in zip(dois.keys(), bibs)]


with open("paper.bib", "w") as out_file:
    fname = "not_on_crossref.bib"
    out_file.write("@preamble{ {\\providecommand{\\BIBYu}{Yu} } }\n\n")
    out_file.write(f"\n% Below is from `{fname}`.\n\n")
    with open(fname) as in_file:
        out_file.write(in_file.read())
    out_file.write("\n% Below is from `paper.yaml`.\n\n")
    for e in entries:
        for line in e.split("\n"):
            # Remove the url line
            if "url = {" not in line:
                out_file.write(f"{line}\n")
        out_file.write("\n")
