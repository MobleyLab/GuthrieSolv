# The Guthrie Hydration Free Energy Database of Experimental Small Molecule Hydration Free Energies

This repository provides access to the late J. Peter Guthrie's small molecule hydration free energy database, which was donated posthumously to the community.
If you are interested in using the data provided here, please read the relevant [background information and disclaimers](#background-information-and-disclaimers) below and consider contributing to [curation of the dataset](#curation-of-the-dataset).

[![DOI](https://zenodo.org/badge/113920871.svg)](https://zenodo.org/badge/latestdoi/113920871)

## Background information and disclaimers

### Death of the primary author
For some years, J. Peter Guthrie (University of Western Ontario) worked passionately on a curating a massive database of experimental hydration free energies that he pulled from the literature.
Some of these were used for the SAMPL series of challenges over the years, and others provided some assistance in curation of [FreeSolv](https://github.com/mobleylab/FreeSolv), which Peter co-authored with me (DLM).
But the project was massive, and the literature immense.
Peter was uniquely qualified for this database curation effort, with deep understanding of the experimental techniques, extrapolations commonly employed, etc.
But the task was vast, and it outlasted him.
He died September 19, 2017, at age 76, after a battle with Guillain-Barre Syndrome.

### Succession plans
Apparently Peter must have expected the task might outlast him, as he left his son, James Guthrie, instructions to contact myself, Anthony Nicholls (OpenEye), and Paul Labute (CCG) in the event of his death.
None of the three of us have many resources to invest in continuing the curation process at present; at the same time, we believe this data and the underlying work and references will have considerable value to the community long term.
So after discussion, we decided the best path forward was simply to make available what Peter and James provided to allow the community to use and curate it.
James gave permission to post this data publicly to allow this effort to continue.

### Disclaimers

#### We provide two different types of data

This dataset consists of two parts which are expected to become significantly different:
1. An original Excel spreadsheet, which is provided exactly as it arrived from the Guthrie family. This is provided in an "as is" format and you should use it as your own risk; we have no information about its contents beyond what is in the spreadsheet itself and in this GitHub repository. No changes to this spreadsheet will be made.
2. A current database, which is initially an export of the contents of the Excel database, but is expected to become an independent entity based on [community curation](#curation-of-the-dataset).

#### Use both versions at your own risk

We make no warranty as to the contents or usefulness of either dataset; both are provided as resources to the community but must be used with caution and with your own consultation of the literature.

## Curation of the dataset

Our hope is that the community will get involved with curation of the dataset provided here -- in particular, the "current database" (the Excel spreadsheet should be left in its original form).
Suggested improvements should come in via pull requests, where each pull request provides proposed modifications (including potentially supporting tools/scripts, data, references, or links to the same) and a clear explanation of these changes.
Thus, over time the current, curated database is expected to move away from simply reflecting the contents of the Excel spreadsheet and become more valuable.

Some specific points of curation which will be needed include:
- Separation of different types of data; for example, the main tab in the database Excel spreadsheet (and the data in `guthrie_database.csv`) contains not just hydration free energies but other properties with other units, e.g. the entries for phenol include values reported in mg/L, g/m^3, etc.
- unit handling; values are present in kJ/mol and kcal/mol
- checking of molecule names against SMILES and stereochemistry; I (DLM) previously gave Peter some tools to help with this but I do not know if he has used them

See also [`usage_notes.md`](usage_notes.md) for some information which relates to the contents.

## Manifest
- `GuthrieDatabase_April14.zip`: Guthrie database (Excel spreadsheet) as it was provided
- `guthrie_database.csv`: Exported csv file of main tab of Excel spreadsheet
- `guthrie_references_and_status.csv`: Additional tab of Excel spreadsheet which provides definitions of the references and reports on Peter's progress in extracting data from those references; may highlight other areas where more data is still available

There is also data/curation work in an additional tab of the spreadsheet, Sheet 2, which may be useful but is not present here as a separate file yet.

## Using the dataset

The data set can be loaded easily in Python using `pandas`, for example as:
```
python
import pandas
db = pandas.read_csv('guthrie_database.csv', encoding='latin1')
data = db[db.Name=='phenol']
```
to load the database and extract all data with a molecule named phenol

## Maintenance

This repository has data quality assurance tests implemented in Python that
can be run with `tox` using the following commands:

```shell
$ git clone git@github.com:MobleyLab/GuthrieSolv.git
$ cd GuthrieSolv
$ pip install tox
$ tox
```

## Authors
### Primary author
- J. Peter Guthrie (University of Western Ontario)

### Other contributors
- David L. Mobley, UC Irvine, who maintains this repository
- Probably students and others who worked with Dr. Guthrie over the years, but I (DLM) do not have their information

## Citing this work
Please cite this GitHub repository, as well as "The Guthrie Hydration Free Energy Database of Experimental Small Molecule Hydration Free Energies," J. Peter Guthrie and David L. Mobley, eScholarship, [https://escholarship.org/uc/item/53n2h10t](https://escholarship.org/uc/item/53n2h10t). 

We maintain archival copies of this repository on eScholarship, administered by the University of California, in order to ensure long term access. New versions will also be posted there.

## Acknowledgments
- James Guthrie, who made this data available and gave permission to post it publicly; he does not want any credit for this, but he should certainly be acknowledged.

(To be updated as people contribute)

## Versions

- Version 0.01 ([10.5281/zenodo.1101258](http://dx.doi.org/10.5281/zenodo.1101258)): Provides initial files.
