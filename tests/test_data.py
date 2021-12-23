# -*- coding: utf-8 -*-

"""Test data."""

import unittest
from collections import defaultdict
from pathlib import Path
from typing import ClassVar

import pandas as pd
from rdkit.Chem import MolFromSmiles

HERE = Path(__file__).parent.resolve()
PATH = HERE.parent.joinpath("guthrie_database.csv")


class TestData(unittest.TestCase):
    """A test case for checking the data."""

    df: ClassVar[pd.DataFrame]

    @classmethod
    def setUpClass(cls) -> None:
        """Prepare the test case with the data."""
        cls.df = pd.read_csv(PATH)

    def test_smiles(self):
        """Test all SMILES can be parsed."""
        dd = defaultdict(list)
        names = {}
        missing = {}
        for idx, (smiles, name) in self.df[["mol", "Name"]].iterrows():
            file_idx = idx + 2
            if pd.isna(smiles):
                missing[file_idx] = name
            else:
                dd[smiles].append(file_idx)
                names[smiles] = name

        for file_idx, name in sorted(missing.items()):
            with self.subTest(name=name, smiles="MISSING", line=file_idx):
                self.fail(msg=f"Lines missing SMILES: {missing}")

        for smiles, lines in sorted(dd.items()):
            with self.subTest(name=names[smiles].strip(), smiles=smiles):
                self.assertIsInstance(smiles, str)
                mol = MolFromSmiles(smiles)
                self.assertIsNotNone(mol, msg=f"Bad smiles for {names[smiles].strip()} on rows: {lines}")
