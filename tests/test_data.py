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
        missing = []
        for idx, smiles in self.df["mol"].items():
            if pd.isna(smiles):
                missing.append(idx)
            else:
                dd[smiles].append(idx)

        with self.subTest(smiles="MISSING"):
            self.assertEqual(0, len(missing), msg=f"Lines missing SMILES: {missing}")

        for smiles, lines in sorted(dd.items()):
            with self.subTest(smiles=smiles):
                self.assertIsInstance(smiles, str)
                mol = MolFromSmiles(smiles)
                self.assertIsNotNone(mol, msg=f"Unable to parse SMILES on rows: {lines}")
