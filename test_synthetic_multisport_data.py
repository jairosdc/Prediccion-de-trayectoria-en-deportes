import unittest
import pandas as pd
import numpy as np
import os
from synthetic_multisport_data import (
    calcular_presion_temporal,
    calcular_presion_marcador,
    calcular_pressure_index,
    clasificar_presion,
    normalizar_probabilidades,
    generar_penaltis_futbol,
    generar_dataset_multideporte
)

class TestSyntheticMultisportData(unittest.TestCase):

    def test_calcular_presion_temporal(self):
        self.assertEqual(calcular_presion_temporal(0), 0)
        self.assertEqual(calcular_presion_temporal(1), 1)
        self.assertEqual(calcular_presion_temporal(0.5), 0.5)

    def test_calcular_presion_marcador(self):
        self.assertEqual(calcular_presion_marcador(0), 1.0)
        self.assertTrue(calcular_presion_marcador(10) < calcular_presion_marcador(1))
        self.assertEqual(calcular_presion_marcador(100), 0.0)

    def test_calcular_pressure_index(self):
        idx_high = calcular_pressure_index(1, 1, 1, 1, 1)
        self.assertTrue(0.99 <= idx_high <= 1.0)
        
        idx_low = calcular_pressure_index(0, 0, 0, 0, 0)
        self.assertEqual(idx_low, 0.0)

    def test_clasificar_presion(self):
        self.assertEqual(clasificar_presion(0.2), 'baja')
        self.assertEqual(clasificar_presion(0.5), 'media')
        self.assertEqual(clasificar_presion(0.8), 'alta')

    def test_normalizar_probabilidades(self):
        probs = [0.1, 0.1, 0.2]
        norm = normalizar_probabilidades(probs)
        self.assertEqual(sum(norm), 1.0)
        self.assertEqual(norm[0], 0.25)
        
        probs_neg = [-0.1, 0.5, 0.5]
        norm_neg = normalizar_probabilidades(probs_neg)
        self.assertEqual(norm_neg[0], 0.0)
        self.assertEqual(norm_neg[1], 0.5)

    def test_generar_penaltis_futbol(self):
        df = generar_penaltis_futbol(n=10)
        self.assertEqual(len(df), 10)
        expected_columns = [
            'event_id', 'sport', 'action_type', 'competition', 'match_time',
            'score_difference', 'is_home_player', 'player', 'opponent',
            'dominant_side', 'fatigue_level', 'competition_importance',
            'pressure_index', 'pressure_level', 'decision_zone', 'action_result'
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns)

    def test_generar_dataset_multideporte(self):
        df_all, df_ftb, df_ten, df_bkb, df_hnd, df_hoc, df_esp = generar_dataset_multideporte(n_por_deporte=5)
        self.assertEqual(len(df_all), 30)
        self.assertEqual(len(df_ftb), 5)
        self.assertEqual(len(df_esp), 5)

if __name__ == '__main__':
    unittest.main()
