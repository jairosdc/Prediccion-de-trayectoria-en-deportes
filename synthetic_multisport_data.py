import pandas as pd
import numpy as np
import os

# =============================================================================
# ADVERTENCIA: 
# Estos datos son sintéticos. Sirven para validar el pipeline, la arquitectura 
# y la demo interactiva. No sirven para extraer conclusiones empíricas reales 
# sobre comportamiento deportivo.
# =============================================================================

def calcular_presion_temporal(match_time_normalized):
    """Calcula la presión por el tiempo de juego (0 = inicio, 1 = final)."""
    return match_time_normalized

def calcular_presion_marcador(score_difference):
    """Calcula la presión por marcador (mayor presión cuando diferencia es 0)."""
    return max(0.0, 1.0 - abs(score_difference) * 0.15)

def calcular_pressure_index(time_pressure, score_pressure, comp_importance, fatigue, critical_moment):
    """Calcula el índice de presión global combinando múltiples factores."""
    index = (0.30 * time_pressure +
             0.25 * score_pressure +
             0.20 * comp_importance +
             0.15 * fatigue +
             0.10 * critical_moment)
    return min(1.0, max(0.0, index))

def clasificar_presion(pressure_index):
    """Clasifica el nivel de presión en baja, media o alta."""
    if pressure_index < 0.35:
        return 'baja'
    elif pressure_index < 0.70:
        return 'media'
    else:
        return 'alta'

def normalizar_probabilidades(probs):
    """Normaliza un array de probabilidades para que sumen 1, evitando negativos."""
    probs = np.array([max(0.0, p) for p in probs])
    suma = np.sum(probs)
    if suma > 0:
        return probs / suma
    return np.ones(len(probs)) / len(probs)

def generar_penaltis_futbol(n=300, seed=42):
    np.random.seed(seed)
    data = []
    zones = ['left', 'center', 'right']
    results = ['goal', 'saved', 'missed']
    
    for i in range(n):
        time_norm = np.random.uniform(0, 1)
        score_diff = np.random.randint(-2, 3)
        comp_imp = np.random.uniform(0.3, 1.0)
        fatigue = np.random.uniform(0.1, 0.9)
        critical = np.random.uniform(0.5, 1.0) if time_norm > 0.8 and abs(score_diff) <= 1 else np.random.uniform(0, 0.5)
        
        t_press = calcular_presion_temporal(time_norm)
        s_press = calcular_presion_marcador(score_diff)
        p_index = calcular_pressure_index(t_press, s_press, comp_imp, fatigue, critical)
        p_level = clasificar_presion(p_index)
        
        dominant_side = np.random.choice(['right', 'left'], p=[0.85, 0.15])
        
        # Regla: diestros cruzan más (left), zurdos cruzan más (right)
        if dominant_side == 'right':
            base_p = [0.45, 0.15, 0.40]
        else:
            base_p = [0.40, 0.15, 0.45]
            
        # Regla: alta presión aumenta la probabilidad del centro
        if p_level == 'alta':
            base_p[1] += 0.15
            base_p[0] -= 0.075
            base_p[2] -= 0.075
            
        decision = np.random.choice(zones, p=normalizar_probabilidades(base_p))
        action_res = np.random.choice(results, p=[0.75, 0.18, 0.07])
        
        data.append({
            'event_id': f"FTB_{i+1:04d}",
            'sport': 'football',
            'action_type': 'penalty',
            'competition': np.random.choice(['World Cup', 'Champions', 'League']),
            'match_time': round(time_norm * 120),
            'score_difference': score_diff,
            'is_home_player': np.random.choice([0, 1]),
            'player': f"Player_{np.random.randint(1, 100)}",
            'opponent': f"GK_{np.random.randint(1, 30)}",
            'dominant_side': dominant_side,
            'fatigue_level': round(fatigue, 2),
            'competition_importance': round(comp_imp, 2),
            'pressure_index': round(p_index, 3),
            'pressure_level': p_level,
            'decision_zone': decision,
            'action_result': action_res
        })
    return pd.DataFrame(data)

def generar_saques_tenis(n=300, seed=43):
    np.random.seed(seed)
    data = []
    zones = ['wide', 'body', 'center']
    results = ['ace', 'returned', 'fault']
    
    for i in range(n):
        time_norm = np.random.uniform(0, 1)
        score_diff = np.random.randint(-1, 2)
        comp_imp = np.random.uniform(0.4, 1.0)
        fatigue = np.random.uniform(0.1, 0.9)
        critical = np.random.uniform(0.6, 1.0) if abs(score_diff) == 0 else np.random.uniform(0, 0.4)
        
        t_press = calcular_presion_temporal(time_norm)
        s_press = calcular_presion_marcador(score_diff)
        p_index = calcular_pressure_index(t_press, s_press, comp_imp, fatigue, critical)
        p_level = clasificar_presion(p_index)
        
        dominant_side = np.random.choice(['right', 'left'], p=[0.9, 0.1])
        base_p = [0.35, 0.30, 0.35]
        
        # Regla: fatiga reduce saques agresivos (wide y center) a favor del body
        if fatigue > 0.7:
            base_p[1] += 0.20
            base_p[0] -= 0.10
            base_p[2] -= 0.10
            
        # Regla: en presión alta aumenta saque al cuerpo (seguridad)
        if p_level == 'alta':
            base_p[1] += 0.15
            
        decision = np.random.choice(zones, p=normalizar_probabilidades(base_p))
        action_res = np.random.choice(results, p=[0.15, 0.65, 0.20])
        
        data.append({
            'event_id': f"TEN_{i+1:04d}",
            'sport': 'tennis',
            'action_type': 'serve',
            'competition': np.random.choice(['Grand Slam', 'Masters 1000', 'ATP 500']),
            'match_time': round(time_norm * 5), # set number approx
            'score_difference': score_diff,
            'is_home_player': np.random.choice([0, 1]),
            'player': f"Player_{np.random.randint(1, 50)}",
            'opponent': f"Opp_{np.random.randint(1, 50)}",
            'dominant_side': dominant_side,
            'fatigue_level': round(fatigue, 2),
            'competition_importance': round(comp_imp, 2),
            'pressure_index': round(p_index, 3),
            'pressure_level': p_level,
            'decision_zone': decision,
            'action_result': action_res
        })
    return pd.DataFrame(data)

def generar_tiros_baloncesto(n=300, seed=44):
    np.random.seed(seed)
    data = []
    zones = ['drive', 'mid_range', 'three_point']
    results = ['scored', 'missed', 'blocked']
    
    for i in range(n):
        time_norm = np.random.uniform(0, 1)
        score_diff = np.random.randint(-5, 5)
        comp_imp = np.random.uniform(0.3, 1.0)
        fatigue = np.random.uniform(0.1, 1.0)
        critical = np.random.uniform(0.8, 1.0) if time_norm > 0.9 and abs(score_diff) <= 3 else np.random.uniform(0, 0.5)
        
        t_press = calcular_presion_temporal(time_norm)
        s_press = calcular_presion_marcador(score_diff)
        p_index = calcular_pressure_index(t_press, s_press, comp_imp, fatigue, critical)
        p_level = clasificar_presion(p_index)
        
        dominant_side = np.random.choice(['right', 'left'], p=[0.88, 0.12])
        base_p = [0.35, 0.35, 0.30]
        
        # Regla: cansancio alto reduce probabilidad de triple y penetración agresiva
        if fatigue > 0.75:
            base_p[2] -= 0.15
            base_p[1] += 0.20
            
        # Regla: alta presión fomenta opciones conservadoras
        if p_level == 'alta':
            base_p[0] += 0.10 # penetración buscando falta
            base_p[2] -= 0.10
            
        decision = np.random.choice(zones, p=normalizar_probabilidades(base_p))
        action_res = np.random.choice(results, p=[0.45, 0.45, 0.10])
        
        data.append({
            'event_id': f"BKB_{i+1:04d}",
            'sport': 'basketball',
            'action_type': 'decisive_shot',
            'competition': np.random.choice(['NBA', 'Euroleague', 'Olympics']),
            'match_time': round(time_norm * 48), # Minute
            'score_difference': score_diff,
            'is_home_player': np.random.choice([0, 1]),
            'player': f"Player_{np.random.randint(1, 80)}",
            'opponent': f"Def_{np.random.randint(1, 80)}",
            'dominant_side': dominant_side,
            'fatigue_level': round(fatigue, 2),
            'competition_importance': round(comp_imp, 2),
            'pressure_index': round(p_index, 3),
            'pressure_level': p_level,
            'decision_zone': decision,
            'action_result': action_res
        })
    return pd.DataFrame(data)

def generar_lanzamientos_balonmano(n=300, seed=45):
    np.random.seed(seed)
    data = []
    zones = ['low_left', 'low_right', 'high_left', 'high_right', 'center']
    results = ['goal', 'saved', 'missed']
    
    for i in range(n):
        time_norm = np.random.uniform(0, 1)
        score_diff = np.random.randint(-3, 3)
        comp_imp = np.random.uniform(0.3, 1.0)
        fatigue = np.random.uniform(0.2, 0.9)
        critical = np.random.uniform(0.5, 1.0) if abs(score_diff) <= 1 else np.random.uniform(0, 0.4)
        
        t_press = calcular_presion_temporal(time_norm)
        s_press = calcular_presion_marcador(score_diff)
        p_index = calcular_pressure_index(t_press, s_press, comp_imp, fatigue, critical)
        p_level = clasificar_presion(p_index)
        
        dominant_side = np.random.choice(['right', 'left'], p=[0.85, 0.15])
        base_p = [0.2, 0.2, 0.2, 0.2, 0.2]
        
        # Regla: mano dominante influye (diestros tienden a la derecha del portero)
        if dominant_side == 'right':
            base_p[1] += 0.1
            base_p[3] += 0.1
        else:
            base_p[0] += 0.1
            base_p[2] += 0.1
            
        # Regla: presión alta aumenta tiros centrales/bajos por seguridad
        if p_level == 'alta':
            base_p[4] += 0.2
            base_p[0] += 0.05
            base_p[1] += 0.05
            
        decision = np.random.choice(zones, p=normalizar_probabilidades(base_p))
        action_res = np.random.choice(results, p=[0.70, 0.22, 0.08])
        
        data.append({
            'event_id': f"HND_{i+1:04d}",
            'sport': 'handball',
            'action_type': '7m_throw',
            'competition': np.random.choice(['World Ch.', 'Euro', 'Champions L.']),
            'match_time': round(time_norm * 60),
            'score_difference': score_diff,
            'is_home_player': np.random.choice([0, 1]),
            'player': f"Player_{np.random.randint(1, 60)}",
            'opponent': f"GK_{np.random.randint(1, 30)}",
            'dominant_side': dominant_side,
            'fatigue_level': round(fatigue, 2),
            'competition_importance': round(comp_imp, 2),
            'pressure_index': round(p_index, 3),
            'pressure_level': p_level,
            'decision_zone': decision,
            'action_result': action_res
        })
    return pd.DataFrame(data)

def generar_shootouts_hockey(n=300, seed=46):
    np.random.seed(seed)
    data = []
    zones = ['left_deke', 'right_deke', 'direct_shot', 'high_shot', 'low_shot']
    results = ['goal', 'saved', 'missed']
    
    for i in range(n):
        time_norm = 1.0 # Shootout siempre es al final
        score_diff = 0 # Siempre es empate en inicio de shootout
        comp_imp = np.random.uniform(0.4, 1.0)
        fatigue = np.random.uniform(0.5, 1.0) # Alta fatiga tras el partido
        critical = np.random.uniform(0.7, 1.0) # Shootout es siempre crítico
        
        t_press = calcular_presion_temporal(time_norm)
        s_press = calcular_presion_marcador(score_diff)
        p_index = calcular_pressure_index(t_press, s_press, comp_imp, fatigue, critical)
        p_level = clasificar_presion(p_index)
        
        dominant_side = np.random.choice(['right', 'left'], p=[0.70, 0.30])
        base_p = [0.2, 0.2, 0.2, 0.2, 0.2]
        
        # Regla: Fatiga alta reduce jugadas complejas (dekes)
        if fatigue > 0.8:
            base_p[0] -= 0.1
            base_p[1] -= 0.1
            base_p[2] += 0.15
            base_p[4] += 0.05
            
        # Regla: En presión alta aumenta el tiro directo
        if p_level == 'alta':
            base_p[2] += 0.2
            
        decision = np.random.choice(zones, p=normalizar_probabilidades(base_p))
        action_res = np.random.choice(results, p=[0.33, 0.55, 0.12])
        
        data.append({
            'event_id': f"HOC_{i+1:04d}",
            'sport': 'hockey',
            'action_type': 'shootout',
            'competition': np.random.choice(['NHL', 'Olympics', 'World Ch.']),
            'match_time': 65, # OT end
            'score_difference': score_diff,
            'is_home_player': np.random.choice([0, 1]),
            'player': f"Player_{np.random.randint(1, 50)}",
            'opponent': f"GK_{np.random.randint(1, 20)}",
            'dominant_side': dominant_side,
            'fatigue_level': round(fatigue, 2),
            'competition_importance': round(comp_imp, 2),
            'pressure_index': round(p_index, 3),
            'pressure_level': p_level,
            'decision_zone': decision,
            'action_result': action_res
        })
    return pd.DataFrame(data)

def generar_duelos_esports(n=300, seed=47):
    np.random.seed(seed)
    data = []
    zones = ['aggressive_push', 'defensive_hold', 'flank', 'bait', 'direct_duel']
    results = ['kill', 'death', 'trade', 'escape']
    
    for i in range(n):
        time_norm = np.random.uniform(0, 1)
        score_diff = np.random.randint(-5, 6)
        comp_imp = np.random.uniform(0.5, 1.0)
        fatigue = np.random.uniform(0.1, 1.0)
        critical = np.random.uniform(0.6, 1.0) if time_norm > 0.8 else np.random.uniform(0, 0.5)
        
        t_press = calcular_presion_temporal(time_norm)
        s_press = calcular_presion_marcador(score_diff)
        p_index = calcular_pressure_index(t_press, s_press, comp_imp, fatigue, critical)
        p_level = clasificar_presion(p_index)
        
        dominant_side = 'right' # Ratón diestro por defecto mayoritario
        base_p = [0.25, 0.25, 0.15, 0.10, 0.25]
        
        # Regla: Fatiga aumenta la pasividad o errores de posicionamiento
        if fatigue > 0.8:
            base_p[1] += 0.20 # Hold pasivo
            base_p[0] -= 0.15
            
        # Regla: Presión alta fomenta juego más defensivo o de bait
        if p_level == 'alta':
            base_p[1] += 0.15
            base_p[3] += 0.10
            base_p[0] -= 0.10
            
        decision = np.random.choice(zones, p=normalizar_probabilidades(base_p))
        action_res = np.random.choice(results, p=[0.40, 0.40, 0.10, 0.10])
        
        data.append({
            'event_id': f"ESP_{i+1:04d}",
            'sport': 'esports',
            'action_type': '1v1_duel',
            'competition': np.random.choice(['Major', 'Pro League', 'Championship']),
            'match_time': round(time_norm * 30), # Rounds or minutes
            'score_difference': score_diff,
            'is_home_player': 0, # Neutral mostly
            'player': f"Player_{np.random.randint(1, 100)}",
            'opponent': f"Opp_{np.random.randint(1, 100)}",
            'dominant_side': dominant_side,
            'fatigue_level': round(fatigue, 2),
            'competition_importance': round(comp_imp, 2),
            'pressure_index': round(p_index, 3),
            'pressure_level': p_level,
            'decision_zone': decision,
            'action_result': action_res
        })
    return pd.DataFrame(data)

def generar_dataset_multideporte(n_por_deporte=300, seed=42):
    """Genera y une todos los dataframes de los distintos deportes."""
    df_ftb = generar_penaltis_futbol(n_por_deporte, seed)
    df_ten = generar_saques_tenis(n_por_deporte, seed+1)
    df_bkb = generar_tiros_baloncesto(n_por_deporte, seed+2)
    df_hnd = generar_lanzamientos_balonmano(n_por_deporte, seed+3)
    df_hoc = generar_shootouts_hockey(n_por_deporte, seed+4)
    df_esp = generar_duelos_esports(n_por_deporte, seed+5)
    
    df_all = pd.concat([df_ftb, df_ten, df_bkb, df_hnd, df_hoc, df_esp], ignore_index=True)
    return df_all, df_ftb, df_ten, df_bkb, df_hnd, df_hoc, df_esp

def guardar_datasets():
    """Genera y guarda todos los datasets en la carpeta data/."""
    os.makedirs('data', exist_ok=True)
    
    print("Iniciando generación de datos sintéticos...")
    df_all, df_ftb, df_ten, df_bkb, df_hnd, df_hoc, df_esp = generar_dataset_multideporte(300)
    
    # Guardar CSVs
    df_ftb.to_csv('data/synthetic_football_penalties.csv', index=False)
    df_ten.to_csv('data/synthetic_tennis_serves.csv', index=False)
    df_bkb.to_csv('data/synthetic_basketball_shots.csv', index=False)
    df_hnd.to_csv('data/synthetic_handball_throws.csv', index=False)
    df_hoc.to_csv('data/synthetic_hockey_shootouts.csv', index=False)
    df_esp.to_csv('data/synthetic_esports_duels.csv', index=False)
    df_all.to_csv('data/synthetic_multisport_decisions.csv', index=False)
    
    print("¡Datos generados y guardados en data/!")
    print("\n--- Resumen de Datos Sintéticos ---")
    print(f"Total registros multideporte: {len(df_all)}")
    print("\nRegistros por deporte:")
    print(df_all['sport'].value_counts())
    
    print("\nDistribución global de nivel de presión:")
    print(df_all['pressure_level'].value_counts(normalize=True).round(3) * 100)
    
    print("\nDistribución de decision_zone por deporte:")
    for sport in df_all['sport'].unique():
        dist = df_all[df_all['sport'] == sport]['decision_zone'].value_counts(normalize=True).round(3) * 100
        print(f"\n{sport.capitalize()}:")
        print(dist.to_string())
        
    print("\nPrimeras filas del dataset multideporte:")
    print(df_all.head())

if __name__ == "__main__":
    guardar_datasets()
