"""
Módulo de políticas de contaminación, Hoy No Circula y medidas de tránsito para CDMX.
"""

# Definición de restricciones según holograma, terminación de placa, día y contingencia
def restringido(holograma, terminacion, dia_semana, contingencia=0):
    # dia_semana: 0=lunes, 1=martes, 2=miércoles, 3=jueves, 4=viernes, 5=sábado, 6=domingo
    
    # Holograma exento (00 y 0 con ciertas características o eléctricos/híbridos)
    if holograma == "exento":
        return False
        
    # Fines de semana (sábado y domingo)
    if dia_semana == 5: # Sábado
        if contingencia > 0:
            return True # Doble Hoy No Circula en contingencia los sábados
        # Sábados según terminación (1er y 3er sábado impares, 2do y 4to pares, etc. Simplificado para demo)
        return terminacion % 2 != 0 
    if dia_semana == 6: # Domingo
        return False # Domingos circula libre generalmente salvo contingencia muy alta

    # Contingencia Fase 1: Restricciones adicionales
    if contingencia > 0:
        # En contingencia descansan más hologramas según terminación o engomado
        if holograma in ("1", "2"):
            return True
        if terminacion % 2 == (dia_semana % 2):
            return True

    # Operación normal (Hoy No Circula regular)
    # Lunes: 5 y 6 (Terminación), Martes: 7 y 8, Miércoles: 3 y 4, Jueves: 1 y 2, Viernes: 9 y 0
    descanso_normal = {
        0: (5, 6),
        1: (7, 8),
        2: (3, 4),
        3: (1, 2),
        4: (9, 0)
    }
    
    if holograma in ("1", "2"):
        if dia_semana in descanso_normal and terminacion in descanso_normal[dia_semana]:
            return True
            
    return False

def medidas_transito(categoria):
    medidas = {
        "Buena": [
            "Operación vehicular normal en toda la ZMVM.",
            "Sin restricciones extraordinarias en vías primarias.",
            "Fomento al uso racional del automóvil particular."
        ],
        "Aceptable": [
            "Monitoreo continuo de concentraciones de ozono y PM2.5.",
            "Supervisión de fuentes móviles altamente contaminantes en puntos clave.",
            "Recomendación de revisión vehicular preventiva."
        ],
        "Mala": [
            "Restricción estricta de circulación a vehículos sin holograma o foráneos en horas pico.",
            "Intensificación de operativos de vialidad para evitar congestionamientos severos en cruces principales.",
            "Suspensión de obras viales que generen emisión de polvos."
        ],
        "Muy mala": [
            "Activación de medidas preventivas de contingencia ambiental.",
            "Restricción reforzada de vehículos administrativos y de combustión interna pesada.",
            "Operativos viales coordinados para priorizar carriles de transporte público."
        ],
        "Extremadamente mala": [
            "Declaratoria de Contingencia Ambiental Fase 1 o superior.",
            "Doble Hoy No Circula obligatorio para reducir drásticamente la flota vehicular circulante.",
            "Suspensión temporal de actividades al aire libre y restricciones severas de velocidad y tránsito."
        ]
    }
    return medidas.get(categoria, ["Mantener vigilancia constante de los índices de calidad del aire."])

def GRUPO_DIA(dia):
    return dia