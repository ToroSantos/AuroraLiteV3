import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path

try:
    from objective_templates_optimized import ConfiguracionCapaV7, TemplateObjetivoV7, CategoriaObjetivo, NivelComplejidad, ModoActivacion
    TEMPLATES_AVAILABLE = True
except ImportError:
    TEMPLATES_AVAILABLE = False
    class ConfiguracionCapaV7:
        def __init__(self, **kwargs):
            for k, v in kwargs.items(): setattr(self, k, v)
    class CategoriaObjetivo(Enum):
        COGNITIVO = "cognitivo"; EMOCIONAL = "emocional"; ESPIRITUAL = "espiritual"; CREATIVO = "creativo"; TERAPEUTICO = "terapeutico"; FISICO = "fisico"; SOCIAL = "social"; EXPERIMENTAL = "experimental"

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class CampoCosciencia(Enum):
    COGNITIVO = "cognitivo"; EMOCIONAL = "emocional"; ESPIRITUAL = "espiritual"; FISICO = "fisico"; ENERGETICO = "energetico"; SOCIAL = "social"; CREATIVO = "creativo"; SANACION = "sanacion"

class NivelActivacion(Enum):
    SUTIL = "sutil"; MODERADO = "moderado"; INTENSO = "intenso"; PROFUNDO = "profundo"; TRASCENDENTE = "trascendente"

class TipoRespuesta(Enum):
    INMEDIATA = "inmediata"; PROGRESIVA = "progresiva"; PROFUNDA = "profunda"; INTEGRATIVA = "integrativa"

@dataclass
class ConfiguracionNeuroacustica:
    beat_primario: float = 10.0; beat_secundario: Optional[float] = None; armónicos: List[float] = field(default_factory=list); modulacion_amplitude: float = 0.5; modulacion_frecuencia: float = 0.1; modulacion_fase: float = 0.0; lateralizacion: float = 0.0; profundidad_espacial: float = 1.0; movimiento_3d: bool = False; patron_movimiento: str = "estatico"; evolucion_activada: bool = False; curva_evolucion: str = "lineal"; tiempo_evolucion_min: float = 5.0; frecuencias_resonancia: List[float] = field(default_factory=list); Q_factor: float = 1.0

@dataclass
class PerfilCampoV7:
    nombre: str; descripcion: str; campo_consciencia: CampoCosciencia; style: str; neurotransmisores_simples: List[str] = field(default_factory=list); beat_base: float = 10.0; neurotransmisores_principales: Dict[str, float] = field(default_factory=dict); neurotransmisores_moduladores: Dict[str, float] = field(default_factory=dict); configuracion_neuroacustica: ConfiguracionNeuroacustica = field(default_factory=ConfiguracionNeuroacustica); ondas_primarias: List[str] = field(default_factory=list); ondas_secundarias: List[str] = field(default_factory=list); patron_ondas: str = "estable"; nivel_activacion: NivelActivacion = NivelActivacion.MODERADO; tipo_respuesta: TipoRespuesta = TipoRespuesta.PROGRESIVA; duracion_efecto_min: int = 15; duracion_optima_min: int = 25; duracion_maxima_min: int = 45; efectos_cognitivos: List[str] = field(default_factory=list); efectos_emocionales: List[str] = field(default_factory=list); efectos_fisicos: List[str] = field(default_factory=list); efectos_energeticos: List[str] = field(default_factory=list); mejores_momentos: List[str] = field(default_factory=list); ambientes_optimos: List[str] = field(default_factory=list); posturas_recomendadas: List[str] = field(default_factory=list); perfiles_sinergicos: List[str] = field(default_factory=list); perfiles_antagonicos: List[str] = field(default_factory=list); secuencia_recomendada: List[str] = field(default_factory=list); parametros_ajustables: List[str] = field(default_factory=list); adaptable_intensidad: bool = True; adaptable_duracion: bool = True; base_cientifica: str = "validado"; estudios_referencia: List[str] = field(default_factory=list); nivel_evidencia: float = 0.8; mecanismo_accion: str = ""; contraindicaciones: List[str] = field(default_factory=list); precauciones: List[str] = field(default_factory=list); poblacion_objetivo: List[str] = field(default_factory=list); version: str = "v7.0"; complejidad_tecnica: str = "medio"; recursos_requeridos: str = "medio"; compatibilidad_v6: bool = True; veces_usado: int = 0; efectividad_promedio: float = 0.0; satisfaccion_usuarios: float = 0.0; reportes_efectos: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self._migrar_datos_v6(); self._validar_configuracion(); self._calcular_metricas_automaticas(); self._configurar_parametros_derivados()
    
    def _migrar_datos_v6(self):
        if self.neurotransmisores_simples and not self.neurotransmisores_principales:
            intensidades = {"Dopamina": 0.8, "Serotonina": 0.7, "GABA": 0.8, "Acetilcolina": 0.7, "Oxitocina": 0.6, "Anandamida": 0.6, "Endorfina": 0.5, "Norepinefrina": 0.6, "Adrenalina": 0.7, "BDNF": 0.5, "Melatonina": 0.8}
            for nt in self.neurotransmisores_simples:
                intensidad = intensidades.get(nt, 0.6)
                if not self.neurotransmisores_principales: self.neurotransmisores_principales[nt.lower()] = intensidad
                else: self.neurotransmisores_moduladores[nt.lower()] = intensidad * 0.7
        if not self.configuracion_neuroacustica.beat_primario: self.configuracion_neuroacustica.beat_primario = self.beat_base
        if not self.ondas_primarias: self.ondas_primarias = self._inferir_ondas_desde_beat(self.beat_base)
    
    def _inferir_ondas_desde_beat(self, beat: float) -> List[str]:
        if beat <= 4: return ["delta"]
        elif beat <= 8: return ["theta"]
        elif beat <= 12: return ["alpha"]
        elif beat <= 30: return ["beta"]
        else: return ["gamma"]
    
    def _validar_configuracion(self):
        if not self.nombre: raise ValueError("Perfil debe tener nombre")
        if self.duracion_efecto_min >= self.duracion_maxima_min: logger.warning(f"Duración efecto >= máxima en {self.nombre}")
        if not 0 <= self.nivel_evidencia <= 1: logger.warning(f"Nivel evidencia fuera de rango en {self.nombre}")
        if self._calcular_coherencia_neuroacustica() < 0.5: logger.warning(f"Baja coherencia neuroacústica en {self.nombre}")
    
    def _calcular_coherencia_neuroacustica(self) -> float:
        mapeo = {"dopamina": ["beta", "gamma"], "serotonina": ["alpha", "theta"], "gaba": ["delta", "theta", "alpha"], "acetilcolina": ["beta", "gamma"], "oxitocina": ["alpha", "theta"], "anandamida": ["theta", "delta"], "endorfina": ["alpha", "beta"], "norepinefrina": ["beta", "gamma"], "adrenalina": ["beta", "gamma"], "bdnf": ["alpha", "beta"], "melatonina": ["delta"]}
        coherencia_total = peso_total = 0.0
        for nt, intensidad in self.neurotransmisores_principales.items():
            if nt.lower() in mapeo:
                ondas_esperadas = mapeo[nt.lower()]
                ondas_actuales = [o.lower() for o in self.ondas_primarias + self.ondas_secundarias]
                coincidencias = len(set(ondas_esperadas) & set(ondas_actuales))
                coherencia_nt = coincidencias / len(ondas_esperadas) if ondas_esperadas else 0
                coherencia_total += coherencia_nt * intensidad; peso_total += intensidad
        return coherencia_total / peso_total if peso_total > 0 else 0.5
    
    def _calcular_metricas_automaticas(self):
        fc = 0
        if len(self.neurotransmisores_principales) > 2: fc += 1
        if len(self.neurotransmisores_moduladores) > 1: fc += 1
        if self.configuracion_neuroacustica.evolucion_activada: fc += 1
        if self.configuracion_neuroacustica.movimiento_3d: fc += 1
        if len(self.configuracion_neuroacustica.frecuencias_resonancia) > 2: fc += 1
        if fc <= 1: self.complejidad_tecnica = self.recursos_requeridos = "bajo"
        elif fc <= 3: self.complejidad_tecnica = self.recursos_requeridos = "medio"
        else: self.complejidad_tecnica = self.recursos_requeridos = "alto"
    
    def _configurar_parametros_derivados(self):
        if not self.configuracion_neuroacustica.armónicos:
            beat = self.configuracion_neuroacustica.beat_primario
            self.configuracion_neuroacustica.armónicos = [beat * 2, beat * 3, beat * 4]
        if not self.configuracion_neuroacustica.beat_secundario: self.configuracion_neuroacustica.beat_secundario = self.configuracion_neuroacustica.beat_primario * 1.618
        if not self.parametros_ajustables:
            self.parametros_ajustables = ["intensidad", "duracion", "profundidad_espacial"]
            if self.configuracion_neuroacustica.evolucion_activada: self.parametros_ajustables.append("velocidad_evolucion")
            if len(self.neurotransmisores_principales) > 1: self.parametros_ajustables.append("balance_neurotransmisores")

class GestorPerfilesCampo:
    def __init__(self):
        self.perfiles: Dict[str, PerfilCampoV7] = {}; self.categorias: Dict[CampoCosciencia, List[str]] = {}; self.sinergias_mapeadas: Dict[str, Dict[str, float]] = {}; self.cache_recomendaciones: Dict[str, Any] = {}
        self._migrar_perfiles_v6(); self._crear_perfiles_v7_exclusivos(); self._calcular_sinergias_automaticas(); self._organizar_por_categorias()
        logger.info(f"GestorPerfilesCampo V7 inicializado con {len(self.perfiles)} perfiles")
    
    def _migrar_perfiles_v6(self):
        pd = {
            "autoestima": {"v6": {"style": "luminoso", "nt": ["Dopamina", "Serotonina"], "beat": 10}, "v7": {"descripcion": "Fortalecimiento del sentido de valor personal y confianza interior", "campo_consciencia": CampoCosciencia.EMOCIONAL, "nivel_activacion": NivelActivacion.MODERADO, "efectos_emocionales": ["Confianza elevada", "Autoestima saludable"], "mejores_momentos": ["mañana", "tarde"], "ambientes_optimos": ["privado"], "nivel_evidencia": 0.88}},
            "memoria": {"v6": {"style": "etereo", "nt": ["BDNF"], "beat": 12}, "v7": {"descripcion": "Optimización de la consolidación y recuperación de memoria", "campo_consciencia": CampoCosciencia.COGNITIVO, "nivel_activacion": NivelActivacion.MODERADO, "efectos_cognitivos": ["Memoria mejorada", "Consolidación acelerada"], "mejores_momentos": ["mañana"], "ambientes_optimos": ["silencioso"], "nivel_evidencia": 0.92}},
            "sueño": {"v6": {"style": "sereno", "nt": ["GABA"], "beat": 4}, "v7": {"descripcion": "Inducción de relajación profunda y preparación para el sueño reparador", "campo_consciencia": CampoCosciencia.FISICO, "nivel_activacion": NivelActivacion.SUTIL, "tipo_respuesta": TipoRespuesta.PROFUNDA, "efectos_fisicos": ["Relajación muscular", "Sueño reparador"], "mejores_momentos": ["noche"], "nivel_evidencia": 0.95}},
            "flow_creativo": {"v6": {"style": "transferencia_datos", "nt": ["Dopamina"], "beat": 13}, "v7": {"descripcion": "Estado de flujo optimizado para creatividad", "campo_consciencia": CampoCosciencia.CREATIVO, "nivel_activacion": NivelActivacion.INTENSO, "efectos_cognitivos": ["Estado de flujo", "Creatividad expandida"], "mejores_momentos": ["mañana"], "nivel_evidencia": 0.85}},
            "meditacion": {"v6": {"style": "vacio_cuantico", "nt": ["Oxitocina", "GABA"], "beat": 6}, "v7": {"descripcion": "Estado meditativo profundo con conexión interior", "campo_consciencia": CampoCosciencia.ESPIRITUAL, "nivel_activacion": NivelActivacion.PROFUNDO, "tipo_respuesta": TipoRespuesta.PROFUNDA, "efectos_emocionales": ["Paz profunda", "Conexión interior"], "mejores_momentos": ["mañana_temprano"], "nivel_evidencia": 0.93}},
            "claridad_mental": {"v6": {"style": "minimalista", "nt": ["Acetilcolina", "Dopamina"], "beat": 14}, "v7": {"descripcion": "Optimización cognitiva para pensamiento claro", "campo_consciencia": CampoCosciencia.COGNITIVO, "nivel_activacion": NivelActivacion.MODERADO, "efectos_cognitivos": ["Claridad mental", "Concentración"], "mejores_momentos": ["mañana"], "nivel_evidencia": 0.90}},
            "expansion_consciente": {"v6": {"style": "mistico", "nt": ["Serotonina", "Anandamida"], "beat": 7}, "v7": {"descripcion": "Expansión de la consciencia", "campo_consciencia": CampoCosciencia.ESPIRITUAL, "nivel_activacion": NivelActivacion.TRASCENDENTE, "tipo_respuesta": TipoRespuesta.INTEGRATIVA, "efectos_emocionales": ["Expansión consciencia"], "mejores_momentos": ["noche"], "nivel_evidencia": 0.75}},
            "enraizamiento": {"v6": {"style": "tribal", "nt": ["Adrenalina", "Endorfina"], "beat": 10}, "v7": {"descripcion": "Conexión profunda con la tierra", "campo_consciencia": CampoCosciencia.ENERGETICO, "nivel_activacion": NivelActivacion.INTENSO, "efectos_fisicos": ["Vitalidad", "Energía terrestre"], "mejores_momentos": ["mañana"], "nivel_evidencia": 0.82}},
            "autocuidado": {"v6": {"style": "organico", "nt": ["Oxitocina", "GABA", "Serotonina"], "beat": 6.5}, "v7": {"descripcion": "Activación del sistema de autocuidado", "campo_consciencia": CampoCosciencia.SANACION, "nivel_activacion": NivelActivacion.SUTIL, "tipo_respuesta": TipoRespuesta.INTEGRATIVA, "efectos_emocionales": ["Amor propio"], "mejores_momentos": ["tarde"], "nivel_evidencia": 0.87}},
            "liberacion_emocional": {"v6": {"style": "warm_dust", "nt": ["GABA", "Norepinefrina"], "beat": 8}, "v7": {"descripcion": "Liberación segura de emociones bloqueadas", "campo_consciencia": CampoCosciencia.SANACION, "nivel_activacion": NivelActivacion.MODERADO, "tipo_respuesta": TipoRespuesta.PROFUNDA, "efectos_emocionales": ["Liberación emocional"], "mejores_momentos": ["tarde"], "nivel_evidencia": 0.88}},
            "conexion_espiritual": {"v6": {"style": "alienigena", "nt": ["Anandamida", "Oxitocina"], "beat": 5}, "v7": {"descripcion": "Conexión profunda con la dimensión espiritual", "campo_consciencia": CampoCosciencia.ESPIRITUAL, "nivel_activacion": NivelActivacion.TRASCENDENTE, "tipo_respuesta": TipoRespuesta.INTEGRATIVA, "efectos_emocionales": ["Amor universal"], "mejores_momentos": ["noche"], "nivel_evidencia": 0.70}},
            "gozo_vital": {"v6": {"style": "sutil", "nt": ["Dopamina", "Endorfina"], "beat": 11}, "v7": {"descripcion": "Activación del gozo natural", "campo_consciencia": CampoCosciencia.EMOCIONAL, "nivel_activacion": NivelActivacion.MODERADO, "efectos_emocionales": ["Gozo natural"], "mejores_momentos": ["mañana"], "nivel_evidencia": 0.84}}
        }
        
        for nombre, data in pd.items():
            v6, v7 = data["v6"], data["v7"]
            config = ConfiguracionNeuroacustica(beat_primario=float(v6["beat"]), modulacion_amplitude=0.5, evolucion_activada=nombre in ["expansion_consciente", "conexion_espiritual"])
            nt_p, nt_m = {}, {}
            for i, nt in enumerate(v6["nt"]):
                intensidad = 0.8 if i == 0 else 0.6
                (nt_p if i == 0 else nt_m)[nt.lower()] = intensidad
            
            self.perfiles[nombre] = PerfilCampoV7(nombre=nombre, style=v6["style"], neurotransmisores_simples=v6["nt"], beat_base=float(v6["beat"]), neurotransmisores_principales=nt_p, neurotransmisores_moduladores=nt_m, configuracion_neuroacustica=config, **v7)
    
    def _crear_perfiles_v7_exclusivos(self):
        pv7 = [
            ("coherencia_cuantica", ConfiguracionNeuroacustica(beat_primario=12.0, beat_secundario=19.47, armónicos=[24.0, 36.0, 48.0], modulacion_amplitude=0.618, evolucion_activada=True, curva_evolucion="fibonacci", movimiento_3d=True, patron_movimiento="espiral_dorada", frecuencias_resonancia=[432.0, 528.0, 741.0]), {"descripcion": "Coherencia cuántica y sincronización", "campo_consciencia": CampoCosciencia.ENERGETICO, "style": "cuantico_cristalino", "neurotransmisores_principales": {"anandamida": 0.9, "dopamina": 0.7}, "ondas_primarias": ["gamma", "theta"], "nivel_activacion": NivelActivacion.TRASCENDENTE, "tipo_respuesta": TipoRespuesta.INTEGRATIVA, "duracion_optima_min": 45, "efectos_cognitivos": ["Pensamiento cuántico"], "mejores_momentos": ["luna_nueva"], "nivel_evidencia": 0.68}),
            ("regeneracion_celular", ConfiguracionNeuroacustica(beat_primario=7.83, armónicos=[15.66, 23.49], evolucion_activada=True, frecuencias_resonancia=[174.0, 285.0]), {"descripcion": "Regeneración celular profunda", "campo_consciencia": CampoCosciencia.SANACION, "style": "medicina_frequencial", "neurotransmisores_principales": {"bdnf": 0.9, "serotonina": 0.8}, "ondas_primarias": ["alpha", "theta"], "nivel_activacion": NivelActivacion.PROFUNDO, "duracion_optima_min": 60, "efectos_fisicos": ["Regeneración celular"], "mejores_momentos": ["noche"], "nivel_evidencia": 0.85}),
            ("hipnosis_generativa", ConfiguracionNeuroacustica(beat_primario=4.5, beat_secundario=6.0, evolucion_activada=True, movimiento_3d=True, patron_movimiento="espiral_descendente"), {"descripcion": "Estado hipnótico profundo", "campo_consciencia": CampoCosciencia.COGNITIVO, "style": "hipnotico_profundo", "neurotransmisores_principales": {"gaba": 0.9, "serotonina": 0.8}, "ondas_primarias": ["theta", "delta"], "nivel_activacion": NivelActivacion.PROFUNDO, "duracion_optima_min": 50, "efectos_cognitivos": ["Estado hipnótico"], "mejores_momentos": ["noche"], "nivel_evidencia": 0.90}),
            ("activacion_pineal", ConfiguracionNeuroacustica(beat_primario=6.3, beat_secundario=111.0, evolucion_activada=True, frecuencias_resonancia=[936.0, 963.0]), {"descripcion": "Activación glándula pineal", "campo_consciencia": CampoCosciencia.ESPIRITUAL, "style": "activacion_glandular", "neurotransmisores_principales": {"melatonina": 0.8, "anandamida": 0.7}, "ondas_primarias": ["theta"], "nivel_activacion": NivelActivacion.INTENSO, "duracion_optima_min": 40, "efectos_cognitivos": ["Intuición expandida"], "mejores_momentos": ["noche"], "nivel_evidencia": 0.72})
        ]
        
        for nombre, config, datos in pv7: self.perfiles[nombre] = PerfilCampoV7(nombre=nombre, configuracion_neuroacustica=config, **datos)
    
    def _calcular_sinergias_automaticas(self):
        np = list(self.perfiles.keys())
        for i, n1 in enumerate(np):
            self.sinergias_mapeadas[n1] = {}
            for j, n2 in enumerate(np):
                if i != j:
                    s = self._calcular_sinergia_entre_perfiles(self.perfiles[n1], self.perfiles[n2])
                    self.sinergias_mapeadas[n1][n2] = s
                    if s > 0.7 and n2 not in self.perfiles[n1].perfiles_sinergicos: self.perfiles[n1].perfiles_sinergicos.append(n2)
                    elif s < 0.3 and n2 not in self.perfiles[n1].perfiles_antagonicos: self.perfiles[n1].perfiles_antagonicos.append(n2)
    
    def _calcular_sinergia_entre_perfiles(self, p1: PerfilCampoV7, p2: PerfilCampoV7) -> float:
        s = 0.0
        nt1, nt2 = set(p1.neurotransmisores_principales.keys()), set(p2.neurotransmisores_principales.keys())
        s += len(nt1 & nt2) * 0.2
        o1 = set([o.lower() for o in p1.ondas_primarias + p1.ondas_secundarias])
        o2 = set([o.lower() for o in p2.ondas_primarias + p2.ondas_secundarias])
        s += len(o1 & o2) * 0.15
        cr = {(CampoCosciencia.COGNITIVO, CampoCosciencia.CREATIVO): 0.3, (CampoCosciencia.EMOCIONAL, CampoCosciencia.SANACION): 0.3, (CampoCosciencia.ESPIRITUAL, CampoCosciencia.ENERGETICO): 0.4, (CampoCosciencia.FISICO, CampoCosciencia.SANACION): 0.25}
        pc = (p1.campo_consciencia, p2.campo_consciencia)
        if pc in cr: s += cr[pc]
        elif pc[::-1] in cr: s += cr[pc[::-1]]
        comp = {(NivelActivacion.SUTIL, NivelActivacion.MODERADO): 0.2, (NivelActivacion.MODERADO, NivelActivacion.INTENSO): 0.2, (NivelActivacion.SUTIL, NivelActivacion.PROFUNDO): 0.15}
        pn = (p1.nivel_activacion, p2.nivel_activacion)
        if pn in comp: s += comp[pn]
        elif pn[::-1] in comp: s += comp[pn[::-1]]
        if abs(p1.configuracion_neuroacustica.beat_primario - p2.configuracion_neuroacustica.beat_primario) > 10: s -= 0.1
        return max(0.0, min(1.0, s))
    
    def _organizar_por_categorias(self):
        for nombre, perfil in self.perfiles.items():
            cat = perfil.campo_consciencia
            if cat not in self.categorias: self.categorias[cat] = []
            self.categorias[cat].append(nombre)
    
    @lru_cache(maxsize=128)
    def obtener_perfil(self, nombre: str) -> Optional[PerfilCampoV7]: return self.perfiles.get(nombre.lower())
    
    def buscar_perfiles(self, criterios: Dict[str, Any]) -> List[PerfilCampoV7]:
        r, p = [], []
        for perfil in self.perfiles.values():
            punt = self._calcular_relevancia_perfil(perfil, criterios)
            if punt > 0.2: r.append(perfil); p.append(punt)
        return [pe for _, pe in sorted(zip(p, r), reverse=True)]
    
    def _calcular_relevancia_perfil(self, perfil: PerfilCampoV7, criterios: Dict[str, Any]) -> float:
        p = 0.0
        if "campo" in criterios and perfil.campo_consciencia.value == criterios["campo"]: p += 0.3
        if "efectos" in criterios:
            eb = [e.lower() for e in criterios["efectos"]]
            te = [e.lower() for e in perfil.efectos_cognitivos + perfil.efectos_emocionales + perfil.efectos_fisicos + perfil.efectos_energeticos]
            p += sum(1 for e in eb if any(e in ef for ef in te)) * 0.2
        if "neurotransmisores" in criterios:
            nb = set(nt.lower() for nt in criterios["neurotransmisores"])
            np = set(perfil.neurotransmisores_principales.keys())
            p += len(nb & np) * 0.15
        if "intensidad" in criterios and perfil.nivel_activacion.value == criterios["intensidad"]: p += 0.2
        if "duracion_max" in criterios and perfil.duracion_efecto_min <= criterios["duracion_max"]: p += 0.1
        if "experiencia" in criterios:
            exp = criterios["experiencia"]
            if ((exp == "principiante" and perfil.complejidad_tecnica == "bajo") or (exp == "avanzado" and perfil.complejidad_tecnica == "alto")): p += 0.1
        return p
    
    def recomendar_secuencia_perfiles(self, objetivo: str, duracion_total: int = 60) -> List[Tuple[str, int]]:
        pr = self._mapear_objetivo_a_perfiles(objetivo)
        if not pr: return []
        sec, tu = [], 0
        for np in pr:
            perfil = self.obtener_perfil(np)
            if not perfil: continue
            tr = duracion_total - tu
            if tr < perfil.duracion_efecto_min: break
            dp = min(perfil.duracion_optima_min, tr)
            sec.append((np, dp)); tu += dp
            if tu >= duracion_total: break
        return sec
    
    def _mapear_objetivo_a_perfiles(self, objetivo: str) -> List[str]:
        ol = objetivo.lower()
        mapeos = {"concentracion": ["claridad_mental", "memoria"], "relajacion": ["sueño", "meditacion"], "creatividad": ["flow_creativo", "expansion_consciente"], "sanacion": ["regeneracion_celular", "liberacion_emocional"], "meditacion": ["meditacion", "expansion_consciente"], "autoestima": ["autoestima", "gozo_vital"], "energia": ["enraizamiento", "gozo_vital"], "transformacion": ["coherencia_cuantica", "activacion_pineal"], "sueño": ["sueño", "regeneracion_celular"], "estudio": ["claridad_mental", "memoria"]}
        for clave, perfiles in mapeos.items():
            if clave in ol: return perfiles
        pf = []
        for nombre, perfil in self.perfiles.items():
            if any(palabra in ol for palabra in nombre.split("_")): pf.append(nombre)
        return pf[:3]
    
    def generar_perfil_personalizado(self, nombre: str, objetivo: str, parametros: Dict[str, Any]) -> PerfilCampoV7:
        campo = self._inferir_campo_consciencia(objetivo); nt = self._inferir_neurotransmisores_objetivo(objetivo); bb = parametros.get("beat_base", 10.0)
        config = ConfiguracionNeuroacustica(beat_primario=bb, modulacion_amplitude=parametros.get("intensidad", 0.5), evolucion_activada=parametros.get("evolucion", False), movimiento_3d=parametros.get("movimiento_3d", False))
        return PerfilCampoV7(nombre=nombre, descripcion=f"Perfil personalizado para {objetivo}", campo_consciencia=campo, style=parametros.get("style", "personalizado"), neurotransmisores_principales=nt, configuracion_neuroacustica=config, nivel_activacion=NivelActivacion(parametros.get("nivel_activacion", "moderado")), duracion_optima_min=parametros.get("duracion", 25), base_cientifica="personalizado", nivel_evidencia=0.7)
    
    def _inferir_campo_consciencia(self, objetivo: str) -> CampoCosciencia:
        ol = objetivo.lower()
        if any(p in ol for p in ["pensar", "concentrar", "estudiar", "memoria"]): return CampoCosciencia.COGNITIVO
        elif any(p in ol for p in ["sentir", "emocional", "autoestima", "amor"]): return CampoCosciencia.EMOCIONAL
        elif any(p in ol for p in ["espiritual", "meditar", "conectar", "transcender"]): return CampoCosciencia.ESPIRITUAL
        elif any(p in ol for p in ["crear", "arte", "innovar", "inspirar"]): return CampoCosciencia.CREATIVO
        elif any(p in ol for p in ["sanar", "curar", "regenerar", "terapia"]): return CampoCosciencia.SANACION
        elif any(p in ol for p in ["energia", "vital", "fuerza", "enraizar"]): return CampoCosciencia.ENERGETICO
        elif any(p in ol for p in ["dormir", "relajar", "fisico", "cuerpo"]): return CampoCosciencia.FISICO
        else: return CampoCosciencia.EMOCIONAL
    
    def _inferir_neurotransmisores_objetivo(self, objetivo: str) -> Dict[str, float]:
        ol, nt = objetivo.lower(), {}
        if any(p in ol for p in ["concentrar", "enfocar", "claridad"]): nt.update({"acetilcolina": 0.8, "dopamina": 0.6})
        if any(p in ol for p in ["relajar", "calmar", "dormir"]): nt.update({"gaba": 0.8, "serotonina": 0.6})
        if any(p in ol for p in ["crear", "inspirar", "motivar"]): nt.update({"dopamina": 0.8, "anandamida": 0.5})
        if any(p in ol for p in ["amar", "conectar", "compasion"]): nt.update({"oxitocina": 0.8, "serotonina": 0.6})
        if any(p in ol for p in ["energia", "fuerza", "vitalidad"]): nt.update({"adrenalina": 0.7, "endorfina": 0.6})
        return nt if nt else {"serotonina": 0.7, "gaba": 0.5}
    
    def exportar_estadisticas(self) -> Dict[str, Any]:
        return {"version": "v7.0", "total_perfiles": len(self.perfiles), "perfiles_v6_migrados": 12, "perfiles_v7_exclusivos": len(self.perfiles) - 12, "perfiles_por_campo": {c.value: len(p) for c, p in self.categorias.items()}, "neurotransmisores_utilizados": len(set().union(*[list(p.neurotransmisores_principales.keys()) for p in self.perfiles.values()])), "promedio_nivel_evidencia": sum(p.nivel_evidencia for p in self.perfiles.values()) / len(self.perfiles), "sinergias_calculadas": sum(len(s) for s in self.sinergias_mapeadas.values()), "perfiles_alta_complejidad": len([p for p in self.perfiles.values() if p.complejidad_tecnica == "alto"])}

def _generar_field_profiles_v6() -> Dict[str, Dict[str, Any]]:
    gestor = GestorPerfilesCampo(); fp = {}
    pvo = ["autoestima", "memoria", "sueño", "flow_creativo", "meditacion", "claridad_mental", "expansion_consciente", "enraizamiento", "autocuidado", "liberacion_emocional", "conexion_espiritual", "gozo_vital"]
    for nombre in pvo:
        perfil = gestor.obtener_perfil(nombre)
        if perfil: fp[nombre] = {"style": perfil.style, "nt": perfil.neurotransmisores_simples, "beat": perfil.beat_base}
    return fp

FIELD_PROFILES = _generar_field_profiles_v6()

def crear_gestor_perfiles() -> GestorPerfilesCampo: return GestorPerfilesCampo()
def obtener_perfil_campo(nombre: str) -> Optional[PerfilCampoV7]: return crear_gestor_perfiles().obtener_perfil(nombre)
def buscar_perfiles_por_efecto(efecto: str) -> List[PerfilCampoV7]: return crear_gestor_perfiles().buscar_perfiles({"efectos": [efecto]})
def recomendar_secuencia_objetivo(objetivo: str, duracion_min: int = 60) -> List[Tuple[str, int]]: return crear_gestor_perfiles().recomendar_secuencia_perfiles(objetivo, duracion_min)
def crear_perfil_personalizado(nombre: str, objetivo: str, **parametros) -> PerfilCampoV7: return crear_gestor_perfiles().generar_perfil_personalizado(nombre, objetivo, parametros)
def obtener_sinergias_perfil(nombre: str) -> Dict[str, float]: return crear_gestor_perfiles().sinergias_mapeadas.get(nombre, {})
def validar_coherencia_perfil(perfil: PerfilCampoV7) -> Dict[str, Any]:
    coherencia = perfil._calcular_coherencia_neuroacustica()
    return {"coherencia_neuroacustica": coherencia, "valido": coherencia > 0.5, "recomendaciones": ["Excelente coherencia" if coherencia > 0.8 else "Buena coherencia" if coherencia > 0.6 else "Coherencia moderada" if coherencia > 0.4 else "Baja coherencia - revisar configuración"]}

_gestor_global = None
def obtener_gestor_global() -> GestorPerfilesCampo:
    global _gestor_global
    if _gestor_global is None: _gestor_global = GestorPerfilesCampo()
    return _gestor_global

def obtener_perfil(nombre: str) -> Optional[PerfilCampoV7]: return obtener_gestor_global().obtener_perfil(nombre)
def recomendar_secuencia(objetivo: str, duracion_total: int = 60) -> List[Tuple[str, int]]: return obtener_gestor_global().recomendar_secuencia_perfiles(objetivo, duracion_total)

if __name__ == "__main__":
    print("🌟 Aurora V7 - Sistema Unificado de Perfiles de Campo"); print("=" * 60)
    gestor = crear_gestor_perfiles(); stats = gestor.exportar_estadisticas()
    print(f"📊 {stats['total_perfiles']} perfiles disponibles"); print(f"📈 Promedio nivel evidencia: {stats['promedio_nivel_evidencia']:.1%}"); print(f"🧬 {stats['neurotransmisores_utilizados']} neurotransmisores diferentes"); print(f"🔗 {stats['sinergias_calculadas']} sinergias calculadas")
    print("\n📋 Perfiles por campo de consciencia:")
    for campo, count in stats["perfiles_por_campo"].items(): print(f"  • {campo.title()}: {count}")
    perfiles_concentracion = buscar_perfiles_por_efecto("concentración")
    print("\n🔍 Búsqueda por efecto 'concentración':")
    for perfil in perfiles_concentracion[:3]: print(f"  • {perfil.nombre} (evidencia: {perfil.nivel_evidencia:.0%})")
    secuencia = recomendar_secuencia_objetivo("estudio intensivo", 45); tiempo_total = 0
    print("\n🎯 Secuencia recomendada para 'estudio intensivo' (45 min):")
    for nombre, duracion in secuencia: print(f"  • {nombre.replace('_', ' ').title()}: {duracion} min"); tiempo_total += duracion
    print(f"    Total: {tiempo_total} minutos")
    sinergias = obtener_sinergias_perfil("claridad_mental"); sinergias_altas = [(n, v) for n, v in sinergias.items() if v > 0.6]
    print("\n🔗 Sinergias del perfil 'claridad_mental':")
    for nombre, valor in sorted(sinergias_altas, key=lambda x: x[1], reverse=True)[:3]: print(f"  • {nombre.replace('_', ' ').title()}: {valor:.0%}")
    print(f"\n🔄 Compatibilidad V6: {len(FIELD_PROFILES)} perfiles disponibles")
    print("✅ Retrocompatibilidad V6:")
    for nombre, config in list(FIELD_PROFILES.items())[:3]: print(f"  • FIELD_PROFILES['{nombre}'] = {config}")
    print(f"\n✅ Sistema V7 Unificado listo con retrocompatibilidad V6 completa")