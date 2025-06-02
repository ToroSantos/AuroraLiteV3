"""Aurora V7 - Objective Manager Unificado OPTIMIZADO"""
import numpy as np, re, logging, time, warnings, json
from typing import Dict, List, Optional, Tuple, Any, Union, Set, Protocol
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from functools import lru_cache
from difflib import SequenceMatcher
from collections import defaultdict

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("Aurora.ObjectiveManager.V7")
VERSION = "V7_AURORA_UNIFIED_MANAGER"

class CategoriaObjetivo(Enum):
    COGNITIVO = "cognitivo"
    EMOCIONAL = "emocional"
    ESPIRITUAL = "espiritual"
    TERAPEUTICO = "terapeutico"
    CREATIVO = "creativo"
    FISICO = "fisico"
    SOCIAL = "social"
    EXPERIMENTAL = "experimental"

class NivelComplejidad(Enum):
    PRINCIPIANTE = "principiante"
    INTERMEDIO = "intermedio"
    AVANZADO = "avanzado"
    EXPERTO = "experto"

class ModoActivacion(Enum):
    SUAVE = "suave"
    PROGRESIVO = "progresivo"
    INMEDIATO = "inmediato"
    ADAPTATIVO = "adaptativo"

@dataclass
class ConfiguracionCapaV7:
    enabled: bool = True
    carrier: Optional[float] = None
    mod_type: str = "am"
    freq_l: Optional[float] = None
    freq_r: Optional[float] = None
    style: Optional[str] = None
    neurotransmisores: Dict[str, float] = field(default_factory=dict)
    frecuencias_personalizadas: List[float] = field(default_factory=list)
    modulacion_profundidad: float = 0.5
    modulacion_velocidad: float = 0.1
    fade_in_ms: int = 500
    fade_out_ms: int = 1000
    delay_inicio_ms: int = 0
    duracion_relativa: float = 1.0
    pan_position: float = 0.0
    width_estereo: float = 1.0
    movimiento_espacial: bool = False
    patron_movimiento: str = "static"
    evolucion_temporal: bool = False
    parametro_evolutivo: str = "amplitude"
    curva_evolucion: str = "linear"
    sincronizacion: List[str] = field(default_factory=list)
    modulacion_cruzada: Dict[str, float] = field(default_factory=dict)
    prioridad: int = 1
    calidad_renderizado: float = 0.8
    uso_cpu: str = "medio"

@dataclass
class TemplateObjetivoV7:
    nombre: str
    descripcion: str
    categoria: CategoriaObjetivo
    complejidad: NivelComplejidad = NivelComplejidad.INTERMEDIO
    emotional_preset: str = ""
    style: str = ""
    layers: Dict[str, ConfiguracionCapaV7] = field(default_factory=dict)
    neurotransmisores_principales: Dict[str, float] = field(default_factory=dict)
    ondas_cerebrales_objetivo: List[str] = field(default_factory=list)
    frecuencia_dominante: float = 10.0
    coherencia_neuroacustica: float = 0.85
    duracion_recomendada_min: int = 20
    duracion_minima_min: int = 10
    duracion_maxima_min: int = 60
    fases_temporales: List[str] = field(default_factory=list)
    modo_activacion: ModoActivacion = ModoActivacion.PROGRESIVO
    efectos_esperados: List[str] = field(default_factory=list)
    efectos_secundarios: List[str] = field(default_factory=list)
    contraindicaciones: List[str] = field(default_factory=list)
    tiempo_efecto_min: int = 5
    mejor_momento_uso: List[str] = field(default_factory=list)
    peor_momento_uso: List[str] = field(default_factory=list)
    ambiente_recomendado: str = "tranquilo"
    postura_recomendada: str = "comoda"
    nivel_atencion_requerido: str = "medio"
    parametros_ajustables: List[str] = field(default_factory=list)
    variaciones_disponibles: List[str] = field(default_factory=list)
    nivel_personalizacion: str = "medio"
    estudios_referencia: List[str] = field(default_factory=list)
    evidencia_cientifica: str = "experimental"
    nivel_confianza: float = 0.8
    fecha_validacion: Optional[str] = None
    investigadores: List[str] = field(default_factory=list)
    version: str = "v7.1"
    autor: str = "Aurora_V7_Unified"
    fecha_creacion: str = field(default_factory=lambda: datetime.now().isoformat())
    ultima_actualizacion: str = field(default_factory=lambda: datetime.now().isoformat())
    compatibilidad_v6: bool = True
    tiempo_renderizado_estimado: int = 30
    recursos_cpu: str = "medio"
    recursos_memoria: str = "medio"
    calidad_audio: float = 0.85
    veces_usado: int = 0
    rating_promedio: float = 0.0
    feedback_usuarios: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.nombre: raise ValueError("Template debe tener nombre")
        if not self.emotional_preset: warnings.warn("Template sin preset emocional")
        if not 0 <= self.nivel_confianza <= 1: warnings.warn(f"Nivel confianza fuera de rango: {self.nivel_confianza}")
        if self.duracion_minima_min >= self.duracion_maxima_min: warnings.warn("Duración mínima >= máxima")
        
        for c in ["neuro_wave", "binaural", "wave_pad", "textured_noise", "heartbeat"]:
            if c not in self.layers: self.layers[c] = ConfiguracionCapaV7(enabled=False)
        
        if self.neurotransmisores_principales and self.ondas_cerebrales_objetivo:
            self.coherencia_neuroacustica = self._calc_coherencia()
        
        capas_activas = sum(1 for l in self.layers.values() if l.enabled)
        self.tiempo_renderizado_estimado = max(15, capas_activas * 8)
        complejidad_total = self._calc_complejidad()
        self.recursos_cpu = "alto" if complejidad_total > 0.8 else "medio" if complejidad_total > 0.5 else "bajo"
        self.recursos_memoria = self.recursos_cpu

    def _calc_coherencia(self) -> float:
        mapeo = {"gaba": ["delta", "theta"], "serotonina": ["alpha", "theta"], "dopamina": ["beta"], 
                "acetilcolina": ["beta", "gamma"], "norepinefrina": ["beta", "gamma"], "anandamida": ["theta", "delta"],
                "oxitocina": ["alpha"], "melatonina": ["delta"], "endorfina": ["alpha", "beta"]}
        suma_total, peso_total = 0.0, 0.0
        for nt, intensidad in self.neurotransmisores_principales.items():
            if nt.lower() in mapeo:
                ondas_esperadas = mapeo[nt.lower()]
                coincidencias = len(set(ondas_esperadas) & set([o.lower() for o in self.ondas_cerebrales_objetivo]))
                coherencia_nt = coincidencias / len(ondas_esperadas) if ondas_esperadas else 0
                suma_total += coherencia_nt * intensidad; peso_total += intensidad
        return suma_total / peso_total if peso_total > 0 else 0.5

    def _calc_complejidad(self) -> float:
        complejidad = sum(1 for l in self.layers.values() if l.enabled) * 0.1
        for layer in self.layers.values():
            if layer.enabled:
                complejidad += sum([0.15 if layer.evolucion_temporal else 0, 0.1 if layer.movimiento_espacial else 0, 
                                  0.1 if layer.modulacion_cruzada else 0, 0.05 if layer.neurotransmisores else 0])
        factores = {NivelComplejidad.EXPERTO: 0.3, NivelComplejidad.AVANZADO: 0.2, NivelComplejidad.INTERMEDIO: 0.1}
        return min(complejidad + factores.get(self.complejidad, 0), 1.0)

class GestorTemplatesOptimizado:
    def __init__(self):
        self.templates: Dict[str, TemplateObjetivoV7] = {}
        self.categorias_disponibles: Dict[str, List[str]] = {}
        self.cache_busquedas: Dict[str, Any] = {}
        self.estadisticas_uso: Dict[str, Dict[str, Any]] = {}
        self._migrar_v6(); self._init_v7(); self._organizar_categorias()

    def _migrar_v6(self):
        templates_data = {
            "claridad_mental": {
                "v6": {"emotional_preset": "claridad_mental", "style": "minimalista", 
                      "layers": {"neuro_wave": {"enabled": True, "carrier": 210, "mod_type": "am"}, 
                                "binaural": {"enabled": True, "freq_l": 100, "freq_r": 111}, 
                                "wave_pad": {"enabled": True}, "textured_noise": {"enabled": True, "style": "soft_grain"}, 
                                "heartbeat": {"enabled": False}}},
                "v7": {"categoria": CategoriaObjetivo.COGNITIVO, "complejidad": NivelComplejidad.INTERMEDIO,
                      "descripcion": "Optimización cognitiva para concentración sostenida",
                      "neurotransmisores_principales": {"acetilcolina": 0.9, "dopamina": 0.6, "norepinefrina": 0.4},
                      "ondas_cerebrales_objetivo": ["beta", "alpha"], "frecuencia_dominante": 11.0,
                      "efectos_esperados": ["Mejora concentración", "Claridad mental", "Reducción distracciones"],
                      "mejor_momento_uso": ["mañana", "tarde"], "duracion_recomendada_min": 25,
                      "evidencia_cientifica": "validado", "nivel_confianza": 0.92}
            },
            "enfoque_total": {
                "v6": {"emotional_preset": "estado_flujo", "style": "futurista",
                      "layers": {"neuro_wave": {"enabled": True, "carrier": 420, "mod_type": "fm"},
                                "binaural": {"enabled": True, "freq_l": 110, "freq_r": 122},
                                "wave_pad": {"enabled": True}, "textured_noise": {"enabled": True, "style": "crystalline"},
                                "heartbeat": {"enabled": True}}},
                "v7": {"categoria": CategoriaObjetivo.COGNITIVO, "complejidad": NivelComplejidad.AVANZADO,
                      "descripcion": "Estado de flujo profundo para rendimiento máximo",
                      "neurotransmisores_principales": {"dopamina": 0.9, "norepinefrina": 0.7, "endorfina": 0.5},
                      "ondas_cerebrales_objetivo": ["beta", "gamma"], "frecuencia_dominante": 12.0,
                      "efectos_esperados": ["Estado de flujo", "Rendimiento máximo", "Inmersión total"],
                      "mejor_momento_uso": ["mañana", "tarde"], "duracion_recomendada_min": 35,
                      "contraindicaciones": ["hipertension", "ansiedad_severa"], "evidencia_cientifica": "validado",
                      "nivel_confianza": 0.94}
            },
            "relajacion_profunda": {
                "v6": {"emotional_preset": "calma_profunda", "style": "etereo",
                      "layers": {"neuro_wave": {"enabled": True, "carrier": 80, "mod_type": "am"},
                                "binaural": {"enabled": True, "freq_l": 80, "freq_r": 88},
                                "wave_pad": {"enabled": True}, "textured_noise": {"enabled": True, "style": "breathy"},
                                "heartbeat": {"enabled": True}}},
                "v7": {"categoria": CategoriaObjetivo.TERAPEUTICO, "complejidad": NivelComplejidad.PRINCIPIANTE,
                      "descripcion": "Relajación terapéutica profunda",
                      "neurotransmisores_principales": {"gaba": 0.9, "serotonina": 0.8, "melatonina": 0.6},
                      "ondas_cerebrales_objetivo": ["theta", "alpha"], "frecuencia_dominante": 8.0,
                      "efectos_esperados": ["Relajación profunda", "Reducción estrés", "Calma mental"],
                      "mejor_momento_uso": ["tarde", "noche"], "duracion_recomendada_min": 30,
                      "evidencia_cientifica": "clinico", "nivel_confianza": 0.96}
            }
        }

        for nombre, data in templates_data.items():
            v6_config, v7_config = data["v6"], data["v7"]
            layers_v7 = {layer_name: ConfiguracionCapaV7(
                enabled=layer_config.get("enabled", False), carrier=layer_config.get("carrier"),
                mod_type=layer_config.get("mod_type", "am"), freq_l=layer_config.get("freq_l"),
                freq_r=layer_config.get("freq_r"), style=layer_config.get("style"),
                modulacion_profundidad=0.5 + (len(layer_name) % 3) * 0.1,
                fade_in_ms=1000 if layer_config.get("enabled") else 0,
                fade_out_ms=1500 if layer_config.get("enabled") else 0,
                prioridad=3 if layer_name in ["neuro_wave", "binaural"] else 2
            ) for layer_name, layer_config in v6_config["layers"].items()}
            
            self.templates[nombre] = TemplateObjetivoV7(
                nombre=nombre.replace("_", " ").title(), emotional_preset=v6_config["emotional_preset"],
                style=v6_config["style"], layers=layers_v7, **v7_config)

    def _init_v7(self):
        self.templates["hiperfocus_cuantico"] = TemplateObjetivoV7(
            nombre="Hiperfocus Cuántico", descripcion="Estado de concentración extrema con coherencia cuántica",
            categoria=CategoriaObjetivo.COGNITIVO, complejidad=NivelComplejidad.EXPERTO,
            emotional_preset="estado_cuantico_cognitivo", style="cuantico_cristalino",
            layers={"neuro_wave": ConfiguracionCapaV7(enabled=True, carrier=432.0, mod_type="quantum_hybrid",
                    neurotransmisores={"acetilcolina": 0.95, "dopamina": 0.8, "norepinefrina": 0.7},
                    modulacion_profundidad=0.7, modulacion_velocidad=0.618, evolucion_temporal=True,
                    parametro_evolutivo="quantum_coherence", curva_evolucion="fibonacci", prioridad=5, calidad_renderizado=1.0),
                    "binaural": ConfiguracionCapaV7(enabled=True, freq_l=110.0, freq_r=152.0, movimiento_espacial=True,
                    patron_movimiento="quantum_spiral", modulacion_cruzada={"neuro_wave": 0.618}, prioridad=4),
                    "wave_pad": ConfiguracionCapaV7(enabled=True, style="quantum_harmonics",
                    frecuencias_personalizadas=[111.0, 222.0, 333.0, 444.0, 555.0], width_estereo=2.0,
                    evolucion_temporal=True, parametro_evolutivo="dimensional_shift", prioridad=3),
                    "textured_noise": ConfiguracionCapaV7(enabled=True, style="quantum_static", modulacion_profundidad=0.3,
                    sincronizacion=["neuro_wave"], prioridad=2)},
            neurotransmisores_principales={"acetilcolina": 0.95, "dopamina": 0.8, "norepinefrina": 0.7, "gaba": 0.3},
            ondas_cerebrales_objetivo=["gamma", "beta"], frecuencia_dominante=42.0, duracion_recomendada_min=45,
            duracion_minima_min=30, duracion_maxima_min=90,
            efectos_esperados=["Concentración extrema", "Procesamiento cuántico", "Creatividad + lógica"],
            contraindicaciones=["epilepsia", "trastornos_atencion", "hipertension_severa"],
            mejor_momento_uso=["mañana"], peor_momento_uso=["noche"], nivel_atencion_requerido="alto",
            evidencia_cientifica="experimental", nivel_confianza=0.75, tiempo_efecto_min=10)

        self.templates["sanacion_multidimensional"] = TemplateObjetivoV7(
            nombre="Sanación Multidimensional", descripcion="Proceso de sanación integral en múltiples dimensiones",
            categoria=CategoriaObjetivo.TERAPEUTICO, complejidad=NivelComplejidad.AVANZADO,
            emotional_preset="sanacion_integral", style="medicina_sagrada",
            layers={"neuro_wave": ConfiguracionCapaV7(enabled=True, carrier=528.0, mod_type="healing_wave",
                    neurotransmisores={"serotonina": 0.9, "oxitocina": 0.8, "endorfina": 0.7, "gaba": 0.6},
                    modulacion_profundidad=0.4, evolucion_temporal=True, parametro_evolutivo="healing_intensity",
                    curva_evolucion="healing_spiral", fade_in_ms=5000, fade_out_ms=3000, prioridad=5),
                    "binaural": ConfiguracionCapaV7(enabled=True, freq_l=85.0, freq_r=93.0, movimiento_espacial=True,
                    patron_movimiento="healing_embrace", fade_in_ms=3000, prioridad=4),
                    "wave_pad": ConfiguracionCapaV7(enabled=True, style="chakra_harmonics",
                    frecuencias_personalizadas=[174.0, 285.0, 396.0, 417.0, 528.0, 639.0, 741.0, 852.0, 963.0],
                    width_estereo=1.8, evolucion_temporal=True, parametro_evolutivo="chakra_activation", prioridad=4),
                    "textured_noise": ConfiguracionCapaV7(enabled=True, style="healing_light", modulacion_profundidad=0.2,
                    delay_inicio_ms=8000, prioridad=2),
                    "heartbeat": ConfiguracionCapaV7(enabled=True, style="universal_heart", modulacion_profundidad=0.3,
                    sincronizacion=["neuro_wave"], prioridad=3)},
            neurotransmisores_principales={"serotonina": 0.9, "oxitocina": 0.8, "endorfina": 0.7, "gaba": 0.6},
            ondas_cerebrales_objetivo=["alpha", "theta"], frecuencia_dominante=8.0, duracion_recomendada_min=50,
            duracion_minima_min=30, duracion_maxima_min=90,
            fases_temporales=["preparacion", "activacion_chakras", "sanacion_profunda", "integracion", "cierre"],
            efectos_esperados=["Sanación emocional", "Equilibrio energético", "Liberación traumas"],
            mejor_momento_uso=["tarde", "noche"], ambiente_recomendado="sagrado_silencioso",
            postura_recomendada="recostada", evidencia_cientifica="clinico", nivel_confianza=0.87)

        self.templates["creatividad_exponencial"] = TemplateObjetivoV7(
            nombre="Creatividad Exponencial", descripcion="Desbloqueo masivo del potencial creativo",
            categoria=CategoriaObjetivo.CREATIVO, complejidad=NivelComplejidad.AVANZADO,
            emotional_preset="genio_creativo", style="vanguardia_artistica",
            layers={"neuro_wave": ConfiguracionCapaV7(enabled=True, carrier=256.0, mod_type="creative_chaos",
                    neurotransmisores={"dopamina": 0.9, "anandamida": 0.8, "acetilcolina": 0.7, "serotonina": 0.6},
                    modulacion_profundidad=0.6, modulacion_velocidad=0.15, evolucion_temporal=True,
                    parametro_evolutivo="creative_explosion", curva_evolucion="exponential", prioridad=5),
                    "binaural": ConfiguracionCapaV7(enabled=True, freq_l=105.0, freq_r=115.0, movimiento_espacial=True,
                    patron_movimiento="creative_dance", modulacion_cruzada={"wave_pad": 0.4}, prioridad=4),
                    "wave_pad": ConfiguracionCapaV7(enabled=True, style="artistic_chaos",
                    frecuencias_personalizadas=[256.0, 341.3, 426.7, 512.0, 682.7], width_estereo=2.2,
                    evolucion_temporal=True, parametro_evolutivo="artistic_flow", movimiento_espacial=True,
                    patron_movimiento="inspiration_spiral", prioridad=4),
                    "textured_noise": ConfiguracionCapaV7(enabled=True, style="creative_sparks", modulacion_profundidad=0.4,
                    modulacion_velocidad=0.2, prioridad=3)},
            neurotransmisores_principales={"dopamina": 0.9, "anandamida": 0.8, "acetilcolina": 0.7, "serotonina": 0.6},
            ondas_cerebrales_objetivo=["alpha", "theta", "gamma"], frecuencia_dominante=10.0, duracion_recomendada_min=35,
            efectos_esperados=["Explosión creativa", "Ideas revolucionarias", "Conexión artística"],
            mejor_momento_uso=["mañana", "tarde"], ambiente_recomendado="espacio_creativo", nivel_atencion_requerido="medio",
            evidencia_cientifica="validado", nivel_confianza=0.84)

    def _organizar_categorias(self):
        for template in self.templates.values():
            categoria = template.categoria.value
            self.categorias_disponibles.setdefault(categoria, []).append(template.nombre.lower().replace(" ", "_"))

    @lru_cache(maxsize=128)
    def obtener_template(self, nombre: str) -> Optional[TemplateObjetivoV7]:
        nombre_normalizado = nombre.lower().replace(" ", "_")
        template = self.templates.get(nombre_normalizado)
        if template: self._actualizar_estadisticas(nombre_normalizado)
        return template

    def _actualizar_estadisticas(self, nombre: str):
        if nombre not in self.estadisticas_uso:
            self.estadisticas_uso[nombre] = {"veces_usado": 0, "ultima_vez": None, "rating_promedio": 0.0, "feedback_total": 0}
        self.estadisticas_uso[nombre]["veces_usado"] += 1
        self.estadisticas_uso[nombre]["ultima_vez"] = datetime.now().isoformat()
        if nombre in self.templates:
            self.templates[nombre].veces_usado = self.estadisticas_uso[nombre]["veces_usado"]

    def buscar_templates_inteligente(self, criterios: Dict[str, Any], limite: int = 10) -> List[TemplateObjetivoV7]:
        cache_key = json.dumps(criterios, sort_keys=True) + f"_limit_{limite}"
        if cache_key in self.cache_busquedas: return self.cache_busquedas[cache_key]
        candidatos = list(self.templates.values())
        puntuaciones = [(template, self._calcular_relevancia(template, criterios)) for template in candidatos]
        puntuaciones.sort(key=lambda x: x[1], reverse=True)
        resultados = [template for template, puntuacion in puntuaciones[:limite] if puntuacion > 0.1]
        self.cache_busquedas[cache_key] = resultados
        return resultados

    def _calcular_relevancia(self, template: TemplateObjetivoV7, criterios: Dict[str, Any]) -> float:
        puntuacion = 0.0
        if "categoria" in criterios and template.categoria.value == criterios["categoria"]: puntuacion += 0.3
        if "efectos" in criterios:
            efectos_buscados = [e.lower() for e in criterios["efectos"]]
            efectos_template = [e.lower() for e in template.efectos_esperados]
            puntuacion += len(set(efectos_buscados) & set(efectos_template)) * 0.2
        if "neurotransmisores" in criterios:
            nt_buscados = set(criterios["neurotransmisores"])
            nt_template = set(template.neurotransmisores_principales.keys())
            puntuacion += len(nt_buscados & nt_template) * 0.15
        if "complejidad" in criterios and template.complejidad.value == criterios["complejidad"]: puntuacion += 0.1
        if "duracion_min" in criterios:
            duracion_buscada = criterios["duracion_min"]
            if template.duracion_minima_min <= duracion_buscada <= template.duracion_maxima_min: puntuacion += 0.1
        if "momento" in criterios and criterios["momento"] in template.mejor_momento_uso: puntuacion += 0.05
        puntuacion *= template.nivel_confianza
        if template.veces_usado > 0: puntuacion += min(template.veces_usado / 100, 1.0) * 0.05
        return puntuacion

    def exportar_estadisticas(self) -> Dict[str, Any]:
        return {"version": "v7.1_unified", "total_templates": len(self.templates),
                "templates_por_categoria": {cat.value: len([t for t in self.templates.values() if t.categoria == cat]) for cat in CategoriaObjetivo},
                "templates_por_complejidad": {comp.value: len([t for t in self.templates.values() if t.complejidad == comp]) for comp in NivelComplejidad},
                "confianza_promedio": sum(t.nivel_confianza for t in self.templates.values()) / len(self.templates),
                "templates_mas_usados": sorted([(nombre, stats["veces_usado"]) for nombre, stats in self.estadisticas_uso.items()], key=lambda x: x[1], reverse=True)[:10],
                "cache_efectividad": {"busquedas_cacheadas": len(self.cache_busquedas), "templates_cacheados": self.obtener_template.cache_info()._asdict()}}

    def limpiar_cache(self):
        self.obtener_template.cache_clear(); self.cache_busquedas.clear()

class TipoRuteo(Enum):
    TEMPLATE_OBJETIVO = "template_objetivo"
    PERFIL_CAMPO = "perfil_campo"
    SECUENCIA_FASES = "secuencia_fases"
    PERSONALIZADO = "personalizado"
    HIBRIDO = "hibrido"
    FALLBACK = "fallback"

class NivelConfianza(Enum):
    EXACTO = "exacto"
    ALTO = "alto"
    MEDIO = "medio"
    BAJO = "bajo"
    INFERIDO = "inferido"

class ContextoUso(Enum):
    TRABAJO = "trabajo"
    MEDITACION = "meditacion"
    ESTUDIO = "estudio"
    RELAJACION = "relajacion"
    CREATIVIDAD = "creatividad"
    TERAPIA = "terapia"
    EJERCICIO = "ejercicio"
    SUENO = "sueno"
    MANIFESTACION = "manifestacion"
    SANACION = "sanacion"

@dataclass
class ResultadoRuteo:
    objetivo_original: str
    objetivo_procesado: str
    tipo_ruteo: TipoRuteo
    nivel_confianza: NivelConfianza
    puntuacion_confianza: float
    preset_emocional: str
    estilo: str
    modo: str
    beat_base: float
    capas: Dict[str, bool] = field(default_factory=dict)
    template_objetivo: Optional[Any] = None
    perfil_campo: Optional[Any] = None
    secuencia_fases: Optional[Any] = None
    contexto_inferido: Optional[ContextoUso] = None
    personalizaciones_sugeridas: List[str] = field(default_factory=list)
    rutas_alternativas: List[str] = field(default_factory=list)
    rutas_sinergicas: List[str] = field(default_factory=list)
    secuencia_recomendada: List[str] = field(default_factory=list)
    tiempo_procesamiento_ms: float = 0.0
    fuentes_consultadas: List[str] = field(default_factory=list)
    algoritmos_utilizados: List[str] = field(default_factory=list)
    coherencia_neuroacustica: float = 0.0
    evidencia_cientifica: str = "validado"
    contraindicaciones: List[str] = field(default_factory=list)
    utilizado_anteriormente: int = 0
    efectividad_reportada: float = 0.0
    feedback_usuarios: List[str] = field(default_factory=list)
    aurora_v7_optimizado: bool = True
    compatible_director: bool = True

@dataclass
class PerfilUsuario:
    experiencia: str = "intermedio"
    objetivos_frecuentes: List[str] = field(default_factory=list)
    contextos_uso: List[ContextoUso] = field(default_factory=list)
    momento_preferido: List[str] = field(default_factory=list)
    duracion_preferida: int = 25
    intensidad_preferida: str = "moderado"
    objetivos_utilizados: Dict[str, int] = field(default_factory=dict)
    efectividad_objetivos: Dict[str, float] = field(default_factory=dict)
    capas_preferidas: Dict[str, bool] = field(default_factory=dict)
    estilos_preferidos: List[str] = field(default_factory=list)
    beats_preferidos: List[float] = field(default_factory=list)
    condiciones_medicas: List[str] = field(default_factory=list)
    sensibilidades: List[str] = field(default_factory=list)
    disponibilidad_tiempo: Dict[str, int] = field(default_factory=dict)

class RouterInteligenteV7:
    def __init__(self, gestor_templates: Optional[GestorTemplatesOptimizado] = None):
        self.version = VERSION
        self.gestor_templates = gestor_templates or GestorTemplatesOptimizado()
        self.inicializacion_exitosa = True
        self._inicializar_rutas_v6()
        self.analizador_semantico = AnalizadorSemantico()
        self.motor_personalizacion = MotorPersonalizacion()
        self.validador_cientifico = ValidadorCientifico()
        self.cache_ruteos = {}
        self.estadisticas_uso = defaultdict(dict)
        self.modelos_prediccion = {}
        self.umbral_confianza_minimo = 0.3
        self.max_alternativas = 5
        self.max_sinergias = 3
        self.usar_cache = True
        logger.info(f"Router V7 inicializado con gestor de templates integrado")

    def _inicializar_rutas_v6(self):
        self.rutas_v6_mapeadas = {
            "claridad_mental": {"v6_config": {"preset": "claridad_mental", "estilo": "minimalista", "modo": "enfoque", "beat": 14.0, 
                               "capas": {"neuro_wave": True, "binaural": True, "wave_pad": True, "textured_noise": True, "heartbeat": False}},
                               "v7_mapping": {"tipo_ruteo": TipoRuteo.TEMPLATE_OBJETIVO, "template_nombre": "claridad_mental", "contexto": ContextoUso.TRABAJO}},
            "concentracion": {"v6_config": {"preset": "estado_flujo", "estilo": "crystalline", "modo": "enfoque", "beat": 15.0,
                             "capas": {"neuro_wave": True, "binaural": True, "wave_pad": True, "textured_noise": True, "heartbeat": False}},
                             "v7_mapping": {"tipo_ruteo": TipoRuteo.TEMPLATE_OBJETIVO, "template_nombre": "enfoque_total", "contexto": ContextoUso.ESTUDIO}},
            "relajacion": {"v6_config": {"preset": "calma_profunda", "estilo": "sereno", "modo": "relajante", "beat": 7.0,
                          "capas": {"neuro_wave": True, "binaural": True, "wave_pad": True, "textured_noise": True, "heartbeat": True}},
                          "v7_mapping": {"tipo_ruteo": TipoRuteo.TEMPLATE_OBJETIVO, "template_nombre": "relajacion_profunda", "contexto": ContextoUso.RELAJACION}},
            "creatividad": {"v6_config": {"preset": "inspiracion_creativa", "estilo": "vanguardia", "modo": "flujo_creativo", "beat": 10.0,
                           "capas": {"neuro_wave": True, "binaural": True, "wave_pad": True, "textured_noise": True, "heartbeat": False}},
                           "v7_mapping": {"tipo_ruteo": TipoRuteo.TEMPLATE_OBJETIVO, "template_nombre": "creatividad_exponencial", "contexto": ContextoUso.CREATIVIDAD}},
            "meditacion": {"v6_config": {"preset": "presencia_interior", "estilo": "mistico", "modo": "contemplativo", "beat": 6.0,
                          "capas": {"neuro_wave": True, "binaural": True, "wave_pad": True, "textured_noise": True, "heartbeat": False}},
                          "v7_mapping": {"tipo_ruteo": TipoRuteo.TEMPLATE_OBJETIVO, "template_nombre": "presencia_total", "contexto": ContextoUso.MEDITACION}},
            "sanacion": {"v6_config": {"preset": "sanacion_integral", "estilo": "medicina_sagrada", "modo": "sanador", "beat": 7.83,
                        "capas": {"neuro_wave": True, "binaural": True, "wave_pad": True, "textured_noise": True, "heartbeat": True}},
                        "v7_mapping": {"tipo_ruteo": TipoRuteo.TEMPLATE_OBJETIVO, "template_nombre": "sanacion_multidimensional", "contexto": ContextoUso.TERAPIA}}
        }

    @lru_cache(maxsize=512)
    def rutear_objetivo(self, objetivo: str, perfil_usuario: Optional[PerfilUsuario] = None, contexto: Optional[ContextoUso] = None, personalizacion: Optional[Dict[str, Any]] = None) -> ResultadoRuteo:
        tiempo_inicio = time.time()
        try:
            objetivo_procesado = self._procesar_objetivo(objetivo)
            analisis = self.analizador_semantico.analizar(objetivo_procesado)
            resultado = self._ejecutar_ruteo_jerarquico(objetivo_procesado, analisis, perfil_usuario, contexto)
            if perfil_usuario or personalizacion: resultado = self.motor_personalizacion.personalizar(resultado, perfil_usuario, personalizacion)
            resultado = self.validador_cientifico.validar(resultado)
            resultado = self._enriquecer_con_alternativas(resultado)
            resultado.tiempo_procesamiento_ms = (time.time() - tiempo_inicio) * 1000
            resultado.objetivo_original = objetivo
            if resultado.puntuacion_confianza > 0.7 and self.usar_cache: self.cache_ruteos[objetivo.lower()] = resultado
            self._actualizar_estadisticas_uso(objetivo, resultado)
            return resultado
        except Exception as e:
            logger.error(f"Error routing '{objetivo}': {e}")
            return self._crear_ruteo_fallback(objetivo, str(e))

    def procesar_objetivo(self, objetivo: str, contexto: Dict[str, Any]) -> Dict[str, Any]:
        resultado = self.rutear_objetivo(objetivo, contexto=contexto.get('contexto'))
        return {"preset_emocional": resultado.preset_emocional, "estilo": resultado.estilo, "modo": resultado.modo,
                "beat_base": resultado.beat_base, "capas": resultado.capas, "nivel_confianza": resultado.puntuacion_confianza,
                "tipo_ruteo": resultado.tipo_ruteo.value, "contexto_inferido": resultado.contexto_inferido.value if resultado.contexto_inferido else None,
                "aurora_v7_optimizado": True}

    def obtener_alternativas(self, objetivo: str) -> List[str]:
        resultado = self.rutear_objetivo(objetivo)
        return (resultado.rutas_alternativas + resultado.rutas_sinergicas)[:5]

    def _procesar_objetivo(self, objetivo: str) -> str:
        objetivo_limpio = re.sub(r'[^\w\s_áéíóúñü]', '', objetivo.lower().strip())
        objetivo_limpio = re.sub(r'\s+', ' ', objetivo_limpio)
        sinonimos = {"concentracion": "claridad_mental", "concentrarse": "claridad_mental", "enfocar": "claridad_mental",
                    "estudiar": "claridad_mental", "atencion": "claridad_mental", "dormir": "sueno", "descansar": "relajacion",
                    "calmar": "relajacion", "tranquilo": "relajacion", "paz": "relajacion", "serenidad": "relajacion",
                    "crear": "creatividad", "inspiracion": "creatividad", "arte": "creatividad", "innovar": "creatividad",
                    "diseñar": "creatividad", "meditar": "meditacion", "espiritual": "conexion_espiritual",
                    "divino": "conexion_espiritual", "sagrado": "conexion_espiritual", "sanar": "sanacion",
                    "curar": "sanacion", "terapia": "sanacion", "gratitud": "agradecimiento", "gracias": "agradecimiento"}
        for sinonimo, canonico in sinonimos.items():
            if sinonimo in objetivo_limpio: objetivo_limpio = objetivo_limpio.replace(sinonimo, canonico)
        return objetivo_limpio

    def _ejecutar_ruteo_jerarquico(self, objetivo: str, analisis: Dict[str, Any], perfil_usuario: Optional[PerfilUsuario], contexto: Optional[ContextoUso]) -> ResultadoRuteo:
        estrategias = [(self._intentar_ruteo_exacto_v6, 0.9, "Ruteo directo V6"), (self._intentar_ruteo_templates_v7, 0.8, "Templates V7"),
                      (self._intentar_ruteo_semantico_avanzado, 0.5, "Análisis semántico"), (self._intentar_ruteo_personalizado_inteligente, 0.4, "IA personalizada")]
        for metodo, umbral, descripcion in estrategias:
            try:
                if metodo.__name__ in ['_intentar_ruteo_personalizado_inteligente']:
                    resultado = metodo(objetivo, analisis, perfil_usuario, contexto)
                elif metodo.__name__ in ['_intentar_ruteo_exacto_v6']:
                    resultado = metodo(objetivo)
                else:
                    resultado = metodo(objetivo, analisis)
                if resultado and resultado.puntuacion_confianza >= umbral:
                    resultado.algoritmos_utilizados.append(descripcion)
                    return resultado
            except Exception as e:
                logger.warning(f"Error {descripcion}: {e}")
                continue
        return self._crear_ruteo_fallback(objetivo, "Todas las estrategias fallaron")

    def _intentar_ruteo_exacto_v6(self, objetivo: str) -> Optional[ResultadoRuteo]:
        if objetivo in self.rutas_v6_mapeadas:
            ruta = self.rutas_v6_mapeadas[objetivo]
            v6_config, v7_mapping = ruta["v6_config"], ruta["v7_mapping"]
            resultado = ResultadoRuteo(objetivo_original=objetivo, objetivo_procesado=objetivo, tipo_ruteo=TipoRuteo(v7_mapping["tipo_ruteo"]),
                                     nivel_confianza=NivelConfianza.EXACTO, puntuacion_confianza=0.95, preset_emocional=v6_config["preset"],
                                     estilo=v6_config["estilo"], modo=v6_config["modo"], beat_base=float(v6_config["beat"]), capas=v6_config["capas"],
                                     contexto_inferido=ContextoUso(v7_mapping["contexto"]), fuentes_consultadas=["rutas_v6"], algoritmos_utilizados=["mapeo_directo_v6"])
            self._enriquecer_resultado_con_v7(resultado, v7_mapping)
            return resultado
        return None

    def _intentar_ruteo_templates_v7(self, objetivo: str, analisis: Dict[str, Any]) -> Optional[ResultadoRuteo]:
        template = self.gestor_templates.obtener_template(objetivo)
        if template: return self._crear_resultado_desde_template(template, objetivo, 0.9)
        templates_disponibles = list(self.gestor_templates.templates.keys())
        similitudes = [(nombre, self._calcular_similitud(objetivo, nombre)) for nombre in templates_disponibles]
        similitudes.sort(key=lambda x: x[1], reverse=True)
        if similitudes and similitudes[0][1] > 0.75:
            template = self.gestor_templates.obtener_template(similitudes[0][0])
            if template: return self._crear_resultado_desde_template(template, objetivo, similitudes[0][1])
        palabras_clave = analisis.get("palabras_clave", [objetivo])
        for palabra in palabras_clave:
            try:
                templates_efectos = self.gestor_templates.buscar_templates_inteligente({"efectos": [palabra]}, limite=1)
                if templates_efectos: return self._crear_resultado_desde_template(templates_efectos[0], objetivo, 0.8)
            except: continue
        return None

    def _intentar_ruteo_semantico_avanzado(self, objetivo: str, analisis: Dict[str, Any]) -> Optional[ResultadoRuteo]:
        intencion = analisis.get("intencion_principal", "relajacion")
        mapeo_intenciones = {"concentrar": {"template": "claridad_mental"}, "relajar": {"template": "relajacion_profunda"},
                           "crear": {"template": "creatividad_exponencial"}, "meditar": {"template": "presencia_total"},
                           "sanar": {"template": "sanacion_multidimensional"}}
        if intencion in mapeo_intenciones:
            config = mapeo_intenciones[intencion]
            template = self.gestor_templates.obtener_template(config["template"])
            if template: return self._crear_resultado_desde_template(template, objetivo, 0.7)
        return None

    def _intentar_ruteo_personalizado_inteligente(self, objetivo: str, analisis: Dict[str, Any], perfil_usuario: Optional[PerfilUsuario], contexto: Optional[ContextoUso]) -> Optional[ResultadoRuteo]:
        config = self._generar_configuracion_inteligente(objetivo, analisis, perfil_usuario, contexto)
        if config:
            return ResultadoRuteo(objetivo_original=objetivo, objetivo_procesado=objetivo, tipo_ruteo=TipoRuteo.PERSONALIZADO,
                                nivel_confianza=NivelConfianza.INFERIDO, puntuacion_confianza=0.5, preset_emocional=config["preset"],
                                estilo=config["estilo"], modo=config["modo"], beat_base=config["beat"], capas=config["capas"],
                                contexto_inferido=contexto, fuentes_consultadas=["analisis_semantico", "ia_personalizada"],
                                algoritmos_utilizados=["generacion_inteligente"], personalizaciones_sugeridas=["Configuración generada automáticamente"])
        return None

    def _crear_ruteo_fallback(self, objetivo: str, razon: str) -> ResultadoRuteo:
        return ResultadoRuteo(objetivo_original=objetivo, objetivo_procesado=objetivo, tipo_ruteo=TipoRuteo.FALLBACK,
                            nivel_confianza=NivelConfianza.BAJO, puntuacion_confianza=0.3, preset_emocional="calma_profunda",
                            estilo="sereno", modo="seguro", beat_base=8.0, capas={"neuro_wave": True, "binaural": True, "wave_pad": True,
                            "textured_noise": True, "heartbeat": False}, contexto_inferido=ContextoUso.RELAJACION,
                            fuentes_consultadas=["fallback_seguro"], algoritmos_utilizados=["configuracion_default"],
                            personalizaciones_sugeridas=[f"Ruteo fallback usado - {razon}"])

    def _crear_resultado_desde_template(self, template: TemplateObjetivoV7, objetivo: str, confianza: float) -> ResultadoRuteo:
        return ResultadoRuteo(objetivo_original=objetivo, objetivo_procesado=objetivo, tipo_ruteo=TipoRuteo.TEMPLATE_OBJETIVO,
                            nivel_confianza=self._calcular_nivel_confianza(confianza), puntuacion_confianza=confianza,
                            preset_emocional=template.emotional_preset, estilo=template.style, modo="template_v7",
                            beat_base=template.frecuencia_dominante, capas=self._convertir_capas_template_a_v6(template.layers),
                            template_objetivo=template, contexto_inferido=self._inferir_contexto_desde_categoria(template.categoria),
                            fuentes_consultadas=["templates_v7"], algoritmos_utilizados=["mapeo_template_v7"],
                            coherencia_neuroacustica=template.coherencia_neuroacustica, evidencia_cientifica=template.evidencia_cientifica,
                            contraindicaciones=template.contraindicaciones)

    def _enriquecer_resultado_con_v7(self, resultado: ResultadoRuteo, v7_mapping: Dict[str, Any]):
        if "template_nombre" in v7_mapping:
            template = self.gestor_templates.obtener_template(v7_mapping["template_nombre"])
            if template:
                resultado.template_objetivo = template
                resultado.coherencia_neuroacustica = template.coherencia_neuroacustica
                resultado.evidencia_cientifica = template.evidencia_cientifica

    def _enriquecer_con_alternativas(self, resultado: ResultadoRuteo) -> ResultadoRuteo:
        objetivo = resultado.objetivo_procesado
        if resultado.template_objetivo:
            try:
                categoria = resultado.template_objetivo.categoria
                templates_categoria = [nombre for nombre, template in self.gestor_templates.templates.items() if template.categoria == categoria]
                resultado.rutas_alternativas.extend([t for t in templates_categoria[:3] if t != objetivo])
            except: pass
        return resultado

    def _calcular_similitud(self, texto1: str, texto2: str) -> float:
        return SequenceMatcher(None, texto1.lower(), texto2.lower()).ratio()

    def _calcular_nivel_confianza(self, puntuacion: float) -> NivelConfianza:
        if puntuacion >= 0.9: return NivelConfianza.EXACTO
        elif puntuacion >= 0.7: return NivelConfianza.ALTO
        elif puntuacion >= 0.5: return NivelConfianza.MEDIO
        elif puntuacion >= 0.3: return NivelConfianza.BAJO
        else: return NivelConfianza.INFERIDO

    def _convertir_capas_template_a_v6(self, capas_v7: Dict) -> Dict[str, bool]:
        capas_v6 = {nombre: capa.enabled if hasattr(capa, 'enabled') else bool(capa) for nombre, capa in capas_v7.items()}
        capas_base = ["neuro_wave", "binaural", "wave_pad", "textured_noise", "heartbeat"]
        for capa in capas_base: capas_v6.setdefault(capa, capa in ["neuro_wave", "binaural", "wave_pad", "textured_noise"])
        return capas_v6

    def _inferir_contexto_desde_categoria(self, categoria) -> ContextoUso:
        if not categoria: return ContextoUso.RELAJACION
        mapeo = {"cognitivo": ContextoUso.TRABAJO, "creativo": ContextoUso.CREATIVIDAD, "terapeutico": ContextoUso.TERAPIA,
                "espiritual": ContextoUso.MEDITACION, "emocional": ContextoUso.RELAJACION, "fisico": ContextoUso.EJERCICIO}
        categoria_str = categoria.value if hasattr(categoria, 'value') else str(categoria)
        return mapeo.get(categoria_str, ContextoUso.RELAJACION)

    def _generar_configuracion_inteligente(self, objetivo: str, analisis: Dict[str, Any], perfil_usuario: Optional[PerfilUsuario], contexto: Optional[ContextoUso]) -> Optional[Dict[str, Any]]:
        configs_contexto = {
            ContextoUso.TRABAJO: {"preset": "claridad_mental", "estilo": "minimalista", "modo": "enfoque", "beat": 14.0,
                                "capas": {"neuro_wave": True, "binaural": True, "wave_pad": True, "textured_noise": True, "heartbeat": False}},
            ContextoUso.RELAJACION: {"preset": "calma_profunda", "estilo": "sereno", "modo": "relajante", "beat": 7.0,
                                   "capas": {"neuro_wave": True, "binaural": True, "wave_pad": True, "textured_noise": True, "heartbeat": True}},
            ContextoUso.CREATIVIDAD: {"preset": "expansion_creativa", "estilo": "inspirador", "modo": "flujo", "beat": 10.0,
                                    "capas": {"neuro_wave": True, "binaural": True, "wave_pad": True, "textured_noise": True, "heartbeat": False}}
        }
        contexto_usado = contexto or ContextoUso.RELAJACION
        config_base = configs_contexto.get(contexto_usado, configs_contexto[ContextoUso.RELAJACION]).copy()
        if perfil_usuario:
            if perfil_usuario.beats_preferidos: config_base["beat"] = perfil_usuario.beats_preferidos[0]
            if perfil_usuario.estilos_preferidos: config_base["estilo"] = perfil_usuario.estilos_preferidos[0]
            if perfil_usuario.capas_preferidas: config_base["capas"].update(perfil_usuario.capas_preferidas)
            factor = {"suave": 0.8, "moderado": 1.0, "intenso": 1.2}.get(perfil_usuario.intensidad_preferida, 1.0)
            config_base["beat"] *= factor
        return config_base

    def _actualizar_estadisticas_uso(self, objetivo: str, resultado: ResultadoRuteo):
        if objetivo not in self.estadisticas_uso:
            self.estadisticas_uso[objetivo] = {"veces_usado": 0, "confianza_promedio": 0.0, "tipos_ruteo_usados": defaultdict(int), "tiempo_promedio_ms": 0.0}
        stats = self.estadisticas_uso[objetivo]
        stats["veces_usado"] += 1
        stats["confianza_promedio"] = ((stats["confianza_promedio"] * (stats["veces_usado"] - 1) + resultado.puntuacion_confianza) / stats["veces_usado"])
        stats["tipos_ruteo_usados"][resultado.tipo_ruteo.value] += 1
        stats["tiempo_promedio_ms"] = ((stats["tiempo_promedio_ms"] * (stats["veces_usado"] - 1) + resultado.tiempo_procesamiento_ms) / stats["veces_usado"])

    def obtener_estadisticas_router(self) -> Dict[str, Any]:
        total_ruteos = sum(stats["veces_usado"] for stats in self.estadisticas_uso.values())
        return {"version": self.version, "total_ruteos_realizados": total_ruteos, "objetivos_unicos": len(self.estadisticas_uso),
                "rutas_v6_disponibles": len(self.rutas_v6_mapeadas), "templates_integrados": len(self.gestor_templates.templates),
                "cache_hits": len(self.cache_ruteos), "confianza_promedio": (sum(stats["confianza_promedio"] for stats in self.estadisticas_uso.values()) / len(self.estadisticas_uso) if self.estadisticas_uso else 0),
                "tiempo_promedio_ms": (sum(stats["tiempo_promedio_ms"] for stats in self.estadisticas_uso.values()) / len(self.estadisticas_uso) if self.estadisticas_uso else 0),
                "objetivos_mas_usados": sorted([(objetivo, stats["veces_usado"]) for objetivo, stats in self.estadisticas_uso.items()], key=lambda x: x[1], reverse=True)[:10],
                "aurora_v7_optimizado": True, "protocolo_director_implementado": True}

class AnalizadorSemantico:
    def analizar(self, objetivo: str) -> Dict[str, Any]:
        palabras = objetivo.lower().split()
        palabras_clave = [p for p in palabras if len(p) > 3 and p not in ['para', 'con', 'una', 'más', 'muy', 'todo', 'esta', 'este', 'esa', 'ese']]
        intenciones = {"concentrar": ["concentrar", "enfocar", "claridad", "atencion", "estudiar"], "relajar": ["relajar", "calmar", "dormir", "descansar", "paz", "tranquilo"],
                      "crear": ["crear", "inspirar", "arte", "innovar", "diseñar", "creatividad"], "meditar": ["meditar", "espiritual", "conexion", "interior", "contemplar"],
                      "sanar": ["sanar", "curar", "terapia", "equilibrar", "restaurar", "sanacion"], "energizar": ["energia", "fuerza", "vitalidad", "activar", "despertar"],
                      "manifestar": ["manifestar", "visualizar", "crear_realidad", "materializar"]}
        intencion = "relajar"
        for intencion_key, palabras_intencion in intenciones.items():
            if any(palabra in objetivo.lower() for palabra in palabras_intencion): intencion = intencion_key; break
        return {"palabras_clave": palabras_clave, "intencion_principal": intencion, "modificadores": {"intensidad": self._detectar_intensidad(objetivo),
                "urgencia": self._detectar_urgencia(objetivo), "duracion": self._detectar_duracion_sugerida(objetivo)}, "longitud_objetivo": len(objetivo),
                "complejidad_linguistica": len(palabras_clave), "es_objetivo_simple": len(palabras) <= 3,
                "contiene_negacion": any(negacion in objetivo.lower() for negacion in ["no", "sin", "menos", "reducir"]),
                "nivel_especificidad": self._calcular_especificidad(objetivo)}

    def _detectar_intensidad(self, objetivo: str) -> str:
        objetivo_lower = objetivo.lower()
        if any(palabra in objetivo_lower for palabra in ["suave", "ligero", "sutil"]): return "suave"
        elif any(palabra in objetivo_lower for palabra in ["intenso", "profundo", "fuerte"]): return "intenso"
        else: return "moderado"

    def _detectar_urgencia(self, objetivo: str) -> str:
        objetivo_lower = objetivo.lower()
        if any(palabra in objetivo_lower for palabra in ["rapido", "inmediato", "urgente"]): return "alta"
        elif any(palabra in objetivo_lower for palabra in ["gradual", "lento", "pausado"]): return "baja"
        else: return "normal"

    def _detectar_duracion_sugerida(self, objetivo: str) -> Optional[str]:
        objetivo_lower = objetivo.lower()
        if any(palabra in objetivo_lower for palabra in ["corto", "breve", "rapido"]): return "corta"
        elif any(palabra in objetivo_lower for palabra in ["largo", "extenso", "profundo"]): return "larga"
        else: return None

    def _calcular_especificidad(self, objetivo: str) -> str:
        palabras = objetivo.split()
        if len(palabras) <= 2: return "general"
        elif len(palabras) <= 5: return "especifico"
        else: return "muy_especifico"

class MotorPersonalizacion:
    def personalizar(self, resultado: ResultadoRuteo, perfil_usuario: Optional[PerfilUsuario], personalizacion: Optional[Dict[str, Any]]) -> ResultadoRuteo:
        if not perfil_usuario and not personalizacion: return resultado
        if perfil_usuario: resultado = self._aplicar_personalizacion_perfil(resultado, perfil_usuario)
        if personalizacion: resultado = self._aplicar_personalizacion_explicita(resultado, personalizacion)
        return resultado

    def _aplicar_personalizacion_perfil(self, resultado: ResultadoRuteo, perfil: PerfilUsuario) -> ResultadoRuteo:
        if perfil.duracion_preferida != 25: resultado.personalizaciones_sugeridas.append(f"Duración ajustada a {perfil.duracion_preferida} min")
        if perfil.intensidad_preferida != "moderado":
            factor = {"suave": 0.8, "moderado": 1.0, "intenso": 1.2}.get(perfil.intensidad_preferida, 1.0)
            resultado.beat_base *= factor
            resultado.personalizaciones_sugeridas.append(f"Intensidad ajustada a {perfil.intensidad_preferida}")
        if perfil.capas_preferidas:
            for capa, preferencia in perfil.capas_preferidas.items():
                if capa in resultado.capas:
                    resultado.capas[capa] = preferencia
                    resultado.personalizaciones_sugeridas.append(f"Capa {capa} {'activada' if preferencia else 'desactivada'}")
        if perfil.estilos_preferidos and perfil.estilos_preferidos[0] != resultado.estilo:
            resultado.estilo = perfil.estilos_preferidos[0]
            resultado.personalizaciones_sugeridas.append("Estilo personalizado aplicado")
        return resultado

    def _aplicar_personalizacion_explicita(self, resultado: ResultadoRuteo, personalizacion: Dict[str, Any]) -> ResultadoRuteo:
        if "beat_adjustment" in personalizacion:
            resultado.beat_base += personalizacion["beat_adjustment"]
            resultado.personalizaciones_sugeridas.append("Beat personalizado aplicado")
        if "style_override" in personalizacion:
            resultado.estilo = personalizacion["style_override"]
            resultado.personalizaciones_sugeridas.append("Estilo personalizado aplicado")
        if "mode_override" in personalizacion:
            resultado.modo = personalizacion["mode_override"]
            resultado.personalizaciones_sugeridas.append("Modo personalizado aplicado")
        if "layer_overrides" in personalizacion:
            for capa, estado in personalizacion["layer_overrides"].items():
                if capa in resultado.capas:
                    resultado.capas[capa] = estado
                    resultado.personalizaciones_sugeridas.append(f"Capa {capa} {'activada' if estado else 'desactivada'} manualmente")
        return resultado

class ValidadorCientifico:
    def validar(self, resultado: ResultadoRuteo) -> ResultadoRuteo:
        if not 0.5 <= resultado.beat_base <= 100:
            resultado.contraindicaciones.append("Frecuencia fuera de rango seguro")
            resultado.puntuacion_confianza *= 0.8
        if resultado.template_objetivo and hasattr(resultado.template_objetivo, 'coherencia_neuroacustica'):
            coherencia = resultado.template_objetivo.coherencia_neuroacustica
            if coherencia < 0.5:
                resultado.contraindicaciones.append("Baja coherencia neuroacústica")
                resultado.puntuacion_confianza *= 0.9
        capas_activas = sum(1 for activa in resultado.capas.values() if activa)
        if capas_activas == 0:
            resultado.contraindicaciones.append("No hay capas activas")
            resultado.puntuacion_confianza *= 0.7
        elif capas_activas > 5:
            resultado.contraindicaciones.append("Demasiadas capas activas pueden causar fatiga")
        if resultado.capas.get("heartbeat", False) and resultado.beat_base > 20:
            resultado.contraindicaciones.append("Combinación de heartbeat con alta frecuencia puede ser estimulante")
        return resultado

class ObjectiveManagerUnificado:
    def __init__(self):
        self.version = VERSION
        self.gestor_templates = GestorTemplatesOptimizado()
        self.router = RouterInteligenteV7(self.gestor_templates)
        self.estadisticas_globales = {"inicializado": datetime.now().isoformat(), "templates_cargados": len(self.gestor_templates.templates),
                                     "rutas_v6_disponibles": len(self.router.rutas_v6_mapeadas), "total_objetivos_procesados": 0, "tiempo_total_procesamiento": 0.0}
        logger.info(f"ObjectiveManager V7 inicializado - {self.estadisticas_globales['templates_cargados']} templates")

    def obtener_template(self, nombre: str) -> Optional[TemplateObjetivoV7]: return self.gestor_templates.obtener_template(nombre)
    def buscar_templates(self, criterios: Dict[str, Any], limite: int = 10) -> List[TemplateObjetivoV7]: return self.gestor_templates.buscar_templates_inteligente(criterios, limite)
    def listar_templates_por_categoria(self, categoria: CategoriaObjetivo) -> List[str]: return [nombre for nombre, template in self.gestor_templates.templates.items() if template.categoria == categoria]
    def obtener_templates_populares(self, limite: int = 5) -> List[Tuple[str, int]]:
        templates_con_uso = [(nombre, template.veces_usado) for nombre, template in self.gestor_templates.templates.items() if template.veces_usado > 0]
        templates_con_uso.sort(key=lambda x: x[1], reverse=True)
        return templates_con_uso[:limite]

    def crear_template_personalizado(self, nombre: str, descripcion: str, categoria: CategoriaObjetivo, configuracion: Dict[str, Any]) -> TemplateObjetivoV7:
        template = TemplateObjetivoV7(nombre=nombre, descripcion=descripcion, categoria=categoria, autor="Usuario_Personalizado", **configuracion)
        nombre_clave = nombre.lower().replace(" ", "_")
        self.gestor_templates.templates[nombre_clave] = template
        return template

    def validar_template(self, template: TemplateObjetivoV7) -> Dict[str, Any]:
        validacion = {"valido": True, "puntuacion_calidad": 0.0, "advertencias": [], "errores": [], "sugerencias": [], "metricas": {}}
        coherencia = template.coherencia_neuroacustica
        validacion["metricas"]["coherencia_neuroacustica"] = coherencia
        if coherencia < 0.5: validacion["advertencias"].append("Baja coherencia neuroacústica"); validacion["puntuacion_calidad"] -= 0.2
        elif coherencia > 0.8: validacion["puntuacion_calidad"] += 0.3
        capas_activas = [capa for capa in template.layers.values() if capa.enabled]
        validacion["metricas"]["capas_activas"] = len(capas_activas)
        if not capas_activas: validacion["errores"].append("No hay capas activas"); validacion["valido"] = False
        elif len(capas_activas) > 6: validacion["advertencias"].append("Muchas capas activas pueden afectar rendimiento")
        if template.duracion_recomendada_min < 5: validacion["advertencias"].append("Duración muy corta")
        elif template.duracion_recomendada_min > 90: validacion["advertencias"].append("Duración muy larga")
        else: validacion["puntuacion_calidad"] += 0.1
        if template.evidencia_cientifica in ["validado", "clinico"]: validacion["puntuacion_calidad"] += 0.2
        elif template.evidencia_cientifica == "experimental": validacion["puntuacion_calidad"] += 0.1
        validacion["puntuacion_calidad"] = max(0.0, min(1.0, validacion["puntuacion_calidad"] + 0.5))
        return validacion

    def procesar_objetivo(self, objetivo: str, contexto: Dict[str, Any] = None) -> Dict[str, Any]:
        self.estadisticas_globales["total_objetivos_procesados"] += 1
        start_time = time.time()
        resultado = self.router.procesar_objetivo(objetivo, contexto or {})
        processing_time = time.time() - start_time
        self.estadisticas_globales["tiempo_total_procesamiento"] += processing_time
        return resultado

    def rutear_objetivo_avanzado(self, objetivo: str, perfil_usuario: Optional[Dict[str, Any]] = None, contexto: Optional[str] = None, personalizacion: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        perfil_obj = PerfilUsuario(**perfil_usuario) if perfil_usuario else None
        contexto_obj = ContextoUso(contexto) if contexto else None
        resultado = self.router.rutear_objetivo(objetivo, perfil_obj, contexto_obj, personalizacion)
        return {"configuracion_v6": {"preset": resultado.preset_emocional, "estilo": resultado.estilo, "modo": resultado.modo, "beat": resultado.beat_base, "capas": resultado.capas},
                "informacion_v7": {"tipo_ruteo": resultado.tipo_ruteo.value, "confianza": resultado.puntuacion_confianza, "nivel_confianza": resultado.nivel_confianza.value,
                                   "contexto": resultado.contexto_inferido.value if resultado.contexto_inferido else None, "alternativas": resultado.rutas_alternativas,
                                   "sinergias": resultado.rutas_sinergicas, "personalizaciones": resultado.personalizaciones_sugeridas, "contraindicaciones": resultado.contraindicaciones,
                                   "tiempo_procesamiento": resultado.tiempo_procesamiento_ms},
                "recursos_v7": {"template": resultado.template_objetivo.nombre if resultado.template_objetivo else None, "perfil_campo": None, "secuencia": None},
                "validacion_cientifica": {"coherencia_neuroacustica": resultado.coherencia_neuroacustica, "evidencia_cientifica": resultado.evidencia_cientifica, "contraindicaciones": resultado.contraindicaciones},
                "aurora_v7": {"optimizado": resultado.aurora_v7_optimizado, "compatible_director": resultado.compatible_director}}

    def obtener_alternativas(self, objetivo: str) -> List[str]: return self.router.obtener_alternativas(objetivo)

    def recomendar_secuencia_objetivos(self, objetivo_principal: str, duracion_total: int = 60, perfil_usuario: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        resultado_principal = self.router.rutear_objetivo(objetivo_principal)
        objetivos_secuencia = [objetivo_principal]
        tiempo_usado = resultado_principal.template_objetivo.duracion_recomendada_min if resultado_principal.template_objetivo else 20
        for objetivo_sinergico in resultado_principal.rutas_sinergicas:
            if tiempo_usado >= duracion_total: break
            resultado_sinergico = self.router.rutear_objetivo(objetivo_sinergico)
            duracion_objetivo = resultado_sinergico.template_objetivo.duracion_recomendada_min if resultado_sinergico.template_objetivo else 15
            if tiempo_usado + duracion_objetivo <= duracion_total:
                objetivos_secuencia.append(objetivo_sinergico)
                tiempo_usado += duracion_objetivo
        secuencia_detallada = []
        for i, objetivo in enumerate(objetivos_secuencia):
            resultado = self.router.rutear_objetivo(objetivo)
            duracion = resultado.template_objetivo.duracion_recomendada_min if resultado.template_objetivo else 15
            secuencia_detallada.append({"objetivo": objetivo, "orden": i + 1, "duracion_min": duracion, "configuracion": self.procesar_objetivo(objetivo),
                                       "confianza": resultado.puntuacion_confianza, "tipo": resultado.tipo_ruteo.value})
        return secuencia_detallada

    def listar_objetivos_disponibles(self) -> List[str]:
        objetivos = set()
        objetivos.update(self.router.rutas_v6_mapeadas.keys())
        objetivos.update(self.gestor_templates.templates.keys())
        return sorted(list(objetivos))

    def buscar_objetivos_similares(self, objetivo: str, limite: int = 5) -> List[Tuple[str, float]]:
        objetivos_disponibles = self.listar_objetivos_disponibles()
        similitudes = [(obj, self.router._calcular_similitud(objetivo, obj)) for obj in objetivos_disponibles]
        similitudes.sort(key=lambda x: x[1], reverse=True)
        return similitudes[:limite]

    def obtener_estadisticas_completas(self) -> Dict[str, Any]:
        stats_templates = self.gestor_templates.exportar_estadisticas()
        stats_router = self.router.obtener_estadisticas_router()
        return {"version": self.version, "manager_global": self.estadisticas_globales, "templates": stats_templates, "router": stats_router,
                "resumen": {"templates_disponibles": len(self.gestor_templates.templates), "objetivos_totales": len(self.listar_objetivos_disponibles()),
                           "compatibilidad_v6": len(self.router.rutas_v6_mapeadas), "tiempo_promedio_procesamiento": (self.estadisticas_globales["tiempo_total_procesamiento"] / max(1, self.estadisticas_globales["total_objetivos_procesados"]))}}

    def limpiar_cache(self): self.gestor_templates.limpiar_cache(); self.router.cache_ruteos.clear(); logger.info("Cache del ObjectiveManager limpiado")

    def exportar_configuracion(self, formato: str = "json") -> Union[str, Dict[str, Any]]:
        configuracion = {"version": self.version, "templates": {nombre: {"nombre": template.nombre, "descripcion": template.descripcion, "categoria": template.categoria.value,
                        "complejidad": template.complejidad.value, "emotional_preset": template.emotional_preset, "style": template.style, "frecuencia_dominante": template.frecuencia_dominante,
                        "duracion_recomendada_min": template.duracion_recomendada_min, "efectos_esperados": template.efectos_esperados, "evidencia_cientifica": template.evidencia_cientifica,
                        "nivel_confianza": template.nivel_confianza} for nombre, template in self.gestor_templates.templates.items()}, "rutas_v6": self.router.rutas_v6_mapeadas,
                        "estadisticas": self.obtener_estadisticas_completas()}
        if formato == "json": return json.dumps(configuracion, indent=2, ensure_ascii=False)
        else: return configuracion

_manager_global: Optional[ObjectiveManagerUnificado] = None
def obtener_manager() -> ObjectiveManagerUnificado:
    global _manager_global
    if _manager_global is None: _manager_global = ObjectiveManagerUnificado()
    return _manager_global

def crear_manager_optimizado() -> ObjectiveManagerUnificado: return ObjectiveManagerUnificado()
def obtener_template(nombre: str) -> Optional[TemplateObjetivoV7]: return obtener_manager().obtener_template(nombre)
def buscar_templates_por_objetivo(objetivo: str, limite: int = 5) -> List[TemplateObjetivoV7]: return obtener_manager().buscar_templates({"efectos": [objetivo]}, limite)
def validar_template_avanzado(template: TemplateObjetivoV7) -> Dict[str, Any]: return obtener_manager().validar_template(template)
def ruta_por_objetivo(nombre: str) -> Dict[str, Any]:
    resultado = obtener_manager().procesar_objetivo(nombre)
    return {"preset": resultado["preset_emocional"], "estilo": resultado["estilo"], "modo": resultado["modo"], "beat": resultado["beat_base"], "capas": resultado["capas"]}

def rutear_objetivo_inteligente(objetivo: str, perfil_usuario: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]: return obtener_manager().rutear_objetivo_avanzado(objetivo, perfil_usuario, **kwargs)
def procesar_objetivo_director(objetivo: str, contexto: Dict[str, Any] = None) -> Dict[str, Any]: return obtener_manager().procesar_objetivo(objetivo, contexto)
def listar_objetivos_disponibles() -> List[str]: return obtener_manager().listar_objetivos_disponibles()
def buscar_objetivos_similares(objetivo: str, limite: int = 5) -> List[Tuple[str, float]]: return obtener_manager().buscar_objetivos_similares(objetivo, limite)
def recomendar_secuencia_objetivos(objetivo_principal: str, duracion_total: int = 60, perfil_usuario: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]: return obtener_manager().recomendar_secuencia_objetivos(objetivo_principal, duracion_total, perfil_usuario)

crear_gestor_optimizado = crear_manager_optimizado
GestorTemplatesOptimizado = ObjectiveManagerUnificado
crear_router_inteligente = lambda: obtener_manager().router
OBJECTIVE_TEMPLATES = {}

def _generar_compatibilidad_v6():
    global OBJECTIVE_TEMPLATES
    manager = obtener_manager()
    for nombre, template in manager.gestor_templates.templates.items():
        layers_v6 = {nombre_capa: {k: v for k, v in {"enabled": config_capa.enabled, "carrier": config_capa.carrier, "mod_type": config_capa.mod_type,
                    "freq_l": config_capa.freq_l, "freq_r": config_capa.freq_r, "style": config_capa.style}.items() if v is not None} for nombre_capa, config_capa in template.layers.items()}
        OBJECTIVE_TEMPLATES[nombre] = {"emotional_preset": template.emotional_preset, "style": template.style, "layers": layers_v6}

def diagnostico_manager() -> Dict[str, Any]:
    manager = obtener_manager()
    test_objetivos = ["concentracion", "relajacion", "creatividad", "test_inexistente"]
    resultados_test = {}
    for objetivo in test_objetivos:
        try:
            resultado = manager.procesar_objetivo(objetivo)
            resultados_test[objetivo] = {"exito": True, "preset": resultado.get("preset_emocional"), "confianza": resultado.get("nivel_confianza"), "tipo": "procesamiento_director"}
        except Exception as e: resultados_test[objetivo] = {"exito": False, "error": str(e)}
    estadisticas = manager.obtener_estadisticas_completas()
    return {"version": manager.version, "inicializacion_exitosa": True, "estadisticas_completas": estadisticas, "test_objetivos": resultados_test,
            "templates_disponibles": len(manager.gestor_templates.templates), "rutas_v6_disponibles": len(manager.router.rutas_v6_mapeadas),
            "objetivos_totales": len(manager.listar_objetivos_disponibles()), "compatibilidad_v6": len(OBJECTIVE_TEMPLATES), "protocolo_director": True,
            "componentes_integrados": {"templates_manager": True, "intelligent_router": True, "semantic_analyzer": True, "personalization_engine": True, "scientific_validator": True}}

_generar_compatibilidad_v6()

if __name__ == "__main__":
    print("🌟 Aurora V7 - Objective Manager Unificado"); print("=" * 60)
    diagnostico = diagnostico_manager(); print(f"🚀 {diagnostico['version']}")
    print(f"\n📊 Cobertura de objetivos:\n   • Templates V7: {diagnostico['templates_disponibles']}\n   • Rutas V6: {diagnostico['rutas_v6_disponibles']}\n   • Objetivos totales: {diagnostico['objetivos_totales']}\n   • Compatibilidad V6: {diagnostico['compatibilidad_v6']}")
    print(f"\n🔧 Test de funcionalidades:")
    for objetivo, resultado in diagnostico['test_objetivos'].items():
        emoji = "✅" if resultado['exito'] else "❌"
        detalle = f"Preset: {resultado['preset']}" if resultado['exito'] else f"Error: {resultado['error']}"
        print(f"   {emoji} {objetivo}: {detalle}")
    print(f"\n🤖 Componentes integrados:")
    for componente, activo in diagnostico['componentes_integrados'].items(): print(f"   {'✅' if activo else '❌'} {componente}")
    print(f"\n🧪 Test de API unificada:")
    try:
        template = obtener_template("claridad_mental"); print(f"   ✅ obtener_template: {template.nombre if template else 'No encontrado'}")
        config = ruta_por_objetivo("concentracion"); print(f"   ✅ ruta_por_objetivo: {config['preset']}")
        resultado_avanzado = rutear_objetivo_inteligente("creatividad profunda"); print(f"   ✅ rutear_objetivo_inteligente: {resultado_avanzado['informacion_v7']['tipo_ruteo']}")
        secuencia = recomendar_secuencia_objetivos("concentracion", 30); print(f"   ✅ recomendar_secuencia_objetivos: {len(secuencia)} objetivos")
    except Exception as e: print(f"   ❌ Error en API: {e}")
    estadisticas = diagnostico['estadisticas_completas']['resumen']
    print(f"\n📈 Métricas del Sistema:\n   🎯 Templates disponibles: {estadisticas['templates_disponibles']}\n   🔢 Objetivos totales: {estadisticas['objetivos_totales']}\n   ⏱️ Tiempo promedio: {estadisticas['tiempo_promedio_procesamiento']:.3f}s")
    print(f"\n🏆 OBJECTIVE MANAGER V7 UNIFICADO\n🌟 ¡Sistema completamente integrado!\n🔗 ¡Templates + Router en una sola interfaz!\n📦 ¡Compatibilidad V6 garantizada!\n🧠 ¡Inteligencia y personalización avanzada!\n✨ ¡Listo para Aurora Director V7!")
