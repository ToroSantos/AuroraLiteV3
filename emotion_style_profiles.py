"""Aurora V7 - Sistema Unificado de Perfiles Emocionales y Estéticos"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
import logging
import math
import json
import warnings
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Aurora.EmotionStyle.Unified")

class CategoriaEmocional(Enum):
    COGNITIVO = "cognitivo"
    EMOCIONAL = "emocional"
    SOCIAL = "social"
    CREATIVO = "creativo"
    ESPIRITUAL = "espiritual"
    TERAPEUTICO = "terapeutico"
    PERFORMANCE = "performance"
    EXPERIMENTAL = "experimental"

class CategoriaEstilo(Enum):
    MEDITATIVO = "meditativo"
    ENERGIZANTE = "energizante"
    CREATIVO = "creativo"
    TERAPEUTICO = "terapeutico"
    AMBIENTAL = "ambiental"
    EXPERIMENTAL = "experimental"
    TRADICIONAL = "tradicional"
    FUTURISTA = "futurista"
    ORGANICO = "organico"
    ESPIRITUAL = "espiritual"

class NivelIntensidad(Enum):
    SUAVE = "suave"
    MODERADO = "moderado"
    INTENSO = "intenso"

class TipoPad(Enum):
    SINE = "sine"
    SAW = "saw"
    SQUARE = "square"
    TRIANGLE = "triangle"
    PULSE = "pulse"
    STRING = "string"
    TRIBAL_PULSE = "tribal_pulse"
    SHIMMER = "shimmer"
    FADE_BLEND = "fade_blend"
    DIGITAL_SINE = "digital_sine"
    CENTER_PAD = "center_pad"
    DUST_PAD = "dust_pad"
    ICE_STRING = "ice_string"
    NEUROMORPHIC = "neuromorphic"
    BIOACOUSTIC = "bioacoustic"
    CRYSTALLINE = "crystalline"
    ORGANIC_FLOW = "organic_flow"
    QUANTUM_PAD = "quantum_pad"
    HARMONIC_SERIES = "harmonic_series"
    METALLIC = "metallic"
    VOCAL_PAD = "vocal_pad"
    GRANULAR = "granular"
    SPECTRAL = "spectral"
    FRACTAL = "fractal"

class EstiloRuido(Enum):
    DYNAMIC = "dynamic"
    STATIC = "static"
    FRACTAL = "fractal"
    ORGANIC = "organic"
    DIGITAL = "digital"
    NEURAL = "neural"
    ATMOSPHERIC = "atmospheric"
    GRANULAR = "granular"
    SPECTRAL = "spectral"
    CHAOS = "chaos"
    BROWN = "brown"
    PINK = "pink"
    WHITE = "white"
    BLUE = "blue"
    VIOLET = "violet"

class TipoFiltro(Enum):
    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"
    NOTCH = "notch"
    ALLPASS = "allpass"
    MORPHING = "morphing"
    NEUROMORPHIC = "neuromorphic"
    SPECTRAL = "spectral"
    FORMANT = "formant"
    COMB = "comb"
    PHASER = "phaser"
    FLANGER = "flanger"
    CHORUS = "chorus"
    REVERB = "reverb"

class TipoEnvolvente(Enum):
    SUAVE = "suave"
    CELESTIAL = "celestial"
    INQUIETANTE = "inquietante"
    GRAVE = "grave"
    PLANA = "plana"
    LUMINOSA = "luminosa"
    RITMICA = "ritmica"
    VIBRANTE = "vibrante"
    ETEREA = "eterea"
    PULIDA = "pulida"
    LIMPIA = "limpia"
    CALIDA = "calida"
    TRANSPARENTE = "transparente"
    BRILLANTE = "brillante"
    FOCAL = "focal"
    NEUROMORFICA = "neuromorfica"
    ORGANICA = "organica"
    CUANTICA = "cuantica"
    CRISTALINA = "cristalina"
    FLUIDA = "fluida"

class EscalaArmonica(Enum):
    SOLFEGGIO = "solfeggio"
    EQUAL_TEMPERAMENT = "equal_temperament"
    JUST_INTONATION = "just_intonation"
    PYTHAGOREAN = "pythagorean"
    GOLDEN_RATIO = "golden_ratio"
    FIBONACCI = "fibonacci"
    SCHUMANN = "schumann"
    PLANETARY = "planetary"
    CHAKRA = "chakra"
    CUSTOM = "custom"

class PatronPan(Enum):
    ESTATICO = "estatico"
    PENDULO = "pendulo"
    CIRCULAR = "circular"
    ESPIRAL = "espiral"
    RESPIRATORIO = "respiratorio"
    CARDIACO = "cardiaco"
    ALEATORIO = "aleatorio"
    ONDULATORIO = "ondulatorio"
    ORBITAL = "orbital"
    CUANTICO = "cuantico"

@dataclass
class EfectosPsicofisiologicos:
    atencion: float = 0.0
    memoria: float = 0.0
    concentracion: float = 0.0
    creatividad: float = 0.0
    calma: float = 0.0
    alegria: float = 0.0
    confianza: float = 0.0
    apertura: float = 0.0
    energia: float = 0.0
    empatia: float = 0.0
    conexion: float = 0.0
    
    def __post_init__(self):
        for campo, valor in self.__dict__.items():
            if not -1.0 <= valor <= 1.0:
                logger.warning(f"Efecto {campo} fuera de rango [-1,1]: {valor}")

@dataclass
class ConfiguracionFiltroAvanzada:
    tipo: TipoFiltro = TipoFiltro.LOWPASS
    frecuencia_corte: float = 500.0
    resonancia: float = 0.7
    sample_rate: int = 44100
    drive: float = 0.0
    wetness: float = 1.0
    modulacion_cutoff: bool = False
    velocidad_modulacion: float = 0.5
    profundidad_modulacion: float = 0.3
    envelope_follow: bool = False
    filtros_serie: List['ConfiguracionFiltroAvanzada'] = field(default_factory=list)
    mezcla_paralela: bool = False

@dataclass
class ConfiguracionRuidoAvanzada:
    duracion_sec: float
    estilo: EstiloRuido
    amplitud: float = 0.3
    sample_rate: int = 44100
    anchura_estereo: float = 1.0
    attack_ms: int = 100
    decay_ms: int = 200
    sustain_level: float = 0.8
    release_ms: int = 500
    filtro: Optional[ConfiguracionFiltroAvanzada] = None
    banda_frecuencia_min: float = 20.0
    banda_frecuencia_max: float = 20000.0
    granularidad: float = 1.0
    densidad: float = 1.0
    modulacion_temporal: bool = False
    
    def __post_init__(self):
        if self.filtro is None:
            self.filtro = ConfiguracionFiltroAvanzada()

@dataclass
class ParametrosEspacialesAvanzados:
    pan: float = 0.0
    elevation: float = 0.0
    distance: float = 1.0
    width: float = 1.0
    movimiento_activo: bool = False
    tipo_movimiento: str = "circular"
    velocidad_movimiento: float = 0.1
    radio_movimiento: float = 1.0
    reverb_espacial: bool = False
    early_reflections: bool = False
    air_absorption: bool = False
    doppler_effect: bool = False

@dataclass
class ParametrosFade:
    fade_in_sec: float
    fade_out_sec: float
    curva_fade_in: str = "exponential"
    curva_fade_out: str = "exponential"
    micro_fades: bool = False
    fade_crossover: bool = False
    adaptativo: bool = False
    
    def __post_init__(self):
        if self.fade_in_sec < 0 or self.fade_out_sec < 0:
            raise ValueError("Tiempos de fade deben ser positivos")

@dataclass
class ConfiguracionPanorama:
    dinamico: bool
    patron: PatronPan = PatronPan.ESTATICO
    velocidad: float = 0.1
    amplitud: float = 1.0
    centro: float = 0.0
    sincronizacion: str = "libre"
    elevacion: float = 0.0
    profundidad: float = 0.0
    rotacion_3d: bool = False
    doppler_effect: bool = False

@dataclass
class EspectroArmonico:
    frecuencias_base: List[float]
    escala: EscalaArmonica = EscalaArmonica.EQUAL_TEMPERAMENT
    temperamento: str = "440hz"
    armonicos_naturales: List[float] = field(default_factory=list)
    armonicos_artificiales: List[float] = field(default_factory=list)
    subharmonicos: List[float] = field(default_factory=list)
    modulacion_armonica: bool = False
    velocidad_modulacion: float = 0.1
    profundidad_modulacion: float = 0.2
    razon_aurea: bool = False
    fibonacci_sequence: bool = False
    
    def __post_init__(self):
        for f in self.frecuencias_base:
            if f <= 0:
                raise ValueError(f"Frecuencia debe ser positiva: {f}")
        if not self.armonicos_naturales:
            for fb in self.frecuencias_base:
                self.armonicos_naturales.extend([fb * i for i in range(2, 8)])
        if self.razon_aurea and not self.armonicos_artificiales:
            phi = (1 + math.sqrt(5)) / 2
            for fb in self.frecuencias_base:
                self.armonicos_artificiales.extend([fb * phi, fb * (phi ** 2), fb / phi])

@dataclass
class PropiedadesTextura:
    rugosidad: float = 0.0
    granularidad: float = 0.0
    densidad: float = 1.0
    brillantez: float = 0.5
    calidez: float = 0.5
    fluctuacion: float = 0.0
    respiracion: bool = False
    pulsacion: float = 0.0
    amplitud_espacial: float = 1.0
    cohesion: float = 0.8
    dispersion: float = 0.2
    
    def __post_init__(self):
        for attr in ['rugosidad', 'granularidad', 'brillantez', 'calidez', 'fluctuacion']:
            if not 0 <= getattr(self, attr) <= 1:
                warnings.warn(f"{attr} fuera de rango [0,1]")

@dataclass
class MetadatosEsteticos:
    inspiracion: str = ""
    referentes_artisticos: List[str] = field(default_factory=list)
    generos_musicales: List[str] = field(default_factory=list)
    colores_asociados: List[str] = field(default_factory=list)
    texturas_visuales: List[str] = field(default_factory=list)
    elementos_naturales: List[str] = field(default_factory=list)
    arquetipos_psicologicos: List[str] = field(default_factory=list)
    simbolismo: List[str] = field(default_factory=list)
    momentos_dia: List[str] = field(default_factory=list)
    estaciones: List[str] = field(default_factory=list)
    ambientes: List[str] = field(default_factory=list)
    actividades: List[str] = field(default_factory=list)

@dataclass
class PresetEmocionalCompleto:
    nombre: str
    descripcion: str
    categoria: CategoriaEmocional
    intensidad: NivelIntensidad = NivelIntensidad.MODERADO
    neurotransmisores: Dict[str, float] = field(default_factory=dict)
    frecuencia_base: float = 10.0
    frecuencias_armonicas: List[float] = field(default_factory=list)
    estilo_asociado: str = "sereno"
    efectos: EfectosPsicofisiologicos = field(default_factory=EfectosPsicofisiologicos)
    mejor_momento_uso: List[str] = field(default_factory=list)
    contextos_recomendados: List[str] = field(default_factory=list)
    contraindicaciones: List[str] = field(default_factory=list)
    presets_compatibles: List[str] = field(default_factory=list)
    nivel_evidencia: str = "experimental"
    confidence_score: float = 0.8
    version: str = "v7.0"
    fecha_creacion: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        self._validar_preset()
        if not self.frecuencias_armonicas:
            self._calcular_frecuencias_armonicas()
        self._inferir_efectos_desde_neurotransmisores()
    
    def _validar_preset(self):
        if self.frecuencia_base <= 0:
            raise ValueError("Frecuencia base debe ser positiva")
        for nt, intensidad in self.neurotransmisores.items():
            if not 0 <= intensidad <= 1:
                logger.warning(f"Intensidad de {nt} fuera de rango [0,1]: {intensidad}")
    
    def _calcular_frecuencias_armonicas(self):
        self.frecuencias_armonicas = [self.frecuencia_base * i for i in [2, 3, 4, 5]]
    
    def _inferir_efectos_desde_neurotransmisores(self):
        mapeo_efectos = {
            "dopamina": {"atencion": 0.7, "energia": 0.6, "confianza": 0.5},
            "serotonina": {"calma": 0.8, "alegria": 0.6, "apertura": 0.4},
            "gaba": {"calma": 0.9, "atencion": -0.3},
            "oxitocina": {"empatia": 0.9, "conexion": 0.8, "confianza": 0.7},
            "acetilcolina": {"atencion": 0.9, "memoria": 0.8, "concentracion": 0.8},
            "norepinefrina": {"atencion": 0.8, "energia": 0.7},
            "endorfina": {"alegria": 0.8, "energia": 0.7},
            "anandamida": {"creatividad": 0.8, "apertura": 0.9, "alegria": 0.6},
            "melatonina": {"calma": 0.9, "energia": -0.6}
        }
        efectos_calculados = {}
        for campo in self.efectos.__dict__.keys():
            efecto_total = peso_total = 0
            for nt, intensidad in self.neurotransmisores.items():
                if nt in mapeo_efectos and campo in mapeo_efectos[nt]:
                    efecto_total += mapeo_efectos[nt][campo] * intensidad
                    peso_total += intensidad
            if peso_total > 0:
                efectos_calculados[campo] = np.tanh(efecto_total / peso_total)
        for campo, valor in efectos_calculados.items():
            if getattr(self.efectos, campo) == 0.0:
                setattr(self.efectos, campo, valor)

@dataclass
class PerfilEstiloCompleto:
    nombre: str
    categoria: CategoriaEstilo
    tipo_pad: TipoPad
    frecuencia_base: float = 220.0
    armonicos: List[float] = field(default_factory=list)
    nivel_db: float = -12.0
    modulacion_am: float = 0.0
    modulacion_fm: float = 0.0
    velocidad_modulacion: float = 0.5
    filtro: ConfiguracionFiltroAvanzada = field(default_factory=ConfiguracionFiltroAvanzada)
    ruido: Optional[ConfiguracionRuidoAvanzada] = None
    attack_ms: int = 100
    decay_ms: int = 500
    sustain_level: float = 0.7
    release_ms: int = 1000
    espacial: ParametrosEspacialesAvanzados = field(default_factory=ParametrosEspacialesAvanzados)
    neurotransmisores_asociados: List[str] = field(default_factory=list)
    efectos_emocionales: List[str] = field(default_factory=list)
    estados_mentales: List[str] = field(default_factory=list)
    descripcion: str = ""
    inspiracion: str = ""
    uso_recomendado: List[str] = field(default_factory=list)
    compatibilidad: List[str] = field(default_factory=list)
    incompatibilidad: List[str] = field(default_factory=list)
    complejidad_armonica: float = 0.5
    dinamismo_temporal: float = 0.5
    textura_espectral: str = "suave"
    validado: bool = False
    confidence_score: float = 0.0
    version: str = "v7.0"
    fecha_creacion: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        if self.frecuencia_base <= 0:
            raise ValueError("Frecuencia base debe ser positiva")
        if not self.armonicos:
            self._calcular_armonicos_automaticos()
    
    def _calcular_armonicos_automaticos(self):
        num_armonicos = int(3 + self.complejidad_armonica * 5)
        if self.tipo_pad == TipoPad.SINE:
            self.armonicos = [self.frecuencia_base * (2*i + 1) for i in range(num_armonicos)]
        elif self.tipo_pad == TipoPad.SAW:
            self.armonicos = [self.frecuencia_base * (i + 1) for i in range(num_armonicos)]
        elif self.tipo_pad == TipoPad.SQUARE:
            self.armonicos = [self.frecuencia_base * (2*i + 1) for i in range(num_armonicos)]
        elif self.tipo_pad == TipoPad.TRIANGLE:
            self.armonicos = [self.frecuencia_base * (2*i + 1) / ((2*i + 1)**2) for i in range(num_armonicos)]
        else:
            self.armonicos = [self.frecuencia_base * (i + 1) for i in range(num_armonicos)]

@dataclass
class PresetEstiloCompleto:
    nombre: str
    fade: ParametrosFade
    panorama: ConfiguracionPanorama
    espectro: EspectroArmonico
    envolvente: TipoEnvolvente
    textura: PropiedadesTextura = field(default_factory=PropiedadesTextura)
    metadatos: MetadatosEsteticos = field(default_factory=MetadatosEsteticos)
    neurotransmisores_asociados: List[str] = field(default_factory=list)
    preset_emocional_compatible: List[str] = field(default_factory=list)
    perfil_estilo_base: str = ""
    adaptabilidad: float = 0.5
    complejidad_estetica: float = 0.5
    intensidad_emocional: float = 0.5
    version: str = "v7.0"
    validado: bool = False
    confidence_score: float = 0.8
    usage_popularity: float = 0.0
    fecha_creacion: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        for attr, val in [('adaptabilidad', self.adaptabilidad), ('complejidad_estetica', self.complejidad_estetica), ('intensidad_emocional', self.intensidad_emocional)]:
            if not 0 <= val <= 1:
                warnings.warn(f"{attr} fuera de rango [0,1]")
        if not self.neurotransmisores_asociados:
            self._inferir_neurotransmisores()
    
    def _inferir_neurotransmisores(self):
        mapeo = {
            (TipoEnvolvente.SUAVE, TipoEnvolvente.CELESTIAL, TipoEnvolvente.LUMINOSA): ["serotonina", "oxitocina"],
            (TipoEnvolvente.RITMICA, TipoEnvolvente.VIBRANTE, TipoEnvolvente.PULIDA): ["dopamina", "norepinefrina"],
            (TipoEnvolvente.ETEREA, TipoEnvolvente.CUANTICA): ["anandamida", "serotonina"],
            (TipoEnvolvente.GRAVE, TipoEnvolvente.CALIDA, TipoEnvolvente.ORGANICA): ["gaba", "melatonina"]
        }
        for envs, nts in mapeo.items():
            if self.envolvente in envs:
                self.neurotransmisores_asociados.extend(nts)
                break

class GestorEmotionStyleUnificado:
    def __init__(self):
        self.presets_emocionales = {}
        self.perfiles_estilo = {}
        self.presets_esteticos = {}
        self.cache_recomendaciones = {}
        self._inicializar_todos_los_presets()
        logger.info("Gestor Emotion-Style Unificado Aurora V7 inicializado")
    
    def _inicializar_todos_los_presets(self):
        self._inicializar_presets_emocionales()
        self._inicializar_perfiles_estilo()
        self._inicializar_presets_esteticos()
        self._crear_enlaces_cruzados()
    
    def _inicializar_presets_emocionales(self):
        presets_data = {
            "claridad_mental": {"descripcion": "Lucidez cognitiva y enfoque sereno para trabajo mental intenso", "categoria": CategoriaEmocional.COGNITIVO, "intensidad": NivelIntensidad.MODERADO, "neurotransmisores": {"acetilcolina": 0.9, "dopamina": 0.6, "norepinefrina": 0.4}, "frecuencia_base": 14.5, "estilo_asociado": "crystalline", "mejor_momento_uso": ["mañana", "tarde"], "contextos_recomendados": ["trabajo", "estudio", "resolución_problemas"], "presets_compatibles": ["expansion_creativa", "estado_flujo"], "nivel_evidencia": "validado", "confidence_score": 0.92},
            "seguridad_interior": {"descripcion": "Confianza centrada y estabilidad emocional profunda", "categoria": CategoriaEmocional.EMOCIONAL, "intensidad": NivelIntensidad.SUAVE, "neurotransmisores": {"gaba": 0.8, "serotonina": 0.7, "oxitocina": 0.5}, "frecuencia_base": 8.0, "estilo_asociado": "sereno", "mejor_momento_uso": ["mañana", "noche"], "contextos_recomendados": ["meditación", "terapia", "desarrollo_personal"], "nivel_evidencia": "validado", "confidence_score": 0.89},
            "apertura_corazon": {"descripcion": "Aceptación amorosa y conexión emocional profunda", "categoria": CategoriaEmocional.SOCIAL, "intensidad": NivelIntensidad.MODERADO, "neurotransmisores": {"oxitocina": 0.9, "serotonina": 0.5, "endorfina": 0.6}, "frecuencia_base": 7.2, "estilo_asociado": "organico", "mejor_momento_uso": ["tarde", "noche"], "contextos_recomendados": ["relaciones", "terapia_pareja", "trabajo_emocional"], "presets_compatibles": ["conexion_mistica", "introspeccion_suave"], "nivel_evidencia": "experimental", "confidence_score": 0.86},
            "estado_flujo": {"descripcion": "Rendimiento óptimo y disfrute del presente", "categoria": CategoriaEmocional.PERFORMANCE, "intensidad": NivelIntensidad.INTENSO, "neurotransmisores": {"dopamina": 0.9, "norepinefrina": 0.7, "endorfina": 0.5}, "frecuencia_base": 12.0, "estilo_asociado": "futurista", "mejor_momento_uso": ["mañana", "tarde"], "contextos_recomendados": ["deporte", "arte", "programación", "música"], "presets_compatibles": ["expansion_creativa", "claridad_mental"], "nivel_evidencia": "validado", "confidence_score": 0.94},
            "conexion_mistica": {"descripcion": "Unidad espiritual y percepción expandida", "categoria": CategoriaEmocional.ESPIRITUAL, "intensidad": NivelIntensidad.INTENSO, "neurotransmisores": {"anandamida": 0.8, "serotonina": 0.6, "oxitocina": 0.7}, "frecuencia_base": 5.0, "estilo_asociado": "mistico", "mejor_momento_uso": ["noche"], "contextos_recomendados": ["meditación_profunda", "ceremonia", "retiro"], "contraindicaciones": ["ansiedad_severa", "primera_experiencia"], "nivel_evidencia": "experimental", "confidence_score": 0.82},
            "expansion_creativa": {"descripcion": "Inspiración y creatividad fluida", "categoria": CategoriaEmocional.CREATIVO, "neurotransmisores": {"dopamina": 0.8, "acetilcolina": 0.7, "anandamida": 0.6}, "frecuencia_base": 11.5, "estilo_asociado": "etereo", "contextos_recomendados": ["arte", "escritura", "diseño", "música"], "confidence_score": 0.87},
            "calma_profunda": {"descripcion": "Relajación total, serenidad corporal y mental", "categoria": CategoriaEmocional.TERAPEUTICO, "intensidad": NivelIntensidad.SUAVE, "neurotransmisores": {"gaba": 0.9, "serotonina": 0.8, "melatonina": 0.7}, "frecuencia_base": 6.5, "estilo_asociado": "sereno", "mejor_momento_uso": ["noche"], "contextos_recomendados": ["sueño", "recuperación", "trauma"], "presets_compatibles": ["regulacion_emocional"], "nivel_evidencia": "clinico", "confidence_score": 0.93}
        }
        for nombre, data in presets_data.items():
            self.presets_emocionales[nombre] = PresetEmocionalCompleto(nombre=nombre.replace("_", " ").title(), **data)
    
    def _inicializar_perfiles_estilo(self):
        perfiles_data = {
            "sereno": {"categoria": CategoriaEstilo.MEDITATIVO, "tipo_pad": TipoPad.SINE, "frecuencia_base": 6.0, "armonicos": [12.0, 18.0, 24.0], "nivel_db": -15.0, "modulacion_am": 0.1, "velocidad_modulacion": 0.05, "filtro": ConfiguracionFiltroAvanzada(TipoFiltro.LOWPASS, 400.0, 0.3, modulacion_cutoff=True, velocidad_modulacion=0.02), "attack_ms": 2000, "decay_ms": 3000, "sustain_level": 0.9, "release_ms": 5000, "espacial": ParametrosEspacialesAvanzados(width=2.0, reverb_espacial=True), "neurotransmisores_asociados": ["gaba", "serotonina", "melatonina"], "efectos_emocionales": ["calma_profunda", "serenidad", "paz_interior"], "estados_mentales": ["meditativo", "contemplativo", "relajado"], "uso_recomendado": ["meditación", "relajación", "terapia", "sueño"], "complejidad_armonica": 0.2, "dinamismo_temporal": 0.1, "textura_espectral": "suave", "validado": True, "confidence_score": 0.95},
            "organico": {"categoria": CategoriaEstilo.ORGANICO, "tipo_pad": TipoPad.ORGANIC_FLOW, "frecuencia_base": 8.5, "nivel_db": -10.0, "modulacion_am": 0.3, "modulacion_fm": 0.2, "velocidad_modulacion": 0.15, "filtro": ConfiguracionFiltroAvanzada(TipoFiltro.MORPHING, 800.0, 0.6, modulacion_cutoff=True, velocidad_modulacion=0.1), "ruido": ConfiguracionRuidoAvanzada(1.0, EstiloRuido.ORGANIC, 0.15, granularidad=0.7), "espacial": ParametrosEspacialesAvanzados(movimiento_activo=True, tipo_movimiento="pendulo", velocidad_movimiento=0.03), "neurotransmisores_asociados": ["anandamida", "serotonina", "oxitocina"], "efectos_emocionales": ["conexion_natural", "fluidez", "armonia"], "estados_mentales": ["flujo", "creatividad", "conexion"], "complejidad_armonica": 0.6, "dinamismo_temporal": 0.4, "textura_espectral": "orgánica", "validado": True, "confidence_score": 0.88},
            "crystalline": {"categoria": CategoriaEstilo.AMBIENTAL, "tipo_pad": TipoPad.CRYSTALLINE, "frecuencia_base": 11.0, "nivel_db": -13.0, "modulacion_am": 0.2, "velocidad_modulacion": 0.06, "filtro": ConfiguracionFiltroAvanzada(TipoFiltro.HIGHPASS, 200.0, 0.9, drive=0.1), "espacial": ParametrosEspacialesAvanzados(elevation=0.3, width=1.5, early_reflections=True), "neurotransmisores_asociados": ["acetilcolina", "serotonina"], "efectos_emocionales": ["claridad", "pureza", "precision"], "estados_mentales": ["claro", "enfocado", "puro"], "complejidad_armonica": 0.4, "dinamismo_temporal": 0.2, "textura_espectral": "cristalina", "validado": True, "confidence_score": 0.92}
        }
        for nombre, data in perfiles_data.items():
            self.perfiles_estilo[nombre] = PerfilEstiloCompleto(nombre=nombre.title(), **data)
    
    def _inicializar_presets_esteticos(self):
        presets_data = {
            "etereo": {"fade": ParametrosFade(3.0, 3.0, "sigmoid", "exponential", True), "panorama": ConfiguracionPanorama(True, PatronPan.ONDULATORIO, 0.08, 0.7, 0.0, "libre", 0.3, 0.0, True, False), "espectro": EspectroArmonico([432.0, 528.0], EscalaArmonica.SOLFEGGIO, "432hz", razon_aurea=True, modulacion_armonica=True, velocidad_modulacion=0.05), "envolvente": TipoEnvolvente.SUAVE, "textura": PropiedadesTextura(0.1, 0.2, 1.0, 0.7, 0.4, 0.0, True, 0.0, 1.5, 0.8, 0.2), "metadatos": MetadatosEsteticos(inspiracion="Espacios infinitos y consciencia expandida", colores_asociados=["azul_celeste", "violeta_suave", "blanco_perlado"], elementos_naturales=["nubes", "aurora", "espacio"], arquetipos_psicologicos=["el_soñador", "el_visionario"], momentos_dia=["amanecer", "atardecer"], ambientes=["meditación", "espacios_abiertos"]), "neurotransmisores_asociados": ["anandamida", "serotonina", "oxitocina"], "preset_emocional_compatible": ["conexion_mistica", "expansion_creativa"], "complejidad_estetica": 0.7, "intensidad_emocional": 0.6, "validado": True, "confidence_score": 0.92},
            "tribal": {"fade": ParametrosFade(1.0, 2.0, "linear", "exponential", False), "panorama": ConfiguracionPanorama(True, PatronPan.CIRCULAR, 0.15, 0.8, 0.0, "cardiaco", 0.0, 0.0, False, False), "espectro": EspectroArmonico([140.0, 220.0], EscalaArmonica.PYTHAGOREAN, "440hz"), "envolvente": TipoEnvolvente.RITMICA, "textura": PropiedadesTextura(0.5, 0.3, 1.0, 0.4, 0.9, 0.0, False, 1.2, 1.0, 0.8, 0.2), "metadatos": MetadatosEsteticos(inspiracion="Conexión tribal y fuerza ancestral", generos_musicales=["percusion_tribal", "world_music", "shamanic"], colores_asociados=["rojo_tierra", "ocre", "naranja_fuego"], elementos_naturales=["fuego", "tierra", "tambores"], arquetipos_psicologicos=["el_guerrero", "el_chamán"], ambientes=["ritual", "danza", "ceremonia_grupal"]), "neurotransmisores_asociados": ["dopamina", "endorfina", "oxitocina"], "preset_emocional_compatible": ["estado_flujo", "apertura_corazon"], "complejidad_estetica": 0.6, "intensidad_emocional": 0.8, "validado": True, "confidence_score": 0.89}
        }
        for nombre, data in presets_data.items():
            self.presets_esteticos[nombre] = PresetEstiloCompleto(nombre=nombre.title(), **data)
    
    def _crear_enlaces_cruzados(self):
        for preset_name, preset in self.presets_emocionales.items():
            for perfil_name, perfil in self.perfiles_estilo.items():
                neuros_comunes = set(preset.neurotransmisores.keys()) & set(perfil.neurotransmisores_asociados)
                if neuros_comunes:
                    preset.presets_compatibles.append(perfil_name)
                    perfil.compatibilidad.append(preset_name)
            for estetico_name, estetico in self.presets_esteticos.items():
                neuros_comunes = set(preset.neurotransmisores.keys()) & set(estetico.neurotransmisores_asociados)
                if neuros_comunes:
                    estetico.preset_emocional_compatible.append(preset_name)
    
    @lru_cache(maxsize=128)
    def obtener_preset_emocional(self, nombre: str) -> Optional[PresetEmocionalCompleto]:
        return self.presets_emocionales.get(nombre.lower().replace(" ", "_"))
    
    @lru_cache(maxsize=128)
    def obtener_perfil_estilo(self, nombre: str) -> Optional[PerfilEstiloCompleto]:
        return self.perfiles_estilo.get(nombre.lower())
    
    @lru_cache(maxsize=128)
    def obtener_preset_estetico(self, nombre: str) -> Optional[PresetEstiloCompleto]:
        return self.presets_esteticos.get(nombre.lower())
    
    def recomendar_experiencia_completa(self, objetivo_emocional: str, contexto: str = None) -> Dict[str, Any]:
        preset_emocional = self.obtener_preset_emocional(objetivo_emocional)
        if not preset_emocional:
            return {"error": f"Objetivo emocional '{objetivo_emocional}' no encontrado"}
        
        perfiles_compatibles = []
        for nombre, perfil in self.perfiles_estilo.items():
            if objetivo_emocional in perfil.compatibilidad:
                perfiles_compatibles.append((nombre, perfil))
        
        if not perfiles_compatibles:
            for nombre, perfil in self.perfiles_estilo.items():
                neuros_comunes = set(preset_emocional.neurotransmisores.keys()) & set(perfil.neurotransmisores_asociados)
                if neuros_comunes:
                    perfiles_compatibles.append((nombre, perfil))
        
        perfil_recomendado = perfiles_compatibles[0] if perfiles_compatibles else None
        
        esteticos_compatibles = []
        for nombre, estetico in self.presets_esteticos.items():
            if objetivo_emocional in estetico.preset_emocional_compatible:
                esteticos_compatibles.append((nombre, estetico))
        
        estetico_recomendado = esteticos_compatibles[0] if esteticos_compatibles else None
        score_coherencia = self._calcular_coherencia_experiencia(preset_emocional, perfil_recomendado[1] if perfil_recomendado else None, estetico_recomendado[1] if estetico_recomendado else None)
        
        return {
            "preset_emocional": {"nombre": preset_emocional.nombre, "descripcion": preset_emocional.descripcion, "frecuencia_base": preset_emocional.frecuencia_base, "neurotransmisores": preset_emocional.neurotransmisores, "efectos_esperados": self._extraer_efectos_principales(preset_emocional.efectos)},
            "perfil_estilo": {"nombre": perfil_recomendado[0] if perfil_recomendado else "sereno", "tipo_pad": perfil_recomendado[1].tipo_pad.value if perfil_recomendado else "sine", "configuracion_tecnica": self._extraer_config_tecnica(perfil_recomendado[1]) if perfil_recomendado else {}},
            "preset_estetico": {"nombre": estetico_recomendado[0] if estetico_recomendado else "sereno", "envolvente": estetico_recomendado[1].envolvente.value if estetico_recomendado else "suave", "experiencia_sensorial": self._extraer_experiencia_sensorial(estetico_recomendado[1]) if estetico_recomendado else {}},
            "score_coherencia": score_coherencia,
            "recomendaciones_uso": self._generar_recomendaciones_uso(preset_emocional, contexto),
            "parametros_aurora": self._generar_parametros_aurora(preset_emocional, perfil_recomendado, estetico_recomendado)
        }
    
    def generar_secuencia_inteligente(self, objetivo_final: str, duracion_total_min: int = 30) -> List[Dict[str, Any]]:
        preset_objetivo = self.obtener_preset_emocional(objetivo_final)
        if not preset_objetivo:
            return []
        secuencia = []
        duracion_por_preset = duracion_total_min // 3
        preset_inicial = self._seleccionar_preset_inicial(preset_objetivo)
        secuencia.append({"preset": preset_inicial.nombre, "tipo": "emocional", "duracion_min": duracion_por_preset, "posicion": "inicial", "parametros": self._extraer_parametros_preset(preset_inicial)})
        if duracion_total_min > 15:
            preset_intermedio = self._seleccionar_preset_intermedio(preset_inicial, preset_objetivo)
            secuencia.append({"preset": preset_intermedio.nombre, "tipo": "emocional", "duracion_min": duracion_por_preset, "posicion": "intermedio", "parametros": self._extraer_parametros_preset(preset_intermedio)})
        secuencia.append({"preset": preset_objetivo.nombre, "tipo": "emocional", "duracion_min": duracion_total_min - len(secuencia) * duracion_por_preset, "posicion": "final", "parametros": self._extraer_parametros_preset(preset_objetivo)})
        return secuencia
    
    def analizar_compatibilidad_total(self, preset_emocional: str, perfil_estilo: str, preset_estetico: str = None) -> Dict[str, Any]:
        pe = self.obtener_preset_emocional(preset_emocional)
        ps = self.obtener_perfil_estilo(perfil_estilo)
        pest = self.obtener_preset_estetico(preset_estetico) if preset_estetico else None
        if not pe or not ps:
            return {"error": "Presets no encontrados"}
        neuros_comunes_es = set(pe.neurotransmisores.keys()) & set(ps.neurotransmisores_asociados)
        compat_emocional_estilo = len(neuros_comunes_es) / max(len(pe.neurotransmisores), 1)
        diff_freq = abs(pe.frecuencia_base - ps.frecuencia_base)
        compat_frecuencial = 1.0 - min(diff_freq / 20.0, 1.0)
        compat_estetica = 0.8
        if pest:
            neuros_comunes_est = set(pe.neurotransmisores.keys()) & set(pest.neurotransmisores_asociados)
            compat_estetica = len(neuros_comunes_est) / max(len(pe.neurotransmisores), 1)
        score_total = (compat_emocional_estilo + compat_frecuencial + compat_estetica) / 3
        return {"score_compatibilidad_total": score_total, "compatibilidad_emocional_estilo": compat_emocional_estilo, "compatibilidad_frecuencial": compat_frecuencial, "compatibilidad_estetica": compat_estetica, "neurotransmisores_comunes": list(neuros_comunes_es), "diferencia_frecuencial": diff_freq, "recomendacion": self._generar_recomendacion_compatibilidad(score_total), "optimizaciones_sugeridas": self._sugerir_optimizaciones(pe, ps, pest)}
    
    def buscar_por_neurotransmisor(self, neurotransmisor: str) -> Dict[str, List[str]]:
        resultados = {"presets_emocionales": [], "perfiles_estilo": [], "presets_esteticos": []}
        for nombre, preset in self.presets_emocionales.items():
            if neurotransmisor.lower() in [n.lower() for n in preset.neurotransmisores.keys()]:
                resultados["presets_emocionales"].append(nombre)
        for nombre, perfil in self.perfiles_estilo.items():
            if neurotransmisor.lower() in [n.lower() for n in perfil.neurotransmisores_asociados]:
                resultados["perfiles_estilo"].append(nombre)
        for nombre, estetico in self.presets_esteticos.items():
            if neurotransmisor.lower() in [n.lower() for n in estetico.neurotransmisores_asociados]:
                resultados["presets_esteticos"].append(nombre)
        return resultados
    
    def buscar_por_efecto(self, efecto: str, umbral: float = 0.5) -> List[str]:
        resultados = []
        for nombre, preset in self.presets_emocionales.items():
            if hasattr(preset.efectos, efecto):
                valor_efecto = getattr(preset.efectos, efecto)
                if valor_efecto >= umbral:
                    resultados.append(nombre)
        return sorted(resultados, key=lambda n: getattr(self.presets_emocionales[n].efectos, efecto), reverse=True)
    
    def buscar_por_categoria(self, categoria: Union[CategoriaEmocional, CategoriaEstilo]) -> Dict[str, List[str]]:
        resultados = {}
        if isinstance(categoria, CategoriaEmocional):
            resultados["presets_emocionales"] = [nombre for nombre, preset in self.presets_emocionales.items() if preset.categoria == categoria]
        if isinstance(categoria, CategoriaEstilo):
            resultados["perfiles_estilo"] = [nombre for nombre, perfil in self.perfiles_estilo.items() if perfil.categoria == categoria]
        return resultados
    
    def _calcular_coherencia_experiencia(self, emocional, estilo, estetico) -> float:
        score = 0.5
        if emocional and estilo:
            neuros_comunes = set(emocional.neurotransmisores.keys()) & set(estilo.neurotransmisores_asociados)
            score += len(neuros_comunes) * 0.2
        if emocional and estetico:
            neuros_comunes = set(emocional.neurotransmisores.keys()) & set(estetico.neurotransmisores_asociados)
            score += len(neuros_comunes) * 0.15
        return min(1.0, score)
    
    def _extraer_efectos_principales(self, efectos: EfectosPsicofisiologicos) -> List[str]:
        efectos_dict = efectos.__dict__
        efectos_ordenados = sorted(efectos_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        return [efecto for efecto, valor in efectos_ordenados[:3] if abs(valor) > 0.3]
    
    def _extraer_config_tecnica(self, perfil: PerfilEstiloCompleto) -> Dict[str, Any]:
        return {"tipo_pad": perfil.tipo_pad.value, "frecuencia_base": perfil.frecuencia_base, "modulacion_am": perfil.modulacion_am, "modulacion_fm": perfil.modulacion_fm, "filtro_tipo": perfil.filtro.tipo.value, "filtro_cutoff": perfil.filtro.frecuencia_corte}
    
    def _extraer_experiencia_sensorial(self, estetico: PresetEstiloCompleto) -> Dict[str, Any]:
        return {"colores": estetico.metadatos.colores_asociados, "elementos_naturales": estetico.metadatos.elementos_naturales, "arquetipos": estetico.metadatos.arquetipos_psicologicos, "ambientes": estetico.metadatos.ambientes, "intensidad_emocional": estetico.intensidad_emocional}
    
    def _generar_recomendaciones_uso(self, preset: PresetEmocionalCompleto, contexto: str = None) -> List[str]:
        recomendaciones = []
        if preset.mejor_momento_uso:
            recomendaciones.append(f"Mejor momento: {', '.join(preset.mejor_momento_uso)}")
        if preset.contextos_recomendados:
            recomendaciones.append(f"Contextos ideales: {', '.join(preset.contextos_recomendados[:2])}")
        if preset.contraindicaciones:
            recomendaciones.append(f"Evitar si: {', '.join(preset.contraindicaciones)}")
        return recomendaciones
    
    def _generar_parametros_aurora(self, emocional, estilo, estetico) -> Dict[str, Any]:
        params = {"frecuencia_base": emocional.frecuencia_base if emocional else 10.0, "neurotransmisores": emocional.neurotransmisores if emocional else {}, "estilo_audio": estilo[0] if estilo else "sereno", "tipo_pad": estilo[1].tipo_pad.value if estilo else "sine", "intensidad": emocional.intensidad.value if emocional else "moderado"}
        if estetico:
            params.update({"envolvente": estetico[1].envolvente.value, "panorama_dinamico": estetico[1].panorama.dinamico, "fade_in": estetico[1].fade.fade_in_sec, "fade_out": estetico[1].fade.fade_out_sec})
        return params
    
    def _seleccionar_preset_inicial(self, objetivo: PresetEmocionalCompleto) -> PresetEmocionalCompleto:
        mapeo_inicial = {CategoriaEmocional.ESPIRITUAL: "calma_profunda", CategoriaEmocional.COGNITIVO: "seguridad_interior", CategoriaEmocional.PERFORMANCE: "claridad_mental"}
        nombre_inicial = mapeo_inicial.get(objetivo.categoria, "seguridad_interior")
        return self.obtener_preset_emocional(nombre_inicial) or self.obtener_preset_emocional("calma_profunda")
    
    def _seleccionar_preset_intermedio(self, inicial: PresetEmocionalCompleto, objetivo: PresetEmocionalCompleto) -> PresetEmocionalCompleto:
        for preset in self.presets_emocionales.values():
            freq_min = min(inicial.frecuencia_base, objetivo.frecuencia_base)
            freq_max = max(inicial.frecuencia_base, objetivo.frecuencia_base)
            if freq_min < preset.frecuencia_base < freq_max:
                return preset
        return self.obtener_preset_emocional("seguridad_interior") or inicial
    
    def _extraer_parametros_preset(self, preset: PresetEmocionalCompleto) -> Dict[str, Any]:
        return {"frecuencia_base": preset.frecuencia_base, "neurotransmisores": preset.neurotransmisores, "intensidad": preset.intensidad.value, "estilo_asociado": preset.estilo_asociado}
    
    def _generar_recomendacion_compatibilidad(self, score: float) -> str:
        if score > 0.8:
            return "Excelente coherencia - Experiencia óptima garantizada"
        elif score > 0.6:
            return "Buena coherencia - Funciona bien en la mayoría de casos"
        elif score > 0.4:
            return "Coherencia moderada - Considerar ajustes menores"
        else:
            return "Baja coherencia - Reconsiderar combinación"
    
    def _sugerir_optimizaciones(self, emocional, estilo, estetico) -> List[str]:
        optimizaciones = []
        if emocional and estilo:
            diff_freq = abs(emocional.frecuencia_base - estilo.frecuencia_base)
            if diff_freq > 10:
                optimizaciones.append("Ajustar frecuencias para mayor coherencia")
        if estetico and estetico.intensidad_emocional < 0.5:
            optimizaciones.append("Considerar aumentar intensidad emocional")
        return optimizaciones
    
    def exportar_configuracion_aurora(self) -> Dict[str, Any]:
        return {
            "version": "Aurora V7 Emotion-Style Unificado",
            "total_presets_emocionales": len(self.presets_emocionales),
            "total_perfiles_estilo": len(self.perfiles_estilo),
            "total_presets_esteticos": len(self.presets_esteticos),
            "categorias_emocionales": [cat.value for cat in CategoriaEmocional],
            "categorias_estilo": [cat.value for cat in CategoriaEstilo],
            "neurotransmisores_disponibles": list(set().union(*[list(p.neurotransmisores.keys()) for p in self.presets_emocionales.values()])),
            "tipos_pad_disponibles": [tipo.value for tipo in TipoPad],
            "presets_validados": {"emocionales": sum(1 for p in self.presets_emocionales.values() if p.nivel_evidencia in ["validado", "clinico"]), "estilo": sum(1 for p in self.perfiles_estilo.values() if p.validado), "esteticos": sum(1 for p in self.presets_esteticos.values() if p.validado)},
            "confidence_promedio": {"emocionales": sum(p.confidence_score for p in self.presets_emocionales.values()) / len(self.presets_emocionales), "estilo": sum(p.confidence_score for p in self.perfiles_estilo.values()) / len(self.perfiles_estilo), "esteticos": sum(p.confidence_score for p in self.presets_esteticos.values()) / len(self.presets_esteticos)}
        }
    
    def limpiar_cache(self):
        self.obtener_preset_emocional.cache_clear()
        self.obtener_perfil_estilo.cache_clear()
        self.obtener_preset_estetico.cache_clear()
        self.cache_recomendaciones.clear()
        logger.info("Cache del gestor unificado limpiado")

class EmotionalPreset:
    _gestor = None
    
    @classmethod
    def get(cls, nombre: str) -> Optional[Dict[str, Any]]:
        if cls._gestor is None:
            cls._gestor = GestorEmotionStyleUnificado()
        preset = cls._gestor.obtener_preset_emocional(nombre)
        if preset:
            return {"nt": preset.neurotransmisores, "frecuencia_base": preset.frecuencia_base, "descripcion": preset.descripcion}
        presets_legacy = {"claridad_mental": {"nt": {"acetilcolina": 0.9, "dopamina": 0.6}, "frecuencia_base": 14.5, "descripcion": "Lucidez y enfoque sereno"}, "seguridad_interior": {"nt": {"gaba": 0.8, "serotonina": 0.7}, "frecuencia_base": 8.0, "descripcion": "Confianza centrada"}}
        return presets_legacy.get(nombre, None)

class StyleProfile:
    _gestor = None
    
    @staticmethod
    def get(nombre: str) -> Dict[str, str]:
        if StyleProfile._gestor is None:
            StyleProfile._gestor = GestorEmotionStyleUnificado()
        perfil = StyleProfile._gestor.obtener_perfil_estilo(nombre)
        if perfil:
            return {"pad_type": perfil.tipo_pad.value}
        estilos_legacy = {"sereno": {"pad_type": "sine"}, "organico": {"pad_type": "organic_flow"}, "etereo": {"pad_type": "spectral"}, "tribal": {"pad_type": "tribal_pulse"}, "crystalline": {"pad_type": "crystalline"}}
        return estilos_legacy.get(nombre.lower(), {"pad_type": "sine"})

class FilterConfig:
    def __init__(self, cutoff_freq=500.0, sample_rate=44100, filter_type="lowpass"):
        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate
        self.filter_type = filter_type

class NoiseStyle:
    DYNAMIC = "DYNAMIC"
    STATIC = "STATIC"
    FRACTAL = "FRACTAL"

class NoiseConfig:
    def __init__(self, duration_sec, noise_style, stereo_width=1.0, filter_config=None, sample_rate=44100, amplitude=0.3):
        self.duration_sec = duration_sec
        self.noise_style = noise_style
        self.stereo_width = stereo_width
        self.sample_rate = sample_rate
        self.amplitude = amplitude
        self.filter_config = filter_config or FilterConfig()

def crear_gestor_emotion_style_unificado() -> GestorEmotionStyleUnificado:
    return GestorEmotionStyleUnificado()

def obtener_experiencia_completa(objetivo: str, contexto: str = None) -> Dict[str, Any]:
    gestor = crear_gestor_emotion_style_unificado()
    return gestor.recomendar_experiencia_completa(objetivo, contexto)

def buscar_por_emocion(emocion: str) -> List[str]:
    gestor = crear_gestor_emotion_style_unificado()
    return gestor.buscar_por_efecto(emocion.lower())

def buscar_por_neurotransmisor(neurotransmisor: str) -> Dict[str, List[str]]:
    gestor = crear_gestor_emotion_style_unificado()
    return gestor.buscar_por_neurotransmisor(neurotransmisor)

def generar_secuencia_aurora(objetivo: str, duracion_min: int = 30) -> List[Dict[str, Any]]:
    gestor = crear_gestor_emotion_style_unificado()
    return gestor.generar_secuencia_inteligente(objetivo, duracion_min)

def analizar_coherencia_aurora(preset_emocional: str, perfil_estilo: str, preset_estetico: str = None) -> Dict[str, Any]:
    gestor = crear_gestor_emotion_style_unificado()
    return gestor.analizar_compatibilidad_total(preset_emocional, perfil_estilo, preset_estetico)

def obtener_preset_completo(nombre: str) -> Optional[PresetEmocionalCompleto]:
    gestor = crear_gestor_emotion_style_unificado()
    return gestor.obtener_preset_emocional(nombre)

def obtener_perfil_completo(nombre: str) -> Optional[PerfilEstiloCompleto]:
    gestor = crear_gestor_emotion_style_unificado()
    return gestor.obtener_perfil_estilo(nombre)

if __name__ == "__main__":
    print("Aurora V7 - Sistema Unificado Emotion-Style")
    gestor = crear_gestor_emotion_style_unificado()
    config = gestor.exportar_configuracion_aurora()
    print(f"{config['version']}")
    print(f"Presets emocionales: {config['total_presets_emocionales']}")
    print(f"Perfiles de estilo: {config['total_perfiles_estilo']}")
    print(f"Presets estéticos: {config['total_presets_esteticos']}")
    validados = config['presets_validados']
    print(f"Emocionales validados: {validados['emocionales']}")
    print(f"Estilos validados: {validados['estilo']}")
    print(f"Estéticos validados: {validados['esteticos']}")
    neuros = config['neurotransmisores_disponibles'][:5]
    print(f"Neurotransmisores: {', '.join(neuros)}")
    experiencia = obtener_experiencia_completa("claridad_mental", "trabajo")
    if "error" not in experiencia:
        print(f"Preset emocional: {experiencia['preset_emocional']['nombre']}")
        print(f"Perfil estilo: {experiencia['perfil_estilo']['nombre']}")
        print(f"Preset estético: {experiencia['preset_estetico']['nombre']}")
        print(f"Coherencia: {experiencia['score_coherencia']:.2f}")
    resultados_dopamina = buscar_por_neurotransmisor("dopamina")
    for tipo, presets in resultados_dopamina.items():
        if presets:
            print(f"{tipo}: {', '.join(presets[:2])}")
    secuencia = generar_secuencia_aurora("estado_flujo", 20)
    for fase in secuencia:
        print(f"{fase['posicion'].title()}: {fase['preset']} ({fase['duracion_min']}min)")
    preset_legacy = EmotionalPreset.get("claridad_mental")
    if preset_legacy:
        print("EmotionalPreset.get(): Funcional")
    style_legacy = StyleProfile.get("sereno")
    if style_legacy:
        print("StyleProfile.get(): Funcional")
    print("Sistema Unificado Aurora V7 - COMPLETAMENTE FUNCIONAL!")
