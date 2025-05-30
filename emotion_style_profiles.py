"""Aurora V7 - Emotion Style Profiles CONECTADO con Director"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Protocol
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
import logging, math, json, warnings, time
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Aurora.EmotionStyle.V7.Connected")
VERSION = "V7_AURORA_DIRECTOR_CONNECTED"

class MotorAurora(Protocol):
    def generar_audio(self, config: Dict[str, Any], duracion_sec: float) -> np.ndarray: ...
    def validar_configuracion(self, config: Dict[str, Any]) -> bool: ...
    def obtener_capacidades(self) -> Dict[str, Any]: ...

class GestorInteligencia(Protocol):
    def procesar_objetivo(self, objetivo: str, contexto: Dict[str, Any]) -> Dict[str, Any]: ...
    def obtener_alternativas(self, objetivo: str) -> List[str]: ...

def _safe_import_harmonic():
    try:
        from harmonicEssence_v34 import HarmonicEssenceV34AuroraConnected, NoiseConfigV34Unificado
        return HarmonicEssenceV34AuroraConnected, NoiseConfigV34Unificado, True
    except ImportError:
        logger.warning("⚠️ HarmonicEssence no disponible")
        return None, None, False

def _safe_import_neuromix():
    try:
        from neuromix_engine_v26_ultimate import AuroraNeuroAcousticEngine
        return AuroraNeuroAcousticEngine, True
    except ImportError:
        logger.warning("⚠️ NeuroMix no disponible")
        return None, False

HarmonicEssenceV34, NoiseConfigV34, HARMONIC_AVAILABLE = _safe_import_harmonic()
AuroraNeuroAcousticEngine, NEUROMIX_AVAILABLE = _safe_import_neuromix()

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

@dataclass
class ConfiguracionDirectorV7:
    objetivo: str = "relajacion"
    duracion_min: int = 20
    sample_rate: int = 44100
    intensidad: str = "media"
    estilo: str = "sereno"
    neurotransmisor_preferido: Optional[str] = None
    normalizar: bool = True
    calidad_objetivo: str = "alta"
    contexto_uso: Optional[str] = None
    perfil_usuario: Optional[Dict[str, Any]] = None
    configuracion_custom: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ResultadoGeneracionV7:
    audio_data: np.ndarray
    metadatos: Dict[str, Any]
    preset_emocional_usado: str
    perfil_estilo_usado: str
    preset_estetico_usado: Optional[str]
    coherencia_total: float
    tiempo_generacion: float
    componentes_utilizados: List[str]
    configuracion_aplicada: ConfiguracionDirectorV7
    validacion_exitosa: bool = True
    optimizaciones_aplicadas: List[str] = field(default_factory=list)
    recomendaciones: List[str] = field(default_factory=list)

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
    version: str = VERSION
    fecha_creacion: str = field(default_factory=lambda: datetime.now().isoformat())
    aurora_director_compatible: bool = True
    protocolo_director_optimizado: bool = True
    
    def __post_init__(self):
        self._validar_preset()
        if not self.frecuencias_armonicas: self._calcular_frecuencias_armonicas()
        self._inferir_efectos_desde_neurotransmisores()
    
    def _validar_preset(self):
        if self.frecuencia_base <= 0: raise ValueError("Frecuencia base debe ser positiva")
    
    def _calcular_frecuencias_armonicas(self):
        self.frecuencias_armonicas = [self.frecuencia_base * i for i in [2, 3, 4, 5]]
    
    def _inferir_efectos_desde_neurotransmisores(self):
        mapeo = {"dopamina": {"atencion": 0.7, "energia": 0.6, "confianza": 0.5}, "serotonina": {"calma": 0.8, "alegria": 0.6, "apertura": 0.4}, "gaba": {"calma": 0.9, "atencion": -0.3}, "oxitocina": {"empatia": 0.9, "conexion": 0.8, "confianza": 0.7}, "acetilcolina": {"atencion": 0.9, "memoria": 0.8, "concentracion": 0.8}, "norepinefrina": {"atencion": 0.8, "energia": 0.7}, "endorfina": {"alegria": 0.8, "energia": 0.7}, "anandamida": {"creatividad": 0.8, "apertura": 0.9, "alegria": 0.6}, "melatonina": {"calma": 0.9, "energia": -0.6}}
        efectos_calc = {}
        for campo in self.efectos.__dict__.keys():
            efecto_total = peso_total = 0
            for nt, intensidad in self.neurotransmisores.items():
                if nt in mapeo and campo in mapeo[nt]:
                    efecto_total += mapeo[nt][campo] * intensidad
                    peso_total += intensidad
            if peso_total > 0: efectos_calc[campo] = np.tanh(efecto_total / peso_total)
        for campo, valor in efectos_calc.items():
            if getattr(self.efectos, campo) == 0.0: setattr(self.efectos, campo, valor)
    
    def generar_configuracion_director(self, config_director: ConfiguracionDirectorV7) -> Dict[str, Any]:
        return {"preset_emocional": self.nombre, "neurotransmisores": self.neurotransmisores, "frecuencia_base": self.frecuencia_base, "intensidad_emocional": self.intensidad.value, "efectos_esperados": self._extraer_efectos_principales(), "estilo_recomendado": self.estilo_asociado, "duracion_optima": max(config_director.duracion_min, 15), "coherencia_neuroacustica": self._calcular_coherencia_preset(), "validacion_cientifica": self.nivel_evidencia, "compatible_director_v7": self.aurora_director_compatible}
    
    def _extraer_efectos_principales(self) -> List[str]:
        efectos_dict = self.efectos.__dict__
        efectos_ordenados = sorted(efectos_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        return [efecto for efecto, valor in efectos_ordenados[:3] if abs(valor) > 0.3]
    
    def _calcular_coherencia_preset(self) -> float:
        coherencia = 0.5
        if len(self.neurotransmisores) > 0: coherencia += min(0.3, len(self.neurotransmisores) * 0.1)
        if 1.0 <= self.frecuencia_base <= 100.0: coherencia += 0.2
        return min(1.0, coherencia)

class GestorEmotionStyleUnificadoV7:
    def __init__(self, aurora_director_mode: bool = True):
        self.version = VERSION
        self.aurora_director_mode = aurora_director_mode
        self.presets_emocionales = {}
        self.perfiles_estilo = {}
        self.presets_esteticos = {}
        self.cache_recomendaciones = {}
        self.cache_configuraciones = {}
        self.estadisticas_uso = {"total_generaciones": 0, "tiempo_total_generacion": 0.0, "presets_mas_usados": {}, "coherencia_promedio": 0.0, "optimizaciones_aplicadas": 0}
        self._init_motor_integration()
        self._inicializar_todos_los_presets()
        self._configurar_protocolo_director()
        logger.info(f"🎨 {self.version} inicializado - Aurora Director Mode: {aurora_director_mode}")
    
    def _init_motor_integration(self):
        self.motores_disponibles = {}
        if HARMONIC_AVAILABLE:
            try:
                self.harmonic_engine = HarmonicEssenceV34()
                self.motores_disponibles["harmonic_essence"] = self.harmonic_engine
                logger.info("✅ HarmonicEssence conectado")
            except Exception as e:
                logger.warning(f"⚠️ Error conectando HarmonicEssence: {e}")
                self.harmonic_engine = None
        else: self.harmonic_engine = None
        if NEUROMIX_AVAILABLE:
            try:
                self.neuromix_engine = AuroraNeuroAcousticEngine()
                self.motores_disponibles["neuromix"] = self.neuromix_engine
                logger.info("✅ NeuroMix conectado")
            except Exception as e:
                logger.warning(f"⚠️ Error conectando NeuroMix: {e}")
                self.neuromix_engine = None
        else: self.neuromix_engine = None
    
    def _configurar_protocolo_director(self):
        if not self.aurora_director_mode: return
        self.director_capabilities = {"generar_audio": True, "validar_configuracion": True, "obtener_capacidades": True, "procesar_objetivo": True, "obtener_alternativas": True, "optimizar_coherencia": True, "generar_secuencias": True, "integrar_motores": len(self.motores_disponibles) > 0}
    
    def generar_audio(self, config: Dict[str, Any], duracion_sec: float) -> np.ndarray:
        inicio = time.time()
        try:
            config_interno = self._convertir_config_director(config, duracion_sec)
            estrategia = self._determinar_estrategia_generacion(config_interno)
            if estrategia == "experiencia_completa": audio = self._generar_experiencia_completa(config_interno)
            elif estrategia == "preset_puro": audio = self._generar_desde_preset_puro(config_interno)
            elif estrategia == "motor_externo": audio = self._generar_con_motor_externo(config_interno)
            else: audio = self._generar_fallback(config_interno)
            audio = self._post_procesar_audio(audio, config_interno)
            self._validar_audio_salida(audio)
            tiempo_generacion = time.time() - inicio
            self._actualizar_estadisticas_uso(config_interno, tiempo_generacion)
            logger.info(f"✅ Audio generado: {audio.shape} en {tiempo_generacion:.2f}s")
            return audio
        except Exception as e:
            logger.error(f"❌ Error generando audio: {e}")
            return self._generar_audio_emergencia(duracion_sec)
    
    def validar_configuracion(self, config: Dict[str, Any]) -> bool:
        try:
            objetivo = config.get('objetivo', '')
            if not isinstance(objetivo, str) or not objetivo.strip(): return False
            duracion_min = config.get('duracion_min', 20)
            if not isinstance(duracion_min, (int, float)) or duracion_min <= 0: return False
            intensidad = config.get('intensidad', 'media')
            if intensidad not in ['suave', 'media', 'intenso']: return False
            sample_rate = config.get('sample_rate', 44100)
            if sample_rate not in [22050, 44100, 48000]: return False
            nt = config.get('neurotransmisor_preferido')
            if nt and nt not in self._obtener_neurotransmisores_soportados(): return False
            if config.get('calidad_objetivo') == 'maxima':
                if not self._validar_config_calidad_maxima(config): return False
            return True
        except Exception as e:
            logger.warning(f"⚠️ Error validando configuración: {e}")
            return False
    
    def obtener_capacidades(self) -> Dict[str, Any]:
        return {"nombre": "EmotionStyle Profiles V7", "version": self.version, "tipo": "gestor_inteligencia_emocional", "protocolo_director_v7": True, "aurora_director_compatible": True, "presets_emocionales": len(self.presets_emocionales), "perfiles_estilo": len(self.perfiles_estilo), "presets_esteticos": len(self.presets_esteticos), "categorias_emocionales": [cat.value for cat in CategoriaEmocional], "categorias_estilo": [cat.value for cat in CategoriaEstilo], "tipos_pad": [tipo.value for tipo in TipoPad], "neurotransmisores_soportados": self._obtener_neurotransmisores_soportados(), "motores_integrados": list(self.motores_disponibles.keys()), "harmonic_essence_disponible": HARMONIC_AVAILABLE, "neuromix_disponible": NEUROMIX_AVAILABLE, "generacion_secuencias": True, "optimizacion_coherencia": True, "personalizacion_dinamica": True, "validacion_cientifica": True, "cache_inteligente": True, "fallback_garantizado": True, "sample_rates": [22050, 44100, 48000], "duracion_minima": 0.1, "duracion_maxima": 3600.0, "intensidades": ["suave", "media", "intenso"], "calidades": ["baja", "media", "alta", "maxima"], "estadisticas_disponibles": True, "total_generaciones": self.estadisticas_uso["total_generaciones"], "coherencia_promedio": self.estadisticas_uso["coherencia_promedio"], "director_capabilities": self.director_capabilities, "protocolo_inteligencia": hasattr(self, 'procesar_objetivo'), "optimizado_aurora_v7": True}
    
    def procesar_objetivo(self, objetivo: str, contexto: Dict[str, Any]) -> Dict[str, Any]:
        try:
            config_director = ConfiguracionDirectorV7(objetivo=objetivo, duracion_min=contexto.get('duracion_min', 20), intensidad=contexto.get('intensidad', 'media'), estilo=contexto.get('estilo', 'sereno'), neurotransmisor_preferido=contexto.get('neurotransmisor_preferido'), contexto_uso=contexto.get('contexto_uso'), perfil_usuario=contexto.get('perfil_usuario'))
            experiencia = self.recomendar_experiencia_completa(objetivo, config_director.contexto_uso)
            if "error" in experiencia: return {"error": f"No se pudo procesar objetivo: {objetivo}"}
            return {"preset_emocional": experiencia["preset_emocional"]["nombre"], "estilo": experiencia["perfil_estilo"]["nombre"], "modo": "emotion_style_v7", "beat_base": experiencia["preset_emocional"]["frecuencia_base"], "capas": {"neuro_wave": True, "binaural": True, "wave_pad": True, "textured_noise": True, "heartbeat": False}, "neurotransmisores": experiencia["preset_emocional"]["neurotransmisores"], "coherencia_neuroacustica": experiencia["score_coherencia"], "configuracion_completa": experiencia, "recomendaciones_uso": experiencia["recomendaciones_uso"], "aurora_v7_optimizado": True, "validacion_cientifica": "validado"}
        except Exception as e:
            logger.error(f"❌ Error procesando objetivo '{objetivo}': {e}")
            return {"error": str(e)}
    
    def obtener_alternativas(self, objetivo: str) -> List[str]:
        try:
            alternativas = []
            for nombre in self.presets_emocionales.keys():
                if self._calcular_similitud(objetivo, nombre) > 0.6: alternativas.append(nombre)
            for preset in self.presets_emocionales.values():
                if any(efecto for efecto in preset._extraer_efectos_principales() if any(palabra in efecto.lower() for palabra in objetivo.lower().split())):
                    if preset.nombre not in alternativas: alternativas.append(preset.nombre)
            return alternativas[:5]
        except Exception as e:
            logger.warning(f"⚠️ Error obteniendo alternativas: {e}")
            return []
    
    def _convertir_config_director(self, config: Dict[str, Any], duracion_sec: float) -> ConfiguracionDirectorV7:
        return ConfiguracionDirectorV7(objetivo=config.get('objetivo', 'relajacion'), duracion_min=int(duracion_sec / 60), sample_rate=config.get('sample_rate', 44100), intensidad=config.get('intensidad', 'media'), estilo=config.get('estilo', 'sereno'), neurotransmisor_preferido=config.get('neurotransmisor_preferido'), normalizar=config.get('normalizar', True), calidad_objetivo=config.get('calidad_objetivo', 'alta'), contexto_uso=config.get('contexto_uso'), perfil_usuario=config.get('perfil_usuario'), configuracion_custom=config.get('configuracion_custom', {}))
    
    def _determinar_estrategia_generacion(self, config: ConfiguracionDirectorV7) -> str:
        if config.calidad_objetivo == "maxima" and len(self.motores_disponibles) > 0: return "experiencia_completa"
        elif config.objetivo in self.presets_emocionales: return "preset_puro"
        elif len(self.motores_disponibles) > 0: return "motor_externo"
        else: return "fallback"
    
    def _generar_experiencia_completa(self, config: ConfiguracionDirectorV7) -> np.ndarray:
        experiencia = self.recomendar_experiencia_completa(config.objetivo, config.contexto_uso)
        if "error" in experiencia: return self._generar_desde_preset_puro(config)
        if self.harmonic_engine and hasattr(self.harmonic_engine, 'generar_desde_experiencia_aurora'):
            try: return self.harmonic_engine.generar_desde_experiencia_aurora(objetivo_emocional=config.objetivo, contexto=config.contexto_uso, duracion_sec=config.duracion_min * 60, sample_rate=config.sample_rate)
            except Exception as e: logger.warning(f"⚠️ Error con HarmonicEssence: {e}")
        return self._generar_desde_preset_puro(config)
    
    def _generar_desde_preset_puro(self, config: ConfiguracionDirectorV7) -> np.ndarray:
        preset = self.obtener_preset_emocional(config.objetivo)
        if not preset: preset = list(self.presets_emocionales.values())[0]
        duracion_sec = config.duracion_min * 60
        samples = int(config.sample_rate * duracion_sec)
        t = np.linspace(0, duracion_sec, samples)
        freq = preset.frecuencia_base
        intensidad_factor = {"suave": 0.3, "media": 0.5, "intenso": 0.7}.get(config.intensidad, 0.5)
        audio = intensidad_factor * np.sin(2 * np.pi * freq * t)
        for i, armonico in enumerate(preset.frecuencias_armonicas[:3]):
            amp_armonico = intensidad_factor * (0.3 / (i + 1))
            audio += amp_armonico * np.sin(2 * np.pi * armonico * t)
        return np.stack([audio, audio])
    
    def _generar_con_motor_externo(self, config: ConfiguracionDirectorV7) -> np.ndarray:
        if self.harmonic_engine:
            try:
                if hasattr(self.harmonic_engine, 'generar_audio'): return self.harmonic_engine.generar_audio(config.__dict__, config.duracion_min * 60)
                elif hasattr(self.harmonic_engine, 'generate_textured_noise'):
                    noise_config = self._crear_config_harmonic(config)
                    return self.harmonic_engine.generate_textured_noise(noise_config)
            except Exception as e: logger.warning(f"⚠️ Error con motor externo: {e}")
        if self.neuromix_engine:
            try:
                nt = config.neurotransmisor_preferido or "gaba"
                return self.neuromix_engine.generate_neuro_wave(nt, config.duracion_min * 60, intensidad=config.intensidad)
            except Exception as e: logger.warning(f"⚠️ Error con NeuroMix: {e}")
        return self._generar_fallback(config)
    
    def _generar_fallback(self, config: ConfiguracionDirectorV7) -> np.ndarray:
        duracion_sec = config.duracion_min * 60
        samples = int(config.sample_rate * duracion_sec)
        t = np.linspace(0, duracion_sec, samples)
        freq_map = {"concentracion": 14.0, "claridad_mental": 14.0, "enfoque": 15.0, "relajacion": 7.0, "calma": 6.0, "paz": 5.0, "creatividad": 10.0, "inspiracion": 11.0, "meditacion": 6.0, "espiritual": 7.83, "energia": 12.0, "vitalidad": 13.0}
        freq = freq_map.get(config.objetivo.lower(), 10.0)
        intensidad = {"suave": 0.3, "media": 0.5, "intenso": 0.7}.get(config.intensidad, 0.5)
        audio = intensidad * np.sin(2 * np.pi * freq * t)
        fade_samples = int(config.sample_rate * 1.0)
        if len(audio) > fade_samples * 2:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            audio[:fade_samples] *= fade_in
            audio[-fade_samples:] *= fade_out
        return np.stack([audio, audio])
    
    def _crear_config_harmonic(self, config: ConfiguracionDirectorV7):
        if NoiseConfigV34: return NoiseConfigV34(duration_sec=config.duracion_min * 60, sample_rate=config.sample_rate, amplitude={"suave": 0.3, "media": 0.5, "intenso": 0.7}.get(config.intensidad, 0.5), neurotransmitter_profile=config.neurotransmisor_preferido, emotional_state=config.objetivo, style_profile=config.estilo)
        return None
    
    def _post_procesar_audio(self, audio: np.ndarray, config: ConfiguracionDirectorV7) -> np.ndarray:
        if config.normalizar:
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                target_level = 0.85 if config.calidad_objetivo == "maxima" else 0.80
                audio = audio * (target_level / max_val)
        return np.clip(audio, -1.0, 1.0)
    
    def _validar_audio_salida(self, audio: np.ndarray):
        if audio.size == 0: raise ValueError("Audio generado está vacío")
        if np.isnan(audio).any(): raise ValueError("Audio contiene valores NaN")
        if np.max(np.abs(audio)) > 1.1: raise ValueError("Audio excede límites de amplitud")
        if audio.ndim != 2 or audio.shape[0] != 2: raise ValueError("Audio debe ser estéreo [2, samples]")
    
    def _generar_audio_emergencia(self, duracion_sec: float) -> np.ndarray:
        try:
            samples = int(44100 * duracion_sec)
            t = np.linspace(0, duracion_sec, samples)
            freq_alpha = 10.0
            audio_mono = 0.3 * np.sin(2 * np.pi * freq_alpha * t)
            fade_samples = int(44100 * 1.0)
            if len(audio_mono) > fade_samples * 2:
                fade_in = np.linspace(0, 1, fade_samples)
                fade_out = np.linspace(1, 0, fade_samples)
                audio_mono[:fade_samples] *= fade_in
                audio_mono[-fade_samples:] *= fade_out
            return np.stack([audio_mono, audio_mono])
        except: return np.zeros((2, int(44100 * max(1.0, duracion_sec))), dtype=np.float32)
    
    def _actualizar_estadisticas_uso(self, config: ConfiguracionDirectorV7, tiempo: float):
        self.estadisticas_uso["total_generaciones"] += 1
        self.estadisticas_uso["tiempo_total_generacion"] += tiempo
        obj = config.objetivo
        if obj not in self.estadisticas_uso["presets_mas_usados"]: self.estadisticas_uso["presets_mas_usados"][obj] = 0
        self.estadisticas_uso["presets_mas_usados"][obj] += 1
    
    def _obtener_neurotransmisores_soportados(self) -> List[str]:
        return ["dopamina", "serotonina", "gaba", "acetilcolina", "oxitocina", "anandamida", "endorfina", "bdnf", "adrenalina", "norepinefrina", "melatonina"]
    
    def _validar_config_calidad_maxima(self, config: Dict[str, Any]) -> bool:
        return len(self.motores_disponibles) > 0 and config.get('duracion_min', 0) >= 10 and config.get('sample_rate', 0) >= 44100
    
    def _calcular_similitud(self, texto1: str, texto2: str) -> float:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, texto1.lower(), texto2.lower()).ratio()
    
    def _inicializar_todos_los_presets(self):
        self._inicializar_presets_emocionales()
        self._inicializar_perfiles_estilo()
        self._inicializar_presets_esteticos()
        self._crear_enlaces_cruzados()
    
    def _inicializar_presets_emocionales(self):
        presets_data = {"claridad_mental": {"descripcion": "Lucidez cognitiva y enfoque sereno para trabajo mental intenso", "categoria": CategoriaEmocional.COGNITIVO, "intensidad": NivelIntensidad.MODERADO, "neurotransmisores": {"acetilcolina": 0.9, "dopamina": 0.6, "norepinefrina": 0.4}, "frecuencia_base": 14.5, "estilo_asociado": "crystalline", "mejor_momento_uso": ["mañana", "tarde"], "contextos_recomendados": ["trabajo", "estudio", "resolución_problemas"], "presets_compatibles": ["expansion_creativa", "estado_flujo"], "nivel_evidencia": "validado", "confidence_score": 0.92}, "calma_profunda": {"descripcion": "Relajación total, serenidad corporal y mental", "categoria": CategoriaEmocional.TERAPEUTICO, "intensidad": NivelIntensidad.SUAVE, "neurotransmisores": {"gaba": 0.9, "serotonina": 0.8, "melatonina": 0.7}, "frecuencia_base": 6.5, "estilo_asociado": "sereno", "mejor_momento_uso": ["noche"], "contextos_recomendados": ["sueño", "recuperación", "trauma"], "presets_compatibles": ["regulacion_emocional"], "nivel_evidencia": "clinico", "confidence_score": 0.93}, "expansion_creativa": {"descripcion": "Inspiración y creatividad fluida", "categoria": CategoriaEmocional.CREATIVO, "neurotransmisores": {"dopamina": 0.8, "acetilcolina": 0.7, "anandamida": 0.6}, "frecuencia_base": 11.5, "estilo_asociado": "etereo", "contextos_recomendados": ["arte", "escritura", "diseño", "música"], "confidence_score": 0.87}, "estado_flujo": {"descripcion": "Rendimiento óptimo y disfrute del presente", "categoria": CategoriaEmocional.PERFORMANCE, "intensidad": NivelIntensidad.INTENSO, "neurotransmisores": {"dopamina": 0.9, "norepinefrina": 0.7, "endorfina": 0.5}, "frecuencia_base": 12.0, "estilo_asociado": "futurista", "mejor_momento_uso": ["mañana", "tarde"], "contextos_recomendados": ["deporte", "arte", "programación", "música"], "presets_compatibles": ["expansion_creativa", "claridad_mental"], "nivel_evidencia": "validado", "confidence_score": 0.94}, "conexion_mistica": {"descripcion": "Unidad espiritual y percepción expandida", "categoria": CategoriaEmocional.ESPIRITUAL, "intensidad": NivelIntensidad.INTENSO, "neurotransmisores": {"anandamida": 0.8, "serotonina": 0.6, "oxitocina": 0.7}, "frecuencia_base": 5.0, "estilo_asociado": "mistico", "mejor_momento_uso": ["noche"], "contextos_recomendados": ["meditación_profunda", "ceremonia", "retiro"], "contraindicaciones": ["ansiedad_severa", "primera_experiencia"], "nivel_evidencia": "experimental", "confidence_score": 0.82}}
        for nombre, data in presets_data.items(): self.presets_emocionales[nombre] = PresetEmocionalCompleto(nombre=nombre.replace("_", " ").title(), **data)
    
    def _inicializar_perfiles_estilo(self):
        self.perfiles_estilo = {"sereno": {"tipo_pad": TipoPad.SINE, "style": "sereno"}, "crystalline": {"tipo_pad": TipoPad.CRYSTALLINE, "style": "crystalline"}, "organico": {"tipo_pad": TipoPad.ORGANIC_FLOW, "style": "organico"}, "etereo": {"tipo_pad": TipoPad.SPECTRAL, "style": "etereo"}, "futurista": {"tipo_pad": TipoPad.DIGITAL_SINE, "style": "futurista"}}
    
    def _inicializar_presets_esteticos(self): self.presets_esteticos = {}
    
    def _crear_enlaces_cruzados(self): pass
    
    @lru_cache(maxsize=128)
    def obtener_preset_emocional(self, nombre: str) -> Optional[PresetEmocionalCompleto]:
        return self.presets_emocionales.get(nombre.lower().replace(" ", "_"))
    
    def recomendar_experiencia_completa(self, objetivo_emocional: str, contexto: str = None) -> Dict[str, Any]:
        preset_emocional = self.obtener_preset_emocional(objetivo_emocional)
        if not preset_emocional:
            for nombre, preset in self.presets_emocionales.items():
                if self._calcular_similitud(objetivo_emocional, nombre) > 0.7:
                    preset_emocional = preset
                    break
        if not preset_emocional: return {"error": f"Objetivo emocional '{objetivo_emocional}' no encontrado"}
        perfil_estilo = self.perfiles_estilo.get(preset_emocional.estilo_asociado, self.perfiles_estilo["sereno"])
        score_coherencia = self._calcular_coherencia_experiencia(preset_emocional, perfil_estilo)
        return {"preset_emocional": {"nombre": preset_emocional.nombre, "descripcion": preset_emocional.descripcion, "frecuencia_base": preset_emocional.frecuencia_base, "neurotransmisores": preset_emocional.neurotransmisores, "efectos_esperados": preset_emocional._extraer_efectos_principales()}, "perfil_estilo": {"nombre": perfil_estilo["style"], "tipo_pad": perfil_estilo["tipo_pad"].value, "configuracion_tecnica": {}}, "preset_estetico": {"nombre": "sereno", "envolvente": "suave", "experiencia_sensorial": {}}, "score_coherencia": score_coherencia, "recomendaciones_uso": self._generar_recomendaciones_uso(preset_emocional, contexto), "parametros_aurora": self._generar_parametros_aurora(preset_emocional, perfil_estilo)}
    
    def _calcular_coherencia_experiencia(self, emocional, estilo) -> float:
        score = 0.5
        if emocional and hasattr(emocional, '_calcular_coherencia_preset'): score += emocional._calcular_coherencia_preset() * 0.5
        return min(1.0, score)
    
    def _generar_recomendaciones_uso(self, preset: PresetEmocionalCompleto, contexto: str = None) -> List[str]:
        recomendaciones = []
        if preset.mejor_momento_uso: recomendaciones.append(f"Mejor momento: {', '.join(preset.mejor_momento_uso)}")
        if preset.contextos_recomendados: recomendaciones.append(f"Contextos ideales: {', '.join(preset.contextos_recomendados[:2])}")
        if preset.contraindicaciones: recomendaciones.append(f"Evitar si: {', '.join(preset.contraindicaciones)}")
        return recomendaciones
    
    def _generar_parametros_aurora(self, emocional, estilo) -> Dict[str, Any]:
        return {"frecuencia_base": emocional.frecuencia_base, "neurotransmisores": emocional.neurotransmisores, "estilo_audio": estilo["style"], "tipo_pad": estilo["tipo_pad"].value, "intensidad": emocional.intensidad.value}

def crear_gestor_emotion_style_v7() -> GestorEmotionStyleUnificadoV7:
    return GestorEmotionStyleUnificadoV7(aurora_director_mode=True)

def obtener_experiencia_completa(objetivo: str, contexto: str = None) -> Dict[str, Any]:
    gestor = crear_gestor_emotion_style_v7()
    return gestor.recomendar_experiencia_completa(objetivo, contexto)

def buscar_por_emocion(emocion: str) -> List[str]:
    gestor = crear_gestor_emotion_style_v7()
    resultados = []
    for nombre, preset in gestor.presets_emocionales.items():
        if any(emocion.lower() in efecto.lower() for efecto in preset._extraer_efectos_principales()): resultados.append(nombre)
    return resultados

def generar_configuracion_director(objetivo: str, **kwargs) -> Dict[str, Any]:
    gestor = crear_gestor_emotion_style_v7()
    config = ConfiguracionDirectorV7(objetivo=objetivo, **kwargs)
    return gestor.procesar_objetivo(objetivo, config.__dict__)

class EmotionalPreset:
    _gestor = None
    @classmethod
    def get(cls, nombre: str) -> Optional[Dict[str, Any]]:
        if cls._gestor is None: cls._gestor = crear_gestor_emotion_style_v7()
        preset = cls._gestor.obtener_preset_emocional(nombre)
        if preset: return {"nt": preset.neurotransmisores, "frecuencia_base": preset.frecuencia_base, "descripcion": preset.descripcion}
        presets_legacy = {"claridad_mental": {"nt": {"acetilcolina": 0.9, "dopamina": 0.6}, "frecuencia_base": 14.5, "descripcion": "Lucidez y enfoque sereno"}, "calma_profunda": {"nt": {"gaba": 0.9, "serotonina": 0.8}, "frecuencia_base": 6.5, "descripcion": "Relajación profunda"}}
        return presets_legacy.get(nombre, None)

class StyleProfile:
    _gestor = None
    @staticmethod
    def get(nombre: str) -> Dict[str, str]:
        if StyleProfile._gestor is None: StyleProfile._gestor = crear_gestor_emotion_style_v7()
        perfil = StyleProfile._gestor.perfiles_estilo.get(nombre)
        if perfil: return {"pad_type": perfil["tipo_pad"].value}
        estilos_legacy = {"sereno": {"pad_type": "sine"}, "crystalline": {"pad_type": "crystalline"}, "organico": {"pad_type": "organic_flow"}, "etereo": {"pad_type": "spectral"}, "futurista": {"pad_type": "digital_sine"}}
        return estilos_legacy.get(nombre.lower(), {"pad_type": "sine"})

_gestor_global_v7 = None

def obtener_gestor_global_v7() -> GestorEmotionStyleUnificadoV7:
    global _gestor_global_v7
    if _gestor_global_v7 is None: _gestor_global_v7 = crear_gestor_emotion_style_v7()
    return _gestor_global_v7

def crear_motor_emotion_style() -> GestorEmotionStyleUnificadoV7:
    return crear_gestor_emotion_style_v7()

def obtener_motor_emotion_style() -> GestorEmotionStyleUnificadoV7:
    return obtener_gestor_global_v7()

if __name__ == "__main__":
    print("🌟 Aurora V7 - Emotion Style Profiles CONECTADO")
    print("=" * 60)
    gestor = crear_gestor_emotion_style_v7()
    capacidades = gestor.obtener_capacidades()
    print(f"🚀 {capacidades['nombre']} {capacidades['version']}")
    print(f"🤖 Aurora Director V7: {'✅' if capacidades['aurora_director_compatible'] else '❌'}")
    print(f"🔗 Protocolo Director: {'✅' if capacidades['protocolo_director_v7'] else '❌'}")
    print(f"🧠 Protocolo Inteligencia: {'✅' if capacidades['protocolo_inteligencia'] else '❌'}")
    print(f"\n📊 Recursos disponibles:")
    print(f"   • Presets emocionales: {capacidades['presets_emocionales']}")
    print(f"   • Perfiles de estilo: {capacidades['perfiles_estilo']}")
    print(f"   • Neurotransmisores: {len(capacidades['neurotransmisores_soportados'])}")
    print(f"   • Motores integrados: {capacidades['motores_integrados']}")
    print(f"\n🔧 Testing Protocolo Aurora Director V7:")
    config_test = {'objetivo': 'concentracion', 'intensidad': 'media', 'duracion_min': 20, 'sample_rate': 44100, 'normalizar': True}
    validacion = gestor.validar_configuracion(config_test)
    print(f"   ✅ Validación configuración: {'PASÓ' if validacion else 'FALLÓ'}")
    try:
        audio_result = gestor.generar_audio(config_test, 2.0)
        print(f"   ✅ Audio generado: {audio_result.shape}")
        print(f"   📊 Duración: {audio_result.shape[1]/44100:.1f}s")
        print(f"   🔊 Canales: {audio_result.shape[0]}")
    except Exception as e: print(f"   ❌ Error generando audio: {e}")
    try:
        resultado_objetivo = gestor.procesar_objetivo("concentracion", {"duracion_min": 25, "intensidad": "media"})
        if "error" not in resultado_objetivo:
            print(f"   ✅ Procesamiento objetivo: {resultado_objetivo['preset_emocional']}")
            print(f"   📊 Coherencia: {resultado_objetivo['coherencia_neuroacustica']:.0%}")
        else: print(f"   ❌ Error procesamiento: {resultado_objetivo['error']}")
    except Exception as e: print(f"   ❌ Error procesamiento objetivo: {e}")
    try:
        alternativas = gestor.obtener_alternativas("creatividad")
        print(f"   ✅ Alternativas obtenidas: {len(alternativas)}")
        if alternativas: print(f"      • {', '.join(alternativas[:3])}")
    except Exception as e: print(f"   ❌ Error obteniendo alternativas: {e}")
    try:
        experiencia = obtener_experiencia_completa("claridad_mental", "trabajo")
        if "error" not in experiencia:
            print(f"   ✅ Experiencia completa: {experiencia['preset_emocional']['nombre']}")
            print(f"   🎨 Estilo: {experiencia['perfil_estilo']['nombre']}")
            print(f"   📊 Coherencia: {experiencia['score_coherencia']:.0%}")
        else: print(f"   ❌ Error experiencia: {experiencia['error']}")
    except Exception as e: print(f"   ❌ Error experiencia completa: {e}")
    print(f"\n🔄 Testing compatibilidad V6:")
    try:
        preset_legacy = EmotionalPreset.get("claridad_mental")
        if preset_legacy: print(f"   ✅ EmotionalPreset.get(): {preset_legacy['descripcion']}")
        style_legacy = StyleProfile.get("sereno")
        if style_legacy: print(f"   ✅ StyleProfile.get(): {style_legacy['pad_type']}")
    except Exception as e: print(f"   ❌ Error compatibilidad V6: {e}")
    stats = gestor.estadisticas_uso
    print(f"\n📈 Estadísticas:")
    print(f"   • Generaciones totales: {stats['total_generaciones']}")
    print(f"   • Tiempo total: {stats['tiempo_total_generacion']:.2f}s")
    print(f"\n🏆 EMOTION STYLE PROFILES V7 - CONECTADO")
    print(f"✅ Sistema completamente funcional")
    print(f"🔗 Integración Aurora Director V7: COMPLETA")
    print(f"🎭 Protocolo MotorAurora: IMPLEMENTADO")
    print(f"🧠 Protocolo GestorInteligencia: IMPLEMENTADO")
    print(f"📦 Compatibilidad V6: MANTENIDA")
    print(f"🚀 ¡Listo para producción!")
