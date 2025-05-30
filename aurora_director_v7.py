"""Aurora Director V7 OPTIMIZADO - CEREBRO PRINCIPAL MEJORADO"""
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Protocol
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import time
import importlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Aurora.Director.V7.Optimized")

class MotorAurora(Protocol):
    def generar_audio(self, config: Dict[str, Any], duracion_sec: float) -> np.ndarray: ...
    def validar_configuracion(self, config: Dict[str, Any]) -> bool: ...
    def obtener_capacidades(self) -> Dict[str, Any]: ...

class GestorInteligencia(Protocol):
    def procesar_objetivo(self, objetivo: str, contexto: Dict[str, Any]) -> Dict[str, Any]: ...
    def obtener_alternativas(self, objetivo: str) -> List[str]: ...

@dataclass
class ComponenteAurora:
    nombre: str
    tipo: str
    modulo: str
    clase_principal: str
    disponible: bool = False
    instancia: Optional[Any] = None
    version: str = "unknown"
    capacidades: Dict[str, Any] = field(default_factory=dict)
    dependencias: List[str] = field(default_factory=list)
    fallback_disponible: bool = False
    nivel_prioridad: int = 1

class DetectorComponentes:
    def __init__(self):
        self.componentes_registrados = self._init_registro()
        self.componentes_activos: Dict[str, ComponenteAurora] = {}
        self.stats = {"total": 0, "exitosos": 0, "fallidos": 0, "fallback": 0}
    
    def _init_registro(self) -> Dict[str, ComponenteAurora]:
        return {
            "field_profiles": ComponenteAurora("field_profiles", "gestor_inteligencia", "field_profiles", "GestorPerfilesCampo", dependencias=[], fallback_disponible=True, nivel_prioridad=2),
            "objective_router": ComponenteAurora("objective_router", "gestor_inteligencia", "objective_router", "RouterInteligenteV7", dependencias=["field_profiles"], fallback_disponible=True, nivel_prioridad=2),
            "neuromix": ComponenteAurora("neuromix", "motor", "neuromix_engine_v26_ultimate", "AuroraNeuroAcousticEngine", dependencias=[], fallback_disponible=True, nivel_prioridad=1),
            "hypermod": ComponenteAurora("hypermod", "motor", "hypermod_engine_v31", "NeuroWaveGenerator", dependencias=[], fallback_disponible=True, nivel_prioridad=1),
            "harmonic_essence": ComponenteAurora("harmonic_essence", "motor", "harmonicEssence_v33py", "HarmonicEssenceV34AuroraConnected", dependencias=[], fallback_disponible=True, nivel_prioridad=1),
            "emotion_style": ComponenteAurora("emotion_style", "gestor_inteligencia", "emotion_style_profiles", "GestorEmotionStyleUnificado", dependencias=[], fallback_disponible=False, nivel_prioridad=3),
            "quality_pipeline": ComponenteAurora("quality_pipeline", "pipeline", "aurora_quality_pipeline", "AuroraQualityPipeline", dependencias=[], fallback_disponible=True, nivel_prioridad=4)
        }
    
    def detectar_todos(self) -> Dict[str, ComponenteAurora]:
        logger.info("🔍 Detectando componentes...")
        for nombre, comp in sorted(self.componentes_registrados.items(), key=lambda x: x[1].nivel_prioridad):
            self._detectar_comp(comp)
        self._log_resultado()
        return self.componentes_activos
    
    def _detectar_comp(self, comp: ComponenteAurora) -> bool:
        self.stats["total"] += 1
        try:
            if not self._check_deps(comp): return False
            modulo = importlib.import_module(comp.modulo)
            instancia = self._init_instancia(modulo, comp)
            if self._validar_instancia(instancia, comp):
                comp.disponible = True
                comp.instancia = instancia
                comp.capacidades = self._get_caps(instancia)
                comp.version = self._get_version(instancia)
                self.componentes_activos[comp.nombre] = comp
                self.stats["exitosos"] += 1
                logger.info(f"✅ {comp.nombre} v{comp.version}")
                return True
            raise Exception("Instancia inválida")
        except Exception as e:
            logger.warning(f"❌ {comp.nombre}: {e}")
            if comp.fallback_disponible and self._crear_fallback(comp):
                self.stats["fallback"] += 1
                logger.info(f"🔄 {comp.nombre} fallback")
                return True
            self.stats["fallidos"] += 1
            return False
    
    def _check_deps(self, comp: ComponenteAurora) -> bool:
        return all(dep in self.componentes_activos for dep in comp.dependencias)
    
    def _init_instancia(self, modulo: Any, comp: ComponenteAurora) -> Any:
        if comp.nombre == "neuromix": return getattr(modulo, "AuroraNeuroAcousticEngine")()
        elif comp.nombre == "hypermod": return modulo
        elif comp.nombre == "harmonic_essence": return getattr(modulo, "HarmonicEssenceV34AuroraConnected")()
        else:
            for func in [f"crear_gestor_{comp.nombre}", f"crear_{comp.nombre}", "crear_gestor", "obtener_gestor"]:
                if hasattr(modulo, func): return getattr(modulo, func)()
            return getattr(modulo, comp.clase_principal)()
    
    def _validar_instancia(self, inst: Any, comp: ComponenteAurora) -> bool:
        try:
            if comp.tipo == "motor": return hasattr(inst, 'generate_neuro_wave') or hasattr(inst, 'generar_bloques') or hasattr(inst, 'generate_textured_noise')
            elif comp.tipo == "gestor_inteligencia": return hasattr(inst, 'obtener_perfil') or hasattr(inst, 'rutear_objetivo') or hasattr(inst, 'procesar_objetivo')
            elif comp.tipo == "pipeline": return hasattr(inst, 'validar_y_normalizar')
            return True
        except: return False
    
    def _get_caps(self, inst: Any) -> Dict[str, Any]:
        try:
            if hasattr(inst, 'obtener_capacidades'): return inst.obtener_capacidades()
            elif hasattr(inst, 'get_capabilities'): return inst.get_capabilities()
        except: pass
        return {}
    
    def _get_version(self, inst: Any) -> str:
        for attr in ['version', 'VERSION', '__version__']:
            if hasattr(inst, attr): return getattr(inst, attr)
        return "unknown"
    
    def _crear_fallback(self, comp: ComponenteAurora) -> bool:
        try:
            fallbacks = {
                "neuromix": self._fallback_neuromix,
                "harmonic_essence": self._fallback_harmonic,
                "quality_pipeline": self._fallback_quality,
                "field_profiles": self._fallback_profiles,
                "objective_router": self._fallback_router
            }
            if comp.nombre in fallbacks:
                comp.instancia = fallbacks[comp.nombre]()
                comp.disponible = True
                comp.version = "fallback"
                self.componentes_activos[comp.nombre] = comp
                return True
        except: pass
        return False
    
    def _fallback_neuromix(self):
        class NeuroMixFallback:
            def generate_neuro_wave(self, nt: str, dur: float, **kw) -> np.ndarray:
                samples = int(44100 * dur)
                t = np.linspace(0, dur, samples)
                wave = 0.3 * np.sin(2 * np.pi * 10.0 * t)
                return np.stack([wave, wave])
            def get_neuro_preset_scientific(self, nt: str, **kw) -> Dict[str, Any]:
                return {"carrier": 10.0, "beat_freq": 6.0, "am_depth": 0.5}
        return NeuroMixFallback()
    
    def _fallback_harmonic(self):
        class HarmonicFallback:
            def generate_textured_noise(self, config, **kw) -> np.ndarray:
                dur = getattr(config, 'duration_sec', 10.0)
                samples = int(44100 * dur)
                noise = np.random.normal(0, 0.1, samples)
                return np.stack([noise, noise])
        return HarmonicFallback()
    
    def _fallback_quality(self):
        class QualityFallback:
            def validar_y_normalizar(self, signal: np.ndarray) -> np.ndarray:
                if signal.ndim == 1: signal = np.stack([signal, signal])
                max_val = np.max(np.abs(signal))
                if max_val > 0: signal = signal * (0.85 / max_val)
                return np.clip(signal, -1.0, 1.0)
        return QualityFallback()
    
    def _fallback_profiles(self):
        class ProfilesFallback:
            def obtener_perfil(self, nombre: str): return None
            def recomendar_secuencia_perfiles(self, obj: str, dur: int): return [(obj, dur)]
        return ProfilesFallback()
    
    def _fallback_router(self):
        class RouterFallback:
            def rutear_objetivo(self, obj: str, **kw): return {"preset_emocional": "calma_profunda", "estilo": "sereno", "modo": "normal", "beat_base": 8.0, "capas": {"neuro_wave": True, "binaural": True}}
        return RouterFallback()
    
    def _log_resultado(self):
        total = len(self.componentes_registrados)
        activos = len(self.componentes_activos)
        logger.info(f"📊 Componentes: {activos}/{total} ({activos/total*100:.0f}%)")
        logger.info(f"  ✅{self.stats['exitosos']} 🔄{self.stats['fallback']} ❌{self.stats['fallidos']}")

class EstrategiaGeneracion(Enum):
    AURORA_COMPLETO = "aurora_completo"
    INTELIGENCIA_ACTIVA = "inteligencia_activa"
    MOTORES_PUROS = "motores_puros"
    FALLBACK_GARANTIZADO = "fallback_garantizado"

@dataclass
class ConfiguracionAurora:
    objetivo: str = "relajacion"
    duracion_min: int = 20
    sample_rate: int = 44100
    estrategia_preferida: Optional[EstrategiaGeneracion] = None
    forzar_componentes: List[str] = field(default_factory=list)
    excluir_componentes: List[str] = field(default_factory=list)
    intensidad: str = "media"
    estilo: str = "sereno"
    neurotransmisor_preferido: Optional[str] = None
    normalizar: bool = True
    calidad_objetivo: str = "alta"
    exportar_wav: bool = True
    nombre_archivo: str = "aurora_experience"
    incluir_metadatos: bool = True
    configuracion_custom: Dict[str, Any] = field(default_factory=dict)
    perfil_usuario: Optional[Dict[str, Any]] = None
    contexto_uso: Optional[str] = None
    
    def validar(self) -> List[str]:
        problemas = []
        if self.duracion_min <= 0: problemas.append("Duración debe ser positiva")
        if self.sample_rate not in [22050, 44100, 48000]: problemas.append("Sample rate no estándar")
        if self.intensidad not in ["suave", "media", "intenso"]: problemas.append("Intensidad inválida")
        return problemas

@dataclass
class ResultadoAurora:
    audio_data: np.ndarray
    metadatos: Dict[str, Any]
    estrategia_usada: EstrategiaGeneracion
    componentes_usados: List[str]
    tiempo_generacion: float
    configuracion: ConfiguracionAurora

class AuroraDirectorV7Optimizado:
    def __init__(self, auto_detectar: bool = True):
        self.version = "Aurora Director V7 Optimizado"
        self.detector = DetectorComponentes()
        self.componentes: Dict[str, ComponenteAurora] = {}
        self.stats = {"experiencias": 0, "tiempo_total": 0.0, "estrategias": {}, "objetivos": {}, "errores": 0}
        if auto_detectar: self._init_sistema()
    
    def _init_sistema(self):
        logger.info(f"🌟 Inicializando {self.version}")
        self.componentes = self.detector.detectar_todos()
        self._log_estado()
        logger.info("🚀 Sistema inicializado")
    
    def _log_estado(self):
        motores = [c for c in self.componentes.values() if c.tipo == "motor"]
        gestores = [c for c in self.componentes.values() if c.tipo == "gestor_inteligencia"]
        pipelines = [c for c in self.componentes.values() if c.tipo == "pipeline"]
        logger.info(f"🔧 Motores:{len(motores)} Gestores:{len(gestores)} Pipelines:{len(pipelines)}")
        logger.info(f"🎯 Estrategias: {', '.join(self._get_estrategias())}")
    
    def _get_estrategias(self) -> List[str]:
        estrategias = []
        gestores = [c for c in self.componentes.values() if c.tipo == "gestor_inteligencia"]
        motores = [c for c in self.componentes.values() if c.tipo == "motor"]
        pipelines = [c for c in self.componentes.values() if c.tipo == "pipeline"]
        
        if len(gestores) >= 1 and len(motores) >= 2 and len(pipelines) >= 1: estrategias.append("aurora_completo")
        if len(gestores) >= 1 and len(motores) >= 1: estrategias.append("inteligencia_activa")
        if len(motores) >= 1: estrategias.append("motores_puros")
        estrategias.append("fallback_garantizado")
        return estrategias
    
    def crear_experiencia(self, objetivo: str, **kwargs) -> ResultadoAurora:
        inicio = time.time()
        try:
            logger.info(f"🎯 Creando: '{objetivo}'")
            config = self._crear_config(objetivo, kwargs)
            problemas = config.validar()
            if problemas: logger.warning(f"⚠️ Config: {problemas}")
            
            estrategia = self._select_estrategia(config)
            logger.info(f"🧠 Estrategia: {estrategia.value}")
            
            resultado = self._ejecutar_estrategia(estrategia, config)
            resultado_final = self._post_procesar(resultado, config)
            self._validar_resultado(resultado_final)
            
            tiempo = time.time() - inicio
            self._update_stats(objetivo, estrategia, tiempo)
            logger.info(f"✅ Experiencia creada en {tiempo:.2f}s")
            return resultado_final
            
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            self.stats["errores"] += 1
            return self._crear_emergencia(objetivo, str(e))
    
    def _crear_config(self, objetivo: str, kwargs: Dict) -> ConfiguracionAurora:
        configs_smart = {
            "concentracion": {"intensidad": "media", "estilo": "crystalline", "neurotransmisor_preferido": "acetilcolina"},
            "claridad_mental": {"intensidad": "media", "estilo": "minimalista", "neurotransmisor_preferido": "dopamina"},
            "enfoque": {"intensidad": "intenso", "estilo": "crystalline", "neurotransmisor_preferido": "norepinefrina"},
            "relajacion": {"intensidad": "suave", "estilo": "sereno", "neurotransmisor_preferido": "gaba"},
            "meditacion": {"intensidad": "suave", "estilo": "mistico", "neurotransmisor_preferido": "serotonina", "duracion_min": 35},
            "gratitud": {"intensidad": "suave", "estilo": "sutil", "neurotransmisor_preferido": "oxitocina"},
            "creatividad": {"intensidad": "media", "estilo": "organico", "neurotransmisor_preferido": "anandamida"},
            "inspiracion": {"intensidad": "media", "estilo": "vanguardia", "neurotransmisor_preferido": "dopamina"},
            "sanacion": {"intensidad": "suave", "estilo": "medicina_sagrada", "neurotransmisor_preferido": "endorfina", "duracion_min": 45},
            "liberacion": {"intensidad": "media", "estilo": "organico", "neurotransmisor_preferido": "gaba"}
        }
        
        config_base = {}
        for key, conf in configs_smart.items():
            if key in objetivo.lower():
                config_base = conf.copy()
                break
        
        parametros_ctx = self._detect_ctx(objetivo)
        config_final = {"objetivo": objetivo, **config_base, **parametros_ctx, **kwargs}
        return ConfiguracionAurora(**config_final)
    
    def _detect_ctx(self, objetivo: str) -> Dict[str, Any]:
        params = {}
        obj_lower = objetivo.lower()
        
        if any(p in obj_lower for p in ["profundo", "intenso", "fuerte"]): params["intensidad"] = "intenso"
        elif any(p in obj_lower for p in ["suave", "ligero", "sutil"]): params["intensidad"] = "suave"
        
        if any(p in obj_lower for p in ["rapido", "corto", "breve"]): params["duracion_min"] = 10
        elif any(p in obj_lower for p in ["largo", "extenso", "profundo"]): params["duracion_min"] = 45
        
        if any(p in obj_lower for p in ["trabajo", "oficina", "estudio"]): params["contexto_uso"] = "trabajo"
        elif any(p in obj_lower for p in ["dormir", "noche", "sueño"]): params["contexto_uso"] = "sueño"
        
        return params
    
    def _select_estrategia(self, config: ConfiguracionAurora) -> EstrategiaGeneracion:
        if config.estrategia_preferida and config.estrategia_preferida.value in self._get_estrategias():
            return config.estrategia_preferida
        
        motores = [c for c in self.componentes.values() if c.tipo == "motor"]
        gestores = [c for c in self.componentes.values() if c.tipo == "gestor_inteligencia"]
        pipelines = [c for c in self.componentes.values() if c.tipo == "pipeline"]
        
        if len(gestores) >= 2 and len(motores) >= 2 and len(pipelines) >= 1 and config.calidad_objetivo in ["alta", "maxima"]:
            return EstrategiaGeneracion.AURORA_COMPLETO
        elif len(gestores) >= 1 and len(motores) >= 1:
            return EstrategiaGeneracion.INTELIGENCIA_ACTIVA
        elif len(motores) >= 1:
            return EstrategiaGeneracion.MOTORES_PUROS
        else:
            return EstrategiaGeneracion.FALLBACK_GARANTIZADO
    
    def _ejecutar_estrategia(self, estrategia: EstrategiaGeneracion, config: ConfiguracionAurora) -> Dict[str, Any]:
        duracion_sec = config.duracion_min * 60
        componentes_usados = []
        
        if estrategia == EstrategiaGeneracion.AURORA_COMPLETO:
            audio, comps = self._generar_aurora_completo(config, duracion_sec)
        elif estrategia == EstrategiaGeneracion.INTELIGENCIA_ACTIVA:
            audio, comps = self._generar_inteligencia_activa(config, duracion_sec)
        elif estrategia == EstrategiaGeneracion.MOTORES_PUROS:
            audio, comps = self._generar_motores_puros(config, duracion_sec)
        else:
            audio, comps = self._generar_fallback(config, duracion_sec)
        
        return {"audio": audio, "componentes": comps, "estrategia": estrategia}
    
    def _generar_aurora_completo(self, config: ConfiguracionAurora, duracion_sec: float) -> Tuple[np.ndarray, List[str]]:
        comps_usados = []
        
        # Usar router para configuración inteligente
        if "objective_router" in self.componentes:
            router = self.componentes["objective_router"].instancia
            config_inteligente = router.rutear_objetivo(config.objetivo)
            comps_usados.append("objective_router")
        else:
            config_inteligente = {"beat_base": 8.0, "preset_emocional": "calma"}
        
        # Generar base con NeuroMix
        if "neuromix" in self.componentes:
            neuromix = self.componentes["neuromix"].instancia
            nt = config.neurotransmisor_preferido or "gaba"
            audio_base = neuromix.generate_neuro_wave(nt, duracion_sec, intensidad=config.intensidad)
            comps_usados.append("neuromix")
        else:
            samples = int(44100 * duracion_sec)
            audio_base = np.random.normal(0, 0.1, (2, samples))
        
        # Agregar texturas con HarmonicEssence
        if "harmonic_essence" in self.componentes:
            harmonic = self.componentes["harmonic_essence"].instancia
            from types import SimpleNamespace
            h_config = SimpleNamespace(duration_sec=duracion_sec, style=config.estilo)
            textura = harmonic.generate_textured_noise(h_config)
            audio_final = audio_base + 0.3 * textura
            comps_usados.append("harmonic_essence")
        else:
            audio_final = audio_base
        
        # Normalizar con Quality Pipeline
        if "quality_pipeline" in self.componentes:
            pipeline = self.componentes["quality_pipeline"].instancia
            audio_final = pipeline.validar_y_normalizar(audio_final)
            comps_usados.append("quality_pipeline")
        
        return audio_final, comps_usados
    
    def _generar_inteligencia_activa(self, config: ConfiguracionAurora, duracion_sec: float) -> Tuple[np.ndarray, List[str]]:
        comps_usados = []
        
        # Usar motor principal disponible
        if "neuromix" in self.componentes:
            motor = self.componentes["neuromix"].instancia
            nt = config.neurotransmisor_preferido or "gaba"
            audio = motor.generate_neuro_wave(nt, duracion_sec, intensidad=config.intensidad)
            comps_usados.append("neuromix")
        elif "harmonic_essence" in self.componentes:
            motor = self.componentes["harmonic_essence"].instancia
            from types import SimpleNamespace
            h_config = SimpleNamespace(duration_sec=duracion_sec)
            audio = motor.generate_textured_noise(h_config)
            comps_usados.append("harmonic_essence")
        else:
            audio = self._generar_audio_basico(duracion_sec)
        
        return audio, comps_usados
    
    def _generar_motores_puros(self, config: ConfiguracionAurora, duracion_sec: float) -> Tuple[np.ndarray, List[str]]:
        comps_usados = []
        motores_disponibles = [c for c in self.componentes.values() if c.tipo == "motor"]
        
        if motores_disponibles:
            motor = motores_disponibles[0]
            if motor.nombre == "neuromix":
                audio = motor.instancia.generate_neuro_wave("gaba", duracion_sec)
            elif motor.nombre == "harmonic_essence":
                from types import SimpleNamespace
                h_config = SimpleNamespace(duration_sec=duracion_sec)
                audio = motor.instancia.generate_textured_noise(h_config)
            else:
                audio = self._generar_audio_basico(duracion_sec)
            comps_usados.append(motor.nombre)
        else:
            audio = self._generar_audio_basico(duracion_sec)
        
        return audio, comps_usados
    
    def _generar_fallback(self, config: ConfiguracionAurora, duracion_sec: float) -> Tuple[np.ndarray, List[str]]:
        return self._generar_audio_basico(duracion_sec), ["fallback_interno"]
    
    def _generar_audio_basico(self, duracion_sec: float) -> np.ndarray:
        samples = int(44100 * duracion_sec)
        t = np.linspace(0, duracion_sec, samples)
        freq = 8.0  # Alpha básico
        audio = 0.2 * np.sin(2 * np.pi * freq * t)
        return np.stack([audio, audio])
    
    def _post_procesar(self, resultado: Dict[str, Any], config: ConfiguracionAurora) -> ResultadoAurora:
        audio = resultado["audio"]
        if config.normalizar:
            max_val = np.max(np.abs(audio))
            if max_val > 0: audio = audio * (0.85 / max_val)
            audio = np.clip(audio, -1.0, 1.0)
        
        metadatos = {
            "objetivo": config.objetivo,
            "duracion_min": config.duracion_min,
            "estrategia": resultado["estrategia"].value,
            "componentes": resultado["componentes"],
            "timestamp": datetime.now().isoformat(),
            "sample_rate": config.sample_rate,
            "version": self.version
        }
        
        return ResultadoAurora(
            audio_data=audio,
            metadatos=metadatos,
            estrategia_usada=resultado["estrategia"],
            componentes_usados=resultado["componentes"],
            tiempo_generacion=0.0,
            configuracion=config
        )
    
    def _validar_resultado(self, resultado: ResultadoAurora):
        if resultado.audio_data.size == 0: raise ValueError("Audio vacío")
        if np.isnan(resultado.audio_data).any(): raise ValueError("Audio contiene NaN")
        if np.max(np.abs(resultado.audio_data)) > 1.1: raise ValueError("Audio fuera de rango")
    
    def _update_stats(self, objetivo: str, estrategia: EstrategiaGeneracion, tiempo: float):
        self.stats["experiencias"] += 1
        self.stats["tiempo_total"] += tiempo
        self.stats["estrategias"][estrategia.value] = self.stats["estrategias"].get(estrategia.value, 0) + 1
        self.stats["objetivos"][objetivo] = self.stats["objetivos"].get(objetivo, 0) + 1
    
    def _crear_emergencia(self, objetivo: str, error: str) -> ResultadoAurora:
        audio_emergencia = self._generar_audio_basico(60.0)  # 1 minuto básico
        config_emergencia = ConfiguracionAurora(objetivo=objetivo, duracion_min=1)
        
        return ResultadoAurora(
            audio_data=audio_emergencia,
            metadatos={"error": error, "modo_emergencia": True, "objetivo": objetivo},
            estrategia_usada=EstrategiaGeneracion.FALLBACK_GARANTIZADO,
            componentes_usados=["emergencia"],
            tiempo_generacion=0.0,
            configuracion=config_emergencia
        )
    
    def obtener_estado_completo(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "componentes_detectados": {n: {"disponible": c.disponible, "version": c.version, "tipo": c.tipo, "fallback": c.version == "fallback"} for n, c in self.componentes.items()},
            "estadisticas_deteccion": self.detector.stats,
            "estadisticas_uso": self.stats,
            "estrategias_disponibles": self._get_estrategias(),
            "timestamp": datetime.now().isoformat()
        }

_director_global = None

def Aurora(objetivo: str = None, **kwargs):
    """🌟 API ULTRA-SIMPLE DE AURORA V7"""
    global _director_global
    if _director_global is None: _director_global = AuroraDirectorV7Optimizado()
    return _director_global.crear_experiencia(objetivo, **kwargs) if objetivo else _director_global

Aurora.rapido = lambda obj, **kw: Aurora(obj, duracion_min=5, calidad_objetivo="media", **kw)
Aurora.largo = lambda obj, **kw: Aurora(obj, duracion_min=60, calidad_objetivo="alta", **kw)
Aurora.terapeutico = lambda obj, **kw: Aurora(obj, duracion_min=45, intensidad="suave", calidad_objetivo="maxima", **kw)
Aurora.estado = lambda: Aurora().obtener_estado_completo()
Aurora.diagnostico = lambda: Aurora().detector.stats

if __name__ == "__main__":
    print("🌟 Aurora Director V7 OPTIMIZADO - Testing")
    director = Aurora()
    estado = director.obtener_estado_completo()
    print(f"🚀 {estado['version']}")
    print(f"📊 Componentes: {len(estado['componentes_detectados'])}")
    for nombre, info in estado['componentes_detectados'].items():
        emoji = "✅" if info['disponible'] and not info['fallback'] else "🔄" if info['fallback'] else "❌"
        print(f"   {emoji} {nombre} v{info['version']}")
    print(f"🎯 Estrategias: {', '.join(estado['estrategias_disponibles'])}")
    
    try:
        print("🎵 Test de generación...")
        resultado = Aurora("test_optimizado", duracion_min=1, exportar_wav=False)
        print(f"✅ ¡Generación exitosa! Audio: {resultado.audio_data.shape} Estrategia: {resultado.estrategia_usada.value}")
    except Exception as e:
        print(f"❌ Error en test: {e}")
    
    print("🏆 AURORA V7 OPTIMIZADO - ¡LISTO!")