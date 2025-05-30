import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Protocol
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import time
import importlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Aurora.Director.V7.Min")

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
        self.componentes_registrados = {
            "field_profiles": ComponenteAurora("field_profiles", "gestor_inteligencia", "field_profiles", "GestorPerfilesCampo", dependencias=[], fallback_disponible=True, nivel_prioridad=2),
            "objective_router": ComponenteAurora("objective_router", "gestor_inteligencia", "objective_router", "RouterInteligenteV7", dependencias=["field_profiles"], fallback_disponible=True, nivel_prioridad=2),
            "neuromix": ComponenteAurora("neuromix", "motor", "neuromix_aurora_v27", "AuroraNeuroAcousticEngineV27", dependencias=[], fallback_disponible=True, nivel_prioridad=1),
            "hypermod": ComponenteAurora("hypermod", "motor", "hypermod_v32", "HyperModEngineV32AuroraConnected", dependencias=[], fallback_disponible=True, nivel_prioridad=1),
            "harmonic_essence": ComponenteAurora("harmonic_essence", "motor", "harmonicEssence_v34", "HarmonicEssenceV34AuroraConnected", dependencias=[], fallback_disponible=True, nivel_prioridad=1),
            "emotion_style": ComponenteAurora("emotion_style", "gestor_inteligencia", "emotion_style_profiles", "GestorEmotionStyleUnificado", dependencias=[], fallback_disponible=False, nivel_prioridad=3),
            "quality_pipeline": ComponenteAurora("quality_pipeline", "pipeline", "aurora_quality_pipeline", "AuroraQualityPipeline", dependencias=[], fallback_disponible=True, nivel_prioridad=4)
        }
        self.componentes_activos: Dict[str, ComponenteAurora] = {}
        self.stats = {"total": 0, "exitosos": 0, "fallidos": 0, "fallback": 0}
    
    def detectar_todos(self) -> Dict[str, ComponenteAurora]:
        logger.info("Detectando componentes Aurora...")
        for nombre, comp in sorted(self.componentes_registrados.items(), key=lambda x: x[1].nivel_prioridad):
            self._detectar_comp(comp)
        self._log_resultado()
        return self.componentes_activos
    
    def _detectar_comp(self, comp: ComponenteAurora) -> bool:
        self.stats["total"] += 1
        try:
            if not all(dep in self.componentes_activos for dep in comp.dependencias): return False
            modulo = importlib.import_module(comp.modulo)
            instancia = self._init_instancia(modulo, comp)
            if instancia and self._validar_instancia(instancia, comp):
                comp.disponible, comp.instancia = True, instancia
                comp.capacidades = self._get_caps(instancia)
                comp.version = self._get_version(instancia)
                self.componentes_activos[comp.nombre] = comp
                self.stats["exitosos"] += 1
                logger.info(f"✅ {comp.nombre} v{comp.version}")
                return True
            else: raise Exception("Instancia inválida")
        except Exception as e:
            logger.warning(f"❌ {comp.nombre}: {e}")
            if comp.fallback_disponible and self._crear_fallback(comp):
                self.stats["fallback"] += 1
                logger.info(f"🔄 {comp.nombre} fallback")
                return True
            self.stats["fallidos"] += 1
            return False
    
    def _init_instancia(self, modulo: Any, comp: ComponenteAurora) -> Any:
        if comp.nombre == "neuromix":
            return (getattr(modulo, "_global_engine", None) or 
                   next((getattr(modulo, fn)() if callable(getattr(modulo, fn)) else getattr(modulo, fn) 
                        for fn in ["crear_gestor_neuromix", "AuroraNeuroAcousticEngineV27", "AuroraNeuroAcousticEngine"] 
                        if hasattr(modulo, fn)), None))
        elif comp.nombre == "hypermod":
            return (getattr(modulo, "_motor_global_v32", None) or 
                   next((getattr(modulo, fn)() if callable(getattr(modulo, fn)) else getattr(modulo, fn) 
                        for fn in ["crear_gestor_hypermod", "HyperModEngineV32AuroraConnected"] 
                        if hasattr(modulo, fn)), modulo))
        elif comp.nombre == "harmonic_essence":
            if hasattr(modulo, "crear_motor_aurora_conectado"): return modulo.crear_motor_aurora_conectado()
            return (getattr(modulo, "_motor_global", None) or 
                   (modulo.HarmonicEssenceV34AuroraConnected() if hasattr(modulo, "HarmonicEssenceV34AuroraConnected") else None))
        else:
            for func in [f"crear_gestor_{comp.nombre}", f"crear_{comp.nombre}", "crear_gestor", "obtener_gestor"]:
                if hasattr(modulo, func): return getattr(modulo, func)()
            if hasattr(modulo, comp.clase_principal): return getattr(modulo, comp.clase_principal)()
        return None
    
    def _validar_instancia(self, inst: Any, comp: ComponenteAurora) -> bool:
        try:
            if comp.tipo == "motor":
                return ((hasattr(inst, 'generar_audio') and hasattr(inst, 'validar_configuracion') and hasattr(inst, 'obtener_capacidades')) or
                       any(hasattr(inst, m) for m in ['generate_neuro_wave', 'generar_bloques', 'generate_textured_noise', 'generar_bloques_aurora_integrado']))
            elif comp.tipo == "gestor_inteligencia":
                return any(hasattr(inst, m) for m in ['obtener_perfil', 'rutear_objetivo', 'procesar_objetivo'])
            elif comp.tipo == "pipeline":
                return hasattr(inst, 'validar_y_normalizar')
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
            fallbacks = {"neuromix": self._fb_neuromix, "hypermod": self._fb_hypermod, "harmonic_essence": self._fb_harmonic, 
                        "quality_pipeline": self._fb_quality, "field_profiles": self._fb_profiles, "objective_router": self._fb_router}
            if comp.nombre in fallbacks:
                comp.instancia, comp.disponible, comp.version = fallbacks[comp.nombre](), True, "fallback"
                self.componentes_activos[comp.nombre] = comp
                return True
        except: pass
        return False
    
    def _fb_neuromix(self):
        class NeuroMixFB:
            def generar_audio(self, config: Dict[str, Any], dur: float) -> np.ndarray:
                s = int(44100 * dur)
                return np.stack([0.3 * np.sin(2 * np.pi * 10.0 * np.linspace(0, dur, s))] * 2)
            def validar_configuracion(self, config: Dict[str, Any]) -> bool: return True
            def obtener_capacidades(self) -> Dict[str, Any]: return {"nombre": "NeuroMix Fallback", "version": "fallback"}
            def generate_neuro_wave(self, nt: str, dur: float, **kw) -> np.ndarray: return self.generar_audio({'objetivo': nt}, dur)
        return NeuroMixFB()
    
    def _fb_hypermod(self):
        class HyperModFB:
            def generar_audio(self, config: Dict[str, Any], dur: float) -> np.ndarray:
                s = int(44100 * dur)
                return np.stack([0.3 * np.sin(2 * np.pi * 8.0 * np.linspace(0, dur, s))] * 2)
            def validar_configuracion(self, config: Dict[str, Any]) -> bool: return True
            def obtener_capacidades(self) -> Dict[str, Any]: return {"nombre": "HyperMod Fallback", "version": "fallback"}
            def generar_bloques(self, dur: int, layers: List, config=None) -> np.ndarray: return self.generar_audio({'objetivo': 'fallback'}, dur * 60)
        return HyperModFB()
    
    def _fb_harmonic(self):
        class HarmonicFB:
            def generar_audio(self, config: Dict[str, Any], dur: float) -> np.ndarray:
                return np.stack([np.random.normal(0, 0.1, int(44100 * dur))] * 2)
            def validar_configuracion(self, config: Dict[str, Any]) -> bool: return True
            def obtener_capacidades(self) -> Dict[str, Any]: return {"nombre": "HarmonicEssence Fallback", "version": "fallback"}
            def generate_textured_noise(self, config, **kw) -> np.ndarray: return self.generar_audio({'objetivo': 'texture'}, getattr(config, 'duration_sec', 10.0))
        return HarmonicFB()
    
    def _fb_quality(self):
        class QualityFB:
            def validar_y_normalizar(self, signal: np.ndarray) -> np.ndarray:
                if signal.ndim == 1: signal = np.stack([signal, signal])
                max_val = np.max(np.abs(signal))
                if max_val > 0: signal = signal * (0.85 / max_val)
                return np.clip(signal, -1.0, 1.0)
        return QualityFB()
    
    def _fb_profiles(self):
        class ProfilesFB:
            def obtener_perfil(self, nombre: str): return None
            def recomendar_secuencia_perfiles(self, obj: str, dur: int): return [(obj, dur)]
        return ProfilesFB()
    
    def _fb_router(self):
        class RouterFB:
            def rutear_objetivo(self, obj: str, **kw): return {"preset_emocional": "calma_profunda", "estilo": "sereno", "modo": "normal", "beat_base": 8.0, "capas": {"neuro_wave": True, "binaural": True}}
        return RouterFB()
    
    def _log_resultado(self):
        total, activos = len(self.componentes_registrados), len(self.componentes_activos)
        logger.info(f"Componentes: {activos}/{total} ({activos/total*100:.0f}%)")
        logger.info(f"✅{self.stats['exitosos']} 🔄{self.stats['fallback']} ❌{self.stats['fallidos']}")

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
        self.version = "Aurora Director V7 Minified"
        self.detector = DetectorComponentes()
        self.componentes: Dict[str, ComponenteAurora] = {}
        self.stats = {"experiencias": 0, "tiempo_total": 0.0, "estrategias": {}, "objetivos": {}, "errores": 0, "motores_usados": {}}
        if auto_detectar: self._init_sistema()
    
    def _init_sistema(self):
        logger.info(f"Inicializando {self.version}")
        self.componentes = self.detector.detectar_todos()
        self._log_estado()
        logger.info("Sistema inicializado")
    
    def _log_estado(self):
        motores = [c for c in self.componentes.values() if c.tipo == "motor"]
        gestores = [c for c in self.componentes.values() if c.tipo == "gestor_inteligencia"]
        pipelines = [c for c in self.componentes.values() if c.tipo == "pipeline"]
        logger.info(f"Motores:{len(motores)} Gestores:{len(gestores)} Pipelines:{len(pipelines)}")
        logger.info(f"Estrategias: {', '.join(self._get_estrategias())}")
    
    def _get_estrategias(self) -> List[str]:
        g = [c for c in self.componentes.values() if c.tipo == "gestor_inteligencia"]
        m = [c for c in self.componentes.values() if c.tipo == "motor"]
        p = [c for c in self.componentes.values() if c.tipo == "pipeline"]
        estrategias = []
        if len(g) >= 1 and len(m) >= 3 and len(p) >= 1: estrategias.append("aurora_completo")
        if len(g) >= 1 and len(m) >= 2: estrategias.append("inteligencia_activa")
        if len(m) >= 1: estrategias.append("motores_puros")
        estrategias.append("fallback_garantizado")
        return estrategias
    
    def crear_experiencia(self, objetivo: str, **kwargs) -> ResultadoAurora:
        inicio = time.time()
        try:
            logger.info(f"Creando: '{objetivo}'")
            config = self._crear_config(objetivo, kwargs)
            if p := config.validar(): logger.warning(f"Config: {p}")
            estrategia = self._select_estrategia(config)
            logger.info(f"Estrategia: {estrategia.value}")
            resultado = self._ejecutar_estrategia(estrategia, config)
            resultado_final = self._post_procesar(resultado, config)
            self._validar_resultado(resultado_final)
            tiempo = time.time() - inicio
            self._update_stats(objetivo, estrategia, tiempo, resultado['componentes'])
            logger.info(f"Experiencia creada en {tiempo:.2f}s")
            return resultado_final
        except Exception as e:
            logger.error(f"Error: {e}")
            self.stats["errores"] += 1
            return self._crear_emergencia(objetivo, str(e))
    
    def _crear_config(self, objetivo: str, kwargs: Dict) -> ConfiguracionAurora:
        configs = {"concentracion": {"intensidad": "media", "estilo": "crystalline", "neurotransmisor_preferido": "acetilcolina"}, 
                  "claridad_mental": {"intensidad": "media", "estilo": "minimalista", "neurotransmisor_preferido": "dopamina"}, 
                  "enfoque": {"intensidad": "intenso", "estilo": "crystalline", "neurotransmisor_preferido": "norepinefrina"}, 
                  "relajacion": {"intensidad": "suave", "estilo": "sereno", "neurotransmisor_preferido": "gaba"}, 
                  "meditacion": {"intensidad": "suave", "estilo": "mistico", "neurotransmisor_preferido": "serotonina", "duracion_min": 35}, 
                  "gratitud": {"intensidad": "suave", "estilo": "sutil", "neurotransmisor_preferido": "oxitocina"}, 
                  "creatividad": {"intensidad": "media", "estilo": "organico", "neurotransmisor_preferido": "anandamida"}, 
                  "inspiracion": {"intensidad": "media", "estilo": "vanguardia", "neurotransmisor_preferido": "dopamina"}, 
                  "sanacion": {"intensidad": "suave", "estilo": "medicina_sagrada", "neurotransmisor_preferido": "endorfina", "duracion_min": 45}, 
                  "liberacion": {"intensidad": "media", "estilo": "organico", "neurotransmisor_preferido": "gaba"}}
        
        config_base = next((conf.copy() for key, conf in configs.items() if key in objetivo.lower()), {})
        params = self._detect_ctx(objetivo)
        return ConfiguracionAurora(**{"objetivo": objetivo, **config_base, **params, **kwargs})
    
    def _detect_ctx(self, objetivo: str) -> Dict[str, Any]:
        o = objetivo.lower()
        params = {}
        if any(p in o for p in ["profundo", "intenso", "fuerte"]): params["intensidad"] = "intenso"
        elif any(p in o for p in ["suave", "ligero", "sutil"]): params["intensidad"] = "suave"
        if any(p in o for p in ["rapido", "corto", "breve"]): params["duracion_min"] = 10
        elif any(p in o for p in ["largo", "extenso", "profundo"]): params["duracion_min"] = 45
        if any(p in o for p in ["trabajo", "oficina", "estudio"]): params["contexto_uso"] = "trabajo"
        elif any(p in o for p in ["dormir", "noche", "sueño"]): params["contexto_uso"] = "sueño"
        return params
    
    def _select_estrategia(self, config: ConfiguracionAurora) -> EstrategiaGeneracion:
        if config.estrategia_preferida and config.estrategia_preferida.value in self._get_estrategias():
            return config.estrategia_preferida
        m = [c for c in self.componentes.values() if c.tipo == "motor"]
        g = [c for c in self.componentes.values() if c.tipo == "gestor_inteligencia"]
        p = [c for c in self.componentes.values() if c.tipo == "pipeline"]
        if len(g) >= 1 and len(m) >= 3 and len(p) >= 1 and config.calidad_objetivo in ["alta", "maxima"]:
            return EstrategiaGeneracion.AURORA_COMPLETO
        elif len(g) >= 1 and len(m) >= 2: return EstrategiaGeneracion.INTELIGENCIA_ACTIVA
        elif len(m) >= 1: return EstrategiaGeneracion.MOTORES_PUROS
        else: return EstrategiaGeneracion.FALLBACK_GARANTIZADO
    
    def _ejecutar_estrategia(self, estrategia: EstrategiaGeneracion, config: ConfiguracionAurora) -> Dict[str, Any]:
        duracion_sec = config.duracion_min * 60
        if estrategia == EstrategiaGeneracion.AURORA_COMPLETO:
            audio, comps = self._gen_completo(config, duracion_sec)
        elif estrategia == EstrategiaGeneracion.INTELIGENCIA_ACTIVA:
            audio, comps = self._gen_inteligencia(config, duracion_sec)
        elif estrategia == EstrategiaGeneracion.MOTORES_PUROS:
            audio, comps = self._gen_motores(config, duracion_sec)
        else: audio, comps = self._gen_fallback(config, duracion_sec)
        return {"audio": audio, "componentes": comps, "estrategia": estrategia}
    
    def _gen_completo(self, config: ConfiguracionAurora, dur: float) -> Tuple[np.ndarray, List[str]]:
        comps = []
        motor_config = {'objetivo': config.objetivo, 'intensidad': config.intensidad, 'estilo': config.estilo, 
                       'sample_rate': config.sample_rate, 'calidad_objetivo': config.calidad_objetivo, 
                       'normalizar': config.normalizar, 'neurotransmisor_preferido': config.neurotransmisor_preferido, 
                       'contexto_uso': config.contexto_uso, 'duracion_min': config.duracion_min}
        
        if "objective_router" in self.componentes:
            try:
                self.componentes["objective_router"].instancia.rutear_objetivo(config.objetivo)
                comps.append("objective_router")
            except Exception as e: logger.warning(f"Error en router: {e}")
        
        audio_base = None
        if "neuromix" in self.componentes:
            try:
                audio_base = self.componentes["neuromix"].instancia.generar_audio(motor_config, dur)
                comps.append("neuromix")
                logger.info("NeuroMix V27 - Base generada")
            except Exception as e: logger.warning(f"Error en NeuroMix: {e}")
        
        if audio_base is None: audio_base = self._gen_basico(dur)
        
        if "hypermod" in self.componentes:
            try:
                audio_hyper = self.componentes["hypermod"].instancia.generar_audio(motor_config, dur)
                audio_base = 0.7 * audio_base + 0.3 * audio_hyper
                comps.append("hypermod")
                logger.info("HyperMod V32 - Capas agregadas")
            except Exception as e: logger.warning(f"Error en HyperMod: {e}")
        
        if "harmonic_essence" in self.componentes:
            try:
                audio_texture = self.componentes["harmonic_essence"].instancia.generar_audio(motor_config, dur)
                audio_final = 0.85 * audio_base + 0.15 * audio_texture
                comps.append("harmonic_essence")
                logger.info("HarmonicEssence V34 - Texturas agregadas")
            except Exception as e:
                logger.warning(f"Error en HarmonicEssence: {e}")
                audio_final = audio_base
        else: audio_final = audio_base
        
        if "quality_pipeline" in self.componentes:
            try:
                audio_final = self.componentes["quality_pipeline"].instancia.validar_y_normalizar(audio_final)
                comps.append("quality_pipeline")
                logger.info("Quality Pipeline - Audio normalizado")
            except Exception as e:
                logger.warning(f"Error en Quality Pipeline: {e}")
                max_val = np.max(np.abs(audio_final))
                if max_val > 0: audio_final = audio_final * (0.85 / max_val)
        else:
            max_val = np.max(np.abs(audio_final))
            if max_val > 0: audio_final = audio_final * (0.85 / max_val)
        
        return audio_final, comps
    
    def _gen_inteligencia(self, config: ConfiguracionAurora, dur: float) -> Tuple[np.ndarray, List[str]]:
        motor_config = {'objetivo': config.objetivo, 'intensidad': config.intensidad, 'estilo': config.estilo, 
                       'sample_rate': config.sample_rate, 'calidad_objetivo': config.calidad_objetivo, 
                       'normalizar': config.normalizar, 'neurotransmisor_preferido': config.neurotransmisor_preferido, 
                       'duracion_min': config.duracion_min}
        audio = None
        for motor_name in ["neuromix", "hypermod", "harmonic_essence"]:
            if motor_name in self.componentes:
                try:
                    audio = self.componentes[motor_name].instancia.generar_audio(motor_config, dur)
                    return audio, [motor_name]
                except Exception as e:
                    logger.warning(f"Error en motor {motor_name}: {e}")
                    continue
        return self._gen_basico(dur), []
    
    def _gen_motores(self, config: ConfiguracionAurora, dur: float) -> Tuple[np.ndarray, List[str]]:
        motores = [c for c in self.componentes.values() if c.tipo == "motor"]
        motor_config = {'objetivo': config.objetivo, 'intensidad': config.intensidad, 'estilo': config.estilo, 
                       'sample_rate': config.sample_rate, 'calidad_objetivo': config.calidad_objetivo, 
                       'normalizar': config.normalizar, 'duracion_min': config.duracion_min}
        if motores:
            motor = sorted(motores, key=lambda m: {"neuromix": 0, "hypermod": 1, "harmonic_essence": 2}.get(m.nombre, 3))[0]
            try:
                return motor.instancia.generar_audio(motor_config, dur), [motor.nombre]
            except Exception as e:
                logger.warning(f"Error en motor {motor.nombre}: {e}")
        return self._gen_basico(dur), []
    
    def _gen_fallback(self, config: ConfiguracionAurora, dur: float) -> Tuple[np.ndarray, List[str]]:
        return self._gen_basico(dur), ["fallback_interno"]
    
    def _gen_basico(self, dur: float) -> np.ndarray:
        s = int(44100 * dur)
        t = np.linspace(0, dur, s)
        audio = 0.2 * np.sin(2 * np.pi * 8.0 * t)
        fade = int(44100 * 2.0)
        if len(audio) > fade * 2:
            audio[:fade] *= np.linspace(0, 1, fade)
            audio[-fade:] *= np.linspace(1, 0, fade)
        return np.stack([audio, audio])
    
    def _post_procesar(self, resultado: Dict[str, Any], config: ConfiguracionAurora) -> ResultadoAurora:
        audio = resultado["audio"]
        if config.normalizar:
            max_val = np.max(np.abs(audio))
            if max_val > 0: audio = audio * (0.85 / max_val)
            audio = np.clip(audio, -1.0, 1.0)
        
        return ResultadoAurora(
            audio_data=audio,
            metadatos={"objetivo": config.objetivo, "duracion_min": config.duracion_min, "estrategia": resultado["estrategia"].value, 
                      "componentes": resultado["componentes"], "timestamp": datetime.now().isoformat(), "sample_rate": config.sample_rate, 
                      "version": self.version, "intensidad": config.intensidad, "estilo": config.estilo, "calidad_objetivo": config.calidad_objetivo},
            estrategia_usada=resultado["estrategia"],
            componentes_usados=resultado["componentes"],
            tiempo_generacion=0.0,
            configuracion=config
        )
    
    def _validar_resultado(self, resultado: ResultadoAurora):
        if resultado.audio_data.size == 0: raise ValueError("Audio vacío")
        if np.isnan(resultado.audio_data).any(): raise ValueError("Audio contiene NaN")
        if np.max(np.abs(resultado.audio_data)) > 1.1: raise ValueError("Audio fuera de rango")
    
    def _update_stats(self, objetivo: str, estrategia: EstrategiaGeneracion, tiempo: float, componentes: List[str]):
        self.stats["experiencias"] += 1
        self.stats["tiempo_total"] += tiempo
        self.stats["estrategias"][estrategia.value] = self.stats["estrategias"].get(estrategia.value, 0) + 1
        self.stats["objetivos"][objetivo] = self.stats["objetivos"].get(objetivo, 0) + 1
        for comp in componentes:
            if comp in ["neuromix", "hypermod", "harmonic_essence"]:
                self.stats["motores_usados"][comp] = self.stats["motores_usados"].get(comp, 0) + 1
    
    def _crear_emergencia(self, objetivo: str, error: str) -> ResultadoAurora:
        return ResultadoAurora(
            audio_data=self._gen_basico(60.0),
            metadatos={"error": error, "modo_emergencia": True, "objetivo": objetivo},
            estrategia_usada=EstrategiaGeneracion.FALLBACK_GARANTIZADO,
            componentes_usados=["emergencia"],
            tiempo_generacion=0.0,
            configuracion=ConfiguracionAurora(objetivo=objetivo, duracion_min=1)
        )
    
    def test_motor_connections(self) -> Dict[str, Any]:
        results = {}
        for motor_name in ["neuromix", "hypermod", "harmonic_essence"]:
            if motor_name in self.componentes:
                motor = self.componentes[motor_name].instancia
                try:
                    config_test = {'objetivo': 'test_conexion', 'intensidad': 'media', 'sample_rate': 44100, 'normalizar': True, 'calidad_objetivo': 'media'}
                    valido = motor.validar_configuracion(config_test)
                    caps = motor.obtener_capacidades()
                    audio_test = motor.generar_audio(config_test, 1.0)
                    results[motor_name] = {'conectado': True, 'protocolo_valido': valido, 'capacidades_ok': len(caps) > 0, 
                                         'genera_audio': audio_test.size > 0, 'forma_audio': audio_test.shape, 
                                         'version': getattr(motor, 'version', 'unknown'), 
                                         'capacidades': caps.get('nombre', 'Motor Aurora') if isinstance(caps, dict) else 'Motor Aurora'}
                except Exception as e: results[motor_name] = {'conectado': False, 'error': str(e)}
            else: results[motor_name] = {'conectado': False, 'error': 'Motor no detectado'}
        return results
    
    def obtener_estado_completo(self) -> Dict[str, Any]:
        motores_info = {}
        for motor_name in ["neuromix", "hypermod", "harmonic_essence"]:
            if motor_name in self.componentes:
                comp = self.componentes[motor_name]
                motores_info[motor_name] = {"disponible": comp.disponible, "version": comp.version, "tipo": comp.tipo, 
                                          "fallback": comp.version == "fallback", "capacidades": len(comp.capacidades) > 0}
            else:
                motores_info[motor_name] = {"disponible": False, "version": "no_detectado", "tipo": "motor", "fallback": False, "capacidades": False}
        
        return {"version": self.version, 
               "componentes_detectados": {n: {"disponible": c.disponible, "version": c.version, "tipo": c.tipo, "fallback": c.version == "fallback"} for n, c in self.componentes.items()}, 
               "motores_principales": motores_info, "estadisticas_deteccion": self.detector.stats, "estadisticas_uso": self.stats, 
               "estrategias_disponibles": self._get_estrategias(), "timestamp": datetime.now().isoformat(), 
               "sistema_listo": len([c for c in self.componentes.values() if c.tipo == "motor"]) > 0}

_director_global = None

def Aurora(objetivo: str = None, **kwargs):
    global _director_global
    if _director_global is None: _director_global = AuroraDirectorV7Optimizado()
    return _director_global.crear_experiencia(objetivo, **kwargs) if objetivo else _director_global

Aurora.rapido = lambda obj, **kw: Aurora(obj, duracion_min=5, calidad_objetivo="media", **kw)
Aurora.largo = lambda obj, **kw: Aurora(obj, duracion_min=60, calidad_objetivo="alta", **kw)
Aurora.terapeutico = lambda obj, **kw: Aurora(obj, duracion_min=45, intensidad="suave", calidad_objetivo="maxima", **kw)
Aurora.estado = lambda: Aurora().obtener_estado_completo()
Aurora.diagnostico = lambda: Aurora().detector.stats
Aurora.test_motores = lambda: Aurora().test_motor_connections()

if __name__ == "__main__":
    print("Aurora Director V7 MINIFIED - Testing Completo")
    print("=" * 60)
    director = Aurora()
    estado = director.obtener_estado_completo()
    print(f"🚀 {estado['version']}")
    print(f"📊 Componentes detectados: {len(estado['componentes_detectados'])}")
    print(f"\n🔧 MOTORES PRINCIPALES:")
    for nombre, info in estado['motores_principales'].items():
        emoji = "✅" if info['disponible'] and not info['fallback'] else "🔄" if info['fallback'] else "❌"
        print(f"   {emoji} {nombre} v{info['version']}")
    print(f"\n🎯 Estrategias disponibles: {', '.join(estado['estrategias_disponibles'])}")
    print(f"\n🧪 Test de Conexiones de Motores:")
    test_results = Aurora.test_motores()
    for motor, result in test_results.items():
        if result.get('conectado'):
            print(f"   ✅ {motor}: {result['capacidades']} (Audio: {result['forma_audio']})")
        else:
            print(f"   ❌ {motor}: {result.get('error', 'No disponible')}")
    try:
        print(f"\n🎵 Test de Generación Aurora Completa:")
        resultado = Aurora("test_optimizado_completo", duracion_min=1, calidad_objetivo="maxima", exportar_wav=False)
        print(f"   ✅ Audio generado: {resultado.audio_data.shape}")
        print(f"   🎯 Estrategia: {resultado.estrategia_usada.value}")
        print(f"   🔧 Componentes: {', '.join(resultado.componentes_usados)}")
        print(f"   📊 Metadatos: {len(resultado.metadatos)} campos")
        stats = Aurora.diagnostico()
        print(f"\n📈 Estadísticas del Sistema:")
        print(f"   • Total detectados: {stats['total']}")
        print(f"   • Exitosos: {stats['exitosos']}")
        print(f"   • Fallbacks: {stats['fallback']}")
        print(f"   • Fallidos: {stats['fallidos']}")
    except Exception as e:
        print(f"   ❌ Error en test: {e}")
    print(f"\n🏆 AURORA V7 MINIFIED - ¡SISTEMA COMPLETO Y OPTIMIZADO!")
