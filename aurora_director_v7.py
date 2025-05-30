"""
🌟 Aurora Director V7 Integrado - CEREBRO PRINCIPAL DEL SISTEMA AURORA
================================================================================

Este archivo REEMPLAZA y UNIFICA:
- Aurora_Master.py (parcialmente)
- aurora_director.py (completamente)  
- Orquestación inteligente de todos los motores
- Integración completa de Field Profiles + Objective Router

✅ CARACTERÍSTICAS PRINCIPALES:
- Detección automática de componentes disponibles
- Mapeo inteligente de objetivos usando Objective Router
- Generación por fases usando Field Profiles V7
- Orquestación de los 3 motores principales
- Compatibilidad total V6 + potencia V7
- API ultra-simple para el usuario final

🎯 OBJETIVO: Un solo archivo que coordine todo el sistema Aurora de forma inteligente
================================================================================
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path
import time
import json

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Aurora.Director.V7.Integrated")

# === DETECCIÓN AUTOMÁTICA DE COMPONENTES ===
COMPONENTES_DISPONIBLES = {}

def detectar_componente(nombre_modulo: str, alias: str = None) -> bool:
    """Detecta si un componente está disponible e inicializa el gestor"""
    try:
        if nombre_modulo == "field_profiles":
            from field_profiles import crear_gestor_perfiles, obtener_perfil_campo, recomendar_secuencia_objetivo
            COMPONENTES_DISPONIBLES["field_profiles"] = {
                "gestor": crear_gestor_perfiles(),
                "obtener_perfil": obtener_perfil_campo,
                "recomendar_secuencia": recomendar_secuencia_objetivo
            }
            return True
            
        elif nombre_modulo == "objective_router":
            from objective_router import obtener_router, rutear_objetivo_inteligente, listar_objetivos_disponibles
            COMPONENTES_DISPONIBLES["objective_router"] = {
                "router": obtener_router(),
                "rutear": rutear_objetivo_inteligente,
                "listar": listar_objetivos_disponibles
            }
            return True
            
        elif nombre_modulo == "neuromix":
            from neuromix_engine_v26_ultimate import generate_contextual_neuro_wave, AuroraNeuroAcousticEngine
            COMPONENTES_DISPONIBLES["neuromix"] = {
                "engine": AuroraNeuroAcousticEngine(),
                "generate": generate_contextual_neuro_wave
            }
            return True
            
        elif nombre_modulo == "hypermod":
            from hypermod_engine_v31 import generar_bloques_aurora_integrado, crear_preset_relajacion
            COMPONENTES_DISPONIBLES["hypermod"] = {
                "generar": generar_bloques_aurora_integrado,
                "preset_relajacion": crear_preset_relajacion
            }
            return True
            
        elif nombre_modulo == "harmonic_essence":
            from harmonicEssence_v33py import HarmonicEssenceV34AuroraConnected, NoiseConfigV34Unificado
            COMPONENTES_DISPONIBLES["harmonic_essence"] = {
                "engine": HarmonicEssenceV34AuroraConnected(),
                "config_class": NoiseConfigV34Unificado
            }
            return True
            
        elif nombre_modulo == "emotion_style":
            from emotion_style_profiles import crear_gestor_emotion_style_unificado, obtener_experiencia_completa
            COMPONENTES_DISPONIBLES["emotion_style"] = {
                "gestor": crear_gestor_emotion_style_unificado(),
                "obtener_experiencia": obtener_experiencia_completa
            }
            return True
            
        elif nombre_modulo == "quality_pipeline":
            try:
                from aurora_quality_pipeline import AuroraQualityPipeline
                COMPONENTES_DISPONIBLES["quality_pipeline"] = AuroraQualityPipeline()
                return True
            except ImportError:
                # Fallback básico
                class QualityFallback:
                    def validar_y_normalizar(self, signal):
                        if signal.ndim == 1: signal = np.stack([signal, signal])
                        max_val = np.max(np.abs(signal))
                        if max_val > 0: signal = signal * (0.85 / max_val)
                        return np.clip(signal, -1.0, 1.0)
                COMPONENTES_DISPONIBLES["quality_pipeline"] = QualityFallback()
                return True
                
        return False
        
    except ImportError as e:
        logger.warning(f"⚠️ Componente {nombre_modulo} no disponible: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error inicializando {nombre_modulo}: {e}")
        return False

# Detectar todos los componentes al importar
DETECCION_COMPONENTES = {
    "field_profiles": detectar_componente("field_profiles"),
    "objective_router": detectar_componente("objective_router"), 
    "neuromix": detectar_componente("neuromix"),
    "hypermod": detectar_componente("hypermod"),
    "harmonic_essence": detectar_componente("harmonic_essence"),
    "emotion_style": detectar_componente("emotion_style"),
    "quality_pipeline": detectar_componente("quality_pipeline")
}

logger.info(f"🔍 Componentes detectados: {sum(DETECCION_COMPONENTES.values())}/7")
for comp, disponible in DETECCION_COMPONENTES.items():
    emoji = "✅" if disponible else "❌"
    logger.info(f"  {emoji} {comp}")

# === CONFIGURACIÓN Y ESTRUCTURAS DE DATOS ===

class EstrategiaGeneracion(Enum):
    """Estrategias de generación según componentes disponibles"""
    FIELD_PROFILES_COMPLETO = "field_profiles_completo"  # Field Profiles + todos los motores
    ROUTER_INTELIGENTE = "router_inteligente"            # Objective Router + motores disponibles
    MOTORES_DIRECTOS = "motores_directos"                # Solo motores sin inteligencia de campo
    FALLBACK_BASICO = "fallback_basico"                  # Generación mínima garantizada

@dataclass 
class ConfiguracionAurora:
    """Configuración unificada para toda la generación Aurora"""
    # Básicos
    objetivo: str = "relajacion"
    duracion_min: int = 20
    sample_rate: int = 44100
    
    # Inteligencia
    usar_field_profiles: bool = True
    usar_objective_router: bool = True
    personalizar_experiencia: bool = True
    
    # Calidad
    intensidad: str = "media"  # suave, media, intenso
    estilo: str = "sereno"     # sereno, crystalline, organico, etc.
    
    # Motores
    usar_neuromix: bool = True
    usar_hypermod: bool = True  
    usar_harmonic_essence: bool = True
    
    # Output
    normalizar: bool = True
    exportar_wav: bool = True
    nombre_archivo: str = "aurora_experience"
    
    # Avanzado
    configuracion_personalizada: Dict[str, Any] = field(default_factory=dict)
    perfil_usuario: Optional[Dict[str, Any]] = None

@dataclass
class ResultadoAurora:
    """Resultado completo de la generación Aurora"""
    audio_data: np.ndarray
    sample_rate: int
    duracion_segundos: float
    estrategia_usada: EstrategiaGeneracion
    componentes_utilizados: List[str]
    fases_generadas: List[Dict[str, Any]]
    configuracion_aplicada: Dict[str, Any]
    tiempo_generacion: float
    calidad_score: float
    recomendaciones: List[str]
    ruta_archivo: Optional[str] = None
    metadata_completa: Dict[str, Any] = field(default_factory=dict)

# === AURORA DIRECTOR V7 INTEGRADO ===

class AuroraDirectorV7Integrado:
    """
    🧠 CEREBRO PRINCIPAL DEL SISTEMA AURORA V7
    
    Funciones principales:
    1. Detección automática de componentes
    2. Mapeo inteligente de objetivos
    3. Generación por fases con Field Profiles
    4. Orquestación de motores
    5. Control de calidad integrado
    """
    
    def __init__(self):
        self.version = "Aurora Director V7 Integrado"
        self.componentes = COMPONENTES_DISPONIBLES.copy()
        self.deteccion = DETECCION_COMPONENTES.copy()
        self.estadisticas = {
            "experiencias_generadas": 0,
            "tiempo_total": 0.0,
            "estrategias_usadas": {},
            "objetivos_mas_usados": {}
        }
        
        logger.info(f"🌟 {self.version} inicializado")
        self._mostrar_estado_componentes()
    
    def _mostrar_estado_componentes(self):
        """Muestra el estado de los componentes detectados"""
        logger.info("🔧 Estado de componentes:")
        total = len(self.deteccion)
        disponibles = sum(self.deteccion.values())
        
        for nombre, disponible in self.deteccion.items():
            estado = "✅ ACTIVO" if disponible else "❌ NO DISPONIBLE"
            logger.info(f"  • {nombre}: {estado}")
        
        logger.info(f"📊 Componentes disponibles: {disponibles}/{total} ({disponibles/total*100:.0f}%)")
    
    def crear_experiencia(self, objetivo: str, **kwargs) -> ResultadoAurora:
        """
        🎯 API PRINCIPAL: Crea una experiencia Aurora completa
        
        Args:
            objetivo: Objetivo emocional/funcional (ej: "concentracion", "relajacion")
            **kwargs: Parámetros opcionales de configuración
        
        Returns:
            ResultadoAurora con audio y metadata completa
        """
        tiempo_inicio = time.time()
        
        try:
            logger.info(f"🎯 Creando experiencia: {objetivo}")
            
            # 1. CONFIGURACIÓN INTELIGENTE
            config = self._crear_configuracion_inteligente(objetivo, kwargs)
            
            # 2. ESTRATEGIA AUTOMÁTICA  
            estrategia = self._detectar_estrategia_optima(config)
            logger.info(f"🧠 Estrategia seleccionada: {estrategia.value}")
            
            # 3. GENERACIÓN CON ESTRATEGIA ÓPTIMA
            audio_data, fases_info, componentes_usados = self._generar_con_estrategia(estrategia, config)
            
            # 4. VALIDACIÓN Y CALIDAD
            audio_final, calidad_score = self._aplicar_control_calidad(audio_data, config)
            
            # 5. EXPORTAR SI ES NECESARIO
            ruta_archivo = self._exportar_audio(audio_final, config) if config.exportar_wav else None
            
            # 6. CREAR RESULTADO COMPLETO
            tiempo_total = time.time() - tiempo_inicio
            resultado = self._crear_resultado_completo(
                audio_final, config, estrategia, fases_info, 
                componentes_usados, calidad_score, tiempo_total, ruta_archivo
            )
            
            # 7. ACTUALIZAR ESTADÍSTICAS
            self._actualizar_estadisticas(objetivo, estrategia, tiempo_total)
            
            logger.info(f"✅ Experiencia '{objetivo}' creada exitosamente!")
            logger.info(f"  📊 Estrategia: {estrategia.value}")
            logger.info(f"  🎵 Duración: {resultado.duracion_segundos:.1f}s")
            logger.info(f"  💯 Calidad: {calidad_score:.1f}/100")
            logger.info(f"  ⚡ Tiempo: {tiempo_total:.2f}s")
            
            return resultado
            
        except Exception as e:
            logger.error(f"❌ Error creando experiencia '{objetivo}': {e}")
            # Generar experiencia de emergencia
            return self._crear_experiencia_emergencia(objetivo, str(e))
    
    def _crear_configuracion_inteligente(self, objetivo: str, kwargs: Dict) -> ConfiguracionAurora:
        """Crea configuración optimizada según el objetivo"""
        
        # Configuraciones base por objetivo
        configuraciones_base = {
            "concentracion": {
                "intensidad": "media",
                "estilo": "crystalline", 
                "usar_neuromix": True,
                "duracion_min": 25
            },
            "relajacion": {
                "intensidad": "suave",
                "estilo": "sereno",
                "usar_harmonic_essence": True,
                "duracion_min": 20
            },
            "creatividad": {
                "intensidad": "media", 
                "estilo": "organico",
                "usar_emotion_style": True,
                "duracion_min": 30
            },
            "meditacion": {
                "intensidad": "suave",
                "estilo": "mistico", 
                "usar_field_profiles": True,
                "duracion_min": 35
            }
        }
        
        # Obtener configuración base
        config_base = {}
        for objetivo_key, config in configuraciones_base.items():
            if objetivo_key in objetivo.lower():
                config_base = config
                break
        
        # Combinar configuración base + kwargs del usuario
        config_final = {
            "objetivo": objetivo,
            **config_base,
            **kwargs  # Los kwargs del usuario tienen prioridad
        }
        
        return ConfiguracionAurora(**config_final)
    
    def _detectar_estrategia_optima(self, config: ConfiguracionAurora) -> EstrategiaGeneracion:
        """Detecta automáticamente la mejor estrategia según componentes disponibles"""
        
        # Estrategia 1: Field Profiles Completo (ideal)
        if (self.deteccion["field_profiles"] and 
            self.deteccion["objective_router"] and
            config.usar_field_profiles and
            config.personalizar_experiencia):
            return EstrategiaGeneracion.FIELD_PROFILES_COMPLETO
        
        # Estrategia 2: Router Inteligente
        elif (self.deteccion["objective_router"] and 
              config.usar_objective_router and
              sum(self.deteccion.values()) >= 4):
            return EstrategiaGeneracion.ROUTER_INTELIGENTE
        
        # Estrategia 3: Motores Directos
        elif sum([self.deteccion["neuromix"], self.deteccion["hypermod"], 
                 self.deteccion["harmonic_essence"]]) >= 2:
            return EstrategiaGeneracion.MOTORES_DIRECTOS
        
        # Estrategia 4: Fallback Básico (siempre funciona)
        else:
            return EstrategiaGeneracion.FALLBACK_BASICO
    
    def _generar_con_estrategia(self, estrategia: EstrategiaGeneracion, 
                               config: ConfiguracionAurora) -> Tuple[np.ndarray, List[Dict], List[str]]:
        """Genera audio usando la estrategia óptima detectada"""
        
        if estrategia == EstrategiaGeneracion.FIELD_PROFILES_COMPLETO:
            return self._generar_field_profiles_completo(config)
        
        elif estrategia == EstrategiaGeneracion.ROUTER_INTELIGENTE:
            return self._generar_router_inteligente(config)
        
        elif estrategia == EstrategiaGeneracion.MOTORES_DIRECTOS:
            return self._generar_motores_directos(config)
        
        else:  # FALLBACK_BASICO
            return self._generar_fallback_basico(config)
    
    def _generar_field_profiles_completo(self, config: ConfiguracionAurora) -> Tuple[np.ndarray, List[Dict], List[str]]:
        """Generación completa usando Field Profiles + Objective Router"""
        logger.info("🌟 Generando con Field Profiles V7 completo")
        
        # 1. MAPEO INTELIGENTE DE OBJETIVO
        router = self.componentes["objective_router"]["router"]
        resultado_ruteo = router.rutear_objetivo(config.objetivo, perfil_usuario=config.perfil_usuario)
        
        # 2. SECUENCIA DE PERFILES RECOMENDADA
        field_profiles = self.componentes["field_profiles"]["gestor"]
        secuencia_perfiles = field_profiles.recomendar_secuencia_perfiles(
            config.objetivo, config.duracion_min
        )
        
        if not secuencia_perfiles:
            logger.warning("Sin secuencia Field Profiles, usando ruteo básico")
            return self._generar_router_inteligente(config)
        
        # 3. GENERACIÓN POR FASES
        capas_fases = []
        fases_info = []
        
        for i, (nombre_perfil, duracion_min) in enumerate(secuencia_perfiles):
            logger.info(f"  🎬 Fase {i+1}: {nombre_perfil} ({duracion_min} min)")
            
            # Obtener perfil de campo
            perfil = field_profiles.obtener_perfil(nombre_perfil)
            if not perfil:
                logger.warning(f"Perfil {nombre_perfil} no encontrado")
                continue
            
            # Generar audio de la fase
            audio_fase = self._generar_fase_field_profile(perfil, duracion_min, config)
            
            if audio_fase is not None:
                capas_fases.append(audio_fase)
                fases_info.append({
                    "numero": i + 1,
                    "nombre": nombre_perfil,
                    "duracion_min": duracion_min,
                    "campo_consciencia": perfil.campo_consciencia.value,
                    "neurotransmisores": list(perfil.neurotransmisores_principales.keys()),
                    "beat_primario": perfil.configuracion_neuroacustica.beat_primario
                })
        
        if not capas_fases:
            logger.warning("No se generaron fases, usando fallback")
            return self._generar_router_inteligente(config)
        
        # 4. CONCATENAR FASES
        audio_final = np.concatenate(capas_fases, axis=1)
        
        componentes_usados = ["field_profiles", "objective_router"]
        if self.deteccion["neuromix"]: componentes_usados.append("neuromix")
        if self.deteccion["harmonic_essence"]: componentes_usados.append("harmonic_essence")
        
        return audio_final, fases_info, componentes_usados
    
    def _generar_fase_field_profile(self, perfil, duracion_min: float, config: ConfiguracionAurora) -> Optional[np.ndarray]:
        """Genera una fase específica usando un Field Profile"""
        try:
            duracion_sec = duracion_min * 60
            capas_fase = []
            
            # Capa neuroacústica principal
            if self.deteccion["neuromix"]:
                contexto_neuro = {
                    "intensidad": config.intensidad,
                    "estilo": perfil.style,
                    "objetivo_funcional": perfil.campo_consciencia.value,
                    "beat_primario": perfil.configuracion_neuroacustica.beat_primario
                }
                
                neurotransmisor_principal = list(perfil.neurotransmisores_principales.keys())[0]
                audio_neuro = self.componentes["neuromix"]["generate"](
                    neurotransmitter=neurotransmisor_principal,
                    duration_sec=duracion_sec,
                    context=contexto_neuro,
                    sample_rate=config.sample_rate
                )
                capas_fase.append(audio_neuro * 0.6)
            
            # Capa de texturas emocionales
            if self.deteccion["harmonic_essence"]:
                config_texturas = self.componentes["harmonic_essence"]["config_class"](
                    duration_sec=duracion_sec,
                    sample_rate=config.sample_rate,
                    amplitude=0.4,
                    neurotransmitter_profile=list(perfil.neurotransmisores_principales.keys())[0],
                    style_profile=perfil.style,
                    texture_complexity=perfil.configuracion_neuroacustica.modulacion_amplitude
                )
                
                audio_texturas = self.componentes["harmonic_essence"]["engine"].generate_textured_noise(config_texturas)
                capas_fase.append(audio_texturas * 0.4)
            
            # Mezclar capas de la fase
            if capas_fase:
                # Normalizar longitud
                min_length = min(capa.shape[-1] for capa in capas_fase)
                capas_norm = []
                
                for capa in capas_fase:
                    if capa.ndim == 1:
                        capa = np.stack([capa, capa])
                    capa_norm = capa[..., :min_length]
                    capas_norm.append(capa_norm)
                
                # Mezcla inteligente según el campo de consciencia
                if perfil.campo_consciencia.value == "cognitivo":
                    # Para objetivos cognitivos, neuro domina
                    audio_fase = capas_norm[0] * 0.7 + np.sum(capas_norm[1:], axis=0) * 0.3
                else:
                    # Para otros objetivos, mezcla equilibrada
                    audio_fase = np.sum(capas_norm, axis=0) / len(capas_norm)
                
                return audio_fase
                
        except Exception as e:
            logger.error(f"Error generando fase Field Profile: {e}")
            return None
        
        return None
    
    def _generar_router_inteligente(self, config: ConfiguracionAurora) -> Tuple[np.ndarray, List[Dict], List[str]]:
        """Generación usando Objective Router + motores disponibles"""
        logger.info("🧠 Generando con Router Inteligente")
        
        # 1. RUTEO INTELIGENTE
        resultado_ruteo = self.componentes["objective_router"]["rutear"](
            config.objetivo, 
            perfil_usuario=config.perfil_usuario
        )
        
        config_v6 = resultado_ruteo["configuracion_v6"]
        
        # 2. GENERACIÓN CON CONFIGURACIÓN RUTEADA
        duracion_sec = config.duracion_min * 60
        capas = []
        
        # Capa principal con NeuroMix
        if self.deteccion["neuromix"]:
            contexto = {
                "intensidad": config_v6["beat"],
                "estilo": config_v6["estilo"],
                "objetivo_funcional": config.objetivo
            }
            
            audio_principal = self.componentes["neuromix"]["generate"](
                neurotransmitter="dopamina",  # Default seguro
                duration_sec=duracion_sec,
                context=contexto,
                sample_rate=config.sample_rate
            )
            capas.append(audio_principal * 0.7)
        
        # Capa de texturas
        if self.deteccion["harmonic_essence"]:
            config_texturas = self.componentes["harmonic_essence"]["config_class"](
                duration_sec=duracion_sec,
                sample_rate=config.sample_rate,
                amplitude=0.3
            )
            
            audio_texturas = self.componentes["harmonic_essence"]["engine"].generate_textured_noise(config_texturas)
            capas.append(audio_texturas * 0.3)
        
        # Mezclar
        audio_final = self._mezclar_capas_inteligente(capas)
        
        fases_info = [{
            "numero": 1,
            "nombre": config.objetivo,
            "duracion_min": config.duracion_min,
            "configuracion_router": config_v6
        }]
        
        componentes_usados = ["objective_router"]
        if self.deteccion["neuromix"]: componentes_usados.append("neuromix")
        if self.deteccion["harmonic_essence"]: componentes_usados.append("harmonic_essence")
        
        return audio_final, fases_info, componentes_usados
    
    def _generar_motores_directos(self, config: ConfiguracionAurora) -> Tuple[np.ndarray, List[Dict], List[str]]:
        """Generación directa usando solo los motores disponibles"""
        logger.info("⚙️ Generando con motores directos")
        
        duracion_sec = config.duracion_min * 60
        capas = []
        componentes_usados = []
        
        # NeuroMix
        if self.deteccion["neuromix"] and config.usar_neuromix:
            audio_neuro = self.componentes["neuromix"]["generate"](
                neurotransmitter="serotonina",
                duration_sec=duracion_sec,
                context={"intensidad": config.intensidad},
                sample_rate=config.sample_rate
            )
            capas.append(audio_neuro * 0.6)
            componentes_usados.append("neuromix")
        
        # HarmonicEssence
        if self.deteccion["harmonic_essence"] and config.usar_harmonic_essence:
            config_texturas = self.componentes["harmonic_essence"]["config_class"](
                duration_sec=duracion_sec,
                sample_rate=config.sample_rate,
                amplitude=0.4
            )
            
            audio_texturas = self.componentes["harmonic_essence"]["engine"].generate_textured_noise(config_texturas)
            capas.append(audio_texturas * 0.4)
            componentes_usados.append("harmonic_essence")
        
        # Si no hay capas, crear audio básico
        if not capas:
            audio_basico = self._crear_audio_basico(duracion_sec, config.sample_rate)
            capas.append(audio_basico)
            componentes_usados.append("generacion_basica")
        
        audio_final = self._mezclar_capas_inteligente(capas)
        
        fases_info = [{
            "numero": 1,
            "nombre": config.objetivo,
            "duracion_min": config.duracion_min,
            "componentes_directos": componentes_usados
        }]
        
        return audio_final, fases_info, componentes_usados
    
    def _generar_fallback_basico(self, config: ConfiguracionAurora) -> Tuple[np.ndarray, List[Dict], List[str]]:
        """Generación de emergencia garantizada"""
        logger.info("🆘 Generando con fallback básico")
        
        audio_basico = self._crear_audio_basico(config.duracion_min * 60, config.sample_rate)
        
        fases_info = [{
            "numero": 1,
            "nombre": config.objetivo,
            "duracion_min": config.duracion_min,
            "tipo": "fallback_basico"
        }]
        
        return audio_basico, fases_info, ["fallback_basico"]
    
    def _crear_audio_basico(self, duracion_sec: float, sample_rate: int) -> np.ndarray:
        """Crea audio básico funcional"""
        samples = int(duracion_sec * sample_rate)
        t = np.linspace(0, duracion_sec, samples)
        
        # Onda base relajante
        freq_base = 10.0  # Alpha waves
        audio = 0.3 * np.sin(2 * np.pi * freq_base * t)
        
        # Modulación suave
        modulacion = 1 + 0.1 * np.sin(2 * np.pi * 0.1 * t)
        audio *= modulacion
        
        # Envolvente de fade
        fade_samples = int(2 * sample_rate)  # 2 segundos
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out
        
        return np.stack([audio, audio])
    
    def _mezclar_capas_inteligente(self, capas: List[np.ndarray]) -> np.ndarray:
        """Mezcla inteligente de capas de audio"""
        if not capas:
            raise ValueError("No hay capas para mezclar")
        
        if len(capas) == 1:
            capa = capas[0]
            return np.stack([capa, capa]) if capa.ndim == 1 else capa
        
        # Normalizar todas las capas al mismo tamaño
        min_length = min(capa.shape[-1] for capa in capas)
        capas_norm = []
        
        for capa in capas:
            if capa.ndim == 1:
                capa = np.stack([capa, capa])
            capa_crop = capa[..., :min_length]
            capas_norm.append(capa_crop)
        
        # Mezcla con ponderación automática
        audio_final = np.sum(capas_norm, axis=0) / len(capas_norm)
        
        return audio_final
    
    def _aplicar_control_calidad(self, audio_data: np.ndarray, config: ConfiguracionAurora) -> Tuple[np.ndarray, float]:
        """Aplica control de calidad y normalización"""
        try:
            if self.deteccion["quality_pipeline"]:
                audio_final = self.componentes["quality_pipeline"].validar_y_normalizar(audio_data)
            else:
                # Control básico
                if audio_data.ndim == 1:
                    audio_data = np.stack([audio_data, audio_data])
                
                max_val = np.max(np.abs(audio_data))
                if max_val > 0:
                    audio_final = audio_data / max_val * 0.85
                else:
                    audio_final = audio_data
                
                audio_final = np.clip(audio_final, -1.0, 1.0)
            
            # Calcular score de calidad
            peak = float(np.max(np.abs(audio_final)))
            rms = float(np.sqrt(np.mean(audio_final**2)))
            
            score = 85  # Base
            if peak > 0.95: score -= 10
            if rms < 0.01: score -= 10
            if 0.1 < rms < 0.5: score += 10
            
            return audio_final, max(70, min(100, score))
            
        except Exception as e:
            logger.warning(f"Error en control de calidad: {e}")
            return audio_data, 75.0
    
    def _exportar_audio(self, audio_data: np.ndarray, config: ConfiguracionAurora) -> Optional[str]:
        """Exporta el audio a archivo WAV"""
        try:
            import scipy.io.wavfile as wav
            
            output_dir = Path("./aurora_output")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_archivo = f"{config.nombre_archivo}_{timestamp}.wav"
            ruta_completa = output_dir / nombre_archivo
            
            # Convertir a int16
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Escribir archivo
            wav.write(str(ruta_completa), config.sample_rate, audio_int16.T)
            
            logger.info(f"💾 Audio exportado: {ruta_completa}")
            return str(ruta_completa)
            
        except ImportError:
            logger.warning("scipy no disponible para exportar WAV")
            return None
        except Exception as e:
            logger.error(f"Error exportando audio: {e}")
            return None
    
    def _crear_resultado_completo(self, audio_data: np.ndarray, config: ConfiguracionAurora,
                                 estrategia: EstrategiaGeneracion, fases_info: List[Dict],
                                 componentes_usados: List[str], calidad_score: float,
                                 tiempo_generacion: float, ruta_archivo: Optional[str]) -> ResultadoAurora:
        """Crea el resultado completo de la generación"""
        
        duracion_segundos = audio_data.shape[-1] / config.sample_rate
        
        # Generar recomendaciones
        recomendaciones = []
        if calidad_score < 80:
            recomendaciones.append("Considerar aumentar duración para mejor calidad")
        if len(componentes_usados) < 3:
            recomendaciones.append("Más componentes disponibles mejorarían la experiencia")
        if calidad_score > 95:
            recomendaciones.append("Excelente calidad - ideal para uso terapéutico")
        
        return ResultadoAurora(
            audio_data=audio_data,
            sample_rate=config.sample_rate,
            duracion_segundos=duracion_segundos,
            estrategia_usada=estrategia,
            componentes_utilizados=componentes_usados,
            fases_generadas=fases_info,
            configuracion_aplicada=config.__dict__.copy(),
            tiempo_generacion=tiempo_generacion,
            calidad_score=calidad_score,
            recomendaciones=recomendaciones,
            ruta_archivo=ruta_archivo,
            metadata_completa={
                "version_aurora": self.version,
                "timestamp": datetime.now().isoformat(),
                "componentes_detectados": self.deteccion.copy(),
                "total_fases": len(fases_info)
            }
        )
    
    def _crear_experiencia_emergencia(self, objetivo: str, error: str) -> ResultadoAurora:
        """Crea experiencia de emergencia cuando todo falla"""
        logger.warning(f"🆘 Creando experiencia de emergencia para '{objetivo}'")
        
        duracion_sec = 20 * 60  # 20 minutos por defecto
        audio_emergencia = self._crear_audio_basico(duracion_sec, 44100)
        
        return ResultadoAurora(
            audio_data=audio_emergencia,
            sample_rate=44100,
            duracion_segundos=duracion_sec,
            estrategia_usada=EstrategiaGeneracion.FALLBACK_BASICO,
            componentes_utilizados=["emergencia"],
            fases_generadas=[{
                "numero": 1,
                "nombre": "emergencia",
                "duracion_min": 20,
                "error": error
            }],
            configuracion_aplicada={"objetivo": objetivo, "modo": "emergencia"},
            tiempo_generacion=0.0,
            calidad_score=70.0,
            recomendaciones=["Audio de emergencia generado - verificar componentes del sistema"],
            metadata_completa={"modo_emergencia": True, "error_original": error}
        )
    
    def _actualizar_estadisticas(self, objetivo: str, estrategia: EstrategiaGeneracion, tiempo: float):
        """Actualiza estadísticas de uso del sistema"""
        self.estadisticas["experiencias_generadas"] += 1
        self.estadisticas["tiempo_total"] += tiempo
        
        # Estrategias usadas
        estrategia_key = estrategia.value
        if estrategia_key not in self.estadisticas["estrategias_usadas"]:
            self.estadisticas["estrategias_usadas"][estrategia_key] = 0
        self.estadisticas["estrategias_usadas"][estrategia_key] += 1
        
        # Objetivos más usados
        if objetivo not in self.estadisticas["objetivos_mas_usados"]:
            self.estadisticas["objetivos_mas_usados"][objetivo] = 0
        self.estadisticas["objetivos_mas_usados"][objetivo] += 1
    
    def obtener_estadisticas(self) -> Dict[str, Any]:
        """Obtiene estadísticas completas del sistema"""
        return {
            "version": self.version,
            "componentes_disponibles": self.deteccion.copy(),
            "estadisticas_uso": self.estadisticas.copy(),
            "componentes_activos": sum(self.deteccion.values()),
            "total_componentes": len(self.deteccion)
        }
    
    def listar_objetivos_disponibles(self) -> List[str]:
        """Lista todos los objetivos disponibles según los componentes activos"""
        objetivos = set()
        
        # Objetivos básicos siempre disponibles
        objetivos.update(["relajacion", "concentracion", "creatividad", "meditacion"])
        
        # Si hay objective router, obtener objetivos completos
        if self.deteccion["objective_router"]:
            try:
                objetivos_router = self.componentes["objective_router"]["listar"]()
                objetivos.update(objetivos_router)
            except Exception:
                pass
        
        # Si hay field profiles, añadir perfiles disponibles
        if self.deteccion["field_profiles"]:
            try:
                perfiles = list(self.componentes["field_profiles"]["gestor"].perfiles.keys())
                objetivos.update(perfiles)
            except Exception:
                pass
        
        return sorted(list(objetivos))

# === API ULTRA-SIMPLE PARA EL USUARIO ===

# Instancia global del director
_director_global = None

def Aurora(objetivo: str = None, **kwargs):
    """
    🌟 API ULTRA-SIMPLE DE AURORA V7
    
    Usage:
        # Crear experiencia
        resultado = Aurora("concentracion", duracion_min=25)
        
        # Obtener información del sistema
        info = Aurora()
        
        # Casos específicos
        Aurora.rapido("relajacion")      # 5 minutos
        Aurora.largo("meditacion")       # 60 minutos
        Aurora.terapeutico("sanacion")   # 45 min + calidad máxima
    """
    global _director_global
    
    if _director_global is None:
        _director_global = AuroraDirectorV7Integrado()
    
    if objetivo:
        return _director_global.crear_experiencia(objetivo, **kwargs)
    else:
        return _director_global

# Métodos de conveniencia
Aurora.rapido = lambda obj, **kw: Aurora(obj, duracion_min=5, **kw)
Aurora.largo = lambda obj, **kw: Aurora(obj, duracion_min=60, **kw)
Aurora.terapeutico = lambda obj, **kw: Aurora(obj, duracion_min=45, intensidad="suave", normalizar=True, **kw)
Aurora.stats = lambda: Aurora().obtener_estadisticas()
Aurora.objetivos = lambda: Aurora().listar_objetivos_disponibles()

# === COMPATIBILIDAD V6 ===

def crear_director_objetivo(objetivo: str, duracion_min: int = 20, **kwargs):
    """Compatibilidad V6: Crea director para un objetivo específico"""
    director = Aurora()
    return director.crear_experiencia(objetivo, duracion_min=duracion_min, **kwargs)

def construir_pista_aurora(objetivo: str, **kwargs):
    """Compatibilidad V6: Construye pista Aurora"""
    resultado = Aurora(objetivo, **kwargs)
    return resultado.audio_data

# === TESTING Y DEMOSTRACIÓN ===

if __name__ == "__main__":
    print("🌟 Aurora Director V7 Integrado - CEREBRO PRINCIPAL")
    print("=" * 70)
    
    # Inicializar sistema
    director = Aurora()
    
    # Mostrar estado
    stats = director.obtener_estadisticas()
    print(f"🚀 {stats['version']}")
    print(f"🔧 Componentes activos: {stats['componentes_activos']}/{stats['total_componentes']}")
    
    for componente, disponible in stats["componentes_disponibles"].items():
        estado = "✅" if disponible else "❌"
        print(f"   • {componente}: {estado}")
    
    # Listar objetivos disponibles
    objetivos = director.listar_objetivos_disponibles()
    print(f"\n🎯 Objetivos disponibles: {len(objetivos)}")
    print(f"   Ejemplos: {', '.join(objetivos[:5])}")
    
    # Demostración de generación
    try:
        print(f"\n🎵 Generando experiencia de demostración...")
        resultado = Aurora("concentracion", duracion_min=1, exportar_wav=False)
        
        print(f"✅ ¡Experiencia generada exitosamente!")
        print(f"   • Estrategia: {resultado.estrategia_usada.value}")
        print(f"   • Componentes usados: {', '.join(resultado.componentes_utilizados)}")
        print(f"   • Duración: {resultado.duracion_segundos:.1f}s")
        print(f"   • Calidad: {resultado.calidad_score:.1f}/100")
        print(f"   • Tiempo generación: {resultado.tiempo_generacion:.3f}s")
        print(f"   • Fases: {len(resultado.fases_generadas)}")
        
        if resultado.recomendaciones:
            print(f"   • Recomendación: {resultado.recomendaciones[0]}")
        
    except Exception as e:
        print(f"❌ Error en demostración: {e}")
    
    print(f"\n🎯 Ejemplos de uso:")
    print(f"   Aurora('relajacion')                    # Experiencia básica")
    print(f"   Aurora.rapido('concentracion')          # 5 minutos")
    print(f"   Aurora.largo('meditacion')              # 60 minutos")
    print(f"   Aurora.terapeutico('sanacion')          # Calidad terapéutica")
    print(f"   Aurora().listar_objetivos_disponibles() # Ver objetivos")
    print(f"   Aurora().obtener_estadisticas()         # Stats del sistema")
    
    print(f"\n🏆 AURORA V7 DIRECTOR INTEGRADO - ¡CEREBRO PRINCIPAL FUNCIONAL!")
    print(f"🔄 Mantiene compatibilidad V6 + Potencia completa V7")
    print(f"🧠 Detección automática + Integración inteligente")
