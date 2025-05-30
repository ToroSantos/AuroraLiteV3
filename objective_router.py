import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from functools import lru_cache
from difflib import SequenceMatcher
from collections import defaultdict

from objective_templates_optimized import GestorTemplatesOptimizado, TemplateObjetivoV7, crear_gestor_optimizado
from field_profiles_v7 import GestorPerfilesCampo, PerfilCampoV7, crear_gestor_perfiles, CampoCosciencia
from presets_fases_v7 import GestorFasesConscientes, SecuenciaConsciente, crear_gestor_fases

class TipoRuteo(Enum):
    TEMPLATE_OBJETIVO = "template_objetivo"
    PERFIL_CAMPO = "perfil_campo"
    SECUENCIA_FASES = "secuencia_fases"
    PERSONALIZADO = "personalizado"
    HIBRIDO = "hibrido"

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
    template_objetivo: Optional[TemplateObjetivoV7] = None
    perfil_campo: Optional[PerfilCampoV7] = None
    secuencia_fases: Optional[SecuenciaConsciente] = None
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
    def __init__(self):
        self.gestor_templates = crear_gestor_optimizado()
        self.gestor_perfiles = crear_gestor_perfiles()
        self.gestor_fases = crear_gestor_fases()
        self._inicializar_rutas_v6()
        self.analizador_semantico = AnalizadorSemantico()
        self.motor_personalizacion = MotorPersonalizacion()
        self.validador_cientifico = ValidadorCientifico()
        self.cache_ruteos: Dict[str, ResultadoRuteo] = {}
        self.estadisticas_uso: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.modelos_prediccion: Dict[str, Any] = {}
        self.umbral_confianza_minimo = 0.5
        self.max_alternativas = 5
        self.max_sinergias = 3
    
    def _inicializar_rutas_v6(self):
        self.rutas_v6_mapeadas = {
            "claridad_mental": {"v6_config": {"preset": "claridad_mental", "estilo": "minimalista", "modo": "ascenso", "beat": 14, "capas": {"neuro_wave": True, "binaural": True, "wave_pad": True, "textured_noise": True, "heartbeat": False}}, "v7_mapping": {"tipo_ruteo": TipoRuteo.TEMPLATE_OBJETIVO, "template_nombre": "claridad_mental", "perfil_nombre": "claridad_mental", "contexto": ContextoUso.TRABAJO}},
            "conexion_astral": {"v6_config": {"preset": "conexion_mistica", "estilo": "alienigena", "modo": "expansivo", "beat": 5, "capas": {"neuro_wave": True, "binaural": True, "wave_pad": True, "textured_noise": True, "heartbeat": False}}, "v7_mapping": {"tipo_ruteo": TipoRuteo.TEMPLATE_OBJETIVO, "template_nombre": "conexion_astral", "perfil_nombre": "conexion_espiritual", "contexto": ContextoUso.MEDITACION}},
            "expansion_consciente": {"v6_config": {"preset": "expansion_creativa", "estilo": "mistico", "modo": "ascenso", "beat": 11.5, "capas": {"neuro_wave": True, "binaural": True, "wave_pad": True, "textured_noise": True, "heartbeat": False}}, "v7_mapping": {"tipo_ruteo": TipoRuteo.PERFIL_CAMPO, "perfil_nombre": "expansion_consciente", "template_nombre": "conexion_mistica", "contexto": ContextoUso.MEDITACION}},
            "relajacion": {"v6_config": {"preset": "calma_profunda", "estilo": "etereo", "modo": "disolucion", "beat": 6.5, "capas": {"neuro_wave": True, "binaural": True, "wave_pad": True, "textured_noise": True, "heartbeat": True}}, "v7_mapping": {"tipo_ruteo": TipoRuteo.TEMPLATE_OBJETIVO, "template_nombre": "relajacion_profunda", "perfil_nombre": "sueno", "contexto": ContextoUso.RELAJACION}},
            "enraizamiento": {"v6_config": {"preset": "fuerza_tribal", "estilo": "tribal", "modo": "ritmico", "beat": 10, "capas": {"neuro_wave": True, "binaural": True, "wave_pad": True, "textured_noise": True, "heartbeat": True}}, "v7_mapping": {"tipo_ruteo": TipoRuteo.PERFIL_CAMPO, "perfil_nombre": "enraizamiento", "template_nombre": "bienestar_tribal", "contexto": ContextoUso.EJERCICIO}},
            "autocuidado": {"v6_config": {"preset": "regulacion_emocional", "estilo": "organico", "modo": "normal", "beat": 6.5, "capas": {"neuro_wave": True, "binaural": False, "wave_pad": True, "textured_noise": True, "heartbeat": False}}, "v7_mapping": {"tipo_ruteo": TipoRuteo.PERFIL_CAMPO, "perfil_nombre": "autocuidado", "template_nombre": "detox_emocional", "contexto": ContextoUso.TERAPIA}},
            "gozo_vital": {"v6_config": {"preset": "alegria_sostenida", "estilo": "sutil", "modo": "ascenso", "beat": 11, "capas": {"neuro_wave": False, "binaural": True, "wave_pad": True, "textured_noise": False, "heartbeat": False}}, "v7_mapping": {"tipo_ruteo": TipoRuteo.PERFIL_CAMPO, "perfil_nombre": "gozo_vital", "contexto": ContextoUso.CREATIVIDAD}},
            "visualizacion": {"v6_config": {"preset": "conexion_mistica", "estilo": "mistico", "modo": "expansivo", "beat": 7, "capas": {"neuro_wave": True, "binaural": True, "wave_pad": True, "textured_noise": True, "heartbeat": False}}, "v7_mapping": {"tipo_ruteo": TipoRuteo.SECUENCIA_FASES, "secuencia_nombre": "manifestacion_clasica", "fase_especifica": "visualizacion", "contexto": ContextoUso.MEDITACION}},
            "soltar": {"v6_config": {"preset": "regulacion_emocional", "estilo": "warm_dust", "modo": "disolucion", "beat": 8, "capas": {"neuro_wave": True, "binaural": False, "wave_pad": True, "textured_noise": True, "heartbeat": False}}, "v7_mapping": {"tipo_ruteo": TipoRuteo.PERFIL_CAMPO, "perfil_nombre": "liberacion_emocional", "template_nombre": "detox_emocional", "contexto": ContextoUso.TERAPIA}},
            "agradecimiento": {"v6_config": {"preset": "apertura_corazon", "estilo": "sutil", "modo": "normal", "beat": 7.2, "capas": {"neuro_wave": False, "binaural": True, "wave_pad": True, "textured_noise": False, "heartbeat": False}}, "v7_mapping": {"tipo_ruteo": TipoRuteo.TEMPLATE_OBJETIVO, "template_nombre": "agradecimiento_activo", "perfil_nombre": "gozo_vital", "contexto": ContextoUso.MEDITACION}}
        }
    
    @lru_cache(maxsize=256)
    def rutear_objetivo(self, objetivo: str, perfil_usuario: Optional[PerfilUsuario] = None, contexto: Optional[ContextoUso] = None, personalizacion: Optional[Dict[str, Any]] = None) -> ResultadoRuteo:
        tiempo_inicio = datetime.now()
        try:
            objetivo_procesado = self._procesar_objetivo(objetivo)
            analisis_semantico = self.analizador_semantico.analizar(objetivo_procesado)
            resultado_ruteo = self._ejecutar_ruteo_jerarquico(objetivo_procesado, analisis_semantico, perfil_usuario, contexto)
            if perfil_usuario or personalizacion:
                resultado_ruteo = self.motor_personalizacion.personalizar(resultado_ruteo, perfil_usuario, personalizacion)
            resultado_ruteo = self.validador_cientifico.validar(resultado_ruteo)
            resultado_ruteo = self._enriquecer_con_alternativas(resultado_ruteo)
            tiempo_final = datetime.now()
            resultado_ruteo.tiempo_procesamiento_ms = (tiempo_final - tiempo_inicio).total_seconds() * 1000
            resultado_ruteo.objetivo_original = objetivo
            if resultado_ruteo.puntuacion_confianza > 0.7:
                self.cache_ruteos[objetivo.lower()] = resultado_ruteo
            self._actualizar_estadisticas_uso(objetivo, resultado_ruteo)
            return resultado_ruteo
        except Exception as e:
            return self._crear_ruteo_fallback(objetivo, str(e))
    
    def _procesar_objetivo(self, objetivo: str) -> str:
        objetivo_limpio = re.sub(r'[^\w\s_]', '', objetivo.lower().strip())
        objetivo_limpio = re.sub(r'\s+', ' ', objetivo_limpio)
        mapeo_sinonimos = {"concentracion": "claridad_mental", "concentrarse": "claridad_mental", "enfocar": "claridad_mental", "estudiar": "claridad_mental", "dormir": "sueño", "descansar": "relajacion", "calmar": "relajacion", "tranquilo": "relajacion", "crear": "flow_creativo", "inspiracion": "flow_creativo", "meditar": "meditacion", "paz": "meditacion", "sanar": "autocuidado", "curar": "liberacion_emocional", "liberar": "liberacion_emocional", "energia": "enraizamiento", "fuerza": "enraizamiento", "alegria": "gozo_vital", "felicidad": "gozo_vital", "gratitud": "agradecimiento", "gracias": "agradecimiento", "espiritual": "conexion_espiritual", "divino": "conexion_espiritual", "astral": "conexion_astral", "viaje": "conexion_astral"}
        for sinonimo, objetivo_canonico in mapeo_sinonimos.items():
            if sinonimo in objetivo_limpio:
                objetivo_limpio = objetivo_limpio.replace(sinonimo, objetivo_canonico)
        return objetivo_limpio
    
    def _ejecutar_ruteo_jerarquico(self, objetivo: str, analisis_semantico: Dict[str, Any], perfil_usuario: Optional[PerfilUsuario], contexto: Optional[ContextoUso]) -> ResultadoRuteo:
        for metodo, umbral in [(self._intentar_ruteo_exacto_v6, 0.8), (self._intentar_ruteo_templates_v7, 0.7), (self._intentar_ruteo_perfiles_campo, 0.6), (self._intentar_ruteo_fases, 0.5), (self._intentar_ruteo_semantico_avanzado, 0.4)]:
            resultado = metodo(objetivo, analisis_semantico) if metodo.__name__ != '_intentar_ruteo_exacto_v6' else metodo(objetivo)
            if resultado and resultado.puntuacion_confianza > umbral:
                return resultado
        resultado = self._intentar_ruteo_personalizado_inteligente(objetivo, analisis_semantico, perfil_usuario, contexto)
        return resultado if resultado and resultado.puntuacion_confianza > 0.3 else self._crear_ruteo_fallback(objetivo, "No se encontró ruteo específico")
    
    def _intentar_ruteo_exacto_v6(self, objetivo: str) -> Optional[ResultadoRuteo]:
        if objetivo in self.rutas_v6_mapeadas:
            ruta_data = self.rutas_v6_mapeadas[objetivo]
            v6_config = ruta_data["v6_config"]
            v7_mapping = ruta_data["v7_mapping"]
            resultado = ResultadoRuteo(objetivo_original=objetivo, objetivo_procesado=objetivo, tipo_ruteo=TipoRuteo(v7_mapping["tipo_ruteo"]), nivel_confianza=NivelConfianza.EXACTO, puntuacion_confianza=0.95, preset_emocional=v6_config["preset"], estilo=v6_config["estilo"], modo=v6_config["modo"], beat_base=float(v6_config["beat"]), capas=v6_config["capas"], contexto_inferido=ContextoUso(v7_mapping["contexto"]), fuentes_consultadas=["rutas_v6"], algoritmos_utilizados=["mapeo_directo_v6"])
            self._enriquecer_resultado_con_v7(resultado, v7_mapping)
            return resultado
        return None
    
    def _intentar_ruteo_templates_v7(self, objetivo: str, analisis: Dict[str, Any]) -> Optional[ResultadoRuteo]:
        template = self.gestor_templates.obtener_template(objetivo)
        if template:
            return self._crear_resultado_desde_template(template, objetivo, 0.9)
        templates_disponibles = list(self.gestor_templates.templates.keys())
        similitudes = [(nombre, self._calcular_similitud(objetivo, nombre)) for nombre in templates_disponibles]
        similitudes.sort(key=lambda x: x[1], reverse=True)
        if similitudes and similitudes[0][1] > 0.7:
            template = self.gestor_templates.obtener_template(similitudes[0][0])
            if template:
                return self._crear_resultado_desde_template(template, objetivo, similitudes[0][1])
        palabras_clave = analisis.get("palabras_clave", [objetivo])
        for palabra in palabras_clave:
            templates_por_efecto = self.gestor_templates.buscar_templates_por_efecto(palabra)
            if templates_por_efecto:
                return self._crear_resultado_desde_template(templates_por_efecto[0], objetivo, 0.75)
        return None
    
    def _intentar_ruteo_perfiles_campo(self, objetivo: str, analisis: Dict[str, Any]) -> Optional[ResultadoRuteo]:
        perfil = self.gestor_perfiles.obtener_perfil(objetivo)
        if perfil:
            return self._crear_resultado_desde_perfil(perfil, objetivo, 0.9)
        perfiles_disponibles = list(self.gestor_perfiles.perfiles.keys())
        similitudes = [(nombre, self._calcular_similitud(objetivo, nombre)) for nombre in perfiles_disponibles]
        similitudes.sort(key=lambda x: x[1], reverse=True)
        if similitudes and similitudes[0][1] > 0.6:
            perfil = self.gestor_perfiles.obtener_perfil(similitudes[0][0])
            if perfil:
                return self._crear_resultado_desde_perfil(perfil, objetivo, similitudes[0][1])
        criterios = self._convertir_analisis_a_criterios(analisis)
        perfiles_encontrados = self.gestor_perfiles.buscar_perfiles(criterios)
        return self._crear_resultado_desde_perfil(perfiles_encontrados[0], objetivo, 0.7) if perfiles_encontrados else None
    
    def _intentar_ruteo_fases(self, objetivo: str, analisis: Dict[str, Any]) -> Optional[ResultadoRuteo]:
        fase = self.gestor_fases.obtener_fase(objetivo)
        if fase:
            return self._crear_resultado_desde_fase(fase, objetivo, 0.85)
        secuencias = list(self.gestor_fases.secuencias_predefinidas.keys())
        for nombre_secuencia in secuencias:
            if self._calcular_similitud(objetivo, nombre_secuencia) > 0.6:
                secuencia = self.gestor_fases.obtener_secuencia(nombre_secuencia)
                if secuencia:
                    return self._crear_resultado_desde_secuencia(secuencia, objetivo, 0.75)
        return None
    
    def _intentar_ruteo_semantico_avanzado(self, objetivo: str, analisis: Dict[str, Any]) -> Optional[ResultadoRuteo]:
        intencion = analisis.get("intencion_principal", "relajacion")
        mapeo_intenciones = {"concentrar": {"template": "claridad_mental", "perfil": "claridad_mental"}, "relajar": {"template": "relajacion_profunda", "perfil": "sueno"}, "crear": {"template": "enfoque_total", "perfil": "flow_creativo"}, "meditar": {"template": "presencia_total", "perfil": "meditacion"}, "sanar": {"template": "detox_emocional", "perfil": "autocuidado"}, "conectar": {"template": "conexion_mistica", "perfil": "conexion_espiritual"}, "energizar": {"template": "bienestar_tribal", "perfil": "enraizamiento"}}
        if intencion in mapeo_intenciones:
            config = mapeo_intenciones[intencion]
            template = self.gestor_templates.obtener_template(config["template"])
            if template:
                return self._crear_resultado_desde_template(template, objetivo, 0.6)
            perfil = self.gestor_perfiles.obtener_perfil(config["perfil"])
            if perfil:
                return self._crear_resultado_desde_perfil(perfil, objetivo, 0.6)
        return None
    
    def _intentar_ruteo_personalizado_inteligente(self, objetivo: str, analisis: Dict[str, Any], perfil_usuario: Optional[PerfilUsuario], contexto: Optional[ContextoUso]) -> Optional[ResultadoRuteo]:
        config_personalizada = self._generar_configuracion_inteligente(objetivo, analisis, perfil_usuario, contexto)
        if config_personalizada:
            return ResultadoRuteo(objetivo_original=objetivo, objetivo_procesado=objetivo, tipo_ruteo=TipoRuteo.PERSONALIZADO, nivel_confianza=NivelConfianza.INFERIDO, puntuacion_confianza=0.4, preset_emocional=config_personalizada["preset"], estilo=config_personalizada["estilo"], modo=config_personalizada["modo"], beat_base=config_personalizada["beat"], capas=config_personalizada["capas"], contexto_inferido=contexto, fuentes_consultadas=["analisis_semantico", "ia_personalizada"], algoritmos_utilizados=["generacion_inteligente"], personalizaciones_sugeridas=["Configuración generada automáticamente"])
        return None
    
    def _crear_ruteo_fallback(self, objetivo: str, razon: str) -> ResultadoRuteo:
        return ResultadoRuteo(objetivo_original=objetivo, objetivo_procesado=objetivo, tipo_ruteo=TipoRuteo.PERSONALIZADO, nivel_confianza=NivelConfianza.BAJO, puntuacion_confianza=0.3, preset_emocional="calma_profunda", estilo="sereno", modo="normal", beat_base=8.0, capas={"neuro_wave": True, "binaural": True, "wave_pad": True, "textured_noise": True, "heartbeat": False}, contexto_inferido=ContextoUso.RELAJACION, fuentes_consultadas=["fallback_seguro"], algoritmos_utilizados=["configuracion_default"], personalizaciones_sugeridas=[f"Ruteo fallback usado - {razon}"])
    
    def _crear_resultado_desde_template(self, template: TemplateObjetivoV7, objetivo: str, confianza: float) -> ResultadoRuteo:
        return ResultadoRuteo(objetivo_original=objetivo, objetivo_procesado=objetivo, tipo_ruteo=TipoRuteo.TEMPLATE_OBJETIVO, nivel_confianza=self._calcular_nivel_confianza(confianza), puntuacion_confianza=confianza, preset_emocional=template.emotional_preset, estilo=template.style, modo="template_v7", beat_base=template.frecuencia_dominante, capas=self._convertir_capas_template_a_v6(template.layers), template_objetivo=template, contexto_inferido=self._inferir_contexto_desde_categoria(template.categoria), fuentes_consultadas=["templates_v7"], algoritmos_utilizados=["mapeo_template_v7"], coherencia_neuroacustica=template.coherencia_neuroacustica, evidencia_cientifica=template.evidencia_cientifica, contraindicaciones=template.contraindicaciones)
    
    def _crear_resultado_desde_perfil(self, perfil: PerfilCampoV7, objetivo: str, confianza: float) -> ResultadoRuteo:
        return ResultadoRuteo(objetivo_original=objetivo, objetivo_procesado=objetivo, tipo_ruteo=TipoRuteo.PERFIL_CAMPO, nivel_confianza=self._calcular_nivel_confianza(confianza), puntuacion_confianza=confianza, preset_emocional=f"campo_{perfil.nombre}", estilo=perfil.style, modo="perfil_campo", beat_base=perfil.configuracion_neuroacustica.beat_primario, capas=self._convertir_capas_perfil_a_v6(perfil), perfil_campo=perfil, contexto_inferido=self._inferir_contexto_desde_campo(perfil.campo_consciencia), fuentes_consultadas=["perfiles_campo_v7"], algoritmos_utilizados=["mapeo_perfil_campo"], coherencia_neuroacustica=perfil._calcular_coherencia_neuroacustica(), evidencia_cientifica=perfil.base_cientifica, contraindicaciones=perfil.contraindicaciones)
    
    def _crear_resultado_desde_fase(self, fase: Any, objetivo: str, confianza: float) -> ResultadoRuteo:
        return ResultadoRuteo(objetivo_original=objetivo, objetivo_procesado=objetivo, tipo_ruteo=TipoRuteo.SECUENCIA_FASES, nivel_confianza=self._calcular_nivel_confianza(confianza), puntuacion_confianza=confianza, preset_emocional=fase.emocional_preset, estilo=fase.estilo, modo="fase_consciente", beat_base=fase.beat_base, capas=self._convertir_capas_fase_a_v6(fase.capas), contexto_inferido=ContextoUso.MEDITACION, fuentes_consultadas=["fases_conscientes_v7"], algoritmos_utilizados=["mapeo_fase_consciente"], evidencia_cientifica=fase.base_cientifica)
    
    def _crear_resultado_desde_secuencia(self, secuencia: SecuenciaConsciente, objetivo: str, confianza: float) -> ResultadoRuteo:
        primera_fase = secuencia.fases[0] if secuencia.fases else None
        if not primera_fase:
            return self._crear_ruteo_fallback(objetivo, "Secuencia vacía")
        resultado = self._crear_resultado_desde_fase(primera_fase, objetivo, confianza)
        resultado.secuencia_fases = secuencia
        resultado.secuencia_recomendada = [fase.nombre for fase in secuencia.fases]
        return resultado
    
    def _enriquecer_resultado_con_v7(self, resultado: ResultadoRuteo, v7_mapping: Dict[str, Any]):
        if "template_nombre" in v7_mapping:
            template = self.gestor_templates.obtener_template(v7_mapping["template_nombre"])
            if template:
                resultado.template_objetivo = template
                resultado.coherencia_neuroacustica = template.coherencia_neuroacustica
                resultado.evidencia_cientifica = template.evidencia_cientifica
        if "perfil_nombre" in v7_mapping:
            perfil = self.gestor_perfiles.obtener_perfil(v7_mapping["perfil_nombre"])
            if perfil:
                resultado.perfil_campo = perfil
        if "secuencia_nombre" in v7_mapping:
            secuencia = self.gestor_fases.obtener_secuencia(v7_mapping["secuencia_nombre"])
            if secuencia:
                resultado.secuencia_fases = secuencia
    
    def _enriquecer_con_alternativas(self, resultado: ResultadoRuteo) -> ResultadoRuteo:
        objetivo = resultado.objetivo_procesado
        if resultado.template_objetivo:
            templates_categoria = self.gestor_templates.obtener_templates_por_categoria(resultado.template_objetivo.categoria)
            resultado.rutas_alternativas.extend([t.nombre.lower().replace(" ", "_") for t in templates_categoria[:3] if t.nombre.lower().replace(" ", "_") != objetivo])
        if resultado.perfil_campo:
            sinergias = self.gestor_perfiles.sinergias_mapeadas.get(resultado.perfil_campo.nombre, {})
            sinergias_altas = [(nombre, valor) for nombre, valor in sinergias.items() if valor > 0.7]
            sinergias_altas.sort(key=lambda x: x[1], reverse=True)
            resultado.rutas_sinergicas.extend([nombre for nombre, _ in sinergias_altas[:3]])
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
        return {nombre: capa.enabled for nombre, capa in capas_v7.items()}
    
    def _convertir_capas_perfil_a_v6(self, perfil: PerfilCampoV7) -> Dict[str, bool]:
        return {"neuro_wave": len(perfil.neurotransmisores_principales) > 0, "binaural": perfil.configuracion_neuroacustica.beat_primario > 0, "wave_pad": len(perfil.configuracion_neuroacustica.armónicos) > 0, "textured_noise": True, "heartbeat": perfil.campo_consciencia.value in ["emocional", "energetico", "sanacion"]}
    
    def _convertir_capas_fase_a_v6(self, capas_fase: Dict) -> Dict[str, bool]:
        return {nombre: capa.enabled for nombre, capa in capas_fase.items()}
    
    def _inferir_contexto_desde_categoria(self, categoria) -> ContextoUso:
        mapeo = {"cognitivo": ContextoUso.TRABAJO, "creativo": ContextoUso.CREATIVIDAD, "terapeutico": ContextoUso.TERAPIA, "espiritual": ContextoUso.MEDITACION, "emocional": ContextoUso.RELAJACION, "fisico": ContextoUso.EJERCICIO}
        return mapeo.get(categoria.value if hasattr(categoria, 'value') else str(categoria), ContextoUso.RELAJACION)
    
    def _inferir_contexto_desde_campo(self, campo: CampoCosciencia) -> ContextoUso:
        mapeo = {CampoCosciencia.COGNITIVO: ContextoUso.TRABAJO, CampoCosciencia.CREATIVO: ContextoUso.CREATIVIDAD, CampoCosciencia.SANACION: ContextoUso.TERAPIA, CampoCosciencia.ESPIRITUAL: ContextoUso.MEDITACION, CampoCosciencia.EMOCIONAL: ContextoUso.RELAJACION, CampoCosciencia.FISICO: ContextoUso.EJERCICIO, CampoCosciencia.ENERGETICO: ContextoUso.EJERCICIO, CampoCosciencia.SOCIAL: ContextoUso.RELAJACION}
        return mapeo.get(campo, ContextoUso.RELAJACION)
    
    def _convertir_analisis_a_criterios(self, analisis: Dict[str, Any]) -> Dict[str, Any]:
        criterios = {}
        if "palabras_clave" in analisis: criterios["efectos"] = analisis["palabras_clave"]
        if "intencion_principal" in analisis: criterios["campo"] = self._mapear_intencion_a_campo(analisis["intencion_principal"])
        return criterios
    
    def _mapear_intencion_a_campo(self, intencion: str) -> str:
        mapeo = {"concentrar": "cognitivo", "crear": "creativo", "sanar": "sanacion", "meditar": "espiritual", "relajar": "emocional", "energizar": "energetico"}
        return mapeo.get(intencion, "emocional")
    
    def _generar_configuracion_inteligente(self, objetivo: str, analisis: Dict[str, Any], perfil_usuario: Optional[PerfilUsuario], contexto: Optional[ContextoUso]) -> Optional[Dict[str, Any]]:
        configs_contexto = {
            ContextoUso.TRABAJO: {"preset": "claridad_mental", "estilo": "minimalista", "modo": "enfoque", "beat": 14.0, "capas": {"neuro_wave": True, "binaural": True, "wave_pad": True, "textured_noise": True, "heartbeat": False}},
            ContextoUso.RELAJACION: {"preset": "calma_profunda", "estilo": "sereno", "modo": "relajante", "beat": 7.0, "capas": {"neuro_wave": True, "binaural": True, "wave_pad": True, "textured_noise": True, "heartbeat": True}},
            ContextoUso.CREATIVIDAD: {"preset": "expansion_creativa", "estilo": "inspirador", "modo": "flujo", "beat": 10.0, "capas": {"neuro_wave": True, "binaural": True, "wave_pad": True, "textured_noise": True, "heartbeat": False}},
            ContextoUso.MEDITACION: {"preset": "conexion_mistica", "estilo": "mistico", "modo": "profundo", "beat": 6.0, "capas": {"neuro_wave": True, "binaural": True, "wave_pad": True, "textured_noise": True, "heartbeat": False}}
        }
        contexto_usado = contexto or ContextoUso.RELAJACION
        config_base = configs_contexto.get(contexto_usado, configs_contexto[ContextoUso.RELAJACION])
        if perfil_usuario:
            if perfil_usuario.beats_preferidos: config_base["beat"] = perfil_usuario.beats_preferidos[0]
            if perfil_usuario.estilos_preferidos: config_base["estilo"] = perfil_usuario.estilos_preferidos[0]
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
        return {"version": "v7.0_inteligente", "total_ruteos_realizados": total_ruteos, "objetivos_unicos": len(self.estadisticas_uso), "rutas_v6_disponibles": len(self.rutas_v6_mapeadas), "templates_integrados": len(self.gestor_templates.templates), "perfiles_integrados": len(self.gestor_perfiles.perfiles), "fases_integradas": len(self.gestor_fases.fases_base), "cache_hits": len(self.cache_ruteos), "confianza_promedio": sum(stats["confianza_promedio"] for stats in self.estadisticas_uso.values()) / len(self.estadisticas_uso) if self.estadisticas_uso else 0, "tiempo_promedio_ms": sum(stats["tiempo_promedio_ms"] for stats in self.estadisticas_uso.values()) / len(self.estadisticas_uso) if self.estadisticas_uso else 0, "objetivos_mas_usados": sorted([(objetivo, stats["veces_usado"]) for objetivo, stats in self.estadisticas_uso.items()], key=lambda x: x[1], reverse=True)[:10]}

class AnalizadorSemantico:
    def analizar(self, objetivo: str) -> Dict[str, Any]:
        palabras = objetivo.lower().split()
        palabras_clave_relevantes = [palabra for palabra in palabras if len(palabra) > 3 and palabra not in ['para', 'con', 'una', 'más', 'muy', 'todo']]
        intenciones = {"concentrar": ["concentrar", "enfocar", "claridad", "atencion"], "relajar": ["relajar", "calmar", "dormir", "descansar", "paz"], "crear": ["crear", "inspirar", "arte", "innovar", "diseñar"], "meditar": ["meditar", "espiritual", "conexion", "interior"], "sanar": ["sanar", "curar", "terapia", "equilibrar", "restaurar"], "energizar": ["energia", "fuerza", "vitalidad", "activar", "despertar"]}
        intencion_principal = "relajar"
        for intencion, palabras_intencion in intenciones.items():
            if any(palabra in objetivo.lower() for palabra in palabras_intencion):
                intencion_principal = intencion
                break
        return {"palabras_clave": palabras_clave_relevantes, "intencion_principal": intencion_principal, "longitud_objetivo": len(objetivo), "complejidad_linguistica": len(palabras_clave_relevantes), "es_objetivo_simple": len(palabras) <= 3, "contiene_negacion": any(neg in objetivo.lower() for neg in ["no", "sin", "menos", "reducir"])}

class MotorPersonalizacion:
    def personalizar(self, resultado: ResultadoRuteo, perfil_usuario: Optional[PerfilUsuario], personalizacion: Optional[Dict[str, Any]]) -> ResultadoRuteo:
        if not perfil_usuario and not personalizacion: return resultado
        if perfil_usuario:
            if perfil_usuario.duracion_preferida != 25: resultado.personalizaciones_sugeridas.append(f"Duración ajustada a {perfil_usuario.duracion_preferida} min")
            if perfil_usuario.intensidad_preferida != "moderado":
                factor_intensidad = {"suave": 0.7, "moderado": 1.0, "intenso": 1.3}.get(perfil_usuario.intensidad_preferida, 1.0)
                resultado.beat_base *= factor_intensidad
                resultado.personalizaciones_sugeridas.append(f"Intensidad ajustada a {perfil_usuario.intensidad_preferida}")
            if perfil_usuario.capas_preferidas:
                for capa, preferencia in perfil_usuario.capas_preferidas.items():
                    if capa in resultado.capas:
                        resultado.capas[capa] = preferencia
                        resultado.personalizaciones_sugeridas.append(f"Capa {capa} {'activada' if preferencia else 'desactivada'}")
        if personalizacion:
            if "beat_adjustment" in personalizacion:
                resultado.beat_base += personalizacion["beat_adjustment"]
                resultado.personalizaciones_sugeridas.append("Beat personalizado aplicado")
            if "style_override" in personalizacion:
                resultado.estilo = personalizacion["style_override"]
                resultado.personalizaciones_sugeridas.append("Estilo personalizado aplicado")
        return resultado

class ValidadorCientifico:
    def validar(self, resultado: ResultadoRuteo) -> ResultadoRuteo:
        if not 0.5 <= resultado.beat_base <= 100:
            resultado.contraindicaciones.append("Frecuencia fuera de rango seguro")
            resultado.puntuacion_confianza *= 0.8
        if resultado.template_objetivo:
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
        return resultado

_router_global = None

def obtener_router() -> RouterInteligenteV7:
    global _router_global
    if _router_global is None: _router_global = RouterInteligenteV7()
    return _router_global

def ruta_por_objetivo(nombre: str) -> Dict[str, Any]:
    router = obtener_router()
    resultado = router.rutear_objetivo(nombre)
    return {"preset": resultado.preset_emocional, "estilo": resultado.estilo, "modo": resultado.modo, "beat": resultado.beat_base, "capas": resultado.capas}

def listar_objetivos_disponibles() -> List[str]:
    router = obtener_router()
    objetivos = set()
    objetivos.update(router.rutas_v6_mapeadas.keys())
    objetivos.update(router.gestor_templates.templates.keys())
    objetivos.update(router.gestor_perfiles.perfiles.keys())
    objetivos.update(router.gestor_fases.fases_base.keys())
    return sorted(list(objetivos))

def rutear_objetivo_inteligente(objetivo: str, perfil_usuario: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    router = obtener_router()
    perfil_obj = PerfilUsuario(**perfil_usuario) if perfil_usuario else None
    resultado = router.rutear_objetivo(objetivo, perfil_obj, **kwargs)
    return {"configuracion_v6": {"preset": resultado.preset_emocional, "estilo": resultado.estilo, "modo": resultado.modo, "beat": resultado.beat_base, "capas": resultado.capas}, "informacion_v7": {"tipo_ruteo": resultado.tipo_ruteo.value, "confianza": resultado.puntuacion_confianza, "contexto": resultado.contexto_inferido.value if resultado.contexto_inferido else None, "alternativas": resultado.rutas_alternativas, "sinergias": resultado.rutas_sinergicas, "personalizaciones": resultado.personalizaciones_sugeridas, "contraindicaciones": resultado.contraindicaciones, "tiempo_procesamiento": resultado.tiempo_procesamiento_ms}, "recursos_v7": {"template": resultado.template_objetivo.nombre if resultado.template_objetivo else None, "perfil_campo": resultado.perfil_campo.nombre if resultado.perfil_campo else None, "secuencia": resultado.secuencia_fases.nombre if resultado.secuencia_fases else None}}

def buscar_objetivos_similares(objetivo: str, limite: int = 5) -> List[Tuple[str, float]]:
    router = obtener_router()
    objetivos_disponibles = listar_objetivos_disponibles()
    similitudes = [(obj, router._calcular_similitud(objetivo, obj)) for obj in objetivos_disponibles]
    similitudes.sort(key=lambda x: x[1], reverse=True)
    return similitudes[:limite]

def recomendar_secuencia_objetivos(objetivo_principal: str, duracion_total: int = 60, perfil_usuario: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    router = obtener_router()
    resultado_principal = router.rutear_objetivo(objetivo_principal)
    objetivos_secuencia = [objetivo_principal]
    tiempo_usado = resultado_principal.template_objetivo.duracion_recomendada_min if resultado_principal.template_objetivo else 20
    for objetivo_sinergico in resultado_principal.rutas_sinergicas:
        if tiempo_usado >= duracion_total: break
        resultado_sinergico = router.rutear_objetivo(objetivo_sinergico)
        duracion_objetivo = resultado_sinergico.template_objetivo.duracion_recomendada_min if resultado_sinergico.template_objetivo else 15
        if tiempo_usado + duracion_objetivo <= duracion_total:
            objetivos_secuencia.append(objetivo_sinergico)
            tiempo_usado += duracion_objetivo
    secuencia_detallada = []
    for i, objetivo in enumerate(objetivos_secuencia):
        resultado = router.rutear_objetivo(objetivo)
        duracion = resultado.template_objetivo.duracion_recomendada_min if resultado.template_objetivo else 15
        secuencia_detallada.append({"objetivo": objetivo, "orden": i + 1, "duracion_min": duracion, "configuracion": ruta_por_objetivo(objetivo), "confianza": resultado.puntuacion_confianza, "tipo": resultado.tipo_ruteo.value})
    return secuencia_detallada

RUTAS_OBJETIVO = {}

def _generar_rutas_objetivo_v6():
    global RUTAS_OBJETIVO
    router = obtener_router()
    for nombre, data in router.rutas_v6_mapeadas.items(): RUTAS_OBJETIVO[nombre] = data["v6_config"]
    for nombre in router.gestor_templates.templates.keys():
        if nombre not in RUTAS_OBJETIVO:
            config = ruta_por_objetivo(nombre)
            RUTAS_OBJETIVO[nombre] = config
    for nombre in router.gestor_perfiles.perfiles.keys():
        if nombre not in RUTAS_OBJETIVO:
            config = ruta_por_objetivo(nombre)
            RUTAS_OBJETIVO[nombre] = config

_generar_rutas_objetivo_v6()
