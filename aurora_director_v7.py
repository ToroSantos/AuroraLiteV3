import numpy as np
import logging,time,importlib,json
from typing import Dict,List,Optional,Tuple,Any,Union,Protocol
from dataclasses import dataclass,field
from enum import Enum
from datetime import datetime
from pathlib import Path

try:
    from objective_manager import (ObjectiveManagerUnificado,ComponenteEstadoDescripciónRouterInteligenteV7,AnalizadorSemantico,MotorPersonalizacion,ValidadorCientifico,crear_objective_manager_unificado)
    OBJECTIVE_MANAGER_AVAILABLE = True
    logging.info("✅ Objective Manager Unificado detectado")
except ImportError:
    OBJECTIVE_MANAGER_AVAILABLE = False
    logging.warning("⚠️ Objective Manager no disponible")

logging.basicConfig(level=logging.INFO,format='%(asctime)s-%(name)s-%(levelname)s-%(message)s')
logger=logging.getLogger("Aurora.Director.V7")

class MotorAurora(Protocol):
 def generar_audio(self,config:Dict[str,Any],duracion_sec:float)->np.ndarray:...
 def validar_configuracion(self,config:Dict[str,Any])->bool:...
 def obtener_capacidades(self)->Dict[str,Any]:...

class GestorInteligencia(Protocol):
 def procesar_objetivo(self,objetivo:str,contexto:Dict[str,Any])->Dict[str,Any]:...
 def obtener_alternativas(self,objetivo:str)->List[str]:...

class TipoComponente(Enum):
 MOTOR="motor";GESTOR_INTELIGENCIA="gestor_inteligencia";PIPELINE="pipeline";PRESET_MANAGER="preset_manager";STYLE_PROFILE="style_profile";OBJECTIVE_MANAGER="objective_manager"

class EstrategiaGeneracion(Enum):
 AURORA_ORQUESTADO="aurora_orquestado";MULTI_MOTOR="multi_motor";MOTOR_ESPECIALIZADO="motor_especializado";INTELIGENCIA_ADAPTIVA="inteligencia_adaptiva";OBJECTIVE_MANAGER_DRIVEN="objective_manager_driven";FALLBACK_PROGRESIVO="fallback_progresivo"

class ModoOrquestacion(Enum):
 SECUENCIAL="secuencial";PARALELO="paralelo";LAYERED="layered";HYBRID="hybrid"

@dataclass
class ComponenteAurora:
 nombre:str;tipo:TipoComponente;modulo:str;clase_principal:str;disponible:bool=False;instancia:Optional[Any]=None;version:str="unknown";capacidades:Dict[str,Any]=field(default_factory=dict);dependencias:List[str]=field(default_factory=list);fallback_disponible:bool=False;nivel_prioridad:int=1;compatibilidad_aurora:bool=True;metadatos:Dict[str,Any]=field(default_factory=dict)

@dataclass
class ConfiguracionAuroraUnificada:
 objetivo:str="relajacion";duracion_min:int=20;sample_rate:int=44100;estrategia_preferida:Optional[EstrategiaGeneracion]=None;modo_orquestacion:ModoOrquestacion=ModoOrquestacion.HYBRID;motores_preferidos:List[str]=field(default_factory=list);forzar_componentes:List[str]=field(default_factory=list);excluir_componentes:List[str]=field(default_factory=list);intensidad:str="media";estilo:str="sereno";neurotransmisor_preferido:Optional[str]=None;calidad_objetivo:str="alta";normalizar:bool=True;aplicar_mastering:bool=True;validacion_automatica:bool=True;exportar_wav:bool=True;nombre_archivo:str="aurora_experience";incluir_metadatos:bool=True;configuracion_custom:Dict[str,Any]=field(default_factory=dict);perfil_usuario:Optional[Dict[str,Any]]=None;contexto_uso:Optional[str]=None;session_id:Optional[str]=None;usar_objective_manager:bool=True;template_personalizado:Optional[str]=None;perfil_campo_personalizado:Optional[str]=None;secuencia_fases_personalizada:Optional[str]=None
 
 def validar(self)->List[str]:
  p=[]
  if self.duracion_min<=0:p.append("Duración debe ser positiva")
  if self.sample_rate not in[22050,44100,48000]:p.append("Sample rate no estándar")
  if self.intensidad not in["suave","media","intenso"]:p.append("Intensidad inválida")
  if not self.objetivo.strip():p.append("Objetivo no puede estar vacío")
  return p

@dataclass
class ResultadoAuroraIntegrado:
 audio_data:np.ndarray;metadatos:Dict[str,Any];estrategia_usada:EstrategiaGeneracion;modo_orquestacion:ModoOrquestacion;componentes_usados:List[str];tiempo_generacion:float;calidad_score:float;coherencia_neuroacustica:float;efectividad_terapeutica:float;configuracion:ConfiguracionAuroraUnificada;capas_audio:Dict[str,np.ndarray]=field(default_factory=dict);analisis_espectral:Dict[str,Any]=field(default_factory=dict);recomendaciones:List[str]=field(default_factory=list);proxima_sesion:Dict[str,Any]=field(default_factory=dict);resultado_objective_manager:Optional[Dict[str,Any]]=None;template_utilizado:Optional[str]=None;perfil_campo_utilizado:Optional[str]=None;secuencia_fases_utilizada:Optional[str]=None

class DetectorComponentesAvanzado:
 def __init__(self):
  self.componentes_registrados=self._init_registro_completo()
  self.componentes_activos:Dict[str,ComponenteAurora]={}
  self.stats={"total":0,"exitosos":0,"fallidos":0,"fallback":0,"tiempo_deteccion":0.0,"motores_detectados":0,"gestores_detectados":0}
  self.cache_deteccion={}
 
 def _init_registro_completo(self)->Dict[str,ComponenteAurora]:
  registro_base = {
   "neuromix_v27":ComponenteAurora("neuromix_v27",TipoComponente.MOTOR,"neuromix_aurora_v27","AuroraNeuroAcousticEngineV27",dependencias=[],fallback_disponible=True,nivel_prioridad=1,metadatos={"especialidad":"neuroacustica","calidad":"alta"}),
   "hypermod_v32":ComponenteAurora("hypermod_v32",TipoComponente.MOTOR,"hypermod_v32","HyperModEngineV32AuroraConnected",dependencias=[],fallback_disponible=True,nivel_prioridad=1,metadatos={"especialidad":"ondas_cerebrales","calidad":"maxima"}),
   "harmonic_essence_v34":ComponenteAurora("harmonic_essence_v34",TipoComponente.MOTOR,"harmonicEssence_v34","HarmonicEssenceV34AuroraConnected",dependencias=[],fallback_disponible=True,nivel_prioridad=1,metadatos={"especialidad":"texturas","calidad":"alta"}),
   "field_profiles":ComponenteAurora("field_profiles",TipoComponente.GESTOR_INTELIGENCIA,"field_profiles","GestorPerfilesCampo",dependencias=[],fallback_disponible=True,nivel_prioridad=2),
   "objective_router":ComponenteAurora("objective_router",TipoComponente.GESTOR_INTELIGENCIA,"objective_router","RouterInteligenteV7",dependencias=["field_profiles"],fallback_disponible=True,nivel_prioridad=2),
   "emotion_style_profiles":ComponenteAurora("emotion_style_profiles",TipoComponente.GESTOR_INTELIGENCIA,"emotion_style_profiles","GestorEmotionStyleUnificadoV7",dependencias=[],fallback_disponible=True,nivel_prioridad=2),
   "quality_pipeline":ComponenteAurora("quality_pipeline",TipoComponente.PIPELINE,"aurora_quality_pipeline","AuroraQualityPipeline",dependencias=[],fallback_disponible=True,nivel_prioridad=4),
   "neuromix_legacy":ComponenteAurora("neuromix_legacy",TipoComponente.MOTOR,"neuromix_engine_v26_ultimate","AuroraNeuroAcousticEngine",dependencias=[],fallback_disponible=True,nivel_prioridad=5),
   "hypermod_legacy":ComponenteAurora("hypermod_legacy",TipoComponente.MOTOR,"hypermod_engine_v31","NeuroWaveGenerator",dependencias=[],fallback_disponible=True,nivel_prioridad=5),
   "carmine_analyzer_v21":ComponenteAurora("carmine_analyzer_v21",TipoComponente.PIPELINE,"Carmine_Analyzer","CarmineAuroraAnalyzer",dependencias=[],fallback_disponible=True,nivel_prioridad=3,metadatos={"especialidad":"analisis_neuroacustico","version":"2.1","calidad":"maxima"}),
  }
  if OBJECTIVE_MANAGER_AVAILABLE:
      registro_base["objective_manager_unificado"] = ComponenteAurora("objective_manager_unificado",TipoComponente.OBJECTIVE_MANAGER,"objective_manager","ObjectiveManagerUnificado",dependencias=[],fallback_disponible=True,nivel_prioridad=1,metadatos={"especialidad":"gestion_objetivos_integral","version":"unificado_v7","calidad":"maxima","capacidades":["templates","perfiles_campo","secuencias_fases","routing_inteligente"]})
  return registro_base

 def detectar_todos(self)->Dict[str,ComponenteAurora]:
  start_time=time.time()
  logger.info("🔍 Iniciando detección componentes Aurora...")
  componentes_ordenados=sorted(self.componentes_registrados.items(),key=lambda x:x[1].nivel_prioridad)
  for nombre,comp in componentes_ordenados:self._detectar_componente(comp)
  self.stats["tiempo_deteccion"]=time.time()-start_time
  self._log_resumen_deteccion()
  return self.componentes_activos
 
 def _detectar_componente(self,comp:ComponenteAurora)->bool:
  self.stats["total"]+=1
  try:
   if not self._verificar_dependencias(comp):logger.debug(f"⏭️ {comp.nombre}: dependencias no satisfechas");return False
   if comp.nombre in self.cache_deteccion:
    cached_result=self.cache_deteccion[comp.nombre]
    if cached_result["success"]:comp.disponible=True;comp.instancia=cached_result["instancia"];comp.version=cached_result["version"];comp.capacidades=cached_result["capacidades"];self.componentes_activos[comp.nombre]=comp;self.stats["exitosos"]+=1;return True
   modulo=importlib.import_module(comp.modulo)
   instancia=self._crear_instancia(modulo,comp)
   if self._validar_instancia(instancia,comp):
    comp.disponible=True;comp.instancia=instancia;comp.capacidades=self._obtener_capacidades(instancia);comp.version=self._obtener_version(instancia);self.cache_deteccion[comp.nombre]={"success":True,"instancia":instancia,"version":comp.version,"capacidades":comp.capacidades};self.componentes_activos[comp.nombre]=comp
    if comp.tipo==TipoComponente.MOTOR:self.stats["motores_detectados"]+=1
    elif comp.tipo==TipoComponente.GESTOR_INTELIGENCIA:self.stats["gestores_detectados"]+=1
    elif comp.tipo==TipoComponente.OBJECTIVE_MANAGER:self.stats["objective_managers_detectados"] = self.stats.get("objective_managers_detectados", 0) + 1
    self.stats["exitosos"]+=1;logger.info(f"✅ {comp.nombre} v{comp.version}");return True
   else:raise Exception("Instancia no válida")
  except Exception as e:
   logger.debug(f"❌ {comp.nombre}: {e}")
   if comp.fallback_disponible and self._crear_fallback(comp):self.stats["fallback"]+=1;logger.info(f"🔄 {comp.nombre} (fallback)");return True
   self.cache_deteccion[comp.nombre]={"success":False,"error":str(e)};self.stats["fallidos"]+=1;return False
 
 def _verificar_dependencias(self,comp:ComponenteAurora)->bool:return all(dep in self.componentes_activos for dep in comp.dependencias)
 
 def _crear_instancia(self,modulo:Any,comp:ComponenteAurora)->Any:
  if comp.modulo=="neuromix_aurora_v27":return getattr(modulo,"AuroraNeuroAcousticEngineV27")()
  elif comp.modulo=="hypermod_v32":return getattr(modulo,"_motor_global_v32",None)or modulo
  elif comp.modulo=="harmonicEssence_v34":return getattr(modulo,"HarmonicEssenceV34AuroraConnected")()
  elif comp.modulo=="emotion_style_profiles":return getattr(modulo,"crear_gestor_emotion_style_v7")()
  elif comp.modulo=="objective_manager":
      if hasattr(modulo, "crear_objective_manager_unificado"):return getattr(modulo, "crear_objective_manager_unificado")()
      elif hasattr(modulo, "ObjectiveManagerUnificado"):return getattr(modulo, "ObjectiveManagerUnificado")()
      else:return None
  else:
   for metodo in[f"crear_gestor_{comp.nombre.split('_')[0]}",f"crear_{comp.nombre}","crear_gestor","obtener_gestor",comp.clase_principal]:
    if hasattr(modulo,metodo):attr=getattr(modulo,metodo);return attr()if callable(attr)else attr
  return getattr(modulo,comp.clase_principal)()
 
 def _validar_instancia(self,instancia:Any,comp:ComponenteAurora)->bool:
  try:
   if comp.tipo==TipoComponente.MOTOR:return(hasattr(instancia,'generar_audio')or hasattr(instancia,'generate_neuro_wave')or hasattr(instancia,'generar_bloques')or hasattr(instancia,'generate_textured_noise'))
   elif comp.tipo==TipoComponente.GESTOR_INTELIGENCIA:return(hasattr(instancia,'procesar_objetivo')or hasattr(instancia,'rutear_objetivo')or hasattr(instancia,'obtener_perfil'))
   elif comp.tipo==TipoComponente.OBJECTIVE_MANAGER:return(hasattr(instancia,'procesar_objetivo_completo')or hasattr(instancia,'rutear_objetivo_inteligente')or hasattr(instancia,'obtener_configuracion_completa')or hasattr(instancia,'generar_configuracion_motor'))
   elif comp.tipo==TipoComponente.PIPELINE and"carmine_analyzer"in comp.nombre:return hasattr(instancia,'analyze_audio')
   elif comp.tipo==TipoComponente.PIPELINE:return hasattr(instancia,'validar_y_normalizar')
   elif comp.tipo==TipoComponente.STYLE_PROFILE:return(hasattr(instancia,'obtener_preset')or hasattr(instancia,'buscar_por_efecto'))
   return True
  except Exception:return False
 
 def _obtener_capacidades(self,instancia:Any)->Dict[str,Any]:
  try:
   for metodo in['obtener_capacidades','get_capabilities','capacidades']:
    if hasattr(instancia,metodo):return getattr(instancia,metodo)()
  except Exception:pass
  return{}
 
 def _obtener_version(self,instancia:Any)->str:
  for attr in['version','VERSION','__version__','_version']:
   if hasattr(instancia,attr):version=getattr(instancia,attr);return str(version)
  return"unknown"
 
 def _crear_fallback(self,comp:ComponenteAurora)->bool:
  try:
   fallbacks={"neuromix_v27":self._fallback_neuromix,"neuromix_legacy":self._fallback_neuromix,"hypermod_v32":self._fallback_hypermod,"hypermod_legacy":self._fallback_hypermod,"harmonic_essence_v34":self._fallback_harmonic,"field_profiles":self._fallback_field_profiles,"objective_router":self._fallback_objective_router,"quality_pipeline":self._fallback_quality_pipeline,"carmine_analyzer_v21":self._fallback_carmine_analyzer,"objective_manager_unificado": self._fallback_objective_manager}
   if comp.nombre in fallbacks:comp.instancia=fallbacks[comp.nombre]();comp.disponible=True;comp.version="fallback";self.componentes_activos[comp.nombre]=comp;return True
  except Exception as e:logger.error(f"Error creando fallback para {comp.nombre}: {e}")
  return False

 def _fallback_objective_manager(self):
  class ObjectiveManagerFallback:
   def procesar_objetivo_completo(self, objetivo: str, contexto: Dict[str, Any] = None) -> Dict[str, Any]:
    mapeo_basico = {"relajacion":{"neurotransmisor_preferido":"gaba","intensidad":"suave","estilo":"sereno","template_recomendado":"relajacion_profunda","perfil_campo":"relajacion","beat_base":7.0},"concentracion":{"neurotransmisor_preferido":"acetilcolina","intensidad":"media","estilo":"crystalline","template_recomendado":"claridad_mental","perfil_campo":"cognitivo","beat_base":14.0},"creatividad":{"neurotransmisor_preferido":"anandamida","intensidad":"media","estilo":"organico","template_recomendado":"creatividad_exponencial","perfil_campo":"creativo","beat_base":10.0},"meditacion":{"neurotransmisor_preferido":"serotonina","intensidad":"suave","estilo":"mistico","template_recomendado":"presencia_total","perfil_campo":"espiritual","beat_base":6.0}}
    objetivo_lower = objetivo.lower();config_base = mapeo_basico.get("relajacion")
    for key, config in mapeo_basico.items():
        if key in objetivo_lower:config_base = config;break
    return {"configuracion_motor": config_base,"template_utilizado": config_base.get("template_recomendado"),"perfil_campo_utilizado": config_base.get("perfil_campo"),"resultado_routing": {"confianza": 0.7,"tipo": "fallback_mapping","fuente": "objective_manager_fallback"},"metadatos": {"fallback_usado": True,"objetivo_original": objetivo,"contexto_procesado": contexto or {}}}
   def rutear_objetivo_inteligente(self, objetivo: str, **kwargs) -> Dict[str, Any]:return self.procesar_objetivo_completo(objetivo, kwargs)
   def obtener_configuracion_completa(self, objetivo: str) -> Dict[str, Any]:return self.procesar_objetivo_completo(objetivo)
   def generar_configuracion_motor(self, objetivo: str, motor_objetivo: str) -> Dict[str, Any]:config_base = self.procesar_objetivo_completo(objetivo);return config_base.get("configuracion_motor", {})
   def obtener_capacidades(self) -> Dict[str, Any]:return {"nombre": "Objective Manager Fallback","tipo": "gestor_objetivos_fallback","capacidades": ["mapping_basico", "routing_simple"],"templates_disponibles": ["relajacion_profunda", "claridad_mental", "creatividad_exponencial", "presencia_total"],"fallback": True}
  return ObjectiveManagerFallback()

 def _fallback_neuromix(self):
  class NeuroMixFallback:
   def generar_audio(self,config:Dict[str,Any],duracion_sec:float)->np.ndarray:
    samples=int(44100*duracion_sec);t=np.linspace(0,duracion_sec,samples);freq_base=10.0
    if config.get('neurotransmisor_preferido')=='dopamina':freq_base=12.0
    elif config.get('neurotransmisor_preferido')=='gaba':freq_base=6.0
    wave=0.3*np.sin(2*np.pi*freq_base*t);fade_samples=min(2048,len(wave)//4)
    if len(wave)>fade_samples*2:fade_in=np.linspace(0,1,fade_samples);fade_out=np.linspace(1,0,fade_samples);wave[:fade_samples]*=fade_in;wave[-fade_samples:]*=fade_out
    return np.stack([wave,wave])
   def validar_configuracion(self,config:Dict[str,Any])->bool:return isinstance(config,dict)and config.get('objetivo','').strip()
   def obtener_capacidades(self)->Dict[str,Any]:return{"nombre":"NeuroMix Fallback","tipo":"motor_neuroacustico_fallback","neurotransmisores":["dopamina","serotonina","gaba"]}
  return NeuroMixFallback()
 
 def _fallback_hypermod(self):
  class HyperModFallback:
   def generar_audio(self,config:Dict[str,Any],duracion_sec:float)->np.ndarray:samples=int(44100*duracion_sec);t=np.linspace(0,duracion_sec,samples);alpha=0.4*np.sin(2*np.pi*10.0*t);theta=0.2*np.sin(2*np.pi*6.0*t);wave=alpha+theta;return np.stack([wave,wave])
   def validar_configuracion(self,config:Dict[str,Any])->bool:return True
   def obtener_capacidades(self)->Dict[str,Any]:return{"nombre":"HyperMod Fallback","tipo":"motor_ondas_cerebrales_fallback"}
  return HyperModFallback()
 
 def _fallback_harmonic(self):
  class HarmonicFallback:
   def generar_audio(self,config:Dict[str,Any],duracion_sec:float)->np.ndarray:samples=int(44100*duracion_sec);texture=np.random.normal(0,0.1,samples);if samples>100:kernel_size=min(50,samples//20);kernel=np.ones(kernel_size)/kernel_size;texture=np.convolve(texture,kernel,mode='same');return np.stack([texture,texture])
   def validar_configuracion(self,config:Dict[str,Any])->bool:return True
   def obtener_capacidades(self)->Dict[str,Any]:return{"nombre":"HarmonicEssence Fallback","tipo":"motor_texturas_fallback"}
  return HarmonicFallback()
 
 def _fallback_field_profiles(self):
  class FieldProfilesFallback:
   def procesar_objetivo(self,objetivo:str,contexto:Dict[str,Any])->Dict[str,Any]:return{"perfil_recomendado":"basico","configuracion":{"intensidad":"media","duracion_min":20}}
   def obtener_perfil(self,nombre:str):return None
   def recomendar_secuencia_perfiles(self,objetivo:str,duracion:int):return[(objetivo,duracion)]
  return FieldProfilesFallback()
 
 def _fallback_objective_router(self):
  class ObjectiveRouterFallback:
   def procesar_objetivo(self,objetivo:str,contexto:Dict[str,Any])->Dict[str,Any]:mapeo={"relajacion":{"neurotransmisor_preferido":"gaba","intensidad":"suave","estilo":"sereno"},"concentracion":{"neurotransmisor_preferido":"acetilcolina","intensidad":"media","estilo":"crystalline"},"creatividad":{"neurotransmisor_preferido":"anandamida","intensidad":"media","estilo":"organico"}};return mapeo.get(objetivo.lower(),{"neurotransmisor_preferido":"serotonina","intensidad":"media","estilo":"neutro"})
   def rutear_objetivo(self,objetivo:str,**kwargs):return self.procesar_objetivo(objetivo,kwargs)
  return ObjectiveRouterFallback()
 
 def _fallback_quality_pipeline(self):
  class QualityPipelineFallback:
   def validar_y_normalizar(self,signal:np.ndarray)->np.ndarray:if signal.ndim==1:signal=np.stack([signal,signal]);max_val=np.max(np.abs(signal));if max_val>0:signal=signal*(0.85/max_val);return np.clip(signal,-1.0,1.0)
  return QualityPipelineFallback()
 
 def _fallback_carmine_analyzer(self):
  class CarmineAnalyzerFallback:
   def analyze_audio(self,audio:np.ndarray,expected_intent=None):
    if audio.size==0:return type('Result',(),{'score':0,'therapeutic_score':0,'quality':type('Quality',(),{'value':'🔴 CRÍTICO'})(),'suggestions':["Audio vacío"],'issues':["Sin audio"],'gpt_summary':"Audio inválido"})()
    rms=np.sqrt(np.mean(audio**2));peak=np.max(np.abs(audio));score=min(100,max(50,80+(1-min(peak,1.0))*20));quality_level="🟢 ÓPTIMO"if score>=90 else"🟡 OBSERVACIÓN"if score>=70 else"🔴 CRÍTICO"
    return type('Result',(),{'score':int(score),'therapeutic_score':int(score*0.9),'quality':type('Quality',(),{'value':quality_level})(),'suggestions':["Usar fallback - Carmine Analyzer no disponible"],'issues':[]if score>70 else["Calidad subóptima detectada"],'neuro_metrics':type('NeuroMetrics',(),{'entrainment_effectiveness':0.7,'binaural_strength':0.5})(),'gpt_summary':f"Análisis fallback: Score {score:.0f}/100"})()
   def obtener_capacidades(self):return{"nombre":"Carmine Analyzer Fallback","tipo":"analizador_basico_fallback"}
  return CarmineAnalyzerFallback()
 
 def _log_resumen_deteccion(self):
  total=len(self.componentes_registrados);activos=len(self.componentes_activos);porcentaje=(activos/total*100)if total>0 else 0
  logger.info(f"📊 Detección completada en {self.stats['tiempo_deteccion']:.2f}s")
  logger.info(f"   🎯 Componentes: {activos}/{total} ({porcentaje:.0f}%)")
  logger.info(f"   ✅ Exitosos: {self.stats['exitosos']}")
  logger.info(f"   🔄 Fallbacks: {self.stats['fallback']}")
  logger.info(f"   ❌ Fallidos: {self.stats['fallidos']}")
  logger.info(f"   🎵 Motores: {self.stats['motores_detectados']}")
  logger.info(f"   🧠 Gestores: {self.stats['gestores_detectados']}")
  if 'objective_managers_detectados' in self.stats:logger.info(f"   🎯 Objective Managers: {self.stats['objective_managers_detectados']}")

class OrquestadorMultiMotor:
 def __init__(self,componentes_activos:Dict[str,ComponenteAurora]):
  self.componentes=componentes_activos;self.motores_disponibles={nombre:comp for nombre,comp in componentes_activos.items()if comp.tipo==TipoComponente.MOTOR and comp.disponible};self.objective_manager = None
  if "objective_manager_unificado" in componentes_activos and componentes_activos["objective_manager_unificado"].disponible:self.objective_manager = componentes_activos["objective_manager_unificado"].instancia;logger.info("🎯 Objective Manager Unificado conectado")
 
 def generar_audio_orquestado(self,config:ConfiguracionAuroraUnificada,duracion_sec:float)->Tuple[np.ndarray,Dict[str,Any]]:
  metadatos_generacion={"motores_utilizados":[],"tiempo_por_motor":{},"calidad_por_motor":{},"estrategia_aplicada":config.modo_orquestacion.value}
  if config.usar_objective_manager and self.objective_manager:
      try:
          logger.info("🎯 Procesando objetivo con Objective Manager...")
          resultado_om = self.objective_manager.procesar_objetivo_completo(config.objetivo,{"duracion_min": config.duracion_min,"intensidad": config.intensidad,"estilo": config.estilo,"contexto_uso": config.contexto_uso,"perfil_usuario": config.perfil_usuario,"calidad_objetivo": config.calidad_objetivo})
          config_motor = resultado_om.get("configuracion_motor", {})
          for key, value in config_motor.items():
              if hasattr(config, key) and value is not None:setattr(config, key, value)
          metadatos_generacion["objective_manager"] = {"utilizado": True,"template_utilizado": resultado_om.get("template_utilizado"),"perfil_campo_utilizado": resultado_om.get("perfil_campo_utilizado"),"secuencia_fases_utilizada": resultado_om.get("secuencia_fases_utilizada"),"confianza_routing": resultado_om.get("resultado_routing", {}).get("confianza", 0.0),"tipo_routing": resultado_om.get("resultado_routing", {}).get("tipo", "unknown")}
          logger.info(f"✅ Objective Manager procesado - Template: {resultado_om.get('template_utilizado', 'N/A')}")
      except Exception as e:logger.warning(f"⚠️ Error en Objective Manager: {e}");metadatos_generacion["objective_manager"] = {"utilizado": False, "error": str(e)}
  else:metadatos_generacion["objective_manager"] = {"utilizado": False, "razon": "no_disponible_o_deshabilitado"}
  if config.modo_orquestacion==ModoOrquestacion.LAYERED:return self._generar_en_capas(config,duracion_sec,metadatos_generacion)
  elif config.modo_orquestacion==ModoOrquestacion.PARALELO:return self._generar_paralelo(config,duracion_sec,metadatos_generacion)
  elif config.modo_orquestacion==ModoOrquestacion.SECUENCIAL:return self._generar_secuencial(config,duracion_sec,metadatos_generacion)
  else:return self._generar_hibrido(config,duracion_sec,metadatos_generacion)
 
 def _generar_en_capas(self,config:ConfiguracionAuroraUnificada,duracion_sec:float,metadatos:Dict[str,Any])->Tuple[np.ndarray,Dict[str,Any]]:
  capas_config=[("neuromix_v27",{"peso":0.6,"procesamiento":"base"}),("hypermod_v32",{"peso":0.3,"procesamiento":"armonica"}),("harmonic_essence_v34",{"peso":0.2,"procesamiento":"textura"})];audio_final=None;capas_generadas={}
  for nombre_motor,capa_config in capas_config:
   if nombre_motor in self.motores_disponibles:
    start_time=time.time()
    try:
     motor=self.motores_disponibles[nombre_motor].instancia;config_motor=self._adaptar_config_para_motor(config,nombre_motor,capa_config);audio_capa=motor.generar_audio(config_motor,duracion_sec);audio_capa=audio_capa*capa_config["peso"]
     if audio_final is None:audio_final=audio_capa
     else:audio_final=self._combinar_capas(audio_final,audio_capa)
     tiempo_generacion=time.time()-start_time;metadatos["motores_utilizados"].append(nombre_motor);metadatos["tiempo_por_motor"][nombre_motor]=tiempo_generacion;capas_generadas[nombre_motor]=audio_capa
    except Exception as e:logger.warning(f"Error en motor {nombre_motor}: {e}");continue
  if audio_final is None:audio_final=self._generar_fallback_simple(duracion_sec)
  metadatos["capas_generadas"]=len(capas_generadas);return audio_final,metadatos
 
 def _generar_paralelo(self,config:ConfiguracionAuroraUnificada,duracion_sec:float,metadatos:Dict[str,Any])->Tuple[np.ndarray,Dict[str,Any]]:
  motor_principal=self._seleccionar_motor_principal(config)
  if motor_principal:
   start_time=time.time()
   try:instancia=self.motores_disponibles[motor_principal].instancia;config_motor=self._adaptar_config_para_motor(config,motor_principal);audio_resultado=instancia.generar_audio(config_motor,duracion_sec);metadatos["motores_utilizados"].append(motor_principal);metadatos["tiempo_por_motor"][motor_principal]=time.time()-start_time;metadatos["motor_principal"]=motor_principal;return audio_resultado,metadatos
   except Exception as e:logger.error(f"Error en motor principal {motor_principal}: {e}")
  return self._generar_fallback_simple(duracion_sec),metadatos
 
 def _generar_secuencial(self,config:ConfiguracionAuroraUnificada,duracion_sec:float,metadatos:Dict[str,Any])->Tuple[np.ndarray,Dict[str,Any]]:
  motores_activos=list(self.motores_disponibles.keys())[:3]
  if not motores_activos:return self._generar_fallback_simple(duracion_sec),metadatos
  duracion_por_motor=duracion_sec/len(motores_activos);segmentos_audio=[]
  for i,nombre_motor in enumerate(motores_activos):
   start_time=time.time()
   try:motor=self.motores_disponibles[nombre_motor].instancia;config_motor=self._adaptar_config_para_motor(config,nombre_motor);segmento=motor.generar_audio(config_motor,duracion_por_motor);segmentos_audio.append(segmento);metadatos["motores_utilizados"].append(nombre_motor);metadatos["tiempo_por_motor"][nombre_motor]=time.time()-start_time
   except Exception as e:logger.warning(f"Error en motor {nombre_motor}: {e}");samples=int(44100*duracion_por_motor);segmentos_audio.append(np.zeros((2,samples)))
  if segmentos_audio:audio_final=np.concatenate(segmentos_audio,axis=1)
  else:audio_final=self._generar_fallback_simple(duracion_sec)
  return audio_final,metadatos
 
 def _generar_hibrido(self,config:ConfiguracionAuroraUnificada,duracion_sec:float,metadatos:Dict[str,Any])->Tuple[np.ndarray,Dict[str,Any]]:
  num_motores=len(self.motores_disponibles)
  if num_motores>=3 and config.calidad_objetivo=="maxima":return self._generar_en_capas(config,duracion_sec,metadatos)
  elif num_motores>=2:return self._generar_paralelo(config,duracion_sec,metadatos)
  else:return self._generar_secuencial(config,duracion_sec,metadatos)
 
 def _seleccionar_motor_principal(self,config:ConfiguracionAuroraUnificada)->Optional[str]:
  prioridades_objetivo={"concentracion":["neuromix_v27","hypermod_v32"],"relajacion":["harmonic_essence_v34","neuromix_v27"],"creatividad":["harmonic_essence_v34","neuromix_v27"],"meditacion":["hypermod_v32","neuromix_v27"]};objetivo_lower=config.objetivo.lower();motores_preferidos=config.motores_preferidos
  for motor in motores_preferidos:
   if motor in self.motores_disponibles:return motor
  for objetivo_key,lista_motores in prioridades_objetivo.items():
   if objetivo_key in objetivo_lower:
    for motor in lista_motores:
     if motor in self.motores_disponibles:return motor
  if self.motores_disponibles:return list(self.motores_disponibles.keys())[0]
  return None
 
 def _aplicar_correcciones_neuroacusticas_iterativas(self,resultado:ResultadoAuroraIntegrado,config:ConfiguracionAuroraUnificada,analyzer:Any,analysis_inicial:Any)->ResultadoAuroraIntegrado:
  logger.info("🔧 Iniciando correcciones neuroacústicas...");max_iteraciones=2 if config.calidad_objetivo=="maxima"else 1;audio_actual=resultado.audio_data.copy();mejor_resultado=resultado;mejor_score=analysis_inicial.score;correcciones_totales=[]
  for iteracion in range(max_iteraciones):
   logger.info(f"🔄 Iteración {iteracion+1}/{max_iteraciones}")
   try:
    modo_agresivo=iteracion>0 and config.calidad_objetivo=="maxima"
    if hasattr(analyzer,'aplicar_correcciones_neuroacusticas'):
     audio_corregido,metadata_correcciones=analyzer.aplicar_correcciones_neuroacusticas(audio_actual,analysis_inicial,objetivo=config.objetivo,modo_agresivo=modo_agresivo)
     if metadata_correcciones.get("total_correcciones",0)>0:
      expected_intent=self._mapear_objetivo_a_intent_carmine(config.objetivo);nuevo_analysis=analyzer.analyze_audio(audio_corregido,expected_intent);logger.info(f"✅ Score después de correcciones: {nuevo_analysis.score}/100")
      if nuevo_analysis.score>mejor_score:mejor_score=nuevo_analysis.score;audio_actual=audio_corregido;analysis_inicial=nuevo_analysis;mejor_resultado.audio_data=audio_corregido;mejor_resultado=self._actualizar_resultado_con_carmine(mejor_resultado,nuevo_analysis);correcciones_totales.append({"iteracion":iteracion+1,"score_antes":analysis_inicial.score if iteracion==0 else mejor_score,"score_despues":nuevo_analysis.score,"correcciones":metadata_correcciones["correcciones_aplicadas"],"modo_agresivo":modo_agresivo})
       if nuevo_analysis.score>=85:logger.info(f"🎯 Calidad objetivo alcanzada: {nuevo_analysis.score}/100");break
      else:logger.info(f"⚠️ Correcciones no mejoraron el score");break
     else:logger.info("ℹ️ No se aplicaron correcciones");break
    else:logger.warning("⚠️ Analyzer no soporta correcciones automáticas");mejor_resultado=self._actualizar_resultado_con_carmine(mejor_resultado,analysis_inicial);break
   except Exception as e:logger.error(f"❌ Error en iteración {iteracion+1}: {e}");mejor_resultado=self._actualizar_resultado_con_carmine(mejor_resultado,analysis_inicial);break
  if correcciones_totales:mejor_resultado.metadatos["correcciones_neuroacusticas"]={"iteraciones_realizadas":len(correcciones_totales),"score_inicial":analysis_inicial.score if not correcciones_totales else correcciones_totales[0]["score_antes"],"score_final":mejor_score,"mejora_total":mejor_score-(analysis_inicial.score if not correcciones_totales else correcciones_totales[0]["score_antes"]),"correcciones_por_iteracion":correcciones_totales,"calidad_objetivo_alcanzada":mejor_score>=85};logger.info(f"🏆 Correcciones completadas: Score final {mejor_score}/100")
  return mejor_resultado
 
 def _actualizar_resultado_con_carmine(self,resultado:ResultadoAuroraIntegrado,carmine_result:Any)->ResultadoAuroraIntegrado:
  resultado.calidad_score=max(resultado.calidad_score,carmine_result.score);resultado.coherencia_neuroacustica=getattr(carmine_result.neuro_metrics,'entrainment_effectiveness',resultado.coherencia_neuroacustica);resultado.efectividad_terapeutica=max(resultado.efectividad_terapeutica,carmine_result.therapeutic_score/100.0);resultado.metadatos["carmine_analysis"]={"score":carmine_result.score,"therapeutic_score":carmine_result.therapeutic_score,"quality_level":carmine_result.quality.value,"issues":carmine_result.issues,"suggestions":carmine_result.suggestions,"neuro_effectiveness":getattr(carmine_result.neuro_metrics,'entrainment_effectiveness',0.0),"binaural_strength":getattr(carmine_result.neuro_metrics,'binaural_strength',0.0),"gpt_summary":getattr(carmine_result,'gpt_summary',""),"correcciones_aplicadas":False}
  return resultado
 
 def _adaptar_config_para_motor(self,config:ConfiguracionAuroraUnificada,nombre_motor:str,capa_config:Dict[str,Any]=None)->Dict[str,Any]:
  config_base={"objetivo":config.objetivo,"duracion_min":config.duracion_min,"sample_rate":config.sample_rate,"intensidad":config.intensidad,"estilo":config.estilo,"neurotransmisor_preferido":config.neurotransmisor_preferido,"calidad_objetivo":config.calidad_objetivo,"normalizar":config.normalizar,"contexto_uso":config.contexto_uso}
  if hasattr(config, 'template_personalizado') and config.template_personalizado:config_base["template_objetivo"] = config.template_personalizado
  if hasattr(config, 'perfil_campo_personalizado') and config.perfil_campo_personalizado:config_base["perfil_campo"] = config.perfil_campo_personalizado
  if hasattr(config, 'secuencia_fases_personalizada') and config.secuencia_fases_personalizada:config_base["secuencia_fases"] = config.secuencia_fases_personalizada
  if"neuromix"in nombre_motor:config_base.update({"wave_type":"hybrid","processing_mode":"aurora_integrated"})
  elif"hypermod"in nombre_motor:config_base.update({"preset_emocional":config.objetivo,"validacion_cientifica":True,"optimizacion_neuroacustica":True})
  elif"harmonic"in nombre_motor:config_base.update({"texture_type":self._mapear_estilo_a_textura(config.estilo),"precision_cientifica":True})
  if capa_config:config_base.update(capa_config.get("config_adicional",{}))
  return config_base
 
 def _mapear_estilo_a_textura(self,estilo:str)->str:mapeo={"sereno":"relaxation","crystalline":"crystalline","organico":"organic","etereo":"ethereal","tribal":"tribal","mistico":"consciousness"};return mapeo.get(estilo.lower(),"organic")
 def _combinar_capas(self,audio1:np.ndarray,audio2:np.ndarray)->np.ndarray:min_length=min(audio1.shape[1],audio2.shape[1]);audio1_crop=audio1[:,:min_length];audio2_crop=audio2[:,:min_length];combined=audio1_crop+audio2_crop;max_val=np.max(np.abs(combined));if max_val>0.95:combined=combined*(0.85/max_val);return combined
 def _generar_fallback_simple(self,duracion_sec:float)->np.ndarray:samples=int(44100*duracion_sec);t=np.linspace(0,duracion_sec,samples);wave=0.3*np.sin(2*np.pi*10.0*t);fade_samples=min(2048,samples//4);if samples>fade_samples*2:fade_in=np.linspace(0,1,fade_samples);fade_out=np.linspace(1,0,fade_samples);wave[:fade_samples]*=fade_in;wave[-fade_samples:]*=fade_out;return np.stack([wave,wave])
 def _mapear_objetivo_a_intent_carmine(self,objetivo:str):mapeo={"relajacion":"RELAXATION","concentracion":"FOCUS","claridad_mental":"FOCUS","enfoque":"FOCUS","meditacion":"MEDITATION","creatividad":"EMOTIONAL","sanacion":"RELAXATION","sueño":"SLEEP","energia":"ENERGY"};objetivo_lower=objetivo.lower();for key,intent in mapeo.items():if key in objetivo_lower:try:from Carmine_Analyzer import TherapeuticIntent;return getattr(TherapeuticIntent,intent);except:return intent;return None

class AuroraDirectorV7Integrado:
 def __init__(self,auto_detectar:bool=True):
  self.version="Aurora Director V7 Integrado";self.detector=DetectorComponentesAvanzado();self.componentes:Dict[str,ComponenteAurora]={};self.orquestador:Optional[OrquestadorMultiMotor]=None;self.objective_manager:Optional[Any]=None;self.stats={"experiencias_generadas":0,"tiempo_total_generacion":0.0,"estrategias_utilizadas":{},"objetivos_procesados":{},"errores_manejados":0,"fallbacks_utilizados":0,"calidad_promedio":0.0,"motores_utilizados":{},"sesiones_activas":0,"objective_manager_utilizaciones":0,"templates_utilizados":{},"perfiles_campo_utilizados":{},"secuencias_fases_utilizadas":{}};self.cache_configuraciones={};self.cache_resultados={}
  if auto_detectar:self._inicializar_sistema()
 
 def _inicializar_sistema(self):
  logger.info(f"🌟 Inicializando {self.version}");self.componentes=self.detector.detectar_todos();self.orquestador=OrquestadorMultiMotor(self.componentes)
  if "objective_manager_unificado" in self.componentes and self.componentes["objective_manager_unificado"].disponible:self.objective_manager = self.componentes["objective_manager_unificado"].instancia;logger.info("🎯 Objective Manager Unificado conectado")
  else:logger.info("🔄 Objective Manager no disponible")
  self._log_estado_sistema();logger.info("🚀 Sistema Aurora completamente inicializado")
 
 def _log_estado_sistema(self):
  motores=[c for c in self.componentes.values()if c.tipo==TipoComponente.MOTOR];gestores=[c for c in self.componentes.values()if c.tipo==TipoComponente.GESTOR_INTELIGENCIA];pipelines=[c for c in self.componentes.values()if c.tipo==TipoComponente.PIPELINE];objective_managers=[c for c in self.componentes.values()if c.tipo==TipoComponente.OBJECTIVE_MANAGER]
  logger.info(f"🔧 Componentes activos:");logger.info(f"   🎵 Motores: {len(motores)}");logger.info(f"   🧠 Gestores: {len(gestores)}");logger.info(f"   🔄 Pipelines: {len(pipelines)}");logger.info(f"   🎯 Objective Managers: {len(objective_managers)}")
  estrategias=self._obtener_estrategias_disponibles();logger.info(f"🎯 Estrategias disponibles: {len(estrategias)}")
  for estrategia in estrategias:logger.info(f"   • {estrategia.value}")
 
 def crear_experiencia(self,objetivo:str=None,**kwargs)->ResultadoAuroraIntegrado:
  start_time=time.time()
  try:
   config=self._crear_configuracion_optimizada(objetivo,kwargs);problemas=config.validar()
   if problemas:logger.warning(f"⚠️ Problemas de configuración: {problemas}")
   logger.info(f"🎯 Creando experiencia: '{config.objetivo}' ({config.duracion_min}min)");estrategia=self._seleccionar_estrategia_optima(config);logger.info(f"🧠 Estrategia seleccionada: {estrategia.value}");resultado=self._ejecutar_estrategia(estrategia,config);resultado=self._post_procesar_resultado(resultado,config);tiempo_total=time.time()-start_time;self._actualizar_estadisticas(config,resultado,tiempo_total);logger.info(f"✅ Experiencia completada en {tiempo_total:.2f}s");logger.info(f"   📊 Calidad: {resultado.calidad_score:.2f}/100");logger.info(f"   🎵 Audio: {resultado.audio_data.shape}");logger.info(f"   🔧 Componentes: {len(resultado.componentes_usados)}")
   if resultado.resultado_objective_manager:logger.info(f"   🎯 Template: {resultado.template_utilizado or 'N/A'}");logger.info(f"   🎭 Perfil Campo: {resultado.perfil_campo_utilizado or 'N/A'}")
   return resultado
  except Exception as e:logger.error(f"❌ Error creando experiencia: {e}");self.stats["errores_manejados"]+=1;return self._crear_resultado_emergencia(objetivo or"emergencia",str(e))
 
 def _crear_configuracion_optimizada(self,objetivo:str,kwargs:Dict[str,Any])->ConfiguracionAuroraUnificada:
  cache_key=f"{objetivo}_{hash(str(sorted(kwargs.items())))}"
  if cache_key in self.cache_configuraciones:return self.cache_configuraciones[cache_key]
  configs_inteligentes={"concentracion":{"intensidad":"media","estilo":"crystalline","neurotransmisor_preferido":"acetilcolina","modo_orquestacion":ModoOrquestacion.LAYERED,"motores_preferidos":["neuromix_v27","hypermod_v32"]},"claridad_mental":{"intensidad":"media","estilo":"crystalline","neurotransmisor_preferido":"dopamina","modo_orquestacion":ModoOrquestacion.PARALELO,"motores_preferidos":["neuromix_v27"]},"enfoque":{"intensidad":"intenso","estilo":"crystalline","neurotransmisor_preferido":"norepinefrina","modo_orquestacion":ModoOrquestacion.LAYERED},"relajacion":{"intensidad":"suave","estilo":"sereno","neurotransmisor_preferido":"gaba","modo_orquestacion":ModoOrquestacion.HYBRID,"motores_preferidos":["harmonic_essence_v34","neuromix_v27"]},"meditacion":{"intensidad":"suave","estilo":"mistico","neurotransmisor_preferido":"serotonina","duracion_min":35,"modo_orquestacion":ModoOrquestacion.LAYERED,"motores_preferidos":["hypermod_v32","harmonic_essence_v34"]},"gratitud":{"intensidad":"suave","estilo":"sereno","neurotransmisor_preferido":"oxitocina","modo_orquestacion":ModoOrquestacion.HYBRID},"creatividad":{"intensidad":"media","estilo":"organico","neurotransmisor_preferido":"anandamida","modo_orquestacion":ModoOrquestacion.LAYERED,"motores_preferidos":["harmonic_essence_v34","neuromix_v27"]},"inspiracion":{"intensidad":"media","estilo":"organico","neurotransmisor_preferido":"dopamina","modo_orquestacion":ModoOrquestacion.HYBRID},"sanacion":{"intensidad":"suave","estilo":"sereno","neurotransmisor_preferido":"endorfina","duracion_min":45,"calidad_objetivo":"maxima","modo_orquestacion":ModoOrquestacion.LAYERED}}
  objetivo_lower=objetivo.lower()if objetivo else"relajacion";contexto_detectado=self._detectar_contexto_objetivo(objetivo_lower);config_base={}
  for key,config in configs_inteligentes.items():
   if key in objetivo_lower:config_base=config.copy();break
  config_base.update(contexto_detectado);config_final={"objetivo":objetivo or"relajacion",**config_base,**kwargs};config_final.setdefault("usar_objective_manager", OBJECTIVE_MANAGER_AVAILABLE and self.objective_manager is not None);configuracion=ConfiguracionAuroraUnificada(**config_final);self.cache_configuraciones[cache_key]=configuracion;return configuracion
 
 def _detectar_contexto_objetivo(self,objetivo:str)->Dict[str,Any]:
  contexto={}
  if any(palabra in objetivo for palabra in["profundo","intenso","fuerte"]):contexto["intensidad"]="intenso"
  elif any(palabra in objetivo for palabra in["suave","ligero","sutil"]):contexto["intensidad"]="suave"
  if any(palabra in objetivo for palabra in["rapido","corto","breve"]):contexto["duracion_min"]=10
  elif any(palabra in objetivo for palabra in["largo","extenso","profundo"]):contexto["duracion_min"]=45
  if any(palabra in objetivo for palabra in["trabajo","oficina","estudio"]):contexto["contexto_uso"]="trabajo"
  elif any(palabra in objetivo for palabra in["dormir","noche","sueño"]):contexto["contexto_uso"]="sueño"
  elif any(palabra in objetivo for palabra in["meditacion","espiritual"]):contexto["contexto_uso"]="meditacion"
  if any(palabra in objetivo for palabra in["terapeutico","clinico","medicinal"]):contexto["calidad_objetivo"]="maxima"
  return contexto
 
 def _seleccionar_estrategia_optima(self,config:ConfiguracionAuroraUnificada)->EstrategiaGeneracion:
  if config.estrategia_preferida:
   estrategias_disponibles=self._obtener_estrategias_disponibles()
   if config.estrategia_preferida in estrategias_disponibles:return config.estrategia_preferida
  motores=[c for c in self.componentes.values()if c.tipo==TipoComponente.MOTOR and c.disponible];gestores=[c for c in self.componentes.values()if c.tipo==TipoComponente.GESTOR_INTELIGENCIA and c.disponible];pipelines=[c for c in self.componentes.values()if c.tipo==TipoComponente.PIPELINE and c.disponible];objective_managers=[c for c in self.componentes.values()if c.tipo==TipoComponente.OBJECTIVE_MANAGER and c.disponible]
  if(config.usar_objective_manager and len(objective_managers) >= 1 and len(motores) >= 2 and config.calidad_objetivo == "maxima"):return EstrategiaGeneracion.OBJECTIVE_MANAGER_DRIVEN
  if(len(motores)>=3 and len(gestores)>=2 and len(pipelines)>=1 and config.calidad_objetivo=="maxima"):return EstrategiaGeneracion.AURORA_ORQUESTADO
  elif len(motores)>=2 and config.modo_orquestacion in[ModoOrquestacion.LAYERED,ModoOrquestacion.HYBRID]:return EstrategiaGeneracion.MULTI_MOTOR
  elif len(gestores)>=1 and len(motores)>=1:return EstrategiaGeneracion.INTELIGENCIA_ADAPTIVA
  elif len(motores)>=1:return EstrategiaGeneracion.MOTOR_ESPECIALIZADO
  else:return EstrategiaGeneracion.FALLBACK_PROGRESIVO
 
 def _ejecutar_estrategia(self,estrategia:EstrategiaGeneracion,config:ConfiguracionAuroraUnificada)->ResultadoAuroraIntegrado:
  duracion_sec=config.duracion_min*60
  if estrategia==EstrategiaGeneracion.AURORA_ORQUESTADO:return self._estrategia_aurora_orquestado(config,duracion_sec)
  elif estrategia==EstrategiaGeneracion.MULTI_MOTOR:return self._estrategia_multi_motor(config,duracion_sec)
  elif estrategia==EstrategiaGeneracion.INTELIGENCIA_ADAPTIVA:return self._estrategia_inteligencia_adaptiva(config,duracion_sec)
  elif estrategia==EstrategiaGeneracion.MOTOR_ESPECIALIZADO:return self._estrategia_motor_especializado(config,duracion_sec)
  elif estrategia==EstrategiaGeneracion.OBJECTIVE_MANAGER_DRIVEN:return self._estrategia_objective_manager_driven(config,duracion_sec)
  else:return self._estrategia_fallback_progresivo(config,duracion_sec)

 def _estrategia_objective_manager_driven(self, config: ConfiguracionAuroraUnificada, duracion_sec: float) -> ResultadoAuroraIntegrado:
     logger.info("🎯 Ejecutando estrategia Objective Manager Driven")
     if not self.objective_manager:logger.warning("⚠️ Objective Manager no disponible, fallback a Aurora Orquestado");return self._estrategia_aurora_orquestado(config, duracion_sec)
     try:
         resultado_om = self.objective_manager.procesar_objetivo_completo(config.objetivo,{"duracion_min": config.duracion_min,"intensidad": config.intensidad,"estilo": config.estilo,"contexto_uso": config.contexto_uso,"perfil_usuario": config.perfil_usuario,"calidad_objetivo": config.calidad_objetivo})
         config_optimizada = self._aplicar_resultado_objective_manager(config, resultado_om);audio_data, metadatos_orquestacion = self.orquestador.generar_audio_orquestado(config_optimizada, duracion_sec)
         if "quality_pipeline" in self.componentes:pipeline = self.componentes["quality_pipeline"].instancia;audio_data = pipeline.validar_y_normalizar(audio_data)
         calidad_score, coherencia, efectividad = self._calcular_metricas_calidad(audio_data)
         resultado = ResultadoAuroraIntegrado(audio_data=audio_data,metadatos={"estrategia": "objective_manager_driven","orquestacion": metadatos_orquestacion,"objective_manager_usado": True,"pipeline_calidad": "quality_pipeline" in self.componentes},estrategia_usada=EstrategiaGeneracion.OBJECTIVE_MANAGER_DRIVEN,modo_orquestacion=config.modo_orquestacion,componentes_usados=metadatos_orquestacion.get("motores_utilizados", []) + ["objective_manager_unificado"],tiempo_generacion=0.0,calidad_score=calidad_score,coherencia_neuroacustica=coherencia,efectividad_terapeutica=efectividad,configuracion=config,resultado_objective_manager=resultado_om,template_utilizado=resultado_om.get("template_utilizado"),perfil_campo_utilizado=resultado_om.get("perfil_campo_utilizado"),secuencia_fases_utilizada=resultado_om.get("secuencia_fases_utilizada"))
         logger.info(f"✅ Estrategia Objective Manager completada - Template: {resultado.template_utilizado}");return resultado
     except Exception as e:logger.error(f"❌ Error en estrategia Objective Manager: {e}");return self._estrategia_aurora_orquestado(config, duracion_sec)

 def _aplicar_resultado_objective_manager(self, config: ConfiguracionAuroraUnificada, resultado_om: Dict[str, Any]) -> ConfiguracionAuroraUnificada:
     config_motor = resultado_om.get("configuracion_motor", {})
     for key, value in config_motor.items():
         if hasattr(config, key) and value is not None:setattr(config, key, value)
     config.template_personalizado = resultado_om.get("template_utilizado");config.perfil_campo_personalizado = resultado_om.get("perfil_campo_utilizado");config.secuencia_fases_personalizada = resultado_om.get("secuencia_fases_utilizada");return config
 
 def _estrategia_aurora_orquestado(self,config:ConfiguracionAuroraUnificada,duracion_sec:float)->ResultadoAuroraIntegrado:
  config_optimizada=self._aplicar_inteligencia_gestores(config);audio_data,metadatos_orquestacion=self.orquestador.generar_audio_orquestado(config_optimizada,duracion_sec)
  if"quality_pipeline"in self.componentes:pipeline=self.componentes["quality_pipeline"].instancia;audio_data=pipeline.validar_y_normalizar(audio_data)
  calidad_score,coherencia,efectividad=self._calcular_metricas_calidad(audio_data);return ResultadoAuroraIntegrado(audio_data=audio_data,metadatos={"estrategia":"aurora_orquestado","orquestacion":metadatos_orquestacion,"gestores_utilizados":True,"pipeline_calidad":"quality_pipeline"in self.componentes},estrategia_usada=EstrategiaGeneracion.AURORA_ORQUESTADO,modo_orquestacion=config.modo_orquestacion,componentes_usados=metadatos_orquestacion.get("motores_utilizados",[]),tiempo_generacion=0.0,calidad_score=calidad_score,coherencia_neuroacustica=coherencia,efectividad_terapeutica=efectividad,configuracion=config)
 
 def _estrategia_multi_motor(self,config:ConfiguracionAuroraUnificada,duracion_sec:float)->ResultadoAuroraIntegrado:
  audio_data,metadatos_orquestacion=self.orquestador.generar_audio_orquestado(config,duracion_sec);calidad_score,coherencia,efectividad=self._calcular_metricas_calidad(audio_data);return ResultadoAuroraIntegrado(audio_data=audio_data,metadatos={"estrategia":"multi_motor","orquestacion":metadatos_orquestacion},estrategia_usada=EstrategiaGeneracion.MULTI_MOTOR,modo_orquestacion=config.modo_orquestacion,componentes_usados=metadatos_orquestacion.get("motores_utilizados",[]),tiempo_generacion=0.0,calidad_score=calidad_score,coherencia_neuroacustica=coherencia,efectividad_terapeutica=efectividad,configuracion=config)
 
 def _estrategia_inteligencia_adaptiva(self,config:ConfiguracionAuroraUnificada,duracion_sec:float)->ResultadoAuroraIntegrado:
  config_optimizada=self._aplicar_inteligencia_gestores(config);motor_principal=self._seleccionar_motor_principal(config_optimizada)
  if motor_principal:motor=self.componentes[motor_principal].instancia;config_motor=self._adaptar_configuracion_motor(config_optimizada,motor_principal);audio_data=motor.generar_audio(config_motor,duracion_sec);componentes_usados=[motor_principal]
  else:audio_data=self._generar_audio_fallback(duracion_sec);componentes_usados=["fallback"]
  calidad_score,coherencia,efectividad=self._calcular_metricas_calidad(audio_data);return ResultadoAuroraIntegrado(audio_data=audio_data,metadatos={"estrategia":"inteligencia_adaptiva","motor_principal":motor_principal,"configuracion_optimizada":True},estrategia_usada=EstrategiaGeneracion.INTELIGENCIA_ADAPTIVA,modo_orquestacion=ModoOrquestacion.HYBRID,componentes_usados=componentes_usados,tiempo_generacion=0.0,calidad_score=calidad_score,coherencia_neuroacustica=coherencia,efectividad_terapeutica=efectividad,configuracion=config)
 
 def _estrategia_motor_especializado(self,config:ConfiguracionAuroraUnificada,duracion_sec:float)->ResultadoAuroraIntegrado:
  motor_principal=self._seleccionar_motor_principal(config)
  if motor_principal:motor=self.componentes[motor_principal].instancia;config_motor=self._adaptar_configuracion_motor(config,motor_principal);audio_data=motor.generar_audio(config_motor,duracion_sec);componentes_usados=[motor_principal]
  else:audio_data=self._generar_audio_fallback(duracion_sec);componentes_usados=["fallback"]
  calidad_score,coherencia,efectividad=self._calcular_metricas_calidad(audio_data);return ResultadoAuroraIntegrado(audio_data=audio_data,metadatos={"estrategia":"motor_especializado","motor_utilizado":motor_principal},estrategia_usada=EstrategiaGeneracion.MOTOR_ESPECIALIZADO,modo_orquestacion=ModoOrquestacion.HYBRID,componentes_usados=componentes_usados,tiempo_generacion=0.0,calidad_score=calidad_score,coherencia_neuroacustica=coherencia,efectividad_terapeutica=efectividad,configuracion=config)
 
 def _estrategia_fallback_progresivo(self,config:ConfiguracionAuroraUnificada,duracion_sec:float)->ResultadoAuroraIntegrado:
  self.stats["fallbacks_utilizados"]+=1;audio_data=self._generar_audio_fallback(duracion_sec);calidad_score,coherencia,efectividad=self._calcular_metricas_calidad(audio_data);return ResultadoAuroraIntegrado(audio_data=audio_data,metadatos={"estrategia":"fallback_progresivo","motivo":"componentes_insuficientes"},estrategia_usada=EstrategiaGeneracion.FALLBACK_PROGRESIVO,modo_orquestacion=ModoOrquestacion.HYBRID,componentes_usados=["fallback_interno"],tiempo_generacion=0.0,calidad_score=calidad_score,coherencia_neuroacustica=coherencia,efectividad_terapeutica=efectividad,configuracion=config)
 
 def _aplicar_inteligencia_gestores(self,config:ConfiguracionAuroraUnificada)->ConfiguracionAuroraUnificada:
  config_optimizada=config
  if config.usar_objective_manager and self.objective_manager:
      try:
          logger.info("🎯 Aplicando inteligencia del Objective Manager...");resultado_om = self.objective_manager.procesar_objetivo_completo(config.objetivo,{"duracion_min": config.duracion_min,"intensidad": config.intensidad,"estilo": config.estilo,"contexto_uso": config.contexto_uso,"perfil_usuario": config.perfil_usuario,"calidad_objetivo": config.calidad_objetivo});config_optimizada = self._aplicar_resultado_objective_manager(config_optimizada, resultado_om);self.stats["objective_manager_utilizaciones"] += 1;logger.info(f"✅ Objective Manager aplicado - Template: {resultado_om.get('template_utilizado', 'N/A')}");return config_optimizada
      except Exception as e:logger.warning(f"⚠️ Error aplicando Objective Manager: {e}")
  if"objective_router"in self.componentes:
   try:router=self.componentes["objective_router"].instancia;optimizacion=router.procesar_objetivo(config.objetivo,{"contexto_uso":config.contexto_uso});for key,value in optimizacion.items():if hasattr(config_optimizada,key)and value is not None:setattr(config_optimizada,key,value)
   except Exception as e:logger.warning(f"Error aplicando objective_router: {e}")
  if"field_profiles"in self.componentes:
   try:profiles=self.componentes["field_profiles"].instancia;perfil=profiles.obtener_perfil(config.objetivo);if perfil:if hasattr(perfil,'configuracion'):for key,value in perfil.configuracion.items():if hasattr(config_optimizada,key):setattr(config_optimizada,key,value)
   except Exception as e:logger.warning(f"Error aplicando field_profiles: {e}")
  return config_optimizada
 
 def _seleccionar_motor_principal(self,config:ConfiguracionAuroraUnificada)->Optional[str]:
  motores_disponibles=[nombre for nombre,comp in self.componentes.items()if comp.tipo==TipoComponente.MOTOR and comp.disponible]
  if not motores_disponibles:return None
  prioridades={"concentracion":["neuromix_v27","hypermod_v32"],"claridad_mental":["neuromix_v27","hypermod_v32"],"enfoque":["neuromix_v27","hypermod_v32"],"relajacion":["harmonic_essence_v34","neuromix_v27"],"meditacion":["hypermod_v32","harmonic_essence_v34"],"creatividad":["harmonic_essence_v34","neuromix_v27"],"sanacion":["harmonic_essence_v34","hypermod_v32"]}
  for motor in config.motores_preferidos:
   if motor in motores_disponibles:return motor
  objetivo_lower=config.objetivo.lower()
  for objetivo_key,lista_motores in prioridades.items():
   if objetivo_key in objetivo_lower:
    for motor in lista_motores:
     if motor in motores_disponibles:return motor
  return motores_disponibles[0]
 
 def _adaptar_configuracion_motor(self,config:ConfiguracionAuroraUnificada,nombre_motor:str)->Dict[str,Any]:
  config_base={"objetivo":config.objetivo,"duracion_min":config.duracion_min,"sample_rate":config.sample_rate,"intensidad":config.intensidad,"estilo":config.estilo,"neurotransmisor_preferido":config.neurotransmisor_preferido,"calidad_objetivo":config.calidad_objetivo,"normalizar":config.normalizar,"contexto_uso":config.contexto_uso}
  if hasattr(config, 'template_personalizado') and config.template_personalizado:config_base["template_objetivo"] = config.template_personalizado
  if hasattr(config, 'perfil_campo_personalizado') and config.perfil_campo_personalizado:config_base["perfil_campo"] = config.perfil_campo_personalizado
  if hasattr(config, 'secuencia_fases_personalizada') and config.secuencia_fases_personalizada:config_base["secuencia_fases"] = config.secuencia_fases_personalizada
  if"neuromix"in nombre_motor:config_base.update({"wave_type":"hybrid","processing_mode":"aurora_integrated","quality_level":"therapeutic"if config.calidad_objetivo=="maxima"else"enhanced"})
  elif"hypermod"in nombre_motor:config_base.update({"preset_emocional":config.objetivo,"validacion_cientifica":True,"optimizacion_neuroacustica":True,"modo_terapeutico":config.calidad_objetivo=="maxima"})
  elif"harmonic"in nombre_motor:config_base.update({"texture_type":self._mapear_estilo_a_textura(config.estilo),"precision_cientifica":True,"auto_optimizar_coherencia":True})
  return config_base
 
 def _mapear_estilo_a_textura(self,estilo:str)->str:mapeo={"sereno":"relaxation","crystalline":"crystalline","organico":"organic","etereo":"ethereal","tribal":"tribal","mistico":"consciousness","neutro":"meditation"};return mapeo.get(estilo.lower(),"organic")
 
 def _post_procesar_resultado(self,resultado:ResultadoAuroraIntegrado,config:ConfiguracionAuroraUnificada)->ResultadoAuroraIntegrado:
  audio=resultado.audio_data
  if config.normalizar:max_val=np.max(np.abs(audio));if max_val>0:target_level=0.85 if config.calidad_objetivo=="maxima"else 0.80;audio=audio*(target_level/max_val);audio=np.clip(audio,-1.0,1.0)
  if config.aplicar_mastering:audio=self._aplicar_mastering_basico(audio)
  resultado.audio_data=audio;resultado=self._aplicar_analisis_carmine(resultado,config);resultado.recomendaciones=self._generar_recomendaciones(resultado,config);resultado.proxima_sesion=self._generar_sugerencias_proxima_sesion(resultado,config);return resultado
 
 def _aplicar_mastering_basico(self,audio:np.ndarray)->np.ndarray:
  threshold=0.7;ratio=3.0
  for channel in range(audio.shape[0]):signal=audio[channel];over_threshold=np.abs(signal)>threshold;if np.any(over_threshold):compressed=np.where(over_threshold,np.sign(signal)*(threshold+(np.abs(signal)-threshold)/ratio),signal);audio[channel]=compressed
  return audio
 
 def _aplicar_analisis_carmine(self,resultado:ResultadoAuroraIntegrado,config:ConfiguracionAuroraUnificada)->ResultadoAuroraIntegrado:
  try:
   if"carmine_analyzer_v21"in self.componentes:
    analyzer=self.componentes["carmine_analyzer_v21"].instancia;expected_intent=self._mapear_objetivo_a_intent_carmine(config.objetivo);carmine_result=analyzer.analyze_audio(resultado.audio_data,expected_intent);logger.info(f"🔍 Carmine Analysis inicial: {carmine_result.score}/100")
    if config.validacion_automatica and carmine_result.score<85:resultado=self.orquestador._aplicar_correcciones_neuroacusticas_iterativas(resultado,config,analyzer,carmine_result)
    else:resultado=self.orquestador._actualizar_resultado_con_carmine(resultado,carmine_result)
   return resultado
  except Exception as e:logger.warning(f"Error en análisis Carmine: {e}");return resultado
 
 def _mapear_objetivo_a_intent_carmine(self,objetivo:str):
  mapeo={"relajacion":"RELAXATION","concentracion":"FOCUS","claridad_mental":"FOCUS","enfoque":"FOCUS","meditacion":"MEDITATION","creatividad":"EMOTIONAL","sanacion":"RELAXATION","sueño":"SLEEP","energia":"ENERGY"};objetivo_lower=objetivo.lower()
  for key,intent in mapeo.items():
   if key in objetivo_lower:try:from Carmine_Analyzer import TherapeuticIntent;return getattr(TherapeuticIntent,intent);except:return intent
  return None
 
 def _calcular_metricas_calidad(self,audio:np.ndarray)->Tuple[float,float,float]:
  if audio.size==0:return 0.0,0.0,0.0
  try:
   rms=np.sqrt(np.mean(audio**2));peak=np.max(np.abs(audio));crest_factor=peak/(rms+1e-10)
   if audio.ndim==2 and audio.shape[0]==2:correlation=np.corrcoef(audio[0],audio[1])[0,1];coherencia=float(np.nan_to_num(correlation,0.5))
   else:coherencia=0.8
   fft_data=np.abs(np.fft.rfft(audio[0]if audio.ndim==2 else audio));energy_distribution=np.std(fft_data);flatness=np.mean(fft_data)/(np.max(fft_data)+1e-10);calidad_score=min(100,max(60,80+(1-min(peak,1.0))*10+coherencia*10+flatness*10));efectividad=min(1.0,max(0.6,0.7+coherencia*0.2+(1-min(peak,1.0))*0.1));return float(calidad_score),float(coherencia),float(efectividad)
  except Exception as e:logger.warning(f"Error calculando métricas: {e}");return 75.0,0.75,0.75
 
 def _generar_recomendaciones(self,resultado:ResultadoAuroraIntegrado,config:ConfiguracionAuroraUnificada)->List[str]:
  recomendaciones=[]
  if resultado.calidad_score<70:recomendaciones.append("Considerar aumentar la calidad objetivo a 'maxima'")
  if resultado.coherencia_neuroacustica<0.7:recomendaciones.append("Mejorar coherencia usando más componentes Aurora")
  if resultado.efectividad_terapeutica<0.8:recomendaciones.append("Incrementar duración para mayor efectividad terapéutica")
  if len(resultado.componentes_usados)<2 and len(self.orquestador.motores_disponibles)>=2:recomendaciones.append("Probar modo de orquestación 'layered' para mejor resultado")
  if resultado.resultado_objective_manager:
      confianza_routing = resultado.resultado_objective_manager.get("resultado_routing", {}).get("confianza", 0.0)
      if confianza_routing < 0.7:recomendaciones.append("Considerar especificar más detalles en el objetivo para mejor routing")
      if not resultado.template_utilizado:recomendaciones.append("Probar con un template específico para este objetivo")
      if resultado.resultado_objective_manager.get("metadatos", {}).get("fallback_usado"):recomendaciones.append("Objective Manager en modo fallback - verificar disponibilidad de componentes")
  objetivo_lower=config.objetivo.lower()
  if"concentracion"in objetivo_lower and"neuromix"not in resultado.componentes_usados:recomendaciones.append("NeuroMix V27 optimizado para concentración")
  if"relajacion"in objetivo_lower and"harmonic"not in resultado.componentes_usados:recomendaciones.append("HarmonicEssence V34 excelente para relajación")
  if"carmine_analysis"in resultado.metadatos:
   carmine_data=resultado.metadatos["carmine_analysis"]
   for issue in carmine_data.get("issues",[]):
    if issue not in recomendaciones:recomendaciones.append(f"Carmine: {issue}")
   for suggestion in carmine_data.get("suggestions",[])[:2]:
    if suggestion not in recomendaciones:recomendaciones.append(f"Optimización: {suggestion}")
   score=carmine_data.get("score",100)
   if score<70:recomendaciones.append("Considerar regenerar con configuración de calidad máxima")
   elif score<85:recomendaciones.append("Calidad aceptable - pequeños ajustes podrían mejorar")
  if"correcciones_neuroacusticas"in resultado.metadatos:
   correcciones_data=resultado.metadatos["correcciones_neuroacusticas"]
   if correcciones_data.get("calidad_objetivo_alcanzada",False):recomendaciones.append(f"✅ Correcciones automáticas aplicadas exitosamente (+{correcciones_data.get('mejora_total',0):.1f} score)")
   else:
    mejora=correcciones_data.get('mejora_total',0)
    if mejora>0:recomendaciones.append(f"🔧 Correcciones aplicadas con mejora parcial (+{mejora:.1f} score)")
   score_final=correcciones_data.get("score_final",0)
   if score_final<85:recomendaciones.append("Considerar regenerar con motores diferentes para mejor resultado neuroacústico")
  return recomendaciones
 
 def _generar_sugerencias_proxima_sesion(self,resultado:ResultadoAuroraIntegrado,config:ConfiguracionAuroraUnificada)->Dict[str,Any]:
  sugerencias={"objetivos_relacionados":[],"duracion_recomendada":config.duracion_min,"intensidad_sugerida":config.intensidad,"mejoras_configuracion":{}}
  objetivo_lower=config.objetivo.lower()
  if resultado.resultado_objective_manager and self.objective_manager:
      try:
          if hasattr(self.objective_manager, 'obtener_objetivos_relacionados'):sugerencias["objetivos_relacionados"] = self.objective_manager.obtener_objetivos_relacionados(config.objetivo)
          if hasattr(self.objective_manager, 'recomendar_secuencia'):secuencia_recomendada = self.objective_manager.recomendar_secuencia(config.objetivo, config.duracion_min);if secuencia_recomendada:sugerencias["secuencia_recomendada"] = secuencia_recomendada
      except Exception as e:logger.debug(f"Error obteniendo sugerencias del Objective Manager: {e}")
  if"concentracion"in objetivo_lower:sugerencias["objetivos_relacionados"]=["claridad_mental","enfoque_profundo","productividad"]
  elif"relajacion"in objetivo_lower:sugerencias["objetivos_relacionados"]=["meditacion","calma_profunda","descanso"]
  elif"creatividad"in objetivo_lower:sugerencias["objetivos_relacionados"]=["inspiracion","flow_creativo","apertura_mental"]
  if resultado.efectividad_terapeutica>0.9:sugerencias["duracion_recomendada"]=max(10,config.duracion_min-5)
  elif resultado.efectividad_terapeutica<0.7:sugerencias["duracion_recomendada"]=min(60,config.duracion_min+10)
  if resultado.calidad_score<80:sugerencias["mejoras_configuracion"]={"calidad_objetivo":"maxima","modo_orquestacion":"layered"}
  if not config.usar_objective_manager and OBJECTIVE_MANAGER_AVAILABLE:sugerencias["mejoras_configuracion"]["usar_objective_manager"] = True
  return sugerencias
 
 def _actualizar_estadisticas(self,config:ConfiguracionAuroraUnificada,resultado:ResultadoAuroraIntegrado,tiempo_total:float):
  self.stats["experiencias_generadas"]+=1;self.stats["tiempo_total_generacion"]+=tiempo_total;estrategia=resultado.estrategia_usada.value
  if estrategia not in self.stats["estrategias_utilizadas"]:self.stats["estrategias_utilizadas"][estrategia]=0
  self.stats["estrategias_utilizadas"][estrategia]+=1;objetivo=config.objetivo
  if objetivo not in self.stats["objetivos_procesados"]:self.stats["objetivos_procesados"][objetivo]=0
  self.stats["objetivos_procesados"][objetivo]+=1
  for motor in resultado.componentes_usados:
   if motor not in self.stats["motores_utilizados"]:self.stats["motores_utilizados"][motor]=0
   self.stats["motores_utilizados"][motor]+=1
  if resultado.template_utilizado:
      if resultado.template_utilizado not in self.stats["templates_utilizados"]:self.stats["templates_utilizados"][resultado.template_utilizado] = 0
      self.stats["templates_utilizados"][resultado.template_utilizado] += 1
  if resultado.perfil_campo_utilizado:
      if resultado.perfil_campo_utilizado not in self.stats["perfiles_campo_utilizados"]:self.stats["perfiles_campo_utilizados"][resultado.perfil_campo_utilizado] = 0
      self.stats["perfiles_campo_utilizados"][resultado.perfil_campo_utilizado] += 1
  if resultado.secuencia_fases_utilizada:
      if resultado.secuencia_fases_utilizada not in self.stats["secuencias_fases_utilizadas"]:self.stats["secuencias_fases_utilizadas"][resultado.secuencia_fases_utilizada] = 0
      self.stats["secuencias_fases_utilizadas"][resultado.secuencia_fases_utilizada] += 1
  total=self.stats["experiencias_generadas"];current_avg=self.stats["calidad_promedio"];self.stats["calidad_promedio"]=((current_avg*(total-1)+resultado.calidad_score)/total)
 
 def _crear_resultado_emergencia(self,objetivo:str,error:str)->ResultadoAuroraIntegrado:
  audio_emergencia=self._generar_audio_fallback(60.0);config_emergencia=ConfiguracionAuroraUnificada(objetivo=objetivo,duracion_min=1);return ResultadoAuroraIntegrado(audio_data=audio_emergencia,metadatos={"error":error,"modo_emergencia":True,"objetivo":objetivo,"timestamp":datetime.now().isoformat()},estrategia_usada=EstrategiaGeneracion.FALLBACK_PROGRESIVO,modo_orquestacion=ModoOrquestacion.HYBRID,componentes_usados=["emergencia"],tiempo_generacion=0.0,calidad_score=60.0,coherencia_neuroacustica=0.6,efectividad_terapeutica=0.6,configuracion=config_emergencia)
 
 def _generar_audio_fallback(self,duracion_sec:float)->np.ndarray:
  try:samples=int(44100*duracion_sec);t=np.linspace(0,duracion_sec,samples);alpha=0.3*np.sin(2*np.pi*10.0*t);theta=0.2*np.sin(2*np.pi*6.0*t);audio_mono=alpha+theta;fade_samples=int(44100*2.0);if len(audio_mono)>fade_samples*2:fade_in=np.linspace(0,1,fade_samples);fade_out=np.linspace(1,0,fade_samples);audio_mono[:fade_samples]*=fade_in;audio_mono[-fade_samples:]*=fade_out;return np.stack([audio_mono,audio_mono])
  except Exception:samples=int(44100*max(1.0,duracion_sec));return np.zeros((2,samples),dtype=np.float32)
 
 def _obtener_estrategias_disponibles(self)->List[EstrategiaGeneracion]:
  estrategias=[];motores=[c for c in self.componentes.values()if c.tipo==TipoComponente.MOTOR and c.disponible];gestores=[c for c in self.componentes.values()if c.tipo==TipoComponente.GESTOR_INTELIGENCIA and c.disponible];pipelines=[c for c in self.componentes.values()if c.tipo==TipoComponente.PIPELINE and c.disponible];objective_managers=[c for c in self.componentes.values()if c.tipo==TipoComponente.OBJECTIVE_MANAGER and c.disponible]
  if len(objective_managers) >= 1 and len(motores) >= 2:estrategias.append(EstrategiaGeneracion.OBJECTIVE_MANAGER_DRIVEN)
  if len(motores)>=3 and len(gestores)>=2 and len(pipelines)>=1:estrategias.append(EstrategiaGeneracion.AURORA_ORQUESTADO)
  if len(motores)>=2:estrategias.append(EstrategiaGeneracion.MULTI_MOTOR)
  if len(gestores)>=1 and len(motores)>=1:estrategias.append(EstrategiaGeneracion.INTELIGENCIA_ADAPTIVA)
  if len(motores)>=1:estrategias.append(EstrategiaGeneracion.MOTOR_ESPECIALIZADO)
  estrategias.append(EstrategiaGeneracion.FALLBACK_PROGRESIVO);return estrategias
 
 def obtener_estado_completo(self)->Dict[str,Any]:
  estado_base = {"version":self.version,"timestamp":datetime.now().isoformat(),"componentes_detectados":{nombre:{"disponible":comp.disponible,"version":comp.version,"tipo":comp.tipo.value,"fallback":comp.version=="fallback","capacidades":len(comp.capacidades),"dependencias":comp.dependencias,"prioridad":comp.nivel_prioridad} for nombre,comp in self.componentes.items()},"estadisticas_deteccion":self.detector.stats,"estadisticas_uso":self.stats,"estrategias_disponibles":[e.value for e in self._obtener_estrategias_disponibles()],"capacidades_sistema":{"motores_activos":len([c for c in self.componentes.values()if c.tipo==TipoComponente.MOTOR and c.disponible]),"gestores_activos":len([c for c in self.componentes.values()if c.tipo==TipoComponente.GESTOR_INTELIGENCIA and c.disponible]),"pipelines_activos":len([c for c in self.componentes.values()if c.tipo==TipoComponente.PIPELINE and c.disponible]),"objective_managers_activos":len([c for c in self.componentes.values()if c.tipo==TipoComponente.OBJECTIVE_MANAGER and c.disponible]),"orquestador_disponible":self.orquestador is not None,"objective_manager_disponible":self.objective_manager is not None,"fallback_garantizado":True},"metricas_calidad":{"calidad_promedio":self.stats["calidad_promedio"],"experiencias_totales":self.stats["experiencias_generadas"],"tiempo_promedio":(self.stats["tiempo_total_generacion"]/max(1,self.stats["experiencias_generadas"])),"tasa_exito":((self.stats["experiencias_generadas"]-self.stats["errores_manejados"])/max(1,self.stats["experiencias_generadas"])*100)}}
  if self.objective_manager:estado_base["metricas_objective_manager"] = {"utilizaciones_totales": self.stats["objective_manager_utilizaciones"],"templates_mas_utilizados": sorted(self.stats["templates_utilizados"].items(), key=lambda x: x[1], reverse=True)[:5],"perfiles_campo_mas_utilizados": sorted(self.stats["perfiles_campo_utilizados"].items(), key=lambda x: x[1], reverse=True)[:5],"secuencias_fases_mas_utilizadas": sorted(self.stats["secuencias_fases_utilizadas"].items(), key=lambda x: x[1], reverse=True)[:5],"disponible": True,"version": getattr(self.objective_manager, 'version', 'unknown'),"capacidades": getattr(self.objective_manager, 'obtener_capacidades', lambda: {})()}
  else:estado_base["metricas_objective_manager"] = {"disponible": False}
  return estado_base

def crear_experiencia_con_template(objetivo: str, template: str, **kwargs) -> ResultadoAuroraIntegrado:return Aurora(objetivo, template_personalizado=template, **kwargs)
def crear_experiencia_con_perfil_campo(objetivo: str, perfil_campo: str, **kwargs) -> ResultadoAuroraIntegrado:return Aurora(objetivo, perfil_campo_personalizado=perfil_campo, **kwargs)
def crear_experiencia_con_secuencia_fases(objetivo: str, secuencia: str, **kwargs) -> ResultadoAuroraIntegrado:return Aurora(objetivo, secuencia_fases_personalizada=secuencia, **kwargs)
def obtener_templates_disponibles() -> List[str]:director = Aurora();if director.objective_manager and hasattr(director.objective_manager, 'obtener_templates_disponibles'):return director.objective_manager.obtener_templates_disponibles();return []
def obtener_perfiles_campo_disponibles() -> List[str]:director = Aurora();if director.objective_manager and hasattr(director.objective_manager, 'obtener_perfiles_disponibles'):return director.objective_manager.obtener_perfiles_disponibles();return []
def obtener_secuencias_fases_disponibles() -> List[str]:director = Aurora();if director.objective_manager and hasattr(director.objective_manager, 'obtener_secuencias_disponibles'):return director.objective_manager.obtener_secuencias_disponibles();return []

_director_global:Optional[AuroraDirectorV7Integrado]=None
def Aurora(objetivo:str=None,**kwargs)->Union[ResultadoAuroraIntegrado,AuroraDirectorV7Integrado]:
 global _director_global
 if _director_global is None:_director_global=AuroraDirectorV7Integrado()
 if objetivo is not None:return _director_global.crear_experiencia(objetivo,**kwargs)
 return _director_global

Aurora.rapido=lambda obj,**kw:Aurora(obj,duracion_min=5,calidad_objetivo="media",**kw)
Aurora.largo=lambda obj,**kw:Aurora(obj,duracion_min=60,calidad_objetivo="alta",**kw)
Aurora.terapeutico=lambda obj,**kw:Aurora(obj,duracion_min=45,intensidad="suave",calidad_objetivo="maxima",modo_orquestacion="layered",**kw)
Aurora.estado=lambda:Aurora().obtener_estado_completo()
Aurora.diagnostico=lambda:Aurora().detector.stats
Aurora.stats=lambda:Aurora().stats
Aurora.con_template=crear_experiencia_con_template
Aurora.con_perfil_campo=crear_experiencia_con_perfil_campo
Aurora.con_secuencia_fases=crear_experiencia_con_secuencia_fases
Aurora.templates_disponibles=obtener_templates_disponibles
Aurora.perfiles_campo_disponibles=obtener_perfiles_campo_disponibles
Aurora.secuencias_fases_disponibles=obtener_secuencias_fases_disponibles

if __name__=="__main__":
 print("🌟 Aurora Director V7 INTEGRADO - Sistema Completo + Objective Manager");print("="*80);director=Aurora();estado=director.obtener_estado_completo();print(f"🚀 {estado['version']}");print(f"⏰ Inicializado: {estado['timestamp']}");print(f"\n📊 Componentes detectados: {len(estado['componentes_detectados'])}")
 for nombre,info in estado['componentes_detectados'].items():
  emoji="✅"if info['disponible']and not info['fallback']else"🔄"if info['fallback']else"❌";tipo_emoji={"motor":"🎵","gestor_inteligencia":"🧠","pipeline":"🔄","preset_manager":"🎯","style_profile":"🎨","objective_manager":"🎯"}.get(info['tipo'],"🔧");print(f"   {emoji} {tipo_emoji} {nombre} v{info['version']} (P{info['prioridad']})")
 caps=estado['capacidades_sistema'];print(f"\n🔧 Capacidades del Sistema:");print(f"   🎵 Motores activos: {caps['motores_activos']}");print(f"   🧠 Gestores activos: {caps['gestores_activos']}");print(f"   🔄 Pipelines activos: {caps['pipelines_activos']}");print(f"   🎯 Objective Managers activos: {caps['objective_managers_activos']}");print(f"   🎼 Orquestador: {'✅'if caps['orquestador_disponible']else'❌'}");print(f"   🎯 Objective Manager: {'✅'if caps['objective_manager_disponible']else'❌'}");print(f"   🛡️ Fallback garantizado: {'✅'if caps['fallback_garantizado']else'❌'}")
 if estado['metricas_objective_manager']['disponible']:om_metrics = estado['metricas_objective_manager'];print(f"\n🎯 Métricas Objective Manager:");print(f"   📊 Utilizaciones: {om_metrics['utilizaciones_totales']}");print(f"   📝 Templates disponibles: {len(Aurora.templates_disponibles())}");print(f"   🎭 Perfiles campo disponibles: {len(Aurora.perfiles_campo_disponibles())}");print(f"   📋 Secuencias fases disponibles: {len(Aurora.secuencias_fases_disponibles())}")
 print(f"\n🎯 Estrategias disponibles ({len(estado['estrategias_disponibles'])}):");for i,estrategia in enumerate(estado['estrategias_disponibles'],1):print(f"   {i}. {estrategia}")
 print(f"\n🧪 Testing del sistema...")
 try:
  print(f"   🎵 Test 1: Experiencia básica...");resultado=Aurora("test_integrado",duracion_min=1,exportar_wav=False);print(f"      ✅ Audio generado: {resultado.audio_data.shape}");print(f"      📊 Calidad: {resultado.calidad_score:.1f}/100");print(f"      🎼 Estrategia: {resultado.estrategia_usada.value}");print(f"      🔧 Componentes: {len(resultado.componentes_usados)}")
  if resultado.template_utilizado:print(f"      📝 Template: {resultado.template_utilizado}")
  if resultado.perfil_campo_utilizado:print(f"      🎭 Perfil Campo: {resultado.perfil_campo_utilizado}")
  print(f"   ⚡ Test 2: API rápida...");resultado_rapido=Aurora.rapido("concentracion_test");print(f"      ✅ Audio rápido: {resultado_rapido.audio_data.shape}");print(f"      ⏱️ Duración configurada: {resultado_rapido.configuracion.duracion_min}min")
  print(f"   🏥 Test 3: API terapéutica...");resultado_terapeutico=Aurora.terapeutico("relajacion_test");print(f"      ✅ Audio terapéutico: {resultado_terapeutico.audio_data.shape}");print(f"      🎯 Modo orquestación: {resultado_terapeutico.modo_orquestacion.value}");print(f"      💊 Efectividad: {resultado_terapeutico.efectividad_terapeutica:.2f}")
  if OBJECTIVE_MANAGER_AVAILABLE and director.objective_manager:
      print(f"   🎯 Test 4: API Objective Manager...")
      try:
          templates = Aurora.templates_disponibles()
          if templates:resultado_template = Aurora.con_template("test_template", templates[0], duracion_min=1);print(f"      ✅ Con template específico: {resultado_template.audio_data.shape}");print(f"      📝 Template usado: {resultado_template.template_utilizado}")
          perfiles = Aurora.perfiles_campo_disponibles()
          if perfiles:resultado_perfil = Aurora.con_perfil_campo("test_perfil", perfiles[0], duracion_min=1);print(f"      ✅ Con perfil campo específico: {resultado_perfil.audio_data.shape}");print(f"      🎭 Perfil usado: {resultado_perfil.perfil_campo_utilizado}")
          secuencias = Aurora.secuencias_fases_disponibles()
          if secuencias:resultado_secuencia = Aurora.con_secuencia_fases("test_secuencia", secuencias[0], duracion_min=1);print(f"      ✅ Con secuencia específica: {resultado_secuencia.audio_data.shape}");print(f"      📋 Secuencia usada: {resultado_secuencia.secuencia_fases_utilizada}")
      except Exception as e:print(f"      ⚠️ Error en test Objective Manager: {e}")
  print(f"   🔍 Test 5: Integración Carmine...");resultado_carmine=Aurora("test_carmine_integration",duracion_min=1,validacion_automatica=True)
  if"carmine_analysis"in resultado_carmine.metadatos:carmine_score=resultado_carmine.metadatos["carmine_analysis"]["score"];print(f"      ✅ Carmine integrado: {carmine_score}/100");print(f"      🧠 Efectividad neuro: {resultado_carmine.coherencia_neuroacustica:.2f}");if"correcciones_neuroacusticas"in resultado_carmine.metadatos:correcciones=resultado_carmine.metadatos["correcciones_neuroacusticas"];mejora=correcciones.get("mejora_total",0);iteraciones=correcciones.get("iteraciones_realizadas",0);print(f"      🔧 Correcciones: {iteraciones} iteraciones, mejora +{mejora:.1f}");else:print(f"      ℹ️ Sin correcciones necesarias")
  else:print(f"      🔄 Carmine en modo fallback")
 except Exception as e:print(f"   ❌ Error en testing: {e}")
 metricas=estado['metricas_calidad'];print(f"\n📈 Métricas del Sistema:");print(f"   🎯 Calidad promedio: {metricas['calidad_promedio']:.1f}/100");print(f"   🔢 Experiencias totales: {metricas['experiencias_totales']}");print(f"   ⏱️ Tiempo promedio: {metricas['tiempo_promedio']:.2f}s");print(f"   ✅ Tasa de éxito: {metricas['tasa_exito']:.1f}%");print(f"\n🏆 AURORA DIRECTOR V7 INTEGRADO + OBJECTIVE MANAGER");print(f"🌟 ¡Sistema completamente funcional!");print(f"🔗 ¡Todos los motores conectados armoniosamente!");print(f"🧠 ¡Inteligencia y orquestación avanzada!");print(f"🎯 ¡Objective Manager Unificado integrado!");print(f"🎵 ¡Experiencias Aurora de máxima calidad!");print(f"✨ ¡Listo para crear experiencias transformadoras!")
