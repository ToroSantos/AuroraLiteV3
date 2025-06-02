import numpy as np,logging,time,importlib,json
from typing import Dict,List,Optional,Tuple,Any,Union,Protocol
from dataclasses import dataclass,field
from enum import Enum
from datetime import datetime
from pathlib import Path

try:
    from sync_and_scheduler import (sincronizar_y_estructurar_capas,aplicar_fade_narrativo,optimizar_coherencia_global,validar_sync_y_estructura_completa,estructura_layer_fase,generar_estructura_inteligente,ParametrosSincronizacion,ConfiguracionScheduling)
    SYNC_SCHEDULER_HIBRIDO_AVAILABLE=True
    logging.info("✅ Sync híbrido detectado")
except ImportError:SYNC_SCHEDULER_HIBRIDO_AVAILABLE=False;logging.warning("⚠️ Sync híbrido no disponible")

try:from objective_manager import ObjectiveManagerUnificado,ComponenteEstadoDescripciónRouterInteligenteV7,AnalizadorSemantico,MotorPersonalizacion,ValidadorCientifico,crear_objective_manager_unificado;OM_AVAIL=True;logging.info("✅ OM detectado")
except ImportError:OM_AVAIL=False;logging.warning("⚠️ OM no disponible")

logging.basicConfig(level=logging.INFO,format='%(asctime)s-%(levelname)s-%(message)s')
logger=logging.getLogger("Aurora.V7")

class MotorAurora(Protocol):
 def generar_audio(self,config:Dict[str,Any],duracion_sec:float)->np.ndarray:...
 def validar_configuracion(self,config:Dict[str,Any])->bool:...
 def obtener_capacidades(self)->Dict[str,Any]:...

class GestorInteligencia(Protocol):
 def procesar_objetivo(self,objetivo:str,contexto:Dict[str,Any])->Dict[str,Any]:...
 def obtener_alternativas(self,objetivo:str)->List[str]:...

class TipoComponente(Enum):MOTOR="motor";GESTOR_INTELIGENCIA="gestor_inteligencia";PIPELINE="pipeline";PRESET_MANAGER="preset_manager";STYLE_PROFILE="style_profile";OBJECTIVE_MANAGER="objective_manager";SYNC_SCHEDULER="sync_scheduler"

class EstrategiaGeneracion(Enum):AURORA_ORQUESTADO="aurora_orquestado";MULTI_MOTOR="multi_motor";MOTOR_ESPECIALIZADO="motor_especializado";INTELIGENCIA_ADAPTIVA="inteligencia_adaptiva";OBJECTIVE_MANAGER_DRIVEN="objective_manager_driven";FALLBACK_PROGRESIVO="fallback_progresivo";SYNC_SCHEDULER_HIBRIDO="sync_scheduler_hibrido"

class ModoOrquestacion(Enum):SECUENCIAL="secuencial";PARALELO="paralelo";LAYERED="layered";HYBRID="hybrid";SYNC_HIBRIDO="sync_hibrido"

@dataclass
class ComponenteAurora:nombre:str;tipo:TipoComponente;modulo:str;clase_principal:str;disponible:bool=False;instancia:Optional[Any]=None;version:str="unknown";capacidades:Dict[str,Any]=field(default_factory=dict);dependencias:List[str]=field(default_factory=list);fallback_disponible:bool=False;nivel_prioridad:int=1;compatibilidad_aurora:bool=True;metadatos:Dict[str,Any]=field(default_factory=dict)

@dataclass
class ConfiguracionAuroraUnificada:objetivo:str="relajacion";duracion_min:int=20;sample_rate:int=44100;estrategia_preferida:Optional[EstrategiaGeneracion]=None;modo_orquestacion:ModoOrquestacion=ModoOrquestacion.HYBRID;motores_preferidos:List[str]=field(default_factory=list);forzar_componentes:List[str]=field(default_factory=list);excluir_componentes:List[str]=field(default_factory=list);intensidad:str="media";estilo:str="sereno";neurotransmisor_preferido:Optional[str]=None;calidad_objetivo:str="alta";normalizar:bool=True;aplicar_mastering:bool=True;validacion_automatica:bool=True;exportar_wav:bool=True;nombre_archivo:str="aurora_experience";incluir_metadatos:bool=True;configuracion_custom:Dict[str,Any]=field(default_factory=dict);perfil_usuario:Optional[Dict[str,Any]]=None;contexto_uso:Optional[str]=None;session_id:Optional[str]=None;usar_objective_manager:bool=True;template_personalizado:Optional[str]=None;perfil_campo_personalizado:Optional[str]=None;secuencia_fases_personalizada:Optional[str]=None;metadatos_emocionales:Dict[str,Any]=field(default_factory=dict);parametros_neuroacusticos:Dict[str,Any]=field(default_factory=dict);efectos_psicodelicos:Dict[str,Any]=field(default_factory=dict);secuencia_perfiles:List[Tuple[str,int]]=field(default_factory=list);frecuencia_base_psicodelica:Optional[float]=None;coherencia_objetivo:float=0.8;configuracion_enriquecida:bool=False;habilitar_sync_hibrido:bool=True;parametros_sync_hibrido:Optional[Dict[str,Any]]=None
 def validar(self)->List[str]:p=[];if self.duracion_min<=0:p.append("Duración debe ser positiva");if self.sample_rate not in[22050,44100,48000]:p.append("Sample rate no estándar");if self.intensidad not in["suave","media","intenso"]:p.append("Intensidad inválida");if not self.objetivo.strip():p.append("Objetivo no puede estar vacío");return p

@dataclass
class ResultadoAuroraIntegrado:audio_data:np.ndarray;metadatos:Dict[str,Any];estrategia_usada:EstrategiaGeneracion;modo_orquestacion:ModoOrquestacion;componentes_usados:List[str];tiempo_generacion:float;calidad_score:float;coherencia_neuroacustica:float;efectividad_terapeutica:float;configuracion:ConfiguracionAuroraUnificada;capas_audio:Dict[str,np.ndarray]=field(default_factory=dict);analisis_espectral:Dict[str,Any]=field(default_factory=dict);recomendaciones:List[str]=field(default_factory=list);proxima_sesion:Dict[str,Any]=field(default_factory=dict);resultado_objective_manager:Optional[Dict[str,Any]]=None;template_utilizado:Optional[str]=None;perfil_campo_utilizado:Optional[str]=None;secuencia_fases_utilizada:Optional[str]=None;resultado_sync_hibrido:Optional[Dict[str,Any]]=None;validacion_sync_scheduler:Optional[Dict[str,Any]]=None;estructura_fases_utilizada:Optional[List[Dict[str,Any]]]=None

class DetectorComponentesAvanzado:
 def __init__(self):self.componentes_registrados=self._init_registro_completo();self.componentes_activos:Dict[str,ComponenteAurora]={};self.stats={"total":0,"exitosos":0,"fallidos":0,"fallback":0,"tiempo_deteccion":0.0,"motores_detectados":0,"gestores_detectados":0,"sync_scheduler_detectado":0};self.cache_deteccion={}
 def _init_registro_completo(self)->Dict[str,ComponenteAurora]:
  r={
   "neuromix_v27":ComponenteAurora("neuromix_v27",TipoComponente.MOTOR,"neuromix_aurora_v27","AuroraNeuroAcousticEngineV27",dependencias=[],fallback_disponible=True,nivel_prioridad=1,metadatos={"especialidad":"neuroacustica","calidad":"alta"}),
   "hypermod_v32":ComponenteAurora("hypermod_v32",TipoComponente.MOTOR,"hypermod_v32","HyperModEngineV32AuroraConnected",dependencias=[],fallback_disponible=True,nivel_prioridad=1,metadatos={"especialidad":"ondas_cerebrales","calidad":"maxima"}),
   "harmonic_essence_v34":ComponenteAurora("harmonic_essence_v34",TipoComponente.MOTOR,"harmonicEssence_v34","HarmonicEssenceV34AuroraConnected",dependencias=[],fallback_disponible=True,nivel_prioridad=1,metadatos={"especialidad":"texturas","calidad":"alta"}),
   "field_profiles":ComponenteAurora("field_profiles",TipoComponente.GESTOR_INTELIGENCIA,"field_profiles","GestorPerfilesCampo",dependencias=[],fallback_disponible=True,nivel_prioridad=2),
   "objective_router":ComponenteAurora("objective_router",TipoComponente.GESTOR_INTELIGENCIA,"objective_router","RouterInteligenteV7",dependencias=["field_profiles"],fallback_disponible=True,nivel_prioridad=2),
   "emotion_style_profiles":ComponenteAurora("emotion_style_profiles",TipoComponente.GESTOR_INTELIGENCIA,"emotion_style_profiles","GestorEmotionStyleUnificadoV7",dependencias=[],fallback_disponible=True,nivel_prioridad=2),
   "quality_pipeline":ComponenteAurora("quality_pipeline",TipoComponente.PIPELINE,"aurora_quality_pipeline","AuroraQualityPipeline",dependencias=[],fallback_disponible=True,nivel_prioridad=4),
   "neuromix_legacy":ComponenteAurora("neuromix_legacy",TipoComponente.MOTOR,"neuromix_engine_v26_ultimate","AuroraNeuroAcousticEngine",dependencias=[],fallback_disponible=True,nivel_prioridad=5),
   "hypermod_legacy":ComponenteAurora("hypermod_legacy",TipoComponente.MOTOR,"hypermod_engine_v31","NeuroWaveGenerator",dependencias=[],fallback_disponible=True,nivel_prioridad=5),
   "carmine_analyzer_v21":ComponenteAurora("carmine_analyzer_v21",TipoComponente.PIPELINE,"Carmine_Analyzer","CarmineAuroraAnalyzer",dependencias=[],fallback_disponible=True,nivel_prioridad=3,metadatos={"especialidad":"analisis_neuroacustico","version":"2.1","calidad":"maxima"})
  }
  if OM_AVAIL:r["objective_manager_unificado"]=ComponenteAurora("objective_manager_unificado",TipoComponente.OBJECTIVE_MANAGER,"objective_manager","ObjectiveManagerUnificado",dependencias=[],fallback_disponible=True,nivel_prioridad=1,metadatos={"especialidad":"gestion_objetivos_integral","version":"unificado_v7","calidad":"maxima","capacidades":["templates","perfiles_campo","secuencias_fases","routing_inteligente"]})
  if SYNC_SCHEDULER_HIBRIDO_AVAILABLE:r["sync_scheduler_v7"]=ComponenteAurora("sync_scheduler_v7",TipoComponente.SYNC_SCHEDULER,"sync_and_scheduler","SyncSchedulerV7",dependencias=[],fallback_disponible=True,nivel_prioridad=1,metadatos={"especialidad":"sincronizacion_hibrida","version":"V7_UNIFIED_OPTIMIZED","calidad":"maxima","capacidades":["sincronizacion_multicapa","estructura_fases","fade_narrativo","coherencia_global"]})
  return r
 def detectar_todos(self)->Dict[str,ComponenteAurora]:st=time.time();logger.info("🔍 Detectando componentes...");co=sorted(self.componentes_registrados.items(),key=lambda x:x[1].nivel_prioridad);[self._detectar_componente(c)for n,c in co];self.stats["tiempo_deteccion"]=time.time()-st;self._log_resumen_deteccion();return self.componentes_activos
 def _detectar_componente(self,comp:ComponenteAurora)->bool:
  self.stats["total"]+=1
  try:
   if not self._verificar_dependencias(comp):return False
   if comp.nombre in self.cache_deteccion:
    cr=self.cache_deteccion[comp.nombre]
    if cr["success"]:comp.disponible=True;comp.instancia=cr["instancia"];comp.version=cr["version"];comp.capacidades=cr["capacidades"];self.componentes_activos[comp.nombre]=comp;self.stats["exitosos"]+=1;return True
   m=importlib.import_module(comp.modulo);i=self._crear_instancia(m,comp)
   if self._validar_instancia(i,comp):
    comp.disponible=True;comp.instancia=i;comp.capacidades=self._obtener_capacidades(i);comp.version=self._obtener_version(i);self.cache_deteccion[comp.nombre]={"success":True,"instancia":i,"version":comp.version,"capacidades":comp.capacidades};self.componentes_activos[comp.nombre]=comp
    if comp.tipo==TipoComponente.MOTOR:self.stats["motores_detectados"]+=1
    elif comp.tipo==TipoComponente.GESTOR_INTELIGENCIA:self.stats["gestores_detectados"]+=1
    elif comp.tipo==TipoComponente.OBJECTIVE_MANAGER:self.stats["objective_managers_detectados"]=self.stats.get("objective_managers_detectados",0)+1
    elif comp.tipo==TipoComponente.SYNC_SCHEDULER:self.stats["sync_scheduler_detectado"]+=1
    self.stats["exitosos"]+=1;logger.info(f"✅ {comp.nombre} v{comp.version}");return True
   else:raise Exception("Instancia no válida")
  except Exception as e:
   if comp.fallback_disponible and self._crear_fallback(comp):self.stats["fallback"]+=1;return True
   self.cache_deteccion[comp.nombre]={"success":False,"error":str(e)};self.stats["fallidos"]+=1;return False
 def _verificar_dependencias(self,comp:ComponenteAurora)->bool:return all(dep in self.componentes_activos for dep in comp.dependencias)
 def _crear_instancia(self,modulo:Any,comp:ComponenteAurora)->Any:
  if comp.modulo=="neuromix_aurora_v27":return getattr(modulo,"AuroraNeuroAcousticEngineV27")()
  elif comp.modulo=="hypermod_v32":return getattr(modulo,"_motor_global_v32",None)or modulo
  elif comp.modulo=="harmonicEssence_v34":return getattr(modulo,"HarmonicEssenceV34AuroraConnected")()
  elif comp.modulo=="emotion_style_profiles":return getattr(modulo,"crear_gestor_emotion_style_v7")()
  elif comp.modulo=="objective_manager":
   if hasattr(modulo,"crear_objective_manager_unificado"):return getattr(modulo,"crear_objective_manager_unificado")()
   elif hasattr(modulo,"ObjectiveManagerUnificado"):return getattr(modulo,"ObjectiveManagerUnificado")()
   else:return None
  elif comp.modulo=="sync_and_scheduler":return modulo
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
   elif comp.tipo==TipoComponente.SYNC_SCHEDULER:return(hasattr(instancia,'sincronizar_y_estructurar_capas')and hasattr(instancia,'aplicar_fade_narrativo')and hasattr(instancia,'optimizar_coherencia_global'))
   return True
  except Exception:return False
 def _obtener_capacidades(self,instancia:Any)->Dict[str,Any]:
  try:
   for metodo in['obtener_capacidades','get_capabilities','capacidades']:
    if hasattr(instancia,metodo):return getattr(instancia,metodo)()
   if hasattr(instancia,'__name__')and'sync_and_scheduler'in instancia.__name__:return{"sincronizacion_multicapa":True,"estructura_fases":True,"fade_narrativo":True,"coherencia_global":True,"validacion_completa":True}
  except Exception:pass
  return{}
 def _obtener_version(self,instancia:Any)->str:
  for attr in['version','VERSION','__version__','_version']:
   if hasattr(instancia,attr):version=getattr(instancia,attr);return str(version)
  if hasattr(instancia,'__name__')and'sync_and_scheduler'in instancia.__name__:return"V7_UNIFIED_OPTIMIZED"
  return"unknown"
 def _crear_fallback(self,comp:ComponenteAurora)->bool:
  try:
   fb={"neuromix_v27":self._fallback_neuromix,"neuromix_legacy":self._fallback_neuromix,"hypermod_v32":self._fallback_hypermod,"hypermod_legacy":self._fallback_hypermod,"harmonic_essence_v34":self._fallback_harmonic,"field_profiles":self._fallback_field_profiles,"objective_router":self._fallback_objective_router,"quality_pipeline":self._fallback_quality_pipeline,"carmine_analyzer_v21":self._fallback_carmine_analyzer,"objective_manager_unificado":self._fallback_objective_manager,"sync_scheduler_v7":self._fallback_sync_scheduler}
   if comp.nombre in fb:comp.instancia=fb[comp.nombre]();comp.disponible=True;comp.version="fallback";self.componentes_activos[comp.nombre]=comp;return True
  except Exception as e:logger.error(f"Error fallback {comp.nombre}: {e}")
  return False
 def _fallback_sync_scheduler(self):
  class SSF:
   def sincronizar_y_estructurar_capas(self,audio_layers,estructura_fases,**kwargs):return audio_layers,{"fallback":True,"coherencia_global":0.7}
   def aplicar_fade_narrativo(self,audio_layers,fase_actual,configuracion):return audio_layers
   def optimizar_coherencia_global(self,audio_layers,estructura_fases,objetivo_coherencia=0.9):return audio_layers,{"coherencia_global":0.8,"fallback_usado":True}
   def validar_sync_y_estructura_completa(self,audio_layers,estructura_fases,**kwargs):return{"validacion_global":True,"puntuacion_global":0.75,"fallback":True}
   def estructura_layer_fase(self,total_bloques,modo="normal",estilo="neutro"):return[{"bloque":i,"gain":1.0,"paneo":0.0,"capas":{"neuro_wave":True}}for i in range(total_bloques)]
   def generar_estructura_inteligente(self,dur_min,config_base=None,**params):return{"configuracion":{"duracion_minutos":dur_min,"total_bloques":max(2,dur_min//2)},"estructura":self.estructura_layer_fase(max(2,dur_min//2)),"validacion_cientifica":{"confianza_global":0.8,"fallback":True}}
  return SSF()
 def _fallback_objective_manager(self):
  class OMF:
   def procesar_objetivo_completo(self,objetivo:str,contexto:Dict[str,Any]=None)->Dict[str,Any]:mb={"relajacion":{"neurotransmisor_preferido":"gaba","intensidad":"suave","estilo":"sereno","template_recomendado":"relajacion_profunda","perfil_campo":"relajacion","beat_base":7.0},"concentracion":{"neurotransmisor_preferido":"acetilcolina","intensidad":"media","estilo":"crystalline","template_recomendado":"claridad_mental","perfil_campo":"cognitivo","beat_base":14.0},"creatividad":{"neurotransmisor_preferido":"anandamida","intensidad":"media","estilo":"organico","template_recomendado":"creatividad_exponencial","perfil_campo":"creativo","beat_base":10.0},"meditacion":{"neurotransmisor_preferido":"serotonina","intensidad":"suave","estilo":"mistico","template_recomendado":"presencia_total","perfil_campo":"espiritual","beat_base":6.0}};ol=objetivo.lower();cb=mb.get("relajacion");[(cb:=c)for k,c in mb.items()if k in ol];return{"configuracion_motor":cb,"template_utilizado":cb.get("template_recomendado"),"perfil_campo_utilizado":cb.get("perfil_campo"),"resultado_routing":{"confianza":0.7,"tipo":"fallback_mapping","fuente":"objective_manager_fallback"},"metadatos":{"fallback_usado":True,"objetivo_original":objetivo,"contexto_procesado":contexto or{}}}
   def rutear_objetivo_inteligente(self,objetivo:str,**kwargs)->Dict[str,Any]:return self.procesar_objetivo_completo(objetivo,kwargs)
   def obtener_configuracion_completa(self,objetivo:str)->Dict[str,Any]:return self.procesar_objetivo_completo(objetivo)
   def generar_configuracion_motor(self,objetivo:str,motor_objetivo:str)->Dict[str,Any]:cb=self.procesar_objetivo_completo(objetivo);return cb.get("configuracion_motor",{})
   def obtener_capacidades(self)->Dict[str,Any]:return{"nombre":"Objective Manager Fallback","tipo":"gestor_objetivos_fallback","capacidades":["mapping_basico","routing_simple"],"templates_disponibles":["relajacion_profunda","claridad_mental","creatividad_exponencial","presencia_total"],"fallback":True}
  return OMF()
 def _fallback_neuromix(self):
  class NMF:
   def generar_audio(self,config:Dict[str,Any],duracion_sec:float)->np.ndarray:s=int(44100*duracion_sec);t=np.linspace(0,duracion_sec,s);fb=10.0;if config.get('neurotransmisor_preferido')=='dopamina':fb=12.0;elif config.get('neurotransmisor_preferido')=='gaba':fb=6.0;w=0.3*np.sin(2*np.pi*fb*t);fs=min(2048,len(w)//4);if len(w)>fs*2:fi=np.linspace(0,1,fs);fo=np.linspace(1,0,fs);w[:fs]*=fi;w[-fs:]*=fo;return np.stack([w,w])
   def validar_configuracion(self,config:Dict[str,Any])->bool:return isinstance(config,dict)and config.get('objetivo','').strip()
   def obtener_capacidades(self)->Dict[str,Any]:return{"nombre":"NeuroMix Fallback","tipo":"motor_neuroacustico_fallback","neurotransmisores":["dopamina","serotonina","gaba"]}
  return NMF()
 def _fallback_hypermod(self):
  class HMF:
   def generar_audio(self,config:Dict[str,Any],duracion_sec:float)->np.ndarray:s=int(44100*duracion_sec);t=np.linspace(0,duracion_sec,s);a=0.4*np.sin(2*np.pi*10.0*t);th=0.2*np.sin(2*np.pi*6.0*t);w=a+th;return np.stack([w,w])
   def validar_configuracion(self,config:Dict[str,Any])->bool:return True
   def obtener_capacidades(self)->Dict[str,Any]:return{"nombre":"HyperMod Fallback","tipo":"motor_ondas_cerebrales_fallback"}
  return HMF()
 def _fallback_harmonic(self):
  class HF:
   def generar_audio(self,config:Dict[str,Any],duracion_sec:float)->np.ndarray:s=int(44100*duracion_sec);tx=np.random.normal(0,0.1,s);if s>100:ks=min(50,s//20);k=np.ones(ks)/ks;tx=np.convolve(tx,k,mode='same');return np.stack([tx,tx])
   def validar_configuracion(self,config:Dict[str,Any])->bool:return True
   def obtener_capacidades(self)->Dict[str,Any]:return{"nombre":"HarmonicEssence Fallback","tipo":"motor_texturas_fallback"}
  return HF()
 def _fallback_field_profiles(self):
  class FPF:
   def procesar_objetivo(self,objetivo:str,contexto:Dict[str,Any])->Dict[str,Any]:return{"perfil_recomendado":"basico","configuracion":{"intensidad":"media","duracion_min":20}}
   def obtener_perfil(self,nombre:str):return None
   def recomendar_secuencia_perfiles(self,objetivo:str,duracion:int):return[(objetivo,duracion)]
  return FPF()
 def _fallback_objective_router(self):
  class ORF:
   def procesar_objetivo(self,objetivo:str,contexto:Dict[str,Any])->Dict[str,Any]:m={"relajacion":{"neurotransmisor_preferido":"gaba","intensidad":"suave","estilo":"sereno"},"concentracion":{"neurotransmisor_preferido":"acetilcolina","intensidad":"media","estilo":"crystalline"},"creatividad":{"neurotransmisor_preferido":"anandamida","intensidad":"media","estilo":"organico"}};return m.get(objetivo.lower(),{"neurotransmisor_preferido":"serotonina","intensidad":"media","estilo":"neutro"})
   def rutear_objetivo(self,objetivo:str,**kwargs):return self.procesar_objetivo(objetivo,kwargs)
  return ORF()
 def _fallback_quality_pipeline(self):
  class QPF:
   def validar_y_normalizar(self,signal:np.ndarray)->np.ndarray:if signal.ndim==1:signal=np.stack([signal,signal]);mv=np.max(np.abs(signal));if mv>0:signal=signal*(0.85/mv);return np.clip(signal,-1.0,1.0)
  return QPF()
 def _fallback_carmine_analyzer(self):
  class CAF:
   def analyze_audio(self,audio:np.ndarray,expected_intent=None):if audio.size==0:return type('Result',(),{'score':0,'therapeutic_score':0,'quality':type('Quality',(),{'value':'🔴 CRÍTICO'})(),'suggestions':["Audio vacío"],'issues':["Sin audio"],'gpt_summary':"Audio inválido"})();rms=np.sqrt(np.mean(audio**2));pk=np.max(np.abs(audio));sc=min(100,max(50,80+(1-min(pk,1.0))*20));ql="🟢 ÓPTIMO"if sc>=90 else"🟡 OBSERVACIÓN"if sc>=70 else"🔴 CRÍTICO";return type('Result',(),{'score':int(sc),'therapeutic_score':int(sc*0.9),'quality':type('Quality',(),{'value':ql})(),'suggestions':["Usar fallback"],'issues':[]if sc>70 else["Calidad subóptima"],'neuro_metrics':type('NeuroMetrics',(),{'entrainment_effectiveness':0.7,'binaural_strength':0.5})(),'gpt_summary':f"Análisis fallback: Score {sc:.0f}/100"})()
   def obtener_capacidades(self):return{"nombre":"Carmine Analyzer Fallback","tipo":"analizador_basico_fallback"}
  return CAF()
 def _log_resumen_deteccion(self):t=len(self.componentes_registrados);a=len(self.componentes_activos);p=(a/t*100)if t>0 else 0;logger.info(f"📊 {self.stats['tiempo_deteccion']:.2f}s - {a}/{t} ({p:.0f}%) - ✅{self.stats['exitosos']} 🔄{self.stats['fallback']} ❌{self.stats['fallidos']} - 🎵{self.stats['motores_detectados']} 🧠{self.stats['gestores_detectados']} 🌟{self.stats['sync_scheduler_detectado']}")

class OrquestadorMultiMotor:
 def __init__(self,componentes_activos:Dict[str,ComponenteAurora]):self.componentes=componentes_activos;self.motores_disponibles={n:c for n,c in componentes_activos.items()if c.tipo==TipoComponente.MOTOR and c.disponible};self.objective_manager=None;self.sync_scheduler=None;if"objective_manager_unificado"in componentes_activos and componentes_activos["objective_manager_unificado"].disponible:self.objective_manager=componentes_activos["objective_manager_unificado"].instancia;if"sync_scheduler_v7"in componentes_activos and componentes_activos["sync_scheduler_v7"].disponible:self.sync_scheduler=componentes_activos["sync_scheduler_v7"].instancia
 def generar_audio_orquestado(self,config:ConfiguracionAuroraUnificada,duracion_sec:float)->Tuple[np.ndarray,Dict[str,Any]]:
  mg={"motores_utilizados":[],"tiempo_por_motor":{},"calidad_por_motor":{},"estrategia_aplicada":config.modo_orquestacion.value}
  if config.usar_objective_manager and self.objective_manager:
   try:
    rom=self.objective_manager.procesar_objetivo_completo(config.objetivo,{"duracion_min":config.duracion_min,"intensidad":config.intensidad,"estilo":config.estilo,"contexto_uso":config.contexto_uso,"perfil_usuario":config.perfil_usuario,"calidad_objetivo":config.calidad_objetivo});cm=rom.get("configuracion_motor",{})
    for k,v in cm.items():
     if hasattr(config,k)and v is not None:setattr(config,k,v)
    mg["objective_manager"]={"utilizado":True,"template_utilizado":rom.get("template_utilizado"),"perfil_campo_utilizado":rom.get("perfil_campo_utilizado"),"secuencia_fases_utilizada":rom.get("secuencia_fases_utilizada"),"confianza_routing":rom.get("resultado_routing",{}).get("confianza",0.0),"tipo_routing":rom.get("resultado_routing",{}).get("tipo","unknown")}
   except Exception as e:mg["objective_manager"]={"utilizado":False,"error":str(e)}
  else:mg["objective_manager"]={"utilizado":False,"razon":"no_disponible_o_deshabilitado"}
  if config.modo_orquestacion==ModoOrquestacion.LAYERED:return self._generar_en_capas(config,duracion_sec,mg)
  elif config.modo_orquestacion==ModoOrquestacion.PARALELO:return self._generar_paralelo(config,duracion_sec,mg)
  elif config.modo_orquestacion==ModoOrquestacion.SECUENCIAL:return self._generar_secuencial(config,duracion_sec,mg)
  elif config.modo_orquestacion==ModoOrquestacion.SYNC_HIBRIDO:return self.generar_audio_orquestado_v7_hibrido(config,duracion_sec,mg)
  else:return self._generar_hibrido(config,duracion_sec,mg)
 def generar_audio_orquestado_v7_hibrido(self,config:ConfiguracionAuroraUnificada,duracion_sec:float,metadatos:Dict[str,Any])->Tuple[np.ndarray,Dict[str,Any]]:
  try:
   capas_audio={};motores_utilizados=[];motores_configuracion=self._configurar_motores_hibridos(config)
   for capa_nombre,motor_config in motores_configuracion.items():
    motor_nombre=motor_config["motor"]
    if motor_nombre in self.motores_disponibles:
     try:
      motor=self.motores_disponibles[motor_nombre].instancia;config_motor=self._adaptar_config_para_motor(config,motor_nombre,motor_config.get("config_adicional",{}));audio_capa=motor.generar_audio(config_motor,duracion_sec);peso=motor_config.get("peso",1.0);audio_capa=audio_capa*peso;capas_audio[capa_nombre]=audio_capa;motores_utilizados.append(motor_nombre)
     except Exception as e:continue
   if not capas_audio:return self._generar_hibrido_fallback(config,duracion_sec,metadatos)
   if self.sync_scheduler and config.habilitar_sync_hibrido:
    resultado_hibrido=self._aplicar_sync_scheduler_hibrido(capas_audio,config,duracion_sec);metadatos.update({"sync_hibrido_aplicado":True,"motores_utilizados":motores_utilizados,"capas_procesadas":len(capas_audio),"coherencia_global":resultado_hibrido.get("coherencia_global",0.0),"validacion_sync_scheduler":resultado_hibrido.get("validacion",{})});return resultado_hibrido["audio_final"],metadatos
   else:audio_final=self._combinar_capas_tradicional(capas_audio);metadatos.update({"sync_hibrido_aplicado":False,"motores_utilizados":motores_utilizados,"capas_procesadas":len(capas_audio),"modo_fallback":"combinacion_tradicional"});return audio_final,metadatos
  except Exception as e:return self._generar_hibrido_fallback(config,duracion_sec,metadatos)
 def _configurar_motores_hibridos(self,config:ConfiguracionAuroraUnificada)->Dict[str,Dict[str,Any]]:
  objetivo=config.objetivo.lower();configuraciones={"concentracion":{"neuroacustica_principal":{"motor":"neuromix_v27","peso":0.6,"config_adicional":{"wave_type":"neural_complex"}},"ondas_cerebrales":{"motor":"hypermod_v32","peso":0.3,"config_adicional":{"preset_emocional":"claridad_mental"}},"textura_ambiente":{"motor":"harmonic_essence_v34","peso":0.2,"config_adicional":{"texture_type":"crystalline"}}},"relajacion":{"textura_principal":{"motor":"harmonic_essence_v34","peso":0.5,"config_adicional":{"texture_type":"relaxation"}},"neuroacustica_suave":{"motor":"neuromix_v27","peso":0.4,"config_adicional":{"wave_type":"therapeutic"}},"ondas_theta":{"motor":"hypermod_v32","peso":0.3,"config_adicional":{"preset_emocional":"calma_profunda"}}},"creatividad":{"texturas_organicas":{"motor":"harmonic_essence_v34","peso":0.5,"config_adicional":{"texture_type":"organic"}},"neuroacustica_creativa":{"motor":"neuromix_v27","peso":0.4,"config_adicional":{"wave_type":"hybrid"}},"ondas_alpha":{"motor":"hypermod_v32","peso":0.3,"config_adicional":{"preset_emocional":"expansion_creativa"}}},"meditacion":{"ondas_profundas":{"motor":"hypermod_v32","peso":0.5,"config_adicional":{"preset_emocional":"conexion_mistica"}},"texturas_espirituales":{"motor":"harmonic_essence_v34","peso":0.4,"config_adicional":{"texture_type":"consciousness"}},"neuroacustica_meditativa":{"motor":"neuromix_v27","peso":0.3,"config_adicional":{"wave_type":"therapeutic"}}}};config_default={"neuroacustica_base":{"motor":"neuromix_v27","peso":0.5},"ondas_equilibrio":{"motor":"hypermod_v32","peso":0.3},"textura_ambiente":{"motor":"harmonic_essence_v34","peso":0.2}}
  for key,config_especifica in configuraciones.items():
   if key in objetivo:return config_especifica
  return config_default
 def _aplicar_sync_scheduler_hibrido(self,capas_audio:Dict[str,np.ndarray],config:ConfiguracionAuroraUnificada,duracion_sec:float)->Dict[str,Any]:
  try:
   estructura_resultado=self.sync_scheduler.generar_estructura_inteligente(dur_min=config.duracion_min,config_base={"objetivo":config.objetivo,"intensidad":config.intensidad,"estilo":config.estilo});estructura_fases=estructura_resultado.get("estructura",[]);parametros_sync=self.sync_scheduler.ParametrosSincronizacion(sample_rate=config.sample_rate,validacion_neuroacustica=True,optimizacion_automatica=True,umbral_coherencia=config.coherencia_objetivo);config_scheduling=self.sync_scheduler.ConfiguracionScheduling(validacion_neuroacustica=True,optimizacion_coherencia=True,intensidad_base={"suave":0.6,"media":0.8,"intenso":1.0}.get(config.intensidad,0.8),patron_espacial=self._mapear_estilo_a_patron_espacial(config.estilo));capas_sincronizadas,metadatos_sync=self.sync_scheduler.sincronizar_y_estructurar_capas(audio_layers=capas_audio,estructura_fases=estructura_fases,parametros_sync=parametros_sync,config_scheduling=config_scheduling);fase_actual=self._determinar_fase_actual(config,duracion_sec);capas_con_fades=self.sync_scheduler.aplicar_fade_narrativo(audio_layers=capas_sincronizadas,fase_actual=fase_actual,configuracion=config_scheduling);capas_optimizadas,metadatos_coherencia=self.sync_scheduler.optimizar_coherencia_global(audio_layers=capas_con_fades,estructura_fases=estructura_fases,objetivo_coherencia=config.coherencia_objetivo);audio_final=self._combinar_capas_optimizadas(capas_optimizadas);validacion=self.sync_scheduler.validar_sync_y_estructura_completa(audio_layers=capas_optimizadas,estructura_fases=estructura_fases,nivel_detalle="completo");return{"audio_final":audio_final,"coherencia_global":metadatos_coherencia.get("coherencia_global",0.0),"metadatos_sync":metadatos_sync,"metadatos_coherencia":metadatos_coherencia,"validacion":validacion,"estructura_fases":estructura_fases,"fase_aplicada":fase_actual,"sync_scheduler_version":"V7_UNIFIED_OPTIMIZED"}
  except Exception as e:audio_final=self._combinar_capas_tradicional(capas_audio);return{"audio_final":audio_final,"coherencia_global":0.7,"error":str(e),"fallback_usado":True}
 def _mapear_estilo_a_patron_espacial(self,estilo:str):mapeo={"sereno":"neutro","crystalline":"cristalino","organico":"organico","etereo":"etereo","tribal":"tribal","mistico":"mistico","cuantico":"cuantico"};return mapeo.get(estilo.lower(),"neutro")
 def _determinar_fase_actual(self,config:ConfiguracionAuroraUnificada,duracion_sec:float)->str:return"centro"if duracion_sec<=300 else"desarrollo"if duracion_sec<=600 else"desarrollo_temprano"if duracion_sec<=1200 else"entrada"
 def _combinar_capas_optimizadas(self,capas_optimizadas:Dict[str,np.ndarray])->np.ndarray:
  if not capas_optimizadas:return np.zeros((2,int(44100*60)),dtype=np.float32)
  longitudes=[capa.shape[1]if capa.ndim==2 else len(capa)for capa in capas_optimizadas.values()];longitud_minima=min(longitudes);audio_combinado=None
  for nombre_capa,audio_capa in capas_optimizadas.items():
   if audio_capa.ndim==1:audio_capa=np.stack([audio_capa,audio_capa])
   elif audio_capa.ndim==2 and audio_capa.shape[0]!=2:audio_capa=audio_capa.T
   audio_capa=audio_capa[:,:longitud_minima]
   if audio_combinado is None:audio_combinado=audio_capa
   else:audio_combinado+=audio_capa
  max_val=np.max(np.abs(audio_combinado))
  if max_val>0:audio_combinado=audio_combinado*(0.85/max_val)
  return audio_combinado
 def _combinar_capas_tradicional(self,capas_audio:Dict[str,np.ndarray])->np.ndarray:
  if not capas_audio:return np.zeros((2,int(44100*60)),dtype=np.float32)
  audio_final=None
  for nombre_capa,audio_capa in capas_audio.items():
   if audio_capa.ndim==1:audio_capa=np.stack([audio_capa,audio_capa])
   elif audio_capa.ndim==2 and audio_capa.shape[0]!=2:audio_capa=audio_capa.T
   if audio_final is None:audio_final=audio_capa
   else:min_len=min(audio_final.shape[1],audio_capa.shape[1]);audio_final=audio_final[:,:min_len]+audio_capa[:,:min_len]
  max_val=np.max(np.abs(audio_final))
  if max_val>0:audio_final=audio_final*(0.85/max_val)
  return audio_final
 def _generar_hibrido_fallback(self,config:ConfiguracionAuroraUnificada,duracion_sec:float,metadatos:Dict[str,Any])->Tuple[np.ndarray,Dict[str,Any]]:audio,meta=self._generar_hibrido(config,duracion_sec,metadatos);meta.update({"sync_hibrido_aplicado":False,"fallback_hibrido_usado":True,"razon_fallback":"error_en_sync_scheduler_hibrido"});return audio,meta
 def _generar_en_capas(self,config:ConfiguracionAuroraUnificada,duracion_sec:float,metadatos:Dict[str,Any])->Tuple[np.ndarray,Dict[str,Any]]:cc=[("neuromix_v27",{"peso":0.6,"procesamiento":"base"}),("hypermod_v32",{"peso":0.3,"procesamiento":"armonica"}),("harmonic_essence_v34",{"peso":0.2,"procesamiento":"textura"})];af=None;cg={}
  for nm,cc_config in cc:
   if nm in self.motores_disponibles:
    st=time.time()
    try:m=self.motores_disponibles[nm].instancia;cm=self._adaptar_config_para_motor(config,nm,cc_config);ac=m.generar_audio(cm,duracion_sec);ac=ac*cc_config["peso"];af=ac if af is None else self._combinar_capas(af,ac);tg=time.time()-st;metadatos["motores_utilizados"].append(nm);metadatos["tiempo_por_motor"][nm]=tg;cg[nm]=ac
    except Exception as e:continue
  if af is None:af=self._generar_fallback_simple(duracion_sec)
  metadatos["capas_generadas"]=len(cg);return af,metadatos
 def _generar_paralelo(self,config:ConfiguracionAuroraUnificada,duracion_sec:float,metadatos:Dict[str,Any])->Tuple[np.ndarray,Dict[str,Any]]:mp=self._seleccionar_motor_principal(config)
  if mp:
   st=time.time()
   try:i=self.motores_disponibles[mp].instancia;cm=self._adaptar_config_para_motor(config,mp);ar=i.generar_audio(cm,duracion_sec);metadatos["motores_utilizados"].append(mp);metadatos["tiempo_por_motor"][mp]=time.time()-st;metadatos["motor_principal"]=mp;return ar,metadatos
   except Exception as e:pass
  return self._generar_fallback_simple(duracion_sec),metadatos
 def _generar_secuencial(self,config:ConfiguracionAuroraUnificada,duracion_sec:float,metadatos:Dict[str,Any])->Tuple[np.ndarray,Dict[str,Any]]:ma=list(self.motores_disponibles.keys())[:3]
  if not ma:return self._generar_fallback_simple(duracion_sec),metadatos
  dpm=duracion_sec/len(ma);sa=[]
  for i,nm in enumerate(ma):
   st=time.time()
   try:m=self.motores_disponibles[nm].instancia;cm=self._adaptar_config_para_motor(config,nm);s=m.generar_audio(cm,dpm);sa.append(s);metadatos["motores_utilizados"].append(nm);metadatos["tiempo_por_motor"][nm]=time.time()-st
   except Exception as e:samples=int(44100*dpm);sa.append(np.zeros((2,samples)))
  af=np.concatenate(sa,axis=1)if sa else self._generar_fallback_simple(duracion_sec);return af,metadatos
 def _generar_hibrido(self,config:ConfiguracionAuroraUnificada,duracion_sec:float,metadatos:Dict[str,Any])->Tuple[np.ndarray,Dict[str,Any]]:nm=len(self.motores_disponibles);return self._generar_en_capas(config,duracion_sec,metadatos)if nm>=3 and config.calidad_objetivo=="maxima"else self._generar_paralelo(config,duracion_sec,metadatos)if nm>=2 else self._generar_secuencial(config,duracion_sec,metadatos)
 def _seleccionar_motor_principal(self,config:ConfiguracionAuroraUnificada)->Optional[str]:po={"concentracion":["neuromix_v27","hypermod_v32"],"relajacion":["harmonic_essence_v34","neuromix_v27"],"creatividad":["harmonic_essence_v34","neuromix_v27"],"meditacion":["hypermod_v32","neuromix_v27"]};ol=config.objetivo.lower();mp=config.motores_preferidos
  for m in mp:
   if m in self.motores_disponibles:return m
  for ok,lm in po.items():
   if ok in ol:
    for m in lm:
     if m in self.motores_disponibles:return m
  return list(self.motores_disponibles.keys())[0]if self.motores_disponibles else None
 def _adaptar_config_para_motor(self,config:ConfiguracionAuroraUnificada,nombre_motor:str,capa_config:Dict[str,Any]=None)->Dict[str,Any]:
  cb={"objetivo":config.objetivo,"duracion_min":config.duracion_min,"sample_rate":config.sample_rate,"intensidad":config.intensidad,"estilo":config.estilo,"neurotransmisor_preferido":config.neurotransmisor_preferido,"calidad_objetivo":config.calidad_objetivo,"normalizar":config.normalizar,"contexto_uso":config.contexto_uso}
  if hasattr(config,'template_personalizado')and config.template_personalizado:cb["template_objetivo"]=config.template_personalizado
  if hasattr(config,'perfil_campo_personalizado')and config.perfil_campo_personalizado:cb["perfil_campo"]=config.perfil_campo_personalizado
  if hasattr(config,'secuencia_fases_personalizada')and config.secuencia_fases_personalizada:cb["secuencia_fases"]=config.secuencia_fases_personalizada
  if hasattr(config,'parametros_neuroacusticos')and config.parametros_neuroacusticos:cb.update({"beat_primario":config.parametros_neuroacusticos.get("beat_primario"),"beat_secundario":config.parametros_neuroacusticos.get("beat_secundario"),"armonicos":config.parametros_neuroacusticos.get("armonicos"),"coherencia_objetivo":config.parametros_neuroacusticos.get("coherencia_objetivo")})
  if hasattr(config,'efectos_psicodelicos')and config.efectos_psicodelicos:cb.update({"frecuencia_fundamental":config.efectos_psicodelicos.get("frecuencia_fundamental"),"modulacion_depth":config.efectos_psicodelicos.get("modulacion_depth"),"modulacion_rate":config.efectos_psicodelicos.get("modulacion_rate"),"intensidad_efecto":config.efectos_psicodelicos.get("intensidad_efecto")})
  if hasattr(config,'frecuencia_base_psicodelica')and config.frecuencia_base_psicodelica:cb["frecuencia_base_psicodelica"]=config.frecuencia_base_psicodelica
  if"neuromix"in nombre_motor:cb.update({"wave_type":"hybrid","processing_mode":"aurora_integrated"})
  elif"hypermod"in nombre_motor:cb.update({"preset_emocional":config.objetivo,"validacion_cientifica":True,"optimizacion_neuroacustica":True})
  elif"harmonic"in nombre_motor:cb.update({"texture_type":self._mapear_estilo_a_textura(config.estilo),"precision_cientifica":True})
  if capa_config:cb.update(capa_config.get("config_adicional",{}))
  return cb
 def _mapear_estilo_a_textura(self,estilo:str)->str:m={"sereno":"relaxation","crystalline":"crystalline","organico":"organic","etereo":"ethereal","tribal":"tribal","mistico":"consciousness"};return m.get(estilo.lower(),"organic")
 def _combinar_capas(self,audio1:np.ndarray,audio2:np.ndarray)->np.ndarray:ml=min(audio1.shape[1],audio2.shape[1]);a1c=audio1[:,:ml];a2c=audio2[:,:ml];c=a1c+a2c;mv=np.max(np.abs(c));return c*(0.85/mv)if mv>0.95 else c
 def _generar_fallback_simple(self,duracion_sec:float)->np.ndarray:s=int(44100*duracion_sec);t=np.linspace(0,duracion_sec,s);w=0.3*np.sin(2*np.pi*10.0*t);fs=min(2048,s//4);if s>fs*2:fi=np.linspace(0,1,fs);fo=np.linspace(1,0,fs);w[:fs]*=fi;w[-fs:]*=fo;return np.stack([w,w])

class AuroraDirectorV7Integrado:
 def __init__(self,auto_detectar:bool=True):self.version="Aurora Director V7 Integrado - Optimizado con Sync Híbrido";self.detector=DetectorComponentesAvanzado();self.componentes:Dict[str,ComponenteAurora]={};self.orquestador:Optional[OrquestadorMultiMotor]=None;self.objective_manager:Optional[Any]=None;self.sync_scheduler:Optional[Any]=None;self.psychedelic_effects:Dict[str,Any]={};self.emotion_profiles_cache:Dict[str,Any]={};self.field_profiles_cache:Dict[str,Any]={};self.stats={"experiencias_generadas":0,"tiempo_total_generacion":0.0,"estrategias_utilizadas":{},"objetivos_procesados":{},"errores_manejados":0,"fallbacks_utilizados":0,"calidad_promedio":0.0,"motores_utilizados":{},"sesiones_activas":0,"objective_manager_utilizaciones":0,"templates_utilizados":{},"perfiles_campo_utilizados":{},"secuencias_fases_utilizadas":{},"emotion_style_utilizaciones":0,"efectos_psicodelicos_aplicados":0,"field_profiles_avanzados_utilizados":0,"integraciones_exitosas":0,"sync_hibrido_utilizaciones":0,"coherencia_global_promedio":0.0};self.cache_configuraciones={};self.cache_resultados={}
  if auto_detectar:self._inicializar_sistema()
 def _inicializar_sistema(self):logger.info(f"🌟 Inicializando {self.version}");self.componentes=self.detector.detectar_todos();self.orquestador=OrquestadorMultiMotor(self.componentes);self.psychedelic_effects=self._cargar_efectos_psicodelicos()
  if"objective_manager_unificado"in self.componentes and self.componentes["objective_manager_unificado"].disponible:self.objective_manager=self.componentes["objective_manager_unificado"].instancia
  if"sync_scheduler_v7"in self.componentes and self.componentes["sync_scheduler_v7"].disponible:self.sync_scheduler=self.componentes["sync_scheduler_v7"].instancia
  self._log_estado_sistema()
 def _cargar_efectos_psicodelicos(self)->Dict[str,Any]:
  try:ep=Path("psychedelic_effects_tables.json");return json.load(open(ep,'r',encoding='utf-8'))if ep.exists()else{}
  except Exception:return{}
 def _log_estado_sistema(self):m=len([c for c in self.componentes.values()if c.tipo==TipoComponente.MOTOR]);g=len([c for c in self.componentes.values()if c.tipo==TipoComponente.GESTOR_INTELIGENCIA]);p=len([c for c in self.componentes.values()if c.tipo==TipoComponente.PIPELINE]);om=len([c for c in self.componentes.values()if c.tipo==TipoComponente.OBJECTIVE_MANAGER]);ss=len([c for c in self.componentes.values()if c.tipo==TipoComponente.SYNC_SCHEDULER]);logger.info(f"🔧 Componentes: 🎵{m} 🧠{g} 🔄{p} 🎯{om} 🌟{ss}")
 def crear_experiencia(self,objetivo:str=None,**kwargs)->ResultadoAuroraIntegrado:
  st=time.time()
  try:
   c=self._crear_configuracion_optimizada(objetivo,kwargs);pr=c.validar()
   if pr:logger.warning(f"⚠️ Problemas: {pr}")
   e=self._seleccionar_estrategia_optima(c);r=self._ejecutar_estrategia(e,c);r=self._post_procesar_resultado(r,c);tt=time.time()-st;self._actualizar_estadisticas(c,r,tt);return r
  except Exception as e:logger.error(f"❌ Error: {e}");self.stats["errores_manejados"]+=1;return self._crear_resultado_emergencia(objetivo or"emergencia",str(e))
 def _crear_configuracion_optimizada(self,objetivo:str,kwargs:Dict[str,Any])->ConfiguracionAuroraUnificada:
  ck=f"{objetivo}_{hash(str(sorted(kwargs.items())))}"
  if ck in self.cache_configuraciones:return self.cache_configuraciones[ck]
  ci={"concentracion":{"intensidad":"media","estilo":"crystalline","neurotransmisor_preferido":"acetilcolina","modo_orquestacion":ModoOrquestacion.LAYERED,"motores_preferidos":["neuromix_v27","hypermod_v32"]},"claridad_mental":{"intensidad":"media","estilo":"crystalline","neurotransmisor_preferido":"dopamina","modo_orquestacion":ModoOrquestacion.PARALELO,"motores_preferidos":["neuromix_v27"]},"enfoque":{"intensidad":"intenso","estilo":"crystalline","neurotransmisor_preferido":"norepinefrina","modo_orquestacion":ModoOrquestacion.LAYERED},"relajacion":{"intensidad":"suave","estilo":"sereno","neurotransmisor_preferido":"gaba","modo_orquestacion":ModoOrquestacion.HYBRID,"motores_preferidos":["harmonic_essence_v34","neuromix_v27"]},"meditacion":{"intensidad":"suave","estilo":"mistico","neurotransmisor_preferido":"serotonina","duracion_min":35,"modo_orquestacion":ModoOrquestacion.LAYERED,"motores_preferidos":["hypermod_v32","harmonic_essence_v34"]},"gratitud":{"intensidad":"suave","estilo":"sereno","neurotransmisor_preferido":"oxitocina","modo_orquestacion":ModoOrquestacion.HYBRID},"creatividad":{"intensidad":"media","estilo":"organico","neurotransmisor_preferido":"anandamida","modo_orquestacion":ModoOrquestacion.LAYERED,"motores_preferidos":["harmonic_essence_v34","neuromix_v27"]},"inspiracion":{"intensidad":"media","estilo":"organico","neurotransmisor_preferido":"dopamina","modo_orquestacion":ModoOrquestacion.HYBRID},"sanacion":{"intensidad":"suave","estilo":"sereno","neurotransmisor_preferido":"endorfina","duracion_min":45,"calidad_objetivo":"maxima","modo_orquestacion":ModoOrquestacion.LAYERED}}
  ol=objetivo.lower()if objetivo else"relajacion";cd=self._detectar_contexto_objetivo(ol);cb={}
  for k,c in ci.items():
   if k in ol:cb=c.copy();break
  cb.update(cd);cf={"objetivo":objetivo or"relajacion",**cb,**kwargs};cf.setdefault("usar_objective_manager",OM_AVAIL and self.objective_manager is not None)
  if cf.get("calidad_objetivo")=="maxima"and SYNC_SCHEDULER_HIBRIDO_AVAILABLE and self.sync_scheduler:cf.setdefault("habilitar_sync_hibrido",True);cf.setdefault("modo_orquestacion",ModoOrquestacion.SYNC_HIBRIDO)
  config=ConfiguracionAuroraUnificada(**cf);self.cache_configuraciones[ck]=config;return config
 def _detectar_contexto_objetivo(self,objetivo:str)->Dict[str,Any]:
  c={}
  if any(p in objetivo for p in["profundo","intenso","fuerte"]):c["intensidad"]="intenso"
  elif any(p in objetivo for p in["suave","ligero","sutil"]):c["intensidad"]="suave"
  if any(p in objetivo for p in["rapido","corto","breve"]):c["duracion_min"]=10
  elif any(p in objetivo for p in["largo","extenso","profundo"]):c["duracion_min"]=45
  if any(p in objetivo for p in["trabajo","oficina","estudio"]):c["contexto_uso"]="trabajo"
  elif any(p in objetivo for p in["dormir","noche","sueño"]):c["contexto_uso"]="sueño"
  elif any(p in objetivo for p in["meditacion","espiritual"]):c["contexto_uso"]="meditacion"
  if any(p in objetivo for p in["terapeutico","clinico","medicinal"]):c["calidad_objetivo"]="maxima"
  if any(p in objetivo for p in["hibrido","sync","coherencia","estructura"]):c["habilitar_sync_hibrido"]=True;c["modo_orquestacion"]=ModoOrquestacion.SYNC_HIBRIDO
  return c
 def _seleccionar_estrategia_optima(self,config:ConfiguracionAuroraUnificada)->EstrategiaGeneracion:
  if config.estrategia_preferida:ed=self._obtener_estrategias_disponibles();return config.estrategia_preferida if config.estrategia_preferida in ed else EstrategiaGeneracion.FALLBACK_PROGRESIVO
  m=len([c for c in self.componentes.values()if c.tipo==TipoComponente.MOTOR and c.disponible]);g=len([c for c in self.componentes.values()if c.tipo==TipoComponente.GESTOR_INTELIGENCIA and c.disponible]);p=len([c for c in self.componentes.values()if c.tipo==TipoComponente.PIPELINE and c.disponible]);om=len([c for c in self.componentes.values()if c.tipo==TipoComponente.OBJECTIVE_MANAGER and c.disponible]);ss=len([c for c in self.componentes.values()if c.tipo==TipoComponente.SYNC_SCHEDULER and c.disponible])
  if(ss>=1 and m>=2 and config.habilitar_sync_hibrido and(config.modo_orquestacion==ModoOrquestacion.SYNC_HIBRIDO or config.calidad_objetivo=="maxima")):return EstrategiaGeneracion.SYNC_SCHEDULER_HIBRIDO
  if(config.usar_objective_manager and om>=1 and m>=2 and config.calidad_objetivo=="maxima"):return EstrategiaGeneracion.OBJECTIVE_MANAGER_DRIVEN
  if(m>=3 and g>=2 and p>=1 and config.calidad_objetivo=="maxima"):return EstrategiaGeneracion.AURORA_ORQUESTADO
  elif m>=2 and config.modo_orquestacion in[ModoOrquestacion.LAYERED,ModoOrquestacion.HYBRID]:return EstrategiaGeneracion.MULTI_MOTOR
  elif g>=1 and m>=1:return EstrategiaGeneracion.INTELIGENCIA_ADAPTIVA
  elif m>=1:return EstrategiaGeneracion.MOTOR_ESPECIALIZADO
  else:return EstrategiaGeneracion.FALLBACK_PROGRESIVO
 def _ejecutar_estrategia(self,estrategia:EstrategiaGeneracion,config:ConfiguracionAuroraUnificada)->ResultadoAuroraIntegrado:
  ds=config.duracion_min*60
  if estrategia==EstrategiaGeneracion.SYNC_SCHEDULER_HIBRIDO:return self._estrategia_sync_scheduler_hibrido(config,ds)
  elif estrategia==EstrategiaGeneracion.AURORA_ORQUESTADO:return self._estrategia_aurora_orquestado_optimizada(config,ds)
  elif estrategia==EstrategiaGeneracion.MULTI_MOTOR:return self._estrategia_multi_motor(config,ds)
  elif estrategia==EstrategiaGeneracion.INTELIGENCIA_ADAPTIVA:return self._estrategia_inteligencia_adaptiva(config,ds)
  elif estrategia==EstrategiaGeneracion.MOTOR_ESPECIALIZADO:return self._estrategia_motor_especializado(config,ds)
  elif estrategia==EstrategiaGeneracion.OBJECTIVE_MANAGER_DRIVEN:return self._estrategia_objective_manager_driven(config,ds)
  else:return self._estrategia_fallback_progresivo(config,ds)
 def _estrategia_sync_scheduler_hibrido(self,config:ConfiguracionAuroraUnificada,duracion_sec:float)->ResultadoAuroraIntegrado:
  if not self.sync_scheduler:return self._estrategia_aurora_orquestado_optimizada(config,duracion_sec)
  try:
   config_optimizado=self._aplicar_inteligencia_gestores_optimizada(config);config_optimizado.modo_orquestacion=ModoOrquestacion.SYNC_HIBRIDO;audio_data,metadatos_orquestacion=self.orquestador.generar_audio_orquestado(config_optimizado,duracion_sec)
   if"quality_pipeline"in self.componentes:pipeline=self.componentes["quality_pipeline"].instancia;audio_data=pipeline.validar_y_normalizar(audio_data)
   calidad_score,coherencia_neuroacustica,efectividad_terapeutica=self._calcular_metricas_calidad(audio_data);resultado_sync_hibrido=metadatos_orquestacion.get("resultado_sync_hibrido");validacion_sync_scheduler=metadatos_orquestacion.get("validacion_sync_scheduler");estructura_fases=metadatos_orquestacion.get("estructura_fases_utilizada")
   resultado=ResultadoAuroraIntegrado(audio_data=audio_data,metadatos={"estrategia":"sync_scheduler_hibrido","orquestacion":metadatos_orquestacion,"sync_hibrido_aplicado":True,"recursos_integrados":self._obtener_recursos_integrados_aplicados(config_optimizado),"pipeline_calidad":"quality_pipeline"in self.componentes,"sync_scheduler_version":resultado_sync_hibrido.get("sync_scheduler_version","unknown")if resultado_sync_hibrido else"unknown"},estrategia_usada=EstrategiaGeneracion.SYNC_SCHEDULER_HIBRIDO,modo_orquestacion=ModoOrquestacion.SYNC_HIBRIDO,componentes_usados=metadatos_orquestacion.get("motores_utilizados",[])+["sync_scheduler_v7"],tiempo_generacion=0.0,calidad_score=calidad_score,coherencia_neuroacustica=coherencia_neuroacustica,efectividad_terapeutica=efectividad_terapeutica,configuracion=config_optimizado,resultado_sync_hibrido=resultado_sync_hibrido,validacion_sync_scheduler=validacion_sync_scheduler,estructura_fases_utilizada=estructura_fases)
   resultado=self._enriquecer_metadatos_generacion(resultado,config_optimizado);self.stats["sync_hibrido_utilizaciones"]+=1
   if resultado_sync_hibrido and"coherencia_global"in resultado_sync_hibrido:coherencia_global=resultado_sync_hibrido["coherencia_global"];total_usos=self.stats["sync_hibrido_utilizaciones"];promedio_actual=self.stats["coherencia_global_promedio"];self.stats["coherencia_global_promedio"]=((promedio_actual*(total_usos-1))+coherencia_global)/total_usos
   self.stats["integraciones_exitosas"]+=1;return resultado
  except Exception as e:return self._estrategia_aurora_orquestado_optimizada(config,duracion_sec)
 def _estrategia_objective_manager_driven(self,config:ConfiguracionAuroraUnificada,duracion_sec:float)->ResultadoAuroraIntegrado:
  if not self.objective_manager:return self._estrategia_aurora_orquestado_optimizada(config,duracion_sec)
  try:
   rom=self.objective_manager.procesar_objetivo_completo(config.objetivo,{"duracion_min":config.duracion_min,"intensidad":config.intensidad,"estilo":config.estilo,"contexto_uso":config.contexto_uso,"perfil_usuario":config.perfil_usuario,"calidad_objetivo":config.calidad_objetivo});co=self._aplicar_resultado_objective_manager(config,rom);ad,mo=self.orquestador.generar_audio_orquestado(co,duracion_sec)
   if"quality_pipeline"in self.componentes:p=self.componentes["quality_pipeline"].instancia;ad=p.validar_y_normalizar(ad)
   cs,coh,ef=self._calcular_metricas_calidad(ad);r=ResultadoAuroraIntegrado(audio_data=ad,metadatos={"estrategia":"objective_manager_driven","orquestacion":mo,"objective_manager_usado":True,"pipeline_calidad":"quality_pipeline"in self.componentes},estrategia_usada=EstrategiaGeneracion.OBJECTIVE_MANAGER_DRIVEN,modo_orquestacion=config.modo_orquestacion,componentes_usados=mo.get("motores_utilizados",[])+["objective_manager_unificado"],tiempo_generacion=0.0,calidad_score=cs,coherencia_neuroacustica=coh,efectividad_terapeutica=ef,configuracion=config,resultado_objective_manager=rom,template_utilizado=rom.get("template_utilizado"),perfil_campo_utilizado=rom.get("perfil_campo_utilizado"),secuencia_fases_utilizada=rom.get("secuencia_fases_utilizada"));return r
  except Exception as e:return self._estrategia_aurora_orquestado_optimizada(config,duracion_sec)
 def _aplicar_resultado_objective_manager(self,config:ConfiguracionAuroraUnificada,resultado_om:Dict[str,Any])->ConfiguracionAuroraUnificada:cm=resultado_om.get("configuracion_motor",{});[setattr(config,k,v)for k,v in cm.items()if hasattr(config,k)and v is not None];config.template_personalizado=resultado_om.get("template_utilizado");config.perfil_campo_personalizado=resultado_om.get("perfil_campo_utilizado");config.secuencia_fases_personalizada=resultado_om.get("secuencia_fases_utilizada");return config
 def _estrategia_aurora_orquestado_optimizada(self,config:ConfiguracionAuroraUnificada,duracion_sec:float)->ResultadoAuroraIntegrado:
  co=self._aplicar_inteligencia_gestores_optimizada(config);ad,mo=self.orquestador.generar_audio_orquestado(co,duracion_sec)
  if hasattr(co,'efectos_psicodelicos')and co.efectos_psicodelicos:ad=self._aplicar_efectos_psicodelicos_audio(ad,co.efectos_psicodelicos);self.stats["efectos_psicodelicos_aplicados"]+=1
  if"quality_pipeline"in self.componentes:p=self.componentes["quality_pipeline"].instancia;ad=p.validar_y_normalizar(ad)
  cs,coh,ef=self._calcular_metricas_calidad(ad);r=ResultadoAuroraIntegrado(audio_data=ad,metadatos={"estrategia":"aurora_orquestado_optimizado","orquestacion":mo,"recursos_integrados":self._obtener_recursos_integrados_aplicados(co),"pipeline_calidad":"quality_pipeline"in self.componentes},estrategia_usada=EstrategiaGeneracion.AURORA_ORQUESTADO,modo_orquestacion=config.modo_orquestacion,componentes_usados=mo.get("motores_utilizados",[]),tiempo_generacion=0.0,calidad_score=cs,coherencia_neuroacustica=coh,efectividad_terapeutica=ef,configuracion=config);r=self._enriquecer_metadatos_generacion(r,co);self.stats["integraciones_exitosas"]+=1;return r
 def _aplicar_inteligencia_gestores_optimizada(self,config:ConfiguracionAuroraUnificada)->ConfiguracionAuroraUnificada:
  co=config;ra=[]
  if config.usar_objective_manager and self.objective_manager:
   try:
    rom=self.objective_manager.procesar_objetivo_completo(config.objetivo,{"duracion_min":config.duracion_min,"intensidad":config.intensidad,"estilo":config.estilo,"contexto_uso":config.contexto_uso,"perfil_usuario":config.perfil_usuario,"calidad_objetivo":config.calidad_objetivo});co=self._aplicar_resultado_objective_manager(co,rom);self.stats["objective_manager_utilizaciones"]+=1;ra.append("objective_manager")
   except Exception:pass
  co=self._aplicar_emotion_style_profiles(co);ra.append("emotion_style_profiles")if hasattr(co,'metadatos_emocionales')and co.metadatos_emocionales else None;co=self._aplicar_field_profiles_avanzado(co);ra.append("field_profiles_avanzado")if hasattr(co,'parametros_neuroacusticos')and co.parametros_neuroacusticos else None;co=self._aplicar_efectos_psicodelicos(co);ra.append("psychedelic_effects")if hasattr(co,'efectos_psicodelicos')and co.efectos_psicodelicos else None;ra.append("objective_manager")if config.template_personalizado else None;ra.append("sync_scheduler_hibrido")if hasattr(config,'habilitar_sync_hibrido')and config.habilitar_sync_hibrido else None;co.configuracion_enriquecida=True;return co
 def _aplicar_emotion_style_profiles(self,config:ConfiguracionAuroraUnificada)->ConfiguracionAuroraUnificada:
  if"emotion_style_profiles"not in self.componentes:return config
  try:
   em=self.componentes["emotion_style_profiles"].instancia;re=em.procesar_objetivo(config.objetivo,{"duracion_min":config.duracion_min,"intensidad":config.intensidad,"contexto_uso":config.contexto_uso})
   if"error"not in re:
    if"neurotransmisores"in re and not config.neurotransmisor_preferido:nt=re["neurotransmisores"];config.neurotransmisor_preferido=max(nt.items(),key=lambda x:x[1])[0]if nt else None
    if re.get("estilo")and not config.estilo:config.estilo=re["estilo"]
    config.metadatos_emocionales={"preset_emocional":re.get("preset_emocional"),"coherencia_neuroacustica":re.get("coherencia_neuroacustica"),"efectos_esperados":re.get("recomendaciones_uso",[]),"neurotransmisores_detectados":re.get("neurotransmisores",{}),"modo_aplicado":"emotion_style_v7"};self.stats["emotion_style_utilizaciones"]+=1
  except Exception:pass
  return config
 def _aplicar_field_profiles_avanzado(self,config:ConfiguracionAuroraUnificada)->ConfiguracionAuroraUnificada:
  if"field_profiles"not in self.componentes:return config
  try:
   pm=self.componentes["field_profiles"].instancia;p=pm.obtener_perfil(config.objetivo)
   if p:
    if not config.neurotransmisor_preferido and hasattr(p,'neurotransmisores_principales'):pr=p.neurotransmisores_principales;config.neurotransmisor_preferido=max(pr.items(),key=lambda x:x[1])[0]if pr else None
    if hasattr(p,'configuracion_neuroacustica'):nc=p.configuracion_neuroacustica;config.parametros_neuroacusticos={"beat_primario":nc.beat_primario,"beat_secundario":nc.beat_secundario,"armonicos":nc.armonicos,"modulacion_amplitude":nc.modulacion_amplitude,"modulacion_frecuencia":nc.modulacion_frecuencia,"coherencia_objetivo":p.calcular_coherencia_neuroacustica(),"evolucion_activada":nc.evolucion_activada,"movimiento_3d":nc.movimiento_3d,"perfil_aplicado":p.nombre}
    if hasattr(p,'duracion_optima_min')and p.duracion_optima_min>config.duracion_min:config.duracion_min=min(p.duracion_optima_min,config.duracion_min+10)
    if hasattr(p,'style')and not config.estilo:config.estilo=p.style
    if hasattr(p,'nivel_activacion')and config.intensidad=="media":mi={"SUTIL":"suave","MODERADO":"media","INTENSO":"intenso","PROFUNDO":"intenso","TRASCENDENTE":"intenso"};config.intensidad=mi.get(p.nivel_activacion.value,config.intensidad)if hasattr(p.nivel_activacion,'value')else config.intensidad
    self.stats["field_profiles_avanzados_utilizados"]+=1
   s=pm.recomendar_secuencia_perfiles(config.objetivo,config.duracion_min);config.secuencia_perfiles=s if s and len(s)>1 else config.secuencia_perfiles
  except Exception:pass
  return config
 def _aplicar_efectos_psicodelicos(self,config:ConfiguracionAuroraUnificada)->ConfiguracionAuroraUnificada:
  if not self.psychedelic_effects or"pe"not in self.psychedelic_effects:return config
  try:
   ed=self.psychedelic_effects["pe"];ol=config.objetivo.lower();mo={"expansion":["Psilocibina","LSD"],"creatividad":["Psilocibina","LSD","DMT"],"meditacion":["Psilocibina","5-MeO-DMT"],"sanacion":["Psilocibina","MDMA","Ayahuasca"],"conexion":["MDMA","Ayahuasca"],"introspection":["Psilocibina","LSD"],"espiritual":["Ayahuasca","5-MeO-DMT","San_Pedro"],"transformacion":["Iboga","Ayahuasca"],"energia":["San_Pedro","DMT"],"calma":["THC","Ketamina"],"relajacion":["THC","Ketamina"],"concentracion":["Psilocibina"],"claridad":["Psilocibina","LSD"],"flujo":["Psilocibina","LSD"],"flow":["Psilocibina","LSD"],"gratitud":["MDMA","San_Pedro"],"amor":["MDMA","Ayahuasca"],"compasion":["MDMA","Ayahuasca"]};es=None;ne=None
   for pc,ef in mo.items():
    if pc in ol:
     for e in ef:
      if e in ed:es=ed[e];ne=e;break
     if es:break
   if es:
    config.efectos_psicodelicos={"sustancia_referencia":ne,"efecto_principal":es.get("effect","unknown"),"tipo":es.get("type","unknown"),"style_override":es.get("style",""),"intensidad_base":es.get("intensity","media")}
    if"freq"in es:config.frecuencia_base_psicodelica=es["freq"]
    if"p7"in es:p7c=es["p7"];config.efectos_psicodelicos.update({"frecuencia_fundamental":p7c.get("ff"),"frecuencia_carrier":p7c.get("fc"),"armonicos":p7c.get("fh",[]),"modulacion_depth":p7c.get("md"),"modulacion_rate":p7c.get("mr"),"duracion_target":p7c.get("dt"),"intensidad_efecto":p7c.get("ei"),"receptores":p7c.get("rp",[]),"estudios_referencia":p7c.get("rr",[])})
    if"nt"in es and not config.neurotransmisor_preferido:ntp=es["nt"];config.neurotransmisor_preferido=ntp[0].lower()if ntp else None
    if es.get("style")and config.estilo=="sereno":config.estilo=es["style"]
  except Exception:pass
  return config
 def _aplicar_efectos_psicodelicos_audio(self,audio:np.ndarray,efectos_config:Dict[str,Any])->np.ndarray:
  try:
   ap=audio.copy();ea=[]
   if"modulacion_depth"in efectos_config and"modulacion_rate"in efectos_config:d=efectos_config["modulacion_depth"];r=efectos_config["modulacion_rate"];s=ap.shape[1];t=np.linspace(0,s/44100,s);m=1.0+d*np.sin(2*np.pi*r*t);[setattr(ap,i,ap[i]*m)for i in range(ap.shape[0])];ea.append("modulacion_profunda")if d and r else None
   if"frecuencia_fundamental"in efectos_config:ff=efectos_config["frecuencia_fundamental"];i=efectos_config.get("intensidad_efecto",0.3);s=ap.shape[1];t=np.linspace(0,s/44100,s);po=float(i)*np.sin(2*np.pi*float(ff)*t);[setattr(ap,ch,ap[ch]+po*0.1)for ch in range(ap.shape[0])];ea.append("frecuencia_fundamental")if ff and i else None
   if"armonicos"in efectos_config and efectos_config["armonicos"]:ar=efectos_config["armonicos"];i=efectos_config.get("intensidad_efecto",0.2);s=ap.shape[1];t=np.linspace(0,s/44100,s);[[setattr(ap,ch,ap[ch]+aa*np.sin(2*np.pi*float(fa)*t)*0.05)for ch in range(ap.shape[0])]for ind,fa in enumerate(ar[:3])if fa for aa in[float(i)*(0.1/(ind+1))]];ea.append("armonicos")
   return np.clip(ap,-1.0,1.0)
  except Exception:return audio
 def _enriquecer_metadatos_generacion(self,resultado:ResultadoAuroraIntegrado,config:ConfiguracionAuroraUnificada)->ResultadoAuroraIntegrado:
  if hasattr(config,'metadatos_emocionales')and config.metadatos_emocionales:resultado.metadatos["emotion_style"]=config.metadatos_emocionales
  if hasattr(config,'parametros_neuroacusticos')and config.parametros_neuroacusticos:resultado.metadatos["field_profile"]=config.parametros_neuroacusticos
  if hasattr(config,'efectos_psicodelicos')and config.efectos_psicodelicos:resultado.metadatos["psychedelic_effects"]=config.efectos_psicodelicos
  if hasattr(config,'secuencia_perfiles')and config.secuencia_perfiles:resultado.metadatos["profile_sequence"]=config.secuencia_perfiles
  if hasattr(config,'configuracion_enriquecida')and config.configuracion_enriquecida:resultado.metadatos["configuracion_enriquecida"]=True;resultado.metadatos["recursos_aplicados"]=self._obtener_recursos_integrados_aplicados(config)
  if hasattr(config,'habilitar_sync_hibrido')and config.habilitar_sync_hibrido:resultado.metadatos["sync_hibrido_habilitado"]=True;resultado.metadatos["parametros_sync_hibrido"]=config.parametros_sync_hibrido if hasattr(config,'parametros_sync_hibrido')and config.parametros_sync_hibrido else None
  return resultado
 def _obtener_recursos_integrados_aplicados(self,config:ConfiguracionAuroraUnificada)->List[str]:
  r=[]
  if hasattr(config,'metadatos_emocionales')and config.metadatos_emocionales:r.append("emotion_style_profiles")
  if hasattr(config,'parametros_neuroacusticos')and config.parametros_neuroacusticos:r.append("field_profiles_avanzado")
  if hasattr(config,'efectos_psicodelicos')and config.efectos_psicodelicos:r.append("psychedelic_effects")
  if config.template_personalizado:r.append("objective_manager")
  if hasattr(config,'habilitar_sync_hibrido')and config.habilitar_sync_hibrido:r.append("sync_scheduler_hibrido")
  return r
 def _estrategia_multi_motor(self,config:ConfiguracionAuroraUnificada,duracion_sec:float)->ResultadoAuroraIntegrado:ad,mo=self.orquestador.generar_audio_orquestado(config,duracion_sec);cs,coh,ef=self._calcular_metricas_calidad(ad);return ResultadoAuroraIntegrado(audio_data=ad,metadatos={"estrategia":"multi_motor","orquestacion":mo},estrategia_usada=EstrategiaGeneracion.MULTI_MOTOR,modo_orquestacion=config.modo_orquestacion,componentes_usados=mo.get("motores_utilizados",[]),tiempo_generacion=0.0,calidad_score=cs,coherencia_neuroacustica=coh,efectividad_terapeutica=ef,configuracion=config)
 def _estrategia_inteligencia_adaptiva(self,config:ConfiguracionAuroraUnificada,duracion_sec:float)->ResultadoAuroraIntegrado:co=self._aplicar_inteligencia_gestores_optimizada(config);mp=self._seleccionar_motor_principal(co);ad,cu=self._generar_con_motor_principal(mp,co,duracion_sec)if mp else(self._generar_audio_fallback(duracion_sec),["fallback"]);cs,coh,ef=self._calcular_metricas_calidad(ad);return ResultadoAuroraIntegrado(audio_data=ad,metadatos={"estrategia":"inteligencia_adaptiva","motor_principal":mp,"configuracion_optimizada":True},estrategia_usada=EstrategiaGeneracion.INTELIGENCIA_ADAPTIVA,modo_orquestacion=ModoOrquestacion.HYBRID,componentes_usados=cu,tiempo_generacion=0.0,calidad_score=cs,coherencia_neuroacustica=coh,efectividad_terapeutica=ef,configuracion=config)
 def _estrategia_motor_especializado(self,config:ConfiguracionAuroraUnificada,duracion_sec:float)->ResultadoAuroraIntegrado:mp=self._seleccionar_motor_principal(config);ad,cu=self._generar_con_motor_principal(mp,config,duracion_sec)if mp else(self._generar_audio_fallback(duracion_sec),["fallback"]);cs,coh,ef=self._calcular_metricas_calidad(ad);return ResultadoAuroraIntegrado(audio_data=ad,metadatos={"estrategia":"motor_especializado","motor_utilizado":mp},estrategia_usada=EstrategiaGeneracion.MOTOR_ESPECIALIZADO,modo_orquestacion=ModoOrquestacion.HYBRID,componentes_usados=cu,tiempo_generacion=0.0,calidad_score=cs,coherencia_neuroacustica=coh,efectividad_terapeutica=ef,configuracion=config)
 def _estrategia_fallback_progresivo(self,config:ConfiguracionAuroraUnificada,duracion_sec:float)->ResultadoAuroraIntegrado:self.stats["fallbacks_utilizados"]+=1;ad=self._generar_audio_fallback(duracion_sec);cs,coh,ef=self._calcular_metricas_calidad(ad);return ResultadoAuroraIntegrado(audio_data=ad,metadatos={"estrategia":"fallback_progresivo","motivo":"componentes_insuficientes"},estrategia_usada=EstrategiaGeneracion.FALLBACK_PROGRESIVO,modo_orquestacion=ModoOrquestacion.HYBRID,componentes_usados=["fallback_interno"],tiempo_generacion=0.0,calidad_score=cs,coherencia_neuroacustica=coh,efectividad_terapeutica=ef,configuracion=config)
 def _seleccionar_motor_principal(self,config:ConfiguracionAuroraUnificada)->Optional[str]:md=[n for n,c in self.componentes.items()if c.tipo==TipoComponente.MOTOR and c.disponible];pr={"concentracion":["neuromix_v27","hypermod_v32"],"claridad_mental":["neuromix_v27","hypermod_v32"],"enfoque":["neuromix_v27","hypermod_v32"],"relajacion":["harmonic_essence_v34","neuromix_v27"],"meditacion":["hypermod_v32","harmonic_essence_v34"],"creatividad":["harmonic_essence_v34","neuromix_v27"],"sanacion":["harmonic_essence_v34","hypermod_v32"]};ol=config.objetivo.lower();return next((m for m in config.motores_preferidos if m in md),None)or next((m for ok,lm in pr.items()if ok in ol for m in lm if m in md),None)or(md[0]if md else None)
 def _generar_con_motor_principal(self,motor_principal:str,config:ConfiguracionAuroraUnificada,duracion_sec:float)->Tuple[np.ndarray,List[str]]:m=self.componentes[motor_principal].instancia;cm=self._adaptar_configuracion_motor(config,motor_principal);ad=m.generar_audio(cm,duracion_sec);return ad,[motor_principal]
 def _adaptar_configuracion_motor(self,config:ConfiguracionAuroraUnificada,nombre_motor:str)->Dict[str,Any]:
  cb={"objetivo":config.objetivo,"duracion_min":config.duracion_min,"sample_rate":config.sample_rate,"intensidad":config.intensidad,"estilo":config.estilo,"neurotransmisor_preferido":config.neurotransmisor_preferido,"calidad_objetivo":config.calidad_objetivo,"normalizar":config.normalizar,"contexto_uso":config.contexto_uso}
  if hasattr(config,'template_personalizado')and config.template_personalizado:cb["template_objetivo"]=config.template_personalizado
  if hasattr(config,'perfil_campo_personalizado')and config.perfil_campo_personalizado:cb["perfil_campo"]=config.perfil_campo_personalizado
  if hasattr(config,'secuencia_fases_personalizada')and config.secuencia_fases_personalizada:cb["secuencia_fases"]=config.secuencia_fases_personalizada
  if hasattr(config,'parametros_neuroacusticos')and config.parametros_neuroacusticos:cb.update({"beat_primario":config.parametros_neuroacusticos.get("beat_primario"),"beat_secundario":config.parametros_neuroacusticos.get("beat_secundario"),"armonicos":config.parametros_neuroacusticos.get("armonicos"),"coherencia_objetivo":config.parametros_neuroacusticos.get("coherencia_objetivo"),"evolucion_activada":config.parametros_neuroacusticos.get("evolucion_activada"),"movimiento_3d":config.parametros_neuroacusticos.get("movimiento_3d")})
  if hasattr(config,'efectos_psicodelicos')and config.efectos_psicodelicos:cb.update({"frecuencia_fundamental":config.efectos_psicodelicos.get("frecuencia_fundamental"),"modulacion_depth":config.efectos_psicodelicos.get("modulacion_depth"),"modulacion_rate":config.efectos_psicodelicos.get("modulacion_rate"),"intensidad_efecto":config.efectos_psicodelicos.get("intensidad_efecto"),"sustancia_referencia":config.efectos_psicodelicos.get("sustancia_referencia"),"receptores":config.efectos_psicodelicos.get("receptores")})
  if hasattr(config,'frecuencia_base_psicodelica')and config.frecuencia_base_psicodelica:cb["frecuencia_base_psicodelica"]=config.frecuencia_base_psicodelica
  if"neuromix"in nombre_motor:cb.update({"wave_type":"hybrid","processing_mode":"aurora_integrated","quality_level":"therapeutic"if config.calidad_objetivo=="maxima"else"enhanced"})
  elif"hypermod"in nombre_motor:cb.update({"preset_emocional":config.objetivo,"validacion_cientifica":True,"optimizacion_neuroacustica":True,"modo_terapeutico":config.calidad_objetivo=="maxima"})
  elif"harmonic"in nombre_motor:cb.update({"texture_type":self._mapear_estilo_a_textura(config.estilo),"precision_cientifica":True,"auto_optimizar_coherencia":True})
  return cb
 def _mapear_estilo_a_textura(self,estilo:str)->str:m={"sereno":"relaxation","crystalline":"crystalline","organico":"organic","etereo":"ethereal","tribal":"tribal","mistico":"consciousness","neutro":"meditation"};return m.get(estilo.lower(),"organic")
 def _post_procesar_resultado(self,resultado:ResultadoAuroraIntegrado,config:ConfiguracionAuroraUnificada)->ResultadoAuroraIntegrado:
  a=resultado.audio_data
  if config.normalizar:mv=np.max(np.abs(a));tl=0.85 if config.calidad_objetivo=="maxima"else 0.80;a=np.clip(a*(tl/mv),-1.0,1.0)if mv>0 else a
  if config.aplicar_mastering:a=self._aplicar_mastering_basico(a)
  resultado.audio_data=a;resultado=self._aplicar_analisis_carmine(resultado,config);resultado.recomendaciones=self._generar_recomendaciones(resultado,config);resultado.proxima_sesion=self._generar_sugerencias_proxima_sesion(resultado,config);return resultado
 def _aplicar_mastering_basico(self,audio:np.ndarray)->np.ndarray:th=0.7;ra=3.0;[[setattr(audio,ch,np.where(np.abs(s)>th,np.sign(s)*(th+(np.abs(s)-th)/ra),s))for s in[audio[ch]]]for ch in range(audio.shape[0])];return audio
 def _aplicar_analisis_carmine(self,resultado:ResultadoAuroraIntegrado,config:ConfiguracionAuroraUnificada)->ResultadoAuroraIntegrado:
  try:
   if"carmine_analyzer_v21"in self.componentes:a=self.componentes["carmine_analyzer_v21"].instancia;ei=self._mapear_objetivo_a_intent_carmine(config.objetivo);cr=a.analyze_audio(resultado.audio_data,ei);resultado=self._actualizar_resultado_con_carmine(resultado,cr)
  except Exception:pass
  return resultado
 def _actualizar_resultado_con_carmine(self,resultado:ResultadoAuroraIntegrado,carmine_result:Any)->ResultadoAuroraIntegrado:resultado.calidad_score=max(resultado.calidad_score,carmine_result.score);resultado.coherencia_neuroacustica=getattr(carmine_result.neuro_metrics,'entrainment_effectiveness',resultado.coherencia_neuroacustica);resultado.efectividad_terapeutica=max(resultado.efectividad_terapeutica,carmine_result.therapeutic_score/100.0);resultado.metadatos["carmine_analysis"]={"score":carmine_result.score,"therapeutic_score":carmine_result.therapeutic_score,"quality_level":carmine_result.quality.value,"issues":carmine_result.issues,"suggestions":carmine_result.suggestions,"neuro_effectiveness":getattr(carmine_result.neuro_metrics,'entrainment_effectiveness',0.0),"binaural_strength":getattr(carmine_result.neuro_metrics,'binaural_strength',0.0),"gpt_summary":getattr(carmine_result,'gpt_summary',""),"correcciones_aplicadas":False};return resultado
 def _mapear_objetivo_a_intent_carmine(self,objetivo:str):m={"relajacion":"RELAXATION","concentracion":"FOCUS","claridad_mental":"FOCUS","enfoque":"FOCUS","meditacion":"MEDITATION","creatividad":"EMOTIONAL","sanacion":"RELAXATION","sueño":"SLEEP","energia":"ENERGY"};ol=objetivo.lower();intent=next((i for k,i in m.items()if k in ol),None);return getattr(__import__('Carmine_Analyzer').TherapeuticIntent,intent)if intent else None
 def _calcular_metricas_calidad(self,audio:np.ndarray)->Tuple[float,float,float]:
  if audio.size==0:return 0.0,0.0,0.0
  try:r=np.sqrt(np.mean(audio**2));p=np.max(np.abs(audio));cf=p/(r+1e-10);coh=float(np.nan_to_num(np.corrcoef(audio[0],audio[1])[0,1],0.5))if audio.ndim==2 and audio.shape[0]==2 else 0.8;fd=np.abs(np.fft.rfft(audio[0]if audio.ndim==2 else audio));ed=np.std(fd);fl=np.mean(fd)/(np.max(fd)+1e-10);cs=min(100,max(60,80+(1-min(p,1.0))*10+coh*10+fl*10));ef=min(1.0,max(0.6,0.7+coh*0.2+(1-min(p,1.0))*0.1));return float(cs),float(coh),float(ef)
  except Exception:return 75.0,0.75,0.75
 def _generar_recomendaciones(self,resultado:ResultadoAuroraIntegrado,config:ConfiguracionAuroraUnificada)->List[str]:
  r=[]
  if resultado.calidad_score<70:r.append("Considerar calidad 'maxima'")
  if resultado.coherencia_neuroacustica<0.7:r.append("Mejorar coherencia con más componentes")
  if resultado.efectividad_terapeutica<0.8:r.append("Incrementar duración")
  if len(resultado.componentes_usados)<2 and len(self.orquestador.motores_disponibles)>=2:r.append("Probar modo 'layered'")
  if resultado.resultado_objective_manager:cr=resultado.resultado_objective_manager.get("resultado_routing",{}).get("confianza",0.0);r.append("Especificar más detalles en objetivo")if cr<0.7 else None;r.append("Probar template específico")if not resultado.template_utilizado else None;r.append("OM en fallback - verificar componentes")if resultado.resultado_objective_manager.get("metadatos",{}).get("fallback_usado")else None
  ol=config.objetivo.lower();r.append("NeuroMix V27 optimizado para concentración")if"concentracion"in ol and"neuromix"not in resultado.componentes_usados else None;r.append("HarmonicEssence V34 excelente para relajación")if"relajacion"in ol and"harmonic"not in resultado.componentes_usados else None;r.append("Sistema optimizado disponible")if not hasattr(config,'configuracion_enriquecida')or not config.configuracion_enriquecida else None;r.append("Emotion Style Profiles disponible")if"emotion_style_profiles"in self.componentes and not hasattr(config,'metadatos_emocionales')else None;r.append("Efectos psicodélicos disponibles")if self.psychedelic_effects and not hasattr(config,'efectos_psicodelicos')else None
  if SYNC_SCHEDULER_HIBRIDO_AVAILABLE and self.sync_scheduler and not hasattr(config,'habilitar_sync_hibrido'):r.append("Sync Scheduler Híbrido disponible")
  elif hasattr(config,'habilitar_sync_hibrido')and config.habilitar_sync_hibrido and resultado.estrategia_usada!=EstrategiaGeneracion.SYNC_SCHEDULER_HIBRIDO:r.append("Sync Híbrido habilitado pero no utilizado")
  elif resultado.estrategia_usada==EstrategiaGeneracion.SYNC_SCHEDULER_HIBRIDO:coherencia_global=resultado.resultado_sync_hibrido.get('coherencia_global',0.0)if hasattr(resultado,'resultado_sync_hibrido')and resultado.resultado_sync_hibrido else 0.0;r.append("Optimizar parámetros de coherencia")if coherencia_global<0.7 else"Excelente coherencia híbrida"if coherencia_global>=0.9 else None
  if"carmine_analysis"in resultado.metadatos:cd=resultado.metadatos["carmine_analysis"];[r.append(f"Carmine: {i}")for i in cd.get("issues",[])if i not in r];[r.append(f"Optimización: {s}")for s in cd.get("suggestions",[])[:2]if s not in r];sc=cd.get("score",100);r.append("Regenerar con calidad máxima")if sc<70 else"Calidad aceptable"if sc<85 else None
  return[x for x in r if x]
 def _generar_sugerencias_proxima_sesion(self,resultado:ResultadoAuroraIntegrado,config:ConfiguracionAuroraUnificada)->Dict[str,Any]:
  s={"objetivos_relacionados":[],"duracion_recomendada":config.duracion_min,"intensidad_sugerida":config.intensidad,"mejoras_configuracion":{},"sync_hibrido_recomendado":False};ol=config.objetivo.lower()
  if resultado.resultado_objective_manager and self.objective_manager:
   try:s["objetivos_relacionados"]=self.objective_manager.obtener_objetivos_relacionados(config.objetivo)if hasattr(self.objective_manager,'obtener_objetivos_relacionados')else[];sr=self.objective_manager.recomendar_secuencia(config.objetivo,config.duracion_min)if hasattr(self.objective_manager,'recomendar_secuencia')else None;s["secuencia_recomendada"]=sr if sr else None
   except Exception:pass
  if"concentracion"in ol:s["objetivos_relacionados"]=["claridad_mental","enfoque_profundo","productividad"]
  elif"relajacion"in ol:s["objetivos_relacionados"]=["meditacion","calma_profunda","descanso"]
  elif"creatividad"in ol:s["objetivos_relacionados"]=["inspiracion","flow_creativo","apertura_mental"]
  s["duracion_recomendada"]=max(10,config.duracion_min-5)if resultado.efectividad_terapeutica>0.9 else min(60,config.duracion_min+10)if resultado.efectividad_terapeutica<0.7 else config.duracion_min;s["mejoras_configuracion"]={"calidad_objetivo":"maxima","modo_orquestacion":"layered"}if resultado.calidad_score<80 else{};s["mejoras_configuracion"]["usar_objective_manager"]=True if not config.usar_objective_manager and OM_AVAIL else None;s["mejoras_configuracion"]["usar_integracion_optimizada"]=True if not hasattr(config,'configuracion_enriquecida')or not config.configuracion_enriquecida else None
  if SYNC_SCHEDULER_HIBRIDO_AVAILABLE and self.sync_scheduler:
   if resultado.estrategia_usada==EstrategiaGeneracion.SYNC_SCHEDULER_HIBRIDO:coherencia_global=resultado.resultado_sync_hibrido.get('coherencia_global',0.0)if hasattr(resultado,'resultado_sync_hibrido')and resultado.resultado_sync_hibrido else 0.0;s["sync_hibrido_recomendado"]=coherencia_global>=0.8;s["mejoras_configuracion"]["mantener_sync_hibrido"]=True if coherencia_global>=0.8 else None;s["mejoras_configuracion"]["optimizar_coherencia_objetivo"]=0.9 if coherencia_global<0.8 else None
   else:s["sync_hibrido_recomendado"]=True;s["mejoras_configuracion"]["habilitar_sync_hibrido"]=True
  s["mejoras_configuracion"]={k:v for k,v in s["mejoras_configuracion"].items()if v is not None};return s
 def _actualizar_estadisticas(self,config:ConfiguracionAuroraUnificada,resultado:ResultadoAuroraIntegrado,tiempo_total:float):self.stats["experiencias_generadas"]+=1;self.stats["tiempo_total_generacion"]+=tiempo_total;e=resultado.estrategia_usada.value;self.stats["estrategias_utilizadas"][e]=self.stats["estrategias_utilizadas"].get(e,0)+1;o=config.objetivo;self.stats["objetivos_procesados"][o]=self.stats["objetivos_procesados"].get(o,0)+1;[setattr(self.stats["motores_utilizados"],m,self.stats["motores_utilizados"].get(m,0)+1)for m in resultado.componentes_usados];self.stats["templates_utilizados"][resultado.template_utilizado]=self.stats["templates_utilizados"].get(resultado.template_utilizado,0)+1 if resultado.template_utilizado else None;self.stats["perfiles_campo_utilizados"][resultado.perfil_campo_utilizado]=self.stats["perfiles_campo_utilizados"].get(resultado.perfil_campo_utilizado,0)+1 if resultado.perfil_campo_utilizado else None;self.stats["secuencias_fases_utilizadas"][resultado.secuencia_fases_utilizada]=self.stats["secuencias_fases_utilizadas"].get(resultado.secuencia_fases_utilizada,0)+1 if resultado.secuencia_fases_utilizada else None;to=self.stats["experiencias_generadas"];ca=self.stats["calidad_promedio"];self.stats["calidad_promedio"]=((ca*(to-1)+resultado.calidad_score)/to)
 def _crear_resultado_emergencia(self,objetivo:str,error:str)->ResultadoAuroraIntegrado:ae=self._generar_audio_fallback(60.0);ce=ConfiguracionAuroraUnificada(objetivo=objetivo,duracion_min=1);return ResultadoAuroraIntegrado(audio_data=ae,metadatos={"error":error,"modo_emergencia":True,"objetivo":objetivo,"timestamp":datetime.now().isoformat()},estrategia_usada=EstrategiaGeneracion.FALLBACK_PROGRESIVO,modo_orquestacion=ModoOrquestacion.HYBRID,componentes_usados=["emergencia"],tiempo_generacion=0.0,calidad_score=60.0,coherencia_neuroacustica=0.6,efectividad_terapeutica=0.6,configuracion=ce)
 def _generar_audio_fallback(self,duracion_sec:float)->np.ndarray:
  try:s=int(44100*duracion_sec);t=np.linspace(0,duracion_sec,s);a=0.3*np.sin(2*np.pi*10.0*t);th=0.2*np.sin(2*np.pi*6.0*t);am=a+th;fs=int(44100*2.0);am[:fs]*=np.linspace(0,1,fs)if len(am)>fs*2 else None;am[-fs:]*=np.linspace(1,0,fs)if len(am)>fs*2 else None;return np.stack([am,am])
  except Exception:s=int(44100*max(1.0,duracion_sec));return np.zeros((2,s),dtype=np.float32)
 def _obtener_estrategias_disponibles(self)->List[EstrategiaGeneracion]:e=[];m=len([c for c in self.componentes.values()if c.tipo==TipoComponente.MOTOR and c.disponible]);g=len([c for c in self.componentes.values()if c.tipo==TipoComponente.GESTOR_INTELIGENCIA and c.disponible]);p=len([c for c in self.componentes.values()if c.tipo==TipoComponente.PIPELINE and c.disponible]);om=len([c for c in self.componentes.values()if c.tipo==TipoComponente.OBJECTIVE_MANAGER and c.disponible]);ss=len([c for c in self.componentes.values()if c.tipo==TipoComponente.SYNC_SCHEDULER and c.disponible]);e.append(EstrategiaGeneracion.SYNC_SCHEDULER_HIBRIDO)if ss>=1 and m>=2 else None;e.append(EstrategiaGeneracion.OBJECTIVE_MANAGER_DRIVEN)if om>=1 and m>=2 else None;e.append(EstrategiaGeneracion.AURORA_ORQUESTADO)if m>=3 and g>=2 and p>=1 else None;e.append(EstrategiaGeneracion.MULTI_MOTOR)if m>=2 else None;e.append(EstrategiaGeneracion.INTELIGENCIA_ADAPTIVA)if g>=1 and m>=1 else None;e.append(EstrategiaGeneracion.MOTOR_ESPECIALIZADO)if m>=1 else None;e.append(EstrategiaGeneracion.FALLBACK_PROGRESIVO);return[x for x in e if x]
 def obtener_estado_completo(self)->Dict[str,Any]:
  eb={"version":self.version,"timestamp":datetime.now().isoformat(),"componentes_detectados":{n:{"disponible":c.disponible,"version":c.version,"tipo":c.tipo.value,"fallback":c.version=="fallback","capacidades":len(c.capacidades),"dependencias":c.dependencias,"prioridad":c.nivel_prioridad}for n,c in self.componentes.items()},"estadisticas_deteccion":self.detector.stats,"estadisticas_uso":self.stats,"estrategias_disponibles":[e.value for e in self._obtener_estrategias_disponibles()],"capacidades_sistema":{"motores_activos":len([c for c in self.componentes.values()if c.tipo==TipoComponente.MOTOR and c.disponible]),"gestores_activos":len([c for c in self.componentes.values()if c.tipo==TipoComponente.GESTOR_INTELIGENCIA and c.disponible]),"pipelines_activos":len([c for c in self.componentes.values()if c.tipo==TipoComponente.PIPELINE and c.disponible]),"objective_managers_activos":len([c for c in self.componentes.values()if c.tipo==TipoComponente.OBJECTIVE_MANAGER and c.disponible]),"sync_schedulers_activos":len([c for c in self.componentes.values()if c.tipo==TipoComponente.SYNC_SCHEDULER and c.disponible]),"orquestador_disponible":self.orquestador is not None,"objective_manager_disponible":self.objective_manager is not None,"sync_scheduler_hibrido_disponible":self.sync_scheduler is not None,"fallback_garantizado":True},"metricas_calidad":{"calidad_promedio":self.stats["calidad_promedio"],"experiencias_totales":self.stats["experiencias_generadas"],"tiempo_promedio":(self.stats["tiempo_total_generacion"]/max(1,self.stats["experiencias_generadas"])),"tasa_exito":((self.stats["experiencias_generadas"]-self.stats["errores_manejados"])/max(1,self.stats["experiencias_generadas"])*100)}}
  eb["metricas_integraciones_optimizadas"]={"efectos_psicodelicos_disponibles":self.stats.get("efectos_psicodelicos_disponibles",0),"efectos_psicodelicos_aplicados":self.stats.get("efectos_psicodelicos_aplicados",0),"emotion_style_utilizaciones":self.stats.get("emotion_style_utilizaciones",0),"field_profiles_avanzados_utilizados":self.stats.get("field_profiles_avanzados_utilizados",0),"integraciones_exitosas":self.stats.get("integraciones_exitosas",0),"psychedelic_effects_cargado":bool(self.psychedelic_effects),"emotion_style_disponible":"emotion_style_profiles"in self.componentes,"field_profiles_disponible":"field_profiles"in self.componentes,"configuracion_optimizada_activa":True,"sync_hibrido_utilizaciones":self.stats.get("sync_hibrido_utilizaciones",0),"coherencia_global_promedio":self.stats.get("coherencia_global_promedio",0.0),"sync_scheduler_hibrido_disponible":SYNC_SCHEDULER_HIBRIDO_AVAILABLE and self.sync_scheduler is not None}
  eb["metricas_objective_manager"]={"utilizaciones_totales":self.stats["objective_manager_utilizaciones"],"templates_mas_utilizados":sorted(self.stats["templates_utilizados"].items(),key=lambda x:x[1],reverse=True)[:5],"perfiles_campo_mas_utilizados":sorted(self.stats["perfiles_campo_utilizados"].items(),key=lambda x:x[1],reverse=True)[:5],"secuencias_fases_mas_utilizadas":sorted(self.stats["secuencias_fases_utilizadas"].items(),key=lambda x:x[1],reverse=True)[:5],"disponible":bool(self.objective_manager),"version":getattr(self.objective_manager,'version','unknown')if self.objective_manager else None,"capacidades":getattr(self.objective_manager,'obtener_capacidades',lambda:{})()if self.objective_manager else{}}if self.objective_manager else{"disponible":False}
  eb["metricas_sync_hibrido"]={"utilizaciones_totales":self.stats.get("sync_hibrido_utilizaciones",0),"coherencia_global_promedio":self.stats.get("coherencia_global_promedio",0.0),"disponible":bool(self.sync_scheduler),"version":"V7_UNIFIED_OPTIMIZED","funciones_disponibles":["sincronizar_y_estructurar_capas","aplicar_fade_narrativo","optimizar_coherencia_global","validar_sync_y_estructura_completa"],"estrategia_hibrida_activa":EstrategiaGeneracion.SYNC_SCHEDULER_HIBRIDO in self._obtener_estrategias_disponibles()}if self.sync_scheduler else{"disponible":False,"razon":"sync_and_scheduler no disponible"}
  return eb

def crear_experiencia_sync_hibrido(objetivo:str,**kwargs)->ResultadoAuroraIntegrado:kwargs.update({"estrategia_preferida":EstrategiaGeneracion.SYNC_SCHEDULER_HIBRIDO,"modo_orquestacion":ModoOrquestacion.SYNC_HIBRIDO,"habilitar_sync_hibrido":True,"calidad_objetivo":"maxima"});return Aurora(objetivo,**kwargs)
def crear_experiencia_coherencia_maxima(objetivo:str,coherencia_objetivo:float=0.95,**kwargs)->ResultadoAuroraIntegrado:kwargs.update({"estrategia_preferida":EstrategiaGeneracion.SYNC_SCHEDULER_HIBRIDO,"habilitar_sync_hibrido":True,"coherencia_objetivo":coherencia_objetivo,"calidad_objetivo":"maxima","modo_orquestacion":ModoOrquestacion.SYNC_HIBRIDO});return Aurora(objetivo,**kwargs)
def crear_experiencia_estructura_inteligente(objetivo:str,duracion_min:int=30,**kwargs)->ResultadoAuroraIntegrado:kwargs.update({"estrategia_preferida":EstrategiaGeneracion.SYNC_SCHEDULER_HIBRIDO,"habilitar_sync_hibrido":True,"duracion_min":duracion_min,"calidad_objetivo":"maxima","validacion_automatica":True});return Aurora(objetivo,**kwargs)
def verificar_capacidades_sync_hibrido()->Dict[str,Any]:director=Aurora();return{"sync_scheduler_disponible":SYNC_SCHEDULER_HIBRIDO_AVAILABLE,"sync_scheduler_conectado":director.sync_scheduler is not None,"estrategia_hibrida_disponible":EstrategiaGeneracion.SYNC_SCHEDULER_HIBRIDO in director._obtener_estrategias_disponibles(),"funciones_hibridas":["sincronizar_y_estructurar_capas","aplicar_fade_narrativo","optimizar_coherencia_global","validar_sync_y_estructura_completa"]if SYNC_SCHEDULER_HIBRIDO_AVAILABLE else[],"motores_compatibles":len([c for c in director.componentes.values()if c.tipo==TipoComponente.MOTOR and c.disponible]),"coherencia_promedio_historica":director.stats.get("coherencia_global_promedio",0.0),"utilizaciones_hibridas":director.stats.get("sync_hibrido_utilizaciones",0)}
def obtener_estadisticas_sync_hibrido()->Dict[str,Any]:director=Aurora();return director.obtener_estado_completo().get("metricas_sync_hibrido",{"disponible":False})
def crear_experiencia_con_template(objetivo:str,template:str,**kwargs)->ResultadoAuroraIntegrado:return Aurora(objetivo,template_personalizado=template,**kwargs)
def crear_experiencia_con_perfil_campo(objetivo:str,perfil_campo:str,**kwargs)->ResultadoAuroraIntegrado:return Aurora(objetivo,perfil_campo_personalizado=perfil_campo,**kwargs)
def crear_experiencia_con_secuencia_fases(objetivo:str,secuencia:str,**kwargs)->ResultadoAuroraIntegrado:return Aurora(objetivo,secuencia_fases_personalizada=secuencia,**kwargs)
def crear_experiencia_optimizada(objetivo:str,**kwargs)->ResultadoAuroraIntegrado:return Aurora(objetivo,calidad_objetivo="maxima",modo_orquestacion="layered",**kwargs)
def crear_experiencia_psicodelica(objetivo:str,efecto_deseado:str=None,**kwargs)->ResultadoAuroraIntegrado:om=f"{objetivo} {efecto_deseado}"if efecto_deseado else objetivo;return Aurora(om,calidad_objetivo="maxima",modo_orquestacion="layered",**kwargs)
def crear_experiencia_emocional(objetivo:str,emocion_objetivo:str=None,**kwargs)->ResultadoAuroraIntegrado:om=f"{objetivo} {emocion_objetivo}"if emocion_objetivo else objetivo;return Aurora(om,intensidad="media",**kwargs)
def obtener_templates_disponibles()->List[str]:d=Aurora();return d.objective_manager.obtener_templates_disponibles()if d.objective_manager and hasattr(d.objective_manager,'obtener_templates_disponibles')else[]
def obtener_perfiles_campo_disponibles()->List[str]:d=Aurora();return d.objective_manager.obtener_perfiles_disponibles()if d.objective_manager and hasattr(d.objective_manager,'obtener_perfiles_disponibles')else[]
def obtener_secuencias_fases_disponibles()->List[str]:d=Aurora();return d.objective_manager.obtener_secuencias_disponibles()if d.objective_manager and hasattr(d.objective_manager,'obtener_secuencias_disponibles')else[]
def obtener_efectos_psicodelicos_disponibles()->List[str]:d=Aurora();return list(d.psychedelic_effects["pe"].keys())if d.psychedelic_effects and"pe"in d.psychedelic_effects else[]
def obtener_estado_integraciones()->Dict[str,Any]:d=Aurora();return d.obtener_estado_completo().get("metricas_integraciones_optimizadas",{})
def verificar_integraciones_optimizadas()->Dict[str,bool]:d=Aurora();return{"psychedelic_effects":bool(d.psychedelic_effects),"emotion_style_profiles":"emotion_style_profiles"in d.componentes,"field_profiles":"field_profiles"in d.componentes,"objective_manager":d.objective_manager is not None,"carmine_analyzer":"carmine_analyzer_v21"in d.componentes,"quality_pipeline":"quality_pipeline"in d.componentes,"sync_scheduler_hibrido":d.sync_scheduler is not None,"sistema_completamente_optimizado":all([bool(d.psychedelic_effects),"emotion_style_profiles"in d.componentes,"field_profiles"in d.componentes,d.sync_scheduler is not None])}

_director_global:Optional[AuroraDirectorV7Integrado]=None
def Aurora(objetivo:str=None,**kwargs)->Union[ResultadoAuroraIntegrado,AuroraDirectorV7Integrado]:global _director_global;_director_global=AuroraDirectorV7Integrado()if _director_global is None else _director_global;return _director_global.crear_experiencia(objetivo,**kwargs)if objetivo is not None else _director_global

Aurora.rapido=lambda obj,**kw:Aurora(obj,duracion_min=5,calidad_objetivo="media",**kw)
Aurora.largo=lambda obj,**kw:Aurora(obj,duracion_min=60,calidad_objetivo="alta",**kw)
Aurora.terapeutico=lambda obj,**kw:Aurora(obj,duracion_min=45,intensidad="suave",calidad_objetivo="maxima",modo_orquestacion="layered",**kw)
Aurora.optimizado=lambda obj,**kw:crear_experiencia_optimizada(obj,**kw)
Aurora.psicodelico=lambda obj,efecto=None,**kw:crear_experiencia_psicodelica(obj,efecto,**kw)
Aurora.emocional=lambda obj,emocion=None,**kw:crear_experiencia_emocional(obj,emocion,**kw)
Aurora.hibrido=lambda obj,**kw:crear_experiencia_sync_hibrido(obj,**kw)
Aurora.coherencia_maxima=lambda obj,coherencia=0.95,**kw:crear_experiencia_coherencia_maxima(obj,coherencia,**kw)
Aurora.estructura_inteligente=lambda obj,duracion=30,**kw:crear_experiencia_estructura_inteligente(obj,duracion,**kw)
Aurora.estado=lambda:Aurora().obtener_estado_completo()
Aurora.diagnostico=lambda:Aurora().detector.stats
Aurora.stats=lambda:Aurora().stats
Aurora.integraciones=lambda:obtener_estado_integraciones()
Aurora.verificar=lambda:verificar_integraciones_optimizadas()
Aurora.verificar_sync_hibrido=lambda:verificar_capacidades_sync_hibrido()
Aurora.stats_sync_hibrido=lambda:obtener_estadisticas_sync_hibrido()
Aurora.con_template=crear_experiencia_con_template
Aurora.con_perfil_campo=crear_experiencia_con_perfil_campo
Aurora.con_secuencia_fases=crear_experiencia_con_secuencia_fases
Aurora.templates_disponibles=obtener_templates_disponibles
Aurora.perfiles_campo_disponibles=obtener_perfiles_campo_disponibles
Aurora.secuencias_fases_disponibles=obtener_secuencias_fases_disponibles
Aurora.efectos_psicodelicos_disponibles=obtener_efectos_psicodelicos_disponibles

if __name__=="__main__":
    print("🌟 Aurora Director V7 INTEGRADO - Sistema con Sync Scheduler Híbrido")
    print("="*95)
    director=Aurora()
    estado=director.obtener_estado_completo()
    print(f"🚀 {estado['version']}")
    print(f"⏰ Inicializado: {estado['timestamp']}")
    print(f"\n📊 Componentes detectados: {len(estado['componentes_detectados'])}")
    for nombre,info in estado['componentes_detectados'].items():
        emoji="✅"if info['disponible']and not info['fallback']else"🔄"if info['fallback']else"❌"
        tipo_emoji={"motor":"🎵","gestor_inteligencia":"🧠","pipeline":"🔄","preset_manager":"🎯","style_profile":"🎨","objective_manager":"🎯","sync_scheduler":"🌟"}.get(info['tipo'],"🔧")
        print(f"   {emoji} {tipo_emoji} {nombre} v{info['version']} (P{info['prioridad']})")
    caps=estado['capacidades_sistema']
    print(f"\n🔧 Capacidades del Sistema:")
    print(f"   🎵 Motores activos: {caps['motores_activos']}")
    print(f"   🧠 Gestores activos: {caps['gestores_activos']}")
    print(f"   🔄 Pipelines activos: {caps['pipelines_activos']}")
    print(f"   🎯 OM activos: {caps['objective_managers_activos']}")
    print(f"   🌟 Sync Schedulers activos: {caps['sync_schedulers_activos']}")
    print(f"   🎼 Orquestador: {'✅'if caps['orquestador_disponible']else'❌'}")
    print(f"   🎯 OM: {'✅'if caps['objective_manager_disponible']else'❌'}")
    print(f"   🌟 Sync Híbrido: {'✅'if caps['sync_scheduler_hibrido_disponible']else'❌'}")
    print(f"   🛡️ Fallback garantizado: {'✅'if caps['fallback_garantizado']else'❌'}")
    integraciones=estado.get('metricas_integraciones_optimizadas',{})
    print(f"\n🌈 Integraciones Optimizadas:")
    print(f"   🌈 Psychedelic Effects: {'✅ Cargado'if integraciones.get('psychedelic_effects_cargado')else'❌ No disponible'}")
    print(f"   🎭 Emotion Style Profiles: {'✅ Disponible'if integraciones.get('emotion_style_disponible')else'❌ No disponible'}")
    print(f"   🎯 Field Profiles: {'✅ Disponible'if integraciones.get('field_profiles_disponible')else'❌ No disponible'}")
    print(f"   🌟 Sync Scheduler Híbrido: {'✅ DISPONIBLE'if integraciones.get('sync_scheduler_hibrido_disponible')else'❌ No disponible'}")
    print(f"   ⚡ Sistema Optimizado: {'✅ ACTIVO'if integraciones.get('configuracion_optimizada_activa')else'❌ Inactivo'}")
    sync_hibrido=estado.get('metricas_sync_hibrido',{})
    if sync_hibrido.get('disponible'):
        print(f"\n🌟 Métricas Sync Scheduler Híbrido:")
        print(f"   📊 Utilizaciones: {sync_hibrido['utilizaciones_totales']}")
        print(f"   🎯 Coherencia promedio: {sync_hibrido['coherencia_global_promedio']:.3f}")
        print(f"   🔧 Funciones disponibles: {len(sync_hibrido['funciones_disponibles'])}")
        print(f"   ⚡ Estrategia híbrida activa: {'✅'if sync_hibrido['estrategia_hibrida_activa']else'❌'}")
    if estado['metricas_objective_manager']['disponible']:
        om_metrics=estado['metricas_objective_manager']
        print(f"\n🎯 Métricas OM:")
        print(f"   📊 Utilizaciones: {om_metrics['utilizaciones_totales']}")
        print(f"   📝 Templates disponibles: {len(Aurora.templates_disponibles())}")
        print(f"   🎭 Perfiles campo disponibles: {len(Aurora.perfiles_campo_disponibles())}")
        print(f"   📋 Secuencias fases disponibles: {len(Aurora.secuencias_fases_disponibles())}")
    print(f"\n🎯 Estrategias disponibles ({len(estado['estrategias_disponibles'])}):")
    for i,estrategia in enumerate(estado['estrategias_disponibles'],1):
        emoji="🌟"if estrategia=="sync_scheduler_hibrido"else f"{i}."
        print(f"   {emoji} {estrategia}")
    print(f"\n🧪 Testing del sistema con Sync Scheduler Híbrido...")
    try:
        print(f"   🌟 Test 1: Verificación capacidades híbridas...")
        capacidades_hibridas=Aurora.verificar_sync_hibrido()
        print(f"      ✅ Sync Scheduler disponible: {capacidades_hibridas['sync_scheduler_disponible']}")
        print(f"      🔧 Sync Scheduler conectado: {capacidades_hibridas['sync_scheduler_conectado']}")
        print(f"      ⚡ Estrategia híbrida disponible: {capacidades_hibridas['estrategia_hibrida_disponible']}")
        print(f"      🎵 Motores compatibles: {capacidades_hibridas['motores_compatibles']}")
        print(f"   🌟 Test 2: Experiencia híbrida básica...")
        if capacidades_hibridas['estrategia_hibrida_disponible']:
            resultado_hibrido=Aurora.hibrido("test_sync_hibrido",duracion_min=1,exportar_wav=False)
            print(f"      ✅ Audio híbrido generado: {resultado_hibrido.audio_data.shape}")
            print(f"      📊 Calidad: {resultado_hibrido.calidad_score:.1f}/100")
            print(f"      🌟 Estrategia: {resultado_hibrido.estrategia_usada.value}")
            print(f"      🎯 Coherencia neuroacústica: {resultado_hibrido.coherencia_neuroacustica:.3f}")
            if hasattr(resultado_hibrido,'resultado_sync_hibrido')and resultado_hibrido.resultado_sync_hibrido:
                print(f"      🌟 Coherencia global: {resultado_hibrido.resultado_sync_hibrido.get('coherencia_global','N/A'):.3f}")
                print(f"      🔧 Sync Scheduler version: {resultado_hibrido.resultado_sync_hibrido.get('sync_scheduler_version','N/A')}")
        else:
            print(f"      ⚠️ Estrategia híbrida no disponible, test omitido")
        print(f"   🎯 Test 3: Experiencia coherencia máxima...")
        resultado_coherencia=Aurora.coherencia_maxima("concentracion_maxima",coherencia=0.95,duracion_min=1)
        print(f"      ✅ Audio coherencia máxima: {resultado_coherencia.audio_data.shape}")
        print(f"      🎯 Coherencia objetivo configurada: 0.95")
        print(f"      📈 Coherencia alcanzada: {resultado_coherencia.coherencia_neuroacustica:.3f}")
        print(f"   🏗️ Test 4: Experiencia estructura inteligente...")
        resultado_estructura=Aurora.estructura_inteligente("meditacion_profunda",duracion=2,duracion_min=2)
        print(f"      ✅ Audio estructura inteligente: {resultado_estructura.audio_data.shape}")
        print(f"      🏗️ Estructura aplicada: {hasattr(resultado_estructura,'estructura_fases_utilizada')}")
        print(f"   🔄 Test 5: Compatibilidad con APIs existentes...")
        resultado_tradicional=Aurora.optimizado("test_compatibilidad",duracion_min=1,exportar_wav=False)
        print(f"      ✅ API tradicional funcionando: {resultado_tradicional.audio_data.shape}")
        print(f"      🔧 Componentes usados: {len(resultado_tradicional.componentes_usados)}")
    except Exception as e:
        print(f"   ❌ Error en testing: {e}")
    metricas=estado['metricas_calidad']
    print(f"\n📈 Métricas del Sistema:")
    print(f"   🎯 Calidad promedio: {metricas['calidad_promedio']:.1f}/100")
    print(f"   🔢 Experiencias totales: {metricas['experiencias_totales']}")
    print(f"   ⏱️ Tiempo promedio: {metricas['tiempo_promedio']:.2f}s")
    print(f"   ✅ Tasa de éxito: {metricas['tasa_exito']:.1f}%")
    if integraciones:
        print(f"\n🌈 Métricas de Integraciones:")
        print(f"   🌈 Efectos psicodélicos aplicados: {integraciones.get('efectos_psicodelicos_aplicados',0)}")
        print(f"   🎭 Emotion Style utilizaciones: {integraciones.get('emotion_style_utilizaciones',0)}")
        print(f"   🎯 Field Profiles avanzados: {integraciones.get('field_profiles_avanzados_utilizados',0)}")
        print(f"   ⚡ Integraciones exitosas: {integraciones.get('integraciones_exitosas',0)}")
        print(f"   🌟 Sync Híbrido utilizaciones: {integraciones.get('sync_hibrido_utilizaciones',0)}")
        print(f"   🎯 Coherencia global promedio: {integraciones.get('coherencia_global_promedio',0.0):.3f}")
    print(f"\n🏆 AURORA DIRECTOR V7 + SYNC SCHEDULER HÍBRIDO")
    print(f"🌟 ¡Sistema con integración TOTAL + Capacidades Híbridas!")
    print(f"🌈 ¡Psychedelic Effects: {'✅ INTEGRADO'if integraciones.get('psychedelic_effects_cargado')else'❌ No disponible'}!")
    print(f"🎭 ¡Emotion Style Profiles: {'✅ INTEGRADO'if integraciones.get('emotion_style_disponible')else'❌ No disponible'}!")
    print(f"🎯 ¡Field Profiles Avanzado: {'✅ INTEGRADO'if integraciones.get('field_profiles_disponible')else'❌ No disponible'}!")
    print(f"🌟 ¡Sync Scheduler Híbrido: {'✅ INTEGRADO'if integraciones.get('sync_scheduler_hibrido_disponible')else'❌ No disponible'}!")
    print(f"🔗 ¡Todos los motores conectados armoniosamente!")
    print(f"🧠 ¡Inteligencia y orquestación híbrida avanzada!")
    print(f"🎯 ¡OM Unificado + Sync Híbrido integrados!")
    print(f"🎵 ¡Experiencias Aurora de máxima coherencia!")
    print(f"✨ ¡Listo para crear experiencias transformadoras con estructura inteligente!")
    estado_verificacion=Aurora.verificar()
    estado_sync_hibrido=Aurora.verificar_sync_hibrido()
    sistema_completo=(estado_verificacion['sistema_completamente_optimizado']and estado_sync_hibrido['estrategia_hibrida_disponible'])
    if sistema_completo:
        print(f"\n🌟 ¡SISTEMA HÍBRIDO COMPLETAMENTE OPTIMIZADO!")
        print(f"✅ ¡TODAS las integraciones están activas!")
        print(f"🌟 ¡Sync Scheduler Híbrido funcionando!")
        print(f"🚀 ¡Rendimiento, calidad y coherencia MÁXIMOS!")
    else:
        print(f"\n⚠️ Sistema parcialmente optimizado")
        print(f"📋 Componentes faltantes para optimización híbrida completa:")
        if not estado_verificacion['psychedelic_effects']:
            print(f"   ❌ psychedelic_effects_tables.json")
        if not estado_verificacion['emotion_style_profiles']:
            print(f"   ❌ emotion_style_profiles.py")
        if not estado_verificacion['field_profiles']:
            print(f"   ❌ field_profiles.py")
        if not estado_sync_hibrido['sync_scheduler_disponible']:
            print(f"   ❌ sync_and_scheduler.py")
    print(f"\n🌟 APIs Híbridas Disponibles:")
    print(f"   • Aurora.hibrido(objetivo) - Experiencia con sync híbrido")
    print(f"   • Aurora.coherencia_maxima(objetivo, coherencia=0.95) - Coherencia máxima")
    print(f"   • Aurora.estructura_inteligente(objetivo, duracion=30) - Estructura adaptiva")
    print(f"   • Aurora.verificar_sync_hibrido() - Verificar capacidades híbridas")
    print(f"   • Aurora.stats_sync_hibrido() - Estadísticas de uso híbrido")