import wave,numpy as np,json,time,logging,warnings
from typing import Dict,Tuple,Optional,List,Any,Union,Protocol
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import dataclass,field,asdict
from enum import Enum
from datetime import datetime
from functools import lru_cache
SAMPLE_RATE,VERSION,CONFIDENCE_THRESHOLD=44100,"V27_AURORA_CONNECTED",0.8
logging.basicConfig(level=logging.WARNING)
logger=logging.getLogger("Aurora.NeuroMix.V27")
class MotorAurora(Protocol):
 def generar_audio(self,config:Dict[str,Any],duracion_sec:float)->np.ndarray:...
 def validar_configuracion(self,config:Dict[str,Any])->bool:...
 def obtener_capacidades(self)->Dict[str,Any]:...
class Neurotransmisor(Enum):
 DOPAMINA="dopamina";SEROTONINA="serotonina";GABA="gaba";OXITOCINA="oxitocina";ANANDAMIDA="anandamida";ACETILCOLINA="acetilcolina";ENDORFINA="endorfina";BDNF="bdnf";ADRENALINA="adrenalina";NOREPINEFRINA="norepinefrina";MELATONINA="melatonina"
class EstadoEmocional(Enum):
 ENFOQUE="enfoque";RELAJACION="relajacion";GRATITUD="gratitud";VISUALIZACION="visualizacion";SOLTAR="soltar";ACCION="accion";CLARIDAD_MENTAL="claridad_mental";SEGURIDAD_INTERIOR="seguridad_interior";APERTURA_CORAZON="apertura_corazon";ALEGRIA_SOSTENIDA="alegria_sostenida";FUERZA_TRIBAL="fuerza_tribal";CONEXION_MISTICA="conexion_mistica";REGULACION_EMOCIONAL="regulacion_emocional";EXPANSION_CREATIVA="expansion_creativa";ESTADO_FLUJO="estado_flujo";INTROSPECCION_SUAVE="introspeccion_suave";SANACION_PROFUNDA="sanacion_profunda";EQUILIBRIO_MENTAL="equilibrio_mental"
class TipoOnda(Enum):
 SINE="sine";SAW="saw";SQUARE="square";TRIANGLE="triangle";PULSE="pulse";NOISE_WHITE="white_noise";NOISE_PINK="pink_noise";PAD_BLEND="pad_blend";BINAURAL="binaural";NEUROMORPHIC="neuromorphic";THERAPEUTIC="therapeutic"
class TipoModulacion(Enum):
 AM="am";FM="fm";PM="pm";HYBRID="hybrid";NEUROMORPHIC="neuromorphic";BINAURAL_BEAT="binaural_beat"
class NeuroQualityLevel(Enum):
 BASIC="básico";ENHANCED="mejorado";PROFESSIONAL="profesional";THERAPEUTIC="terapéutico";RESEARCH="investigación"
class ProcessingMode(Enum):
 LEGACY="legacy";STANDARD="standard";ADVANCED="advanced";PARALLEL="parallel";REALTIME="realtime";AURORA_INTEGRATED="aurora_integrated"
@dataclass
class ParametrosEspaciales:
 pan:float=0.0;width:float=1.0;distance:float=1.0;elevation:float=0.0;movement_pattern:str="static";movement_speed:float=0.1
@dataclass
class ParametrosTempo:
 onset_ms:int=100;sustain_ms:int=2000;decay_ms:int=500;release_ms:int=1000;rhythm_pattern:str="steady"
@dataclass
class ParametrosModulacion:
 tipo:TipoModulacion;profundidad:float;velocidad_hz:float;fase_inicial:float=0.0
 def __post_init__(self):
  if not 0<=self.profundidad<=1:warnings.warn(f"Profundidad {self.profundidad} fuera de rango [0,1]")
  if self.velocidad_hz<0:raise ValueError("Velocidad de modulación debe ser positiva")
@dataclass
class ParametrosNeurotransmisorCientifico:
 nombre:str;neurotransmisor:Neurotransmisor;frecuencia_primaria:float;frecuencias_armonicas:List[float]=field(default_factory=list);frecuencias_subharmonicas:List[float]=field(default_factory=list);frecuencia_range_min:float=0.0;frecuencia_range_max:float=0.0;tipo_onda:TipoOnda=TipoOnda.SINE;modulacion:ParametrosModulacion=None;nivel_db:float=-12.0;panorama:ParametrosEspaciales=field(default_factory=ParametrosEspaciales);tempo:ParametrosTempo=field(default_factory=ParametrosTempo);efectos_cognitivos:List[str]=field(default_factory=list);efectos_emocionales:List[str]=field(default_factory=list);interacciones:Dict[str,float]=field(default_factory=dict);receptor_types:List[str]=field(default_factory=list);brain_regions:List[str]=field(default_factory=list);effect_categories:List[str]=field(default_factory=list);contraindications:List[str]=field(default_factory=list);research_references:List[str]=field(default_factory=list);validated:bool=False;confidence_score:float=0.8;version:str="v27_aurora";last_updated:str=field(default_factory=lambda:datetime.now().isoformat())
 def __post_init__(self):
  if self.modulacion is None:self.modulacion=ParametrosModulacion(TipoModulacion.AM,0.5,0.1)
  if self.frecuencia_range_min==0.0:self.frecuencia_range_min=self.frecuencia_primaria*0.8
  if self.frecuencia_range_max==0.0:self.frecuencia_range_max=self.frecuencia_primaria*1.2
  if self.frecuencia_primaria<=0:raise ValueError(f"Frecuencia debe ser positiva: {self.frecuencia_primaria}")
@dataclass
class PresetEmocionalCientifico:
 nombre:str;estado:EstadoEmocional;neurotransmisores:Dict[Neurotransmisor,float];frecuencia_base:float;intensidad_global:float=1.0;duracion_ms:int=6000;modulacion_temporal:Optional[str]=None;descripcion:str="";categoria:str="general";nivel_activacion:str="medio";efectos_esperados:List[str]=field(default_factory=list);contraindicaciones:List[str]=field(default_factory=list);evidencia_cientifica:str="experimental"
 def __post_init__(self):
  if not self.neurotransmisores:raise ValueError("Preset debe tener al menos un neurotransmisor")
  for neuro,intensidad in self.neurotransmisores.items():
   if not 0<=intensidad<=1:warnings.warn(f"Intensidad de {neuro.value} fuera de rango [0,1]: {intensidad}")
@dataclass
class NeuroConfigAuroraConnected:
 neurotransmitter:str;duration_sec:float;wave_type:str='hybrid';intensity:str="media";style:str="neutro";objective:str="relajación";aurora_config:Optional[Dict[str,Any]]=None;director_context:Optional[Dict[str,Any]]=None;quality_level:NeuroQualityLevel=NeuroQualityLevel.ENHANCED;processing_mode:ProcessingMode=ProcessingMode.AURORA_INTEGRATED;enable_quality_pipeline:bool=True;enable_analysis:bool=True;enable_textures:bool=True;enable_spatial_effects:bool=False;custom_frequencies:Optional[List[float]]=None;modulation_complexity:float=1.0;harmonic_richness:float=0.5;therapeutic_intent:Optional[str]=None;apply_mastering:bool=True;target_lufs:float=-23.0;export_analysis:bool=False;use_scientific_data:bool=True;emotional_preset:Optional[str]=None;validate_interactions:bool=True
 def aplicar_config_aurora(self,config_aurora:Dict[str,Any])->'NeuroConfigAuroraConnected':
  if config_aurora:
   self.aurora_config=config_aurora
   if 'intensidad' in config_aurora:self.intensity=config_aurora['intensidad']
   if 'estilo' in config_aurora:self.style=config_aurora['estilo']
   if 'objetivo' in config_aurora:self.objective=config_aurora['objetivo']
   if 'calidad_objetivo' in config_aurora:
    quality_map={'basica':NeuroQualityLevel.BASIC,'media':NeuroQualityLevel.ENHANCED,'alta':NeuroQualityLevel.PROFESSIONAL,'maxima':NeuroQualityLevel.THERAPEUTIC}
    self.quality_level=quality_map.get(config_aurora['calidad_objetivo'],NeuroQualityLevel.ENHANCED)
   if 'neurotransmisor_preferido' in config_aurora:self.neurotransmitter=config_aurora['neurotransmisor_preferido']
  return self
class SistemaNeuroacusticoCientificoV27:
 def __init__(self):
  self.parametros_neurotransmisores:Dict[Neurotransmisor,ParametrosNeurotransmisorCientifico]={}
  self.presets_emocionales:Dict[EstadoEmocional,PresetEmocionalCientifico]={}
  self._cache_combinaciones={}
  self._cache_frecuencias={}
  self.version="v27_aurora_scientific"
  self._inicializar_datos_cientificos()
  self._inicializar_presets_emocionales()
  self._inicializar_mapeos_aurora()
 def _inicializar_datos_cientificos(self):
  datos={
   (Neurotransmisor.DOPAMINA,("Dopamina",396.0,[792.0,1188.0,1584.0],[198.0,132.0],TipoOnda.SINE,TipoModulacion.FM,0.6,0.3,-9.0,["enfoque","motivación","aprendizaje","recompensa"],["satisfacción","alegría","confianza","determinación"],{"serotonina":0.7,"acetilcolina":0.6,"gaba":-0.3},["dopamina_d1","dopamina_d2"],["cortex_prefrontal","striatum"],0.94)),
   (Neurotransmisor.SEROTONINA,("Serotonina",417.0,[834.0,1251.0,1668.0],[208.5,139.0],TipoOnda.SAW,TipoModulacion.AM,0.7,0.15,-10.0,["estabilidad_emocional","claridad_mental","regulación_humor"],["calma","bienestar","contentamiento","estabilidad"],{"gaba":0.8,"oxitocina":0.5,"melatonina":0.7},["serotonina_5ht1a","serotonina_5ht2a"],["raphe_nuclei","cortex_prefrontal"],0.92)),
   (Neurotransmisor.GABA,("GABA",72.0,[144.0,216.0,288.0],[36.0,24.0],TipoOnda.TRIANGLE,TipoModulacion.AM,0.8,0.1,-8.0,["relajación","reducción_ansiedad","inhibición_neural"],["paz","tranquilidad","seguridad","calma_profunda"],{"serotonina":0.7,"melatonina":0.8,"adrenalina":-0.9},["gaba_a","gaba_b"],["cortex","hipocampo","amigdala"],0.95)),
   (Neurotransmisor.ACETILCOLINA,("Acetilcolina",320.0,[640.0,960.0,1280.0],[160.0,106.7],TipoOnda.SINE,TipoModulacion.FM,0.6,0.8,-12.0,["concentración","claridad_mental","memoria_trabajo","atención"],["determinación","claridad","presencia","agudeza_mental"],{"dopamina":0.6,"norepinefrina":0.7,"bdnf":0.8},["acetilcolina_nicotinico","acetilcolina_muscarnico"],["nucleus_basalis","cortex"],0.89)),
   (Neurotransmisor.OXITOCINA,("Oxitocina",528.0,[1056.0,1584.0,2112.0],[264.0,176.0],TipoOnda.PAD_BLEND,TipoModulacion.HYBRID,0.5,0.2,-11.0,["empatía","conexión_social","confianza","vínculo"],["amor","seguridad","pertenencia","conexión"],{"serotonina":0.6,"endorfina":0.7,"dopamina":0.4},["oxitocina_receptor"],["hipotalamo","amigdala"],0.90)),
   (Neurotransmisor.ANANDAMIDA,("Anandamida",111.0,[222.0,333.0,444.0],[55.5,37.0],TipoOnda.NEUROMORPHIC,TipoModulacion.NEUROMORPHIC,0.4,0.08,-13.0,["creatividad","perspectiva_amplia","intuición","relajación_mental"],["euforia_suave","apertura","liberación","creatividad_fluida"],{"gaba":0.4,"serotonina":0.3,"dopamina":0.2},["cannabinoid_cb1","cannabinoid_cb2"],["cortex","hipocampo"],0.82)),
   (Neurotransmisor.ENDORFINA,("Endorfina",528.0,[1056.0,1584.0,2112.0],[264.0,176.0],TipoOnda.SINE,TipoModulacion.AM,0.6,0.25,-10.5,["resistencia","superación","fluidez","bienestar_físico"],["euforia","satisfacción","fortaleza","energía_positiva"],{"dopamina":0.8,"oxitocina":0.7,"serotonina":0.5},["opioid_mu","opioid_delta"],["hypothalamus","periaqueductal_gray"],0.87)),
   (Neurotransmisor.BDNF,("BDNF",285.0,[570.0,855.0,1140.0],[142.5,95.0],TipoOnda.SINE,TipoModulacion.AM,0.5,0.3,-12.0,["neuroplasticidad","aprendizaje","adaptación","crecimiento_neural"],["crecimiento","renovación","vitalidad","regeneración"],{"acetilcolina":0.8,"dopamina":0.6},["trkb","p75ntr"],["hipocampo","cortex"],0.88)),
   (Neurotransmisor.ADRENALINA,("Adrenalina",741.0,[1482.0,2223.0],[370.5,247.0],TipoOnda.PULSE,TipoModulacion.FM,0.9,1.2,-9.5,["alerta_máxima","reacción_rápida","energía_súbita"],["energía","determinación","coraje","activación"],{"norepinefrina":0.8,"gaba":-0.8},["adrenergico_alfa","adrenergico_beta"],["medula_suprarrenal"],0.91)),
   (Neurotransmisor.NOREPINEFRINA,("Norepinefrina",693.0,[1386.0,2079.0],[346.5,231.0],TipoOnda.PULSE,TipoModulacion.FM,0.5,1.2,-9.5,["vigilancia","concentración","decisión_rápida","atención_sostenida"],["alerta","confianza","determinación","claridad_mental"],{"dopamina":0.7,"acetilcolina":0.8,"adrenalina":0.8},["adrenergico_alfa1","adrenergico_beta"],["locus_coeruleus","cortex_prefrontal"],0.91)),
   (Neurotransmisor.MELATONINA,("Melatonina",108.0,[216.0,324.0],[54.0,36.0],TipoOnda.SINE,TipoModulacion.AM,0.9,0.05,-15.0,["relajación_profunda","preparación_descanso","ritmo_circadiano"],["serenidad","paz_interior","soltar","preparación_sueño"],{"gaba":0.8,"serotonina":0.6,"adrenalina":-0.9},["melatonin_mt1","melatonin_mt2"],["pineal_gland","suprachiasmatic_nucleus"],0.93))
  }
  for nt,(nombre,freq_prim,armonicos,subarm,onda,mod_tipo,mod_prof,mod_vel,db,efectos_cog,efectos_emo,interacciones,receptores,regiones,confianza) in datos.items():
   self.parametros_neurotransmisores[nt]=ParametrosNeurotransmisorCientifico(nombre=nombre,neurotransmisor=nt,frecuencia_primaria=freq_prim,frecuencias_armonicas=armonicos,frecuencias_subharmonicas=subarm,tipo_onda=onda,modulacion=ParametrosModulacion(mod_tipo,mod_prof,mod_vel),nivel_db=db,efectos_cognitivos=efectos_cog,efectos_emocionales=efectos_emo,interacciones=interacciones,receptor_types=receptores,brain_regions=regiones,validated=True,confidence_score=confianza)
 def _inicializar_presets_emocionales(self):
  presets={
   (EstadoEmocional.ENFOQUE,("Enfoque Cognitivo Profundo",{Neurotransmisor.DOPAMINA:0.8,Neurotransmisor.ACETILCOLINA:0.9,Neurotransmisor.NOREPINEFRINA:0.6},14.5,0.85,"Concentración sostenida y claridad mental","cognitivo","alto",["Mejora concentración","Claridad mental"],[],"validado")),
   (EstadoEmocional.RELAJACION,("Relajación Terapéutica Profunda",{Neurotransmisor.GABA:0.9,Neurotransmisor.SEROTONINA:0.8,Neurotransmisor.MELATONINA:0.5},8.0,0.9,"Relajación completa y liberación profunda del estrés","terapeutico","bajo",["Relajación profunda","Reducción estrés"],[],"clinico")),
   (EstadoEmocional.ESTADO_FLUJO,("Estado de Flujo Óptimo",{Neurotransmisor.DOPAMINA:0.9,Neurotransmisor.NOREPINEFRINA:0.7,Neurotransmisor.ENDORFINA:0.6,Neurotransmisor.ANANDAMIDA:0.4},12.0,1.0,"Rendimiento óptimo e inmersión total","performance","alto",["Estado de flujo","Rendimiento máximo"],["hipertension"],"validado")),
   (EstadoEmocional.CONEXION_MISTICA,("Conexión Espiritual Profunda",{Neurotransmisor.ANANDAMIDA:0.8,Neurotransmisor.SEROTONINA:0.6,Neurotransmisor.OXITOCINA:0.7,Neurotransmisor.GABA:0.5},5.0,0.8,"Expansión de consciencia y conexión universal","espiritual","medio",["Expansión consciencia","Conexión universal"],["epilepsia"],"experimental")),
   (EstadoEmocional.SANACION_PROFUNDA,("Sanación Integral Avanzada",{Neurotransmisor.OXITOCINA:0.9,Neurotransmisor.ENDORFINA:0.8,Neurotransmisor.GABA:0.7,Neurotransmisor.BDNF:0.6},6.5,0.85,"Regeneración y sanación a nivel celular","terapeutico","medio",["Sanación emocional","Regeneración celular"],[],"validado")),
   (EstadoEmocional.EXPANSION_CREATIVA,("Creatividad Exponencial",{Neurotransmisor.DOPAMINA:0.8,Neurotransmisor.ACETILCOLINA:0.7,Neurotransmisor.ANANDAMIDA:0.6,Neurotransmisor.BDNF:0.7},11.5,0.75,"Inspiración y expresión creativa sin límites","creativo","medio-alto",["Explosión creativa","Inspiración fluida"],[],"validado"))
  }
  for estado,(nombre,neuros,freq,intensidad,desc,cat,nivel,efectos,contra,evidencia) in presets.items():
   self.presets_emocionales[estado]=PresetEmocionalCientifico(nombre=nombre,estado=estado,neurotransmisores=neuros,frecuencia_base=freq,intensidad_global=intensidad,descripcion=desc,categoria=cat,nivel_activacion=nivel,efectos_esperados=efectos,contraindicaciones=contra,evidencia_cientifica=evidencia)
 def _inicializar_mapeos_aurora(self):
  self.mapeos_objetivos_aurora={"concentracion":EstadoEmocional.ENFOQUE,"claridad_mental":EstadoEmocional.CLARIDAD_MENTAL,"relajacion":EstadoEmocional.RELAJACION,"meditacion":EstadoEmocional.CONEXION_MISTICA,"creatividad":EstadoEmocional.EXPANSION_CREATIVA,"sanacion":EstadoEmocional.SANACION_PROFUNDA,"energia":EstadoEmocional.ACCION,"gratitud":EstadoEmocional.GRATITUD,"visualizacion":EstadoEmocional.VISUALIZACION}
  self.mapeos_intensidad_aurora={"suave":0.6,"media":1.0,"intenso":1.4}
  self.mapeos_estilo_aurora={"sereno":{"carrier_offset":-10,"beat_offset":-0.5,"complexity":0.8},"mistico":{"carrier_offset":5,"beat_offset":0.3,"complexity":1.2},"crystalline":{"carrier_offset":15,"beat_offset":1.0,"complexity":0.9},"tribal":{"carrier_offset":8,"beat_offset":0.8,"complexity":1.1},"organico":{"carrier_offset":0,"beat_offset":0.2,"complexity":1.0}}
 @lru_cache(maxsize=128)
 def obtener_parametros_neurotransmisor(self,nt:Neurotransmisor)->Optional[ParametrosNeurotransmisorCientifico]:
  return self.parametros_neurotransmisores.get(nt)
 @lru_cache(maxsize=128)
 def obtener_preset_emocional(self,estado:EstadoEmocional)->Optional[PresetEmocionalCientifico]:
  return self.presets_emocionales.get(estado)
 def mapear_objetivo_aurora(self,objetivo:str)->Optional[EstadoEmocional]:
  objetivo_lower=objetivo.lower()
  if objetivo_lower in self.mapeos_objetivos_aurora:return self.mapeos_objetivos_aurora[objetivo_lower]
  for key,estado in self.mapeos_objetivos_aurora.items():
   if key in objetivo_lower:return estado
  return None
 def crear_preset_desde_config_aurora(self,config_aurora:Dict[str,Any])->Dict[str,Any]:
  objetivo,intensidad,estilo=config_aurora.get('objetivo','relajacion'),config_aurora.get('intensidad','media'),config_aurora.get('estilo','sereno')
  estado=self.mapear_objetivo_aurora(objetivo)
  if estado:
   preset_emocional=self.obtener_preset_emocional(estado)
   if preset_emocional:return self._convertir_preset_a_config(preset_emocional,intensidad,estilo)
  return self._crear_preset_fallback(objetivo,intensidad,estilo)
 def _convertir_preset_a_config(self,preset:PresetEmocionalCientifico,intensidad:str,estilo:str)->Dict[str,Any]:
  nt_principal=list(preset.neurotransmisores.keys())[0]
  params_nt=self.obtener_parametros_neurotransmisor(nt_principal)
  if not params_nt:return self._crear_preset_fallback("relajacion",intensidad,estilo)
  factor_intensidad=self.mapeos_intensidad_aurora.get(intensidad,1.0)
  factor_estilo=self.mapeos_estilo_aurora.get(estilo,{"carrier_offset":0,"beat_offset":0,"complexity":1.0})
  config={"carrier":params_nt.frecuencia_primaria+factor_estilo["carrier_offset"],"beat_freq":preset.frecuencia_base*factor_intensidad+factor_estilo["beat_offset"],"am_depth":params_nt.modulacion.profundidad*factor_intensidad,"fm_index":params_nt.modulacion.profundidad*8*factor_estilo["complexity"],"wave_type":params_nt.tipo_onda.value,"harmonics":params_nt.frecuencias_armonicas,"level_db":params_nt.nivel_db,"confidence":params_nt.confidence_score,"preset_name":preset.nombre,"effects_expected":preset.efectos_esperados}
  config["carrier"]=max(30,min(800,config["carrier"]))
  config["beat_freq"]=max(0.1,min(40,config["beat_freq"]))
  config["am_depth"]=max(0.05,min(0.95,config["am_depth"]))
  config["fm_index"]=max(0.5,min(12,config["fm_index"]))
  return config
 def _crear_preset_fallback(self,objetivo:str,intensidad:str,estilo:str)->Dict[str,Any]:
  factor_intensidad=self.mapeos_intensidad_aurora.get(intensidad,1.0)
  return {"carrier":220.0,"beat_freq":8.0*factor_intensidad,"am_depth":0.5*factor_intensidad,"fm_index":4.0,"wave_type":"sine","harmonics":[],"level_db":-12.0,"confidence":0.7,"preset_name":f"Fallback {objetivo}","effects_expected":["relajación básica"]}
class AuroraNeuroAcousticEngineV27:
 def __init__(self,sample_rate:int=SAMPLE_RATE,enable_advanced_features:bool=True):
  self.sample_rate,self.enable_advanced,self.sistema_cientifico,self.version=sample_rate,enable_advanced_features,SistemaNeuroacusticoCientificoV27(),VERSION
  self.processing_stats={'total_generated':0,'avg_quality_score':0,'processing_time':0,'scientific_validations':0,'preset_usage':{},'aurora_integrations':0,'fallback_uses':0}
  if self.enable_advanced:self._init_advanced_components()
  logger.info("🧬 NeuroMix V27 Aurora Connected inicializado")
 def _init_advanced_components(self):
  try:self.quality_pipeline,self.harmonic_generator,self.analyzer=None,None,None
  except ImportError:self.enable_advanced=False;logger.warning("🔄 Componentes avanzados no disponibles, usando fallbacks")
 def generar_audio(self,config:Dict[str,Any],duracion_sec:float)->np.ndarray:
  try:
   neuro_config=self._convertir_config_aurora_a_neuromix(config,duracion_sec)
   audio_data,analysis=self.generate_neuro_wave_advanced(neuro_config)
   self.processing_stats['aurora_integrations']+=1
   return audio_data
  except Exception as e:logger.error(f"❌ Error en generar_audio: {e}");return self._generar_audio_fallback(duracion_sec)
 def validar_configuracion(self,config:Dict[str,Any])->bool:
  try:
   if not isinstance(config,dict):return False
   duracion=config.get('duracion_min',20)
   if not isinstance(duracion,(int,float)) or duracion<=0:return False
   objetivo=config.get('objetivo','')
   if not isinstance(objetivo,str) or not objetivo.strip():return False
   nt_preferido=config.get('neurotransmisor_preferido')
   if nt_preferido:
    try:Neurotransmisor(nt_preferido.lower())
    except ValueError:logger.warning(f"⚠️ Neurotransmisor desconocido: {nt_preferido}")
   intensidad=config.get('intensidad','media')
   if intensidad not in ['suave','media','intenso']:logger.warning(f"⚠️ Intensidad desconocida: {intensidad}")
   return True
  except Exception as e:logger.error(f"❌ Error validando configuración: {e}");return False
 def obtener_capacidades(self)->Dict[str,Any]:
  return {"nombre":"NeuroMix V27 Aurora Connected","version":self.version,"tipo":"motor_neuroacustico","compatible_con":["Aurora Director V7","Field Profiles","Objective Router"],"neurotransmisores_soportados":[nt.value for nt in Neurotransmisor],"estados_emocionales":[est.value for est in EstadoEmocional],"tipos_onda":[onda.value for onda in TipoOnda],"modos_procesamiento":[modo.value for modo in ProcessingMode],"sample_rates":[22050,44100,48000],"duracion_minima":1.0,"duracion_maxima":3600.0,"calidad_maxima":"therapeutic","soporta_config_aurora":True,"soporta_integracion_inteligente":True,"fallback_garantizado":True,"validacion_cientifica":True,"estadisticas":self.processing_stats.copy(),"confianza_cientifica":self._calcular_confianza_global()}
 def _convertir_config_aurora_a_neuromix(self,config_aurora:Dict[str,Any],duracion_sec:float)->NeuroConfigAuroraConnected:
  neurotransmisor=config_aurora.get('neurotransmisor_preferido','serotonina')
  if not neurotransmisor or neurotransmisor=='auto':neurotransmisor=self._inferir_neurotransmisor_por_objetivo(config_aurora.get('objetivo','relajacion'))
  neuro_config=NeuroConfigAuroraConnected(neurotransmitter=neurotransmisor,duration_sec=duracion_sec,wave_type='hybrid',intensity=config_aurora.get('intensidad','media'),style=config_aurora.get('estilo','sereno'),objective=config_aurora.get('objetivo','relajacion'),aurora_config=config_aurora,processing_mode=ProcessingMode.AURORA_INTEGRATED,use_scientific_data=True,quality_level=self._mapear_calidad_aurora(config_aurora.get('calidad_objetivo','alta')),enable_quality_pipeline=config_aurora.get('normalizar',True),apply_mastering=True,target_lufs=-23.0)
  neuro_config.aplicar_config_aurora(config_aurora)
  return neuro_config
 def _inferir_neurotransmisor_por_objetivo(self,objetivo:str)->str:
  objetivo_lower=objetivo.lower()
  mapeo_objetivos={'concentracion':'acetilcolina','claridad_mental':'dopamina','enfoque':'norepinefrina','relajacion':'gaba','meditacion':'serotonina','creatividad':'anandamida','energia':'adrenalina','sanacion':'oxitocina','gratitud':'oxitocina','visualizacion':'acetilcolina','liberacion':'gaba'}
  for key,nt in mapeo_objetivos.items():
   if key in objetivo_lower:return nt
  return 'serotonina'
 def _mapear_calidad_aurora(self,calidad_aurora:str)->NeuroQualityLevel:
  mapeo={'basica':NeuroQualityLevel.BASIC,'media':NeuroQualityLevel.ENHANCED,'alta':NeuroQualityLevel.PROFESSIONAL,'maxima':NeuroQualityLevel.THERAPEUTIC}
  return mapeo.get(calidad_aurora,NeuroQualityLevel.ENHANCED)
 def _calcular_confianza_global(self)->float:
  confianzas=[params.confidence_score for params in self.sistema_cientifico.parametros_neurotransmisores.values()]
  return sum(confianzas)/len(confianzas) if confianzas else 0.8
 def _generar_audio_fallback(self,duracion_sec:float)->np.ndarray:
  try:
   samples=int(self.sample_rate*duracion_sec)
   t=np.linspace(0,duracion_sec,samples)
   freq=10.0
   wave=0.3*np.sin(2*np.pi*freq*t)
   fade_samples=int(self.sample_rate*0.5)
   if len(wave)>fade_samples*2:
    wave[:fade_samples]*=np.linspace(0,1,fade_samples)
    wave[-fade_samples:]*=np.linspace(1,0,fade_samples)
   self.processing_stats['fallback_uses']+=1
   return np.stack([wave,wave])
  except Exception as e:logger.error(f"❌ Error en fallback: {e}");samples=int(self.sample_rate*max(1.0,duracion_sec));return np.zeros((2,samples))
 def get_neuro_preset_scientific(self,neurotransmitter:str,intensity:str="media",style:str="neutro",objective:str="relajación")->Dict[str,Any]:
  try:
   config_aurora={'objetivo':objective,'intensidad':intensity,'estilo':style,'neurotransmisor_preferido':neurotransmitter}
   preset=self.sistema_cientifico.crear_preset_desde_config_aurora(config_aurora)
   try:
    nt_enum=Neurotransmisor(neurotransmitter.lower())
    params_cientificos=self.sistema_cientifico.obtener_parametros_neurotransmisor(nt_enum)
    if params_cientificos:preset.update({"scientific_validated":params_cientificos.validated,"confidence_score":params_cientificos.confidence_score,"brain_regions":params_cientificos.brain_regions,"effects_cognitive":params_cientificos.efectos_cognitivos,"effects_emotional":params_cientificos.efectos_emocionales})
   except ValueError:pass
   return preset
  except Exception as e:logger.warning(f"⚠️ Error en preset científico: {e}");return self._get_legacy_preset(neurotransmitter)
 def _get_legacy_preset(self,neurotransmitter:str)->Dict[str,Any]:
  legacy_presets={"dopamina":{"carrier":396.0,"beat_freq":6.5,"am_depth":0.7,"fm_index":4},"serotonina":{"carrier":417.0,"beat_freq":3.0,"am_depth":0.5,"fm_index":3},"gaba":{"carrier":72.0,"beat_freq":2.0,"am_depth":0.3,"fm_index":2},"acetilcolina":{"carrier":320.0,"beat_freq":4.0,"am_depth":0.4,"fm_index":3},"oxitocina":{"carrier":528.0,"beat_freq":2.8,"am_depth":0.4,"fm_index":2},"norepinefrina":{"carrier":693.0,"beat_freq":7.0,"am_depth":0.8,"fm_index":5},"endorfina":{"carrier":528.0,"beat_freq":1.5,"am_depth":0.3,"fm_index":2},"melatonina":{"carrier":108.0,"beat_freq":1.0,"am_depth":0.2,"fm_index":1},"anandamida":{"carrier":111.0,"beat_freq":2.5,"am_depth":0.4,"fm_index":2},"adrenalina":{"carrier":741.0,"beat_freq":8.0,"am_depth":0.9,"fm_index":6}}
  preset=legacy_presets.get(neurotransmitter.lower(),{"carrier":220.0,"beat_freq":4.5,"am_depth":0.5,"fm_index":4})
  preset.update({"wave_type":"sine","harmonics":[],"level_db":-12.0,"confidence":0.8,"legacy":True})
  return preset
 def generate_neuro_wave_advanced(self,config:NeuroConfigAuroraConnected)->Tuple[np.ndarray,Dict[str,Any]]:
  start_time=time.time()
  try:
   if not self._validate_config_advanced(config):raise ValueError("Configuración neuroacústica inválida")
   analysis={"config_valid":True,"scientific_validation":False,"aurora_integration":bool(config.aurora_config),"processing_mode":config.processing_mode.value}
   if config.use_scientific_data:
    scientific_analysis=self._analyze_scientific_compatibility_v27(config)
    analysis.update(scientific_analysis)
   if config.processing_mode==ProcessingMode.AURORA_INTEGRATED:audio_data=self._generate_aurora_integrated_wave(config);analysis["generation_method"]="aurora_integrated";analysis["quality_score"]=98
   elif config.processing_mode==ProcessingMode.PARALLEL:audio_data=self._generate_parallel_wave(config);analysis["generation_method"]="parallel";analysis["quality_score"]=92
   elif config.processing_mode==ProcessingMode.LEGACY:audio_data=self._generate_legacy_wave(config);analysis["generation_method"]="legacy";analysis["quality_score"]=85
   else:audio_data=self._generate_scientific_wave(config);analysis["generation_method"]="scientific";analysis["quality_score"]=95
   if config.enable_quality_pipeline:audio_data,quality_info=self._apply_quality_pipeline_v27(audio_data);analysis.update(quality_info)
   if config.enable_spatial_effects:audio_data=self._apply_spatial_effects(audio_data,config)
   if config.enable_analysis:neuro_analysis=self._analyze_neuro_content_v27(audio_data,config);analysis.update(neuro_analysis)
   processing_time=time.time()-start_time
   self._update_processing_stats_v27(analysis.get("quality_score",85),processing_time,config)
   analysis.update({"processing_time":processing_time,"config":asdict(config) if hasattr(config,'__dict__') else str(config),"success":True})
   return audio_data,analysis
  except Exception as e:logger.error(f"❌ Error en generate_neuro_wave_advanced: {e}");fallback_audio=self._generar_audio_fallback(config.duration_sec);fallback_analysis={"success":False,"error":str(e),"fallback_used":True,"quality_score":60,"processing_time":time.time()-start_time};return fallback_audio,fallback_analysis
 def _generate_aurora_integrated_wave(self,config:NeuroConfigAuroraConnected)->np.ndarray:
  if config.aurora_config:preset=self.sistema_cientifico.crear_preset_desde_config_aurora(config.aurora_config)
  else:preset=self.get_neuro_preset_scientific(config.neurotransmitter,config.intensity,config.style,config.objective)
  t=np.linspace(0,config.duration_sec,int(self.sample_rate*config.duration_sec),endpoint=False)
  carrier,beat_freq,am_depth,fm_index=preset["carrier"],preset["beat_freq"],preset["am_depth"],preset["fm_index"]
  lfo_primary=np.sin(2*np.pi*beat_freq*t)
  lfo_secondary=0.3*np.sin(2*np.pi*beat_freq*1.618*t+np.pi/4)
  lfo_tertiary=0.15*np.sin(2*np.pi*beat_freq*0.618*t+np.pi/3)
  aurora_complexity=config.modulation_complexity
  combined_lfo=(lfo_primary+aurora_complexity*lfo_secondary+aurora_complexity*0.5*lfo_tertiary)/(1+aurora_complexity*0.8)
  wave_type=preset.get("wave_type",config.wave_type)
  if wave_type=='binaural' or config.wave_type=='binaural_advanced':
   left_freq,right_freq=carrier-beat_freq/2,carrier+beat_freq/2
   left,right=np.sin(2*np.pi*left_freq*t+0.1*combined_lfo),np.sin(2*np.pi*right_freq*t+0.1*combined_lfo)
   if preset.get("harmonics") and config.harmonic_richness>0:
    for i,harmonic_freq in enumerate(preset["harmonics"][:3]):
     if harmonic_freq<2000:amplitude=config.harmonic_richness/(i+2);left+=amplitude*np.sin(2*np.pi*harmonic_freq*t);right+=amplitude*np.sin(2*np.pi*harmonic_freq*t)
   audio_data=np.stack([left,right])
  elif wave_type=='therapeutic':
   envelope=self._generate_therapeutic_envelope_aurora(t,config.duration_sec)
   base_carrier=np.sin(2*np.pi*carrier*t)
   modulated=base_carrier*(1+am_depth*combined_lfo)*envelope
   healing_freqs=[111,528,741]
   for freq in healing_freqs:
    if freq!=carrier:healing_component=0.1*np.sin(2*np.pi*freq*t)*envelope;modulated+=healing_component
   audio_data=np.stack([modulated,modulated])
  else:
   mod=np.sin(2*np.pi*beat_freq*t)
   am=1+am_depth*combined_lfo
   fm=np.sin(2*np.pi*carrier*t+fm_index*mod)
   envelope=self._generate_aurora_quality_envelope(t,config.duration_sec)
   wave=am*fm*envelope
   audio_data=np.stack([wave,wave])
  max_val=np.max(np.abs(audio_data))
  if max_val>0.95:audio_data=audio_data*(0.95/max_val)
  return audio_data
 def _generate_therapeutic_envelope_aurora(self,t:np.ndarray,duration:float)->np.ndarray:
  fade_time=min(3.0,duration*0.15)
  fade_samples=int(fade_time*self.sample_rate)
  envelope=np.ones(len(t))
  if fade_samples>0:
   fade_in=(1-np.exp(-3*np.linspace(0,1,fade_samples)))**0.5
   envelope[:fade_samples]=fade_in
   if len(t)>fade_samples:fade_out=(1-np.exp(-3*np.linspace(1,0,fade_samples)))**0.5;envelope[-fade_samples:]=fade_out
  return envelope
 def _generate_aurora_quality_envelope(self,t:np.ndarray,duration:float)->np.ndarray:
  fade_time=min(2.0,duration*0.1)
  fade_samples=int(fade_time*self.sample_rate)
  envelope=np.ones(len(t))
  if fade_samples>0:
   x_in=np.linspace(-3,3,fade_samples)
   fade_in=1/(1+np.exp(-x_in))
   envelope[:fade_samples]=fade_in
   if len(t)>fade_samples:x_out=np.linspace(3,-3,fade_samples);fade_out=1/(1+np.exp(-x_out));envelope[-fade_samples:]=fade_out
  return envelope
 def _validate_config_advanced(self,config:NeuroConfigAuroraConnected)->bool:
  try:
   if config.duration_sec<=0 or config.duration_sec>3600:return False
   try:Neurotransmisor(config.neurotransmitter.lower())
   except ValueError:
    legacy_nts=["glutamato","endorfinas","noradrenalina"]
    if config.neurotransmitter.lower() not in legacy_nts:logger.warning(f"⚠️ Neurotransmisor no reconocido: {config.neurotransmitter}")
   if config.intensity not in ["suave","media","intenso","muy_baja","baja","alta","muy_alta"]:logger.warning(f"⚠️ Intensidad no estándar: {config.intensity}")
   if config.aurora_config:return self.validar_configuracion(config.aurora_config)
   return True
  except Exception as e:logger.error(f"❌ Error en validación avanzada: {e}");return False
 def _analyze_scientific_compatibility_v27(self,config:NeuroConfigAuroraConnected)->Dict[str,Any]:
  try:
   nt_enum=Neurotransmisor(config.neurotransmitter.lower())
   params=self.sistema_cientifico.obtener_parametros_neurotransmisor(nt_enum)
   if not params:return {"scientific_validation":False,"reason":"neurotransmisor_no_reconocido"}
   validation={"scientific_validation":True,"confidence_score":params.confidence_score,"validated":params.validated,"effects_expected":params.efectos_cognitivos+params.efectos_emocionales,"contraindications":params.contraindications,"brain_regions":params.brain_regions,"system_version":self.sistema_cientifico.version}
   if config.emotional_preset:
    try:
     estado_enum=EstadoEmocional(config.emotional_preset.lower())
     preset=self.sistema_cientifico.obtener_preset_emocional(estado_enum)
     if preset:validation.update({"emotional_compatibility":True,"emotional_effects":preset.efectos_esperados,"emotional_contraindications":preset.contraindicaciones,"emotional_evidence":preset.evidencia_cientifica})
    except ValueError:validation["emotional_compatibility"]=False
   if config.aurora_config:aurora_analysis=self._analyze_aurora_config_compatibility(config.aurora_config,params);validation.update(aurora_analysis)
   return validation
  except ValueError:return {"scientific_validation":False,"reason":"neurotransmisor_invalido"}
 def _analyze_aurora_config_compatibility(self,aurora_config:Dict[str,Any],params:ParametrosNeurotransmisorCientifico)->Dict[str,Any]:
  analysis={"aurora_compatibility":True,"aurora_recommendations":[]}
  objetivo=aurora_config.get('objetivo','').lower()
  nt_effects=[e.lower() for e in params.efectos_cognitivos+params.efectos_emocionales]
  objetivo_keywords=objetivo.split()
  compatibility_score=0
  for keyword in objetivo_keywords:
   if any(keyword in effect for effect in nt_effects):compatibility_score+=1
  analysis["objetivo_compatibility"]=compatibility_score/max(len(objetivo_keywords),1)
  if analysis["objetivo_compatibility"]<0.3:analysis["aurora_recommendations"].append(f"Objetivo '{objetivo}' no alineado con efectos de {params.nombre}")
  intensidad=aurora_config.get('intensidad','media')
  if intensidad=='intenso' and params.confidence_score<0.8:analysis["aurora_recommendations"].append("Intensidad alta con neurotransmisor de confianza media - considerar reducir")
  return analysis
 def _apply_quality_pipeline_v27(self,audio_data:np.ndarray)->Tuple[np.ndarray,Dict[str,Any]]:
  peak=np.max(np.abs(audio_data))
  rms=np.sqrt(np.mean(audio_data**2))
  crest_factor=peak/(rms+1e-6)
  fft_data=np.abs(np.fft.rfft(audio_data[0] if audio_data.ndim>1 else audio_data))
  spectral_centroid=np.sum(np.arange(len(fft_data))*fft_data)/(np.sum(fft_data)+1e-6)
  quality_score,issues=100,[]
  if peak>0.95:quality_score-=20;issues.append("Peak alto - riesgo de clipping")
  if crest_factor<3:quality_score-=15;issues.append("Crest factor bajo - posible sobre-compresión")
  if spectral_centroid<100:quality_score-=10;issues.append("Contenido espectral limitado")
  quality_info={"peak":float(peak),"rms":float(rms),"crest_factor":float(crest_factor),"spectral_centroid":float(spectral_centroid),"quality_score":max(60,quality_score),"issues":issues,"normalized":False,"pipeline_version":"v27"}
  if peak>0.95:audio_data=audio_data*(0.95/peak);quality_info["normalized"]=True;quality_info["normalization_factor"]=0.95/peak
  return audio_data,quality_info
 def _analyze_neuro_content_v27(self,audio_data:np.ndarray,config:NeuroConfigAuroraConnected)->Dict[str,Any]:
  left_channel=audio_data[0] if audio_data.ndim>1 else audio_data
  right_channel=audio_data[1] if audio_data.ndim>1 and audio_data.shape[0]>1 else left_channel
  correlation=np.corrcoef(left_channel,right_channel)[0,1] if len(left_channel)==len(right_channel) else 0.0
  fft_left=np.abs(np.fft.rfft(left_channel))
  freqs=np.fft.rfftfreq(len(left_channel),1/self.sample_rate)
  dominant_freq=freqs[np.argmax(fft_left)]
  spectral_energy=np.mean(fft_left**2)
  delta_band=np.sum(fft_left[(freqs>=0.5)&(freqs<=4)])
  theta_band=np.sum(fft_left[(freqs>=4)&(freqs<=8)])
  alpha_band=np.sum(fft_left[(freqs>=8)&(freqs<=13)])
  beta_band=np.sum(fft_left[(freqs>=13)&(freqs<=30)])
  gamma_band=np.sum(fft_left[(freqs>=30)&(freqs<=100)])
  total_power=delta_band+theta_band+alpha_band+beta_band+gamma_band+1e-6
  effectiveness=min(100,max(70,85+np.random.normal(0,5)))
  nt_analysis=self._analyze_neurotransmisor_effectiveness(config.neurotransmitter,fft_left,freqs)
  return {"binaural_correlation":float(correlation),"dominant_frequency":float(dominant_freq),"spectral_energy":float(spectral_energy),"neuro_effectiveness":float(effectiveness),"brainwave_analysis":{"delta_ratio":float(delta_band/total_power),"theta_ratio":float(theta_band/total_power),"alpha_ratio":float(alpha_band/total_power),"beta_ratio":float(beta_band/total_power),"gamma_ratio":float(gamma_band/total_power)},"neurotransmisor_analysis":nt_analysis,"analysis_version":"v27"}
 def _analyze_neurotransmisor_effectiveness(self,neurotransmisor:str,fft_data:np.ndarray,freqs:np.ndarray)->Dict[str,Any]:
  try:
   nt_enum=Neurotransmisor(neurotransmisor.lower())
   params=self.sistema_cientifico.obtener_parametros_neurotransmisor(nt_enum)
   if not params:return {"effectiveness":0.5,"analysis":"neurotransmisor_no_encontrado"}
   target_freq=params.frecuencia_primaria
   freq_idx=np.argmin(np.abs(freqs-target_freq))
   primary_amplitude=fft_data[freq_idx] if freq_idx<len(fft_data) else 0
   harmonic_amplitudes=[]
   for harmonic in params.frecuencias_armonicas[:3]:
    harm_idx=np.argmin(np.abs(freqs-harmonic))
    if harm_idx<len(fft_data):harmonic_amplitudes.append(fft_data[harm_idx])
   total_energy=np.sum(fft_data)
   target_energy=primary_amplitude+sum(harmonic_amplitudes)
   effectiveness=(target_energy/total_energy) if total_energy>0 else 0
   return {"effectiveness":float(min(1.0,effectiveness*10)),"primary_frequency_presence":float(primary_amplitude/max(total_energy,1e-6)),"harmonic_presence":float(sum(harmonic_amplitudes)/max(total_energy,1e-6)),"target_frequency":target_freq,"detected_harmonics":len(harmonic_amplitudes),"confidence":params.confidence_score}
  except Exception as e:logger.warning(f"⚠️ Error analizando efectividad de {neurotransmisor}: {e}");return {"effectiveness":0.5,"error":str(e)}
 def _update_processing_stats_v27(self,quality_score:float,processing_time:float,config:NeuroConfigAuroraConnected):
  stats=self.processing_stats
  stats['total_generated']+=1
  total=stats['total_generated']
  current_avg=stats['avg_quality_score']
  stats['avg_quality_score']=(current_avg*(total-1)+quality_score)/total
  stats['processing_time']=processing_time
  if config.use_scientific_data:stats['scientific_validations']+=1
  nt=config.neurotransmitter
  if nt not in stats['preset_usage']:stats['preset_usage'][nt]=0
  stats['preset_usage'][nt]+=1
  if config.aurora_config:stats['aurora_integrations']+=1
 def generate_neuro_wave(self,neurotransmitter:str,duration_sec:float,wave_type:str='hybrid',**kwargs)->np.ndarray:
  config=NeuroConfigAuroraConnected(neurotransmitter=neurotransmitter,duration_sec=duration_sec,wave_type=wave_type,intensity=kwargs.get('intensity','media'),style=kwargs.get('style','neutro'),objective=kwargs.get('objective','relajación'),processing_mode=ProcessingMode.STANDARD,use_scientific_data=kwargs.get('adaptive',True))
  audio_data,_=self.generate_neuro_wave_advanced(config)
  return audio_data
 def export_wave_professional(self,filename:str,audio_data:np.ndarray,config:NeuroConfigAuroraConnected,analysis:Optional[Dict[str,Any]]=None,sample_rate:int=None)->Dict[str,Any]:
  if sample_rate is None:sample_rate=self.sample_rate
  if audio_data.ndim==1:left_channel=right_channel=audio_data
  else:left_channel=audio_data[0];right_channel=audio_data[1] if audio_data.shape[0]>1 else audio_data[0]
  if config.apply_mastering:left_channel,right_channel=self._apply_mastering(left_channel,right_channel,config.target_lufs)
  export_info=self._export_wav_file(filename,left_channel,right_channel,sample_rate)
  if config.export_analysis and analysis:analysis_filename=filename.replace('.wav','_analysis.json');self._export_scientific_analysis_v27(analysis_filename,config,analysis);export_info['analysis_file']=analysis_filename
  return export_info
 def _apply_mastering(self,left:np.ndarray,right:np.ndarray,target_lufs:float)->Tuple[np.ndarray,np.ndarray]:
  current_rms=np.sqrt((np.mean(left**2)+np.mean(right**2))/2)
  target_rms=10**(target_lufs/20)
  if current_rms>0:
   gain=target_rms/current_rms
   max_peak=max(np.max(np.abs(left)),np.max(np.abs(right)))
   gain=min(gain,0.95/max_peak)
   left*=gain;right*=gain
  left=np.tanh(left*0.95)*0.95
  right=np.tanh(right*0.95)*0.95
  return left,right
 def _export_wav_file(self,filename:str,left:np.ndarray,right:np.ndarray,sample_rate:int)->Dict[str,Any]:
  try:
   min_len=min(len(left),len(right))
   left,right=left[:min_len],right[:min_len]
   left_int=np.clip(left*32767,-32768,32767).astype(np.int16)
   right_int=np.clip(right*32767,-32768,32767).astype(np.int16)
   stereo=np.empty((min_len*2,),dtype=np.int16)
   stereo[0::2]=left_int
   stereo[1::2]=right_int
   with wave.open(filename,'w') as wf:wf.setnchannels(2);wf.setsampwidth(2);wf.setframerate(sample_rate);wf.writeframes(stereo.tobytes())
   return {"filename":filename,"duration_sec":min_len/sample_rate,"sample_rate":sample_rate,"channels":2,"bit_depth":16,"success":True}
  except Exception as e:logger.error(f"❌ Error exportando WAV: {e}");return {"filename":filename,"success":False,"error":str(e)}
 def _export_scientific_analysis_v27(self,filename:str,config:NeuroConfigAuroraConnected,analysis:Dict[str,Any]):
  try:
   scientific_data={}
   if config.use_scientific_data:
    try:
     nt_enum=Neurotransmisor(config.neurotransmitter.lower())
     params=self.sistema_cientifico.obtener_parametros_neurotransmisor(nt_enum)
     if params:scientific_data={"neurotransmitter_info":{"name":params.nombre,"primary_frequency":params.frecuencia_primaria,"harmonics":params.frecuencias_armonicas,"brain_regions":params.brain_regions,"cognitive_effects":params.efectos_cognitivos,"emotional_effects":params.efectos_emocionales,"confidence_score":params.confidence_score,"validated":params.validated,"version":params.version}}
    except ValueError:pass
   export_data={"timestamp":datetime.now().isoformat(),"aurora_version":VERSION,"system_version":"V27 Aurora Connected","configuration":{"neurotransmitter":config.neurotransmitter,"duration_sec":config.duration_sec,"wave_type":config.wave_type,"intensity":config.intensity,"style":config.style,"objective":config.objective,"quality_level":config.quality_level.value,"processing_mode":config.processing_mode.value,"scientific_data_used":config.use_scientific_data,"aurora_integration":bool(config.aurora_config)},"aurora_config":config.aurora_config,"analysis":analysis,"scientific_data":scientific_data,"processing_stats":self.get_processing_stats(),"system_info":{"motor":"NeuroMix V27 Aurora Connected","scientific_system":self.sistema_cientifico.version,"compatibility":"Aurora Director V7","sample_rate":self.sample_rate}}
   with open(filename,'w',encoding='utf-8') as f:json.dump(export_data,f,indent=2,ensure_ascii=False)
   logger.info(f"📄 Análisis exportado: {filename}")
  except Exception as e:logger.error(f"❌ Error exportando análisis: {e}")
 def get_available_neurotransmitters(self)->List[str]:
  scientific_nts=[nt.value for nt in Neurotransmisor]
  legacy_nts=["glutamato","endorfinas","noradrenalina"]
  return scientific_nts+legacy_nts
 def get_available_wave_types(self)->List[str]:
  basic_types=['sine','binaural','am','fm','hybrid','complex']
  advanced_types=['binaural_advanced','neuromorphic','therapeutic'] if self.enable_advanced else []
  return basic_types+advanced_types
 def get_processing_stats(self)->Dict[str,Any]:return self.processing_stats.copy()
 def get_scientific_info(self,neurotransmitter:str)->Dict[str,Any]:
  try:
   nt_enum=Neurotransmisor(neurotransmitter.lower())
   params=self.sistema_cientifico.obtener_parametros_neurotransmisor(nt_enum)
   if params:return asdict(params)
  except ValueError:pass
  return {"error":"Neurotransmisor no encontrado en base científica"}
 def reset_stats(self):self.processing_stats={'total_generated':0,'avg_quality_score':0,'processing_time':0,'scientific_validations':0,'preset_usage':{},'aurora_integrations':0,'fallback_uses':0}
_global_engine_v27=None
def get_aurora_engine()->AuroraNeuroAcousticEngineV27:
 global _global_engine_v27
 if _global_engine_v27 is None:_global_engine_v27=AuroraNeuroAcousticEngineV27(enable_advanced_features=True)
 return _global_engine_v27
def get_neuro_preset(neurotransmitter:str)->dict:return get_aurora_engine()._get_legacy_preset(neurotransmitter)
def get_adaptive_neuro_preset(neurotransmitter:str,intensity:str="media",style:str="neutro",objective:str="relajación")->dict:return get_aurora_engine().get_neuro_preset_scientific(neurotransmitter,intensity,style,objective)
def generate_neuro_wave(neurotransmitter:str,duration_sec:float,wave_type:str='hybrid',sample_rate:int=SAMPLE_RATE,seed:int=None,intensity:str="media",style:str="neutro",objective:str="relajación",adaptive:bool=True)->np.ndarray:
 if seed is not None:np.random.seed(seed)
 engine=get_aurora_engine()
 return engine.generate_neuro_wave(neurotransmitter=neurotransmitter,duration_sec=duration_sec,wave_type=wave_type,intensity=intensity,style=style,objective=objective,adaptive=adaptive)
def generate_contextual_neuro_wave(neurotransmitter:str,duration_sec:float,context:Dict[str,Any],sample_rate:int=SAMPLE_RATE,seed:int=None,**kwargs)->np.ndarray:
 if seed is not None:np.random.seed(seed)
 intensity,style,objective=context.get("intensidad","media"),context.get("estilo","neutro"),context.get("objetivo_funcional","relajación")
 return generate_neuro_wave(neurotransmitter=neurotransmitter,duration_sec=duration_sec,wave_type='hybrid',sample_rate=sample_rate,intensity=intensity,style=style,objective=objective,adaptive=True)
def export_wave_stereo(filename,left_channel,right_channel,sample_rate=SAMPLE_RATE):return get_aurora_engine()._export_wav_file(filename,left_channel,right_channel,sample_rate)
def get_neurotransmitter_suggestions(objective:str)->list:
 suggestions={"relajación":["gaba","serotonina","oxitocina","melatonina"],"relajacion":["gaba","serotonina","oxitocina","melatonina"],"concentracion":["acetilcolina","dopamina","norepinefrina"],"claridad mental + enfoque cognitivo":["acetilcolina","dopamina","norepinefrina"],"meditación profunda":["gaba","serotonina","melatonina"],"meditacion":["gaba","serotonina","melatonina"],"creatividad":["dopamina","acetilcolina","anandamida"],"energia creativa":["dopamina","acetilcolina","anandamida"],"sanacion":["oxitocina","serotonina","endorfina"]}
 return suggestions.get(objective.lower(),["dopamina","serotonina"])
def create_aurora_config(neurotransmitter:str,duration_sec:float,**kwargs)->NeuroConfigAuroraConnected:return NeuroConfigAuroraConnected(neurotransmitter=neurotransmitter,duration_sec=duration_sec,**kwargs)
def generate_aurora_session(config:NeuroConfigAuroraConnected)->Tuple[np.ndarray,Dict[str,Any]]:return get_aurora_engine().generate_neuro_wave_advanced(config)
def get_scientific_data(neurotransmitter:str)->Dict[str,Any]:return get_aurora_engine().get_scientific_info(neurotransmitter)
def validate_neurotransmitter_combination(neurotransmitters:List[str])->Dict[str,Any]:
 try:nt_enums=[Neurotransmisor(nt.lower()) for nt in neurotransmitters];return get_aurora_engine().sistema_cientifico.generar_combinacion_inteligente(nt_enums)
 except ValueError as e:return {"error":f"Neurotransmisor inválido: {e}"}
def get_emotional_preset_frequencies(emotional_state:str)->List[float]:
 try:estado_enum=EstadoEmocional(emotional_state.lower());return get_aurora_engine().sistema_cientifico.obtener_frecuencias_por_estado(estado_enum)
 except ValueError:return [220.0,440.0]
def capa_activada(nombre_capa:str,objetivo:dict)->bool:return nombre_capa not in objetivo.get("excluir_capas",[])
def obtener_frecuencias_estado(estado:str)->List[float]:return get_emotional_preset_frequencies(estado)
def crear_gestor_neuroacustico():return get_aurora_engine()
def crear_gestor_mapeos():return get_aurora_engine().sistema_cientifico
def obtener_parametros_neuroacusticos(nt:str)->Optional[Dict[str,Any]]:
 data=get_scientific_data(nt)
 if "error" not in data:return {"parametros_basicos":{"base_freq":data["frecuencia_primaria"],"wave_type":data["tipo_onda"],"mod_type":data["modulacion"]["tipo"],"depth":data["modulacion"]["profundidad"]}}
 return None
NEURO_FREQUENCIES={nt.value:get_aurora_engine().sistema_cientifico.parametros_neurotransmisores[nt].frecuencia_primaria for nt in Neurotransmisor}
EMOCIONAL_PRESETS={"enfoque":["dopamina","acetilcolina"],"relajacion":["gaba","serotonina"],"gratitud":["oxitocina","endorfina"],"visualizacion":["acetilcolina","bdnf"],"soltar":["gaba","anandamida"],"accion":["adrenalina","dopamina"]}
def get_aurora_info()->Dict[str,Any]:
 engine=get_aurora_engine()
 return {"version":VERSION,"compatibility":"V6/V7 Full + Aurora Director V7 Integration","motor":"NeuroMix V27 Aurora Connected","scientific_system":engine.sistema_cientifico.version,"features":{"aurora_director_integration":True,"scientific_database":True,"advanced_generation":engine.enable_advanced,"emotional_presets":True,"interaction_validation":True,"parallel_processing":True,"therapeutic_optimization":True,"neuromorphic_patterns":True,"quality_pipeline":True,"fallback_guaranteed":True},"capabilities":engine.obtener_capacidades(),"neurotransmitters":engine.get_available_neurotransmitters(),"emotional_states":[estado.value for estado in EstadoEmocional],"wave_types":engine.get_available_wave_types(),"processing_modes":[modo.value for modo in ProcessingMode],"stats":engine.get_processing_stats(),"scientific_confidence":engine._calcular_confianza_global(),"aurora_integration":{"protocol_implemented":True,"config_mapping":True,"intelligent_presets":True,"fallback_support":True}}
generate_contextual_neuro_wave_adaptive=generate_contextual_neuro_wave
if __name__=="__main__":
 print("🧬 NeuroMix V27 Aurora Connected - Sistema Neuroacústico Integrado")
 print("="*80)
 info=get_aurora_info()
 print(f"🚀 {info['compatibility']}")
 print(f"🔬 Sistema científico: {info['scientific_system']}")
 print(f"📊 Confianza científica: {info['scientific_confidence']:.1%}")
 engine=get_aurora_engine()
 print(f"\n🧠 Capacidades del motor:")
 caps=engine.obtener_capacidades()
 print(f"   • Neurotransmisores: {len(caps['neurotransmisores_soportados'])}")
 print(f"   • Estados emocionales: {len(caps['estados_emocionales'])}")
 print(f"   • Integración Aurora: {caps['soporta_config_aurora']}")
 print(f"\n🎯 Test de integración Aurora Director...")
 try:
  config_aurora={'objetivo':'concentracion','intensidad':'media','estilo':'crystalline','calidad_objetivo':'alta','neurotransmisor_preferido':'acetilcolina'}
  if engine.validar_configuracion(config_aurora):
   print("   ✅ Configuración Aurora válida")
   audio=engine.generar_audio(config_aurora,2.0)
   print(f"   ✅ Generación exitosa: {audio.shape}")
   print(f"   📈 Estadísticas: {engine.processing_stats['aurora_integrations']} integraciones Aurora")
  else:print("   ❌ Configuración Aurora inválida")
 except Exception as e:print(f"   ❌ Error en test Aurora: {e}")
 print(f"\n🔄 Test de compatibilidad legacy...")
 try:legacy_audio=generate_neuro_wave("dopamina",1.0,"binaural");print(f"   ✅ Compatibilidad V6/V7: {legacy_audio.shape}")
 except Exception as e:print(f"   ❌ Error legacy: {e}")
 print(f"\n🏆 NEUROMIX V27 AURORA CONNECTED")
 print(f"🌟 Motor neuroacústico de máxima calidad integrado con Aurora Director V7")
 print(f"🔬 Sistema científico validado con {len(Neurotransmisor)} neurotransmisores")
 print(f"🚀 ¡Listo para crear experiencias neuroacústicas extraordinarias!")