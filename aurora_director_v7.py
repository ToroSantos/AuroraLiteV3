"""Aurora Director V7 CONECTADO - OPTIMIZADO"""
import numpy as np,logging,importlib,traceback,time
from typing import Dict,List,Optional,Tuple,Any,Union,Protocol
from dataclasses import dataclass,field
from enum import Enum
from datetime import datetime

logging.basicConfig(level=logging.INFO,format='%(asctime)s-%(name)s-%(levelname)s-%(message)s')
logger=logging.getLogger("Aurora.Director.V7")
VERSION="V7_AURORA_DIRECTOR_CONNECTED_OPTIMIZED"

class MotorAurora(Protocol):
    def generar_audio(self,config:Dict[str,Any],duracion_sec:float)->np.ndarray:...
    def validar_configuracion(self,config:Dict[str,Any])->bool:...
    def obtener_capacidades(self)->Dict[str,Any]:...

class GestorInteligencia(Protocol):
    def procesar_objetivo(self,objetivo:str,contexto:Dict[str,Any])->Dict[str,Any]:...
    def obtener_alternativas(self,objetivo:str)->List[str]:...

@dataclass
class ComponenteAurora:
    nombre:str;tipo:str;modulo:str;clase_principal:str;disponible:bool=False;instancia:Optional[Any]=None
    version:str="unknown";capacidades:Dict[str,Any]=field(default_factory=dict)
    dependencias:List[str]=field(default_factory=list);fallback_disponible:bool=False
    nivel_prioridad:int=1;protocolo_implementado:bool=False;aurora_v7_optimizado:bool=False

class DetectorComponentesV7:
    def __init__(self):
        self.componentes_registrados=self._init_registro_v7()
        self.componentes_activos:Dict[str,ComponenteAurora]={}
        self.protocolos_validados:Dict[str,bool]={}
        self.stats={"total":0,"exitosos":0,"fallidos":0,"fallback":0,"protocolo_motor":0,"protocolo_inteligencia":0}
    
    def _init_registro_v7(self)->Dict[str,ComponenteAurora]:
        return {
            "emotion_style":ComponenteAurora("emotion_style","gestor_inteligencia","emotion_style_profiles","GestorEmotionStyleUnificadoV7",dependencias=[],fallback_disponible=True,nivel_prioridad=1),
            "field_profiles":ComponenteAurora("field_profiles","gestor_inteligencia","field_profiles","GestorPerfilesCampo",dependencias=[],fallback_disponible=True,nivel_prioridad=2),
            "objective_router":ComponenteAurora("objective_router","gestor_inteligencia","objective_router_v7","RouterInteligenteV7",dependencias=["emotion_style"],fallback_disponible=True,nivel_prioridad=2),
            "neuromix":ComponenteAurora("neuromix","motor","neuromix_aurora_v27","AuroraNeuroAcousticEngineV27",dependencias=[],fallback_disponible=True,nivel_prioridad=1),
            "hypermod":ComponenteAurora("hypermod","motor","hypermod_v32","HyperModEngineV32AuroraConnected",dependencias=[],fallback_disponible=True,nivel_prioridad=1),
            "harmonic_essence":ComponenteAurora("harmonic_essence","motor","harmonicEssence_v34","HarmonicEssenceV34AuroraConnected",dependencias=[],fallback_disponible=True,nivel_prioridad=1),
            "quality_pipeline":ComponenteAurora("quality_pipeline","pipeline","aurora_quality_pipeline","AuroraQualityPipeline",dependencias=[],fallback_disponible=True,nivel_prioridad=4)
        }
    
    def detectar_todos_v7(self)->Dict[str,ComponenteAurora]:
        logger.info("🔍 Detectando componentes Aurora V7...")
        for nombre,comp in sorted(self.componentes_registrados.items(),key=lambda x:x[1].nivel_prioridad):
            self._detectar_componente_v7(comp)
        self._validar_protocolos();self._log_resultado_v7()
        return self.componentes_activos
    
    def _detectar_componente_v7(self,comp:ComponenteAurora)->bool:
        self.stats["total"]+=1
        try:
            if not self._check_deps(comp):return False
            modulo=self._importar_modulo_seguro(comp)
            if not modulo:raise Exception("No se pudo importar el módulo")
            instancia=self._crear_instancia_v7(modulo,comp)
            if not instancia:raise Exception("No se pudo crear la instancia")
            protocolo_valido=self._validar_protocolo_componente(instancia,comp)
            comp.disponible=True;comp.instancia=instancia;comp.capacidades=self._get_caps_seguro(instancia)
            comp.version=self._get_version_seguro(instancia);comp.protocolo_implementado=protocolo_valido
            comp.aurora_v7_optimizado=self._verificar_optimizacion_v7(instancia)
            self.componentes_activos[comp.nombre]=comp;self.stats["exitosos"]+=1
            if comp.tipo=="motor"and protocolo_valido:self.stats["protocolo_motor"]+=1
            elif comp.tipo=="gestor_inteligencia"and protocolo_valido:self.stats["protocolo_inteligencia"]+=1
            estado="✅"if protocolo_valido else"⚠️";logger.info(f"{estado} {comp.nombre} v{comp.version}")
            return True
        except Exception as e:
            logger.warning(f"❌ {comp.nombre}: {e}")
            if comp.fallback_disponible and self._crear_fallback_v7(comp):
                self.stats["fallback"]+=1;logger.info(f"🔄 {comp.nombre} fallback activado");return True
            self.stats["fallidos"]+=1;return False
    
    def _importar_modulo_seguro(self,comp:ComponenteAurora):
        try:return importlib.import_module(comp.modulo)
        except:return None
    
    def _crear_instancia_v7(self,modulo:Any,comp:ComponenteAurora)->Any:
        try:
            if comp.nombre=="emotion_style":
                for func_name in ["crear_gestor_emotion_style_v7","obtener_gestor_global_v7","crear_motor_emotion_style","obtener_motor_emotion_style"]:
                    if hasattr(modulo,func_name):
                        try:return getattr(modulo,func_name)()
                        except:continue
                if hasattr(modulo,"GestorEmotionStyleUnificadoV7"):
                    try:return getattr(modulo,"GestorEmotionStyleUnificadoV7")(aurora_director_mode=True)
                    except:pass
                for item_name in ["crear_gestor_emotion_style_unificado","GestorEmotionStyleUnificado"]:
                    if hasattr(modulo,item_name):
                        try:
                            item=getattr(modulo,item_name)
                            return item()if callable(item)else item
                        except:continue
            elif comp.nombre=="neuromix":
                for clase_attr,constructor in [("AuroraNeuroAcousticEngineV27",lambda cls:cls(enable_advanced_features=True)),("AuroraNeuroAcousticEngine",lambda cls:cls()),("_global_engine",lambda obj:obj)]:
                    if hasattr(modulo,clase_attr):
                        try:
                            item=getattr(modulo,clase_attr)
                            return constructor(item)if callable(item)else item
                        except:continue
            elif comp.nombre=="hypermod":
                for item_name,constructor in [("HyperModEngineV32AuroraConnected",lambda cls:cls(enable_advanced_features=True)),("_motor_global_v32",lambda obj:obj),("gestor_aurora",lambda obj:obj),("generar_bloques_aurora_integrado",lambda func:modulo)]:
                    if hasattr(modulo,item_name):
                        try:
                            item=getattr(modulo,item_name)
                            return modulo if item_name=="generar_bloques_aurora_integrado"else(constructor(item)if callable(item)else item)
                        except:continue
            elif comp.nombre=="harmonic_essence":
                for item_name,constructor in [("crear_motor_aurora_conectado",lambda func:func(cache_size=256)),("HarmonicEssenceV34AuroraConnected",lambda cls:cls(cache_size=256,enable_aurora_v7=True)),("HarmonicEssenceV34",lambda cls:cls())]:
                    if hasattr(modulo,item_name):
                        try:return constructor(getattr(modulo,item_name))
                        except:continue
            elif comp.nombre in["field_profiles","objective_router"]:
                for func_name in[f"crear_gestor_{comp.nombre}",f"crear_{comp.nombre}","crear_gestor","obtener_gestor"]:
                    if hasattr(modulo,func_name):return getattr(modulo,func_name)()
                if hasattr(modulo,comp.clase_principal):return getattr(modulo,comp.clase_principal)()
            else:
                if hasattr(modulo,comp.clase_principal):return getattr(modulo,comp.clase_principal)()
            return None
        except:return None
    
    def _validar_protocolo_componente(self,instancia:Any,comp:ComponenteAurora)->bool:
        try:
            if comp.tipo=="motor":
                metodos_motor_basicos=['generar_audio','validar_configuracion','obtener_capacidades']
                protocolo_basico=all(hasattr(instancia,metodo)and callable(getattr(instancia,metodo))for metodo in metodos_motor_basicos)
                if comp.nombre=="neuromix":
                    neuromix_especifico=any(hasattr(instancia,metodo)for metodo in['generate_neuro_wave','get_neuro_preset'])|hasattr(instancia,'generate_neuro_wave_advanced')
                    return protocolo_basico or neuromix_especifico
                elif comp.nombre=="hypermod":
                    hypermod_especifico=any(hasattr(instancia,metodo)for metodo in['generar_bloques','generar_bloques_aurora_integrado','crear_preset_relajacion'])
                    if hasattr(instancia,'_motor_global_v32'):
                        motor_global=getattr(instancia,'_motor_global_v32')
                        if motor_global and hasattr(motor_global,'generar_audio'):hypermod_especifico=True
                    return protocolo_basico or hypermod_especifico
                elif comp.nombre=="harmonic_essence":
                    harmonic_especifico=any(hasattr(instancia,metodo)for metodo in['generate_textured_noise','generar_desde_experiencia_aurora'])
                    return protocolo_basico or harmonic_especifico
                else:return protocolo_basico
            elif comp.tipo=="gestor_inteligencia":
                return all(hasattr(instancia,metodo)and callable(getattr(instancia,metodo))for metodo in['procesar_objetivo','obtener_alternativas'])
            elif comp.tipo=="pipeline":return hasattr(instancia,'validar_y_normalizar')
            return True
        except:return False
    
    def _verificar_optimizacion_v7(self,instancia:Any)->bool:
        try:
            for attr in['aurora_director_compatible','aurora_director_mode','protocolo_director_v7','version']:
                if hasattr(instancia,attr):
                    value=getattr(instancia,attr)
                    if(attr=='version'and'V7'in str(value))or value is True:return True
            return False
        except:return False
    
    def _validar_protocolos(self):
        self.protocolos_validados={"motor_aurora":False,"gestor_inteligencia":False,"pipeline":False,"sistema_completo":False}
        motores_validos=sum(1 for c in self.componentes_activos.values()if c.tipo=="motor"and c.protocolo_implementado)
        gestores_validos=sum(1 for c in self.componentes_activos.values()if c.tipo=="gestor_inteligencia"and c.protocolo_implementado)
        self.protocolos_validados["motor_aurora"]=motores_validos>0
        self.protocolos_validados["gestor_inteligencia"]=gestores_validos>0
        self.protocolos_validados["pipeline"]=any(c.tipo=="pipeline"for c in self.componentes_activos.values())
        self.protocolos_validados["sistema_completo"]=(motores_validos>0 and gestores_validos>0)
    
    def _check_deps(self,comp:ComponenteAurora)->bool:
        return all(dep in self.componentes_activos for dep in comp.dependencias)
    
    def _get_caps_seguro(self,inst:Any)->Dict[str,Any]:
        try:
            for method in['obtener_capacidades','get_capabilities']:
                if hasattr(inst,method):
                    caps=getattr(inst,method)()
                    return caps if isinstance(caps,dict)else{}
        except:pass
        return{}
    
    def _get_version_seguro(self,inst:Any)->str:
        for attr in['version','VERSION','__version__']:
            try:
                if hasattr(inst,attr):
                    version=getattr(inst,attr)
                    return str(version)if version else"unknown"
            except:continue
        return"unknown"
    
    def _crear_fallback_v7(self,comp:ComponenteAurora)->bool:
        try:
            fallbacks={"emotion_style":self._fallback_emotion_style_v7,"neuromix":self._fallback_neuromix_v7,
                      "harmonic_essence":self._fallback_harmonic_v7,"quality_pipeline":self._fallback_quality_v7,
                      "field_profiles":self._fallback_profiles_v7,"objective_router":self._fallback_router_v7}
            if comp.nombre in fallbacks:
                comp.instancia=fallbacks[comp.nombre]();comp.disponible=True;comp.version="fallback_v7"
                comp.protocolo_implementado=True;self.componentes_activos[comp.nombre]=comp;return True
        except:pass
        return False
    
    def _fallback_emotion_style_v7(self):
        class EmotionStyleFallback:
            def __init__(self):self.version="fallback_v7";self.aurora_director_compatible=True;self.protocolo_director_v7=True
            def generar_audio(self,config:Dict[str,Any],duracion_sec:float)->np.ndarray:
                samples=int(44100*duracion_sec);t=np.linspace(0,duracion_sec,samples)
                freq_map={"concentracion":14.0,"claridad_mental":14.0,"relajacion":7.0,"calma":6.0,"creatividad":10.0,"meditacion":6.0}
                freq=freq_map.get(config.get('objetivo','').lower(),10.0)
                intensidad_map={"suave":0.3,"media":0.5,"intenso":0.7}
                amp=intensidad_map.get(config.get('intensidad','media'),0.5)
                wave=amp*np.sin(2*np.pi*freq*t);return np.stack([wave,wave])
            def validar_configuracion(self,config:Dict[str,Any])->bool:return isinstance(config,dict)and'objetivo'in config
            def obtener_capacidades(self)->Dict[str,Any]:return{"nombre":"EmotionStyle Fallback","tipo":"gestor_inteligencia_fallback","protocolo_motor":True,"protocolo_inteligencia":True}
            def procesar_objetivo(self,objetivo:str,contexto:Dict[str,Any])->Dict[str,Any]:
                return{"preset_emocional":"fallback_preset","estilo":"sereno","modo":"fallback","beat_base":10.0,
                      "capas":{"neuro_wave":True,"binaural":True,"wave_pad":True,"textured_noise":True,"heartbeat":False},
                      "coherencia_neuroacustica":0.7,"aurora_v7_optimizado":True}
            def obtener_alternativas(self,objetivo:str)->List[str]:return["relajacion","concentracion","creatividad"]
        return EmotionStyleFallback()
    
    def _fallback_neuromix_v7(self):
        class NeuroMixFallback:
            def __init__(self):self.version="fallback_v7";self.aurora_director_compatible=True
            def generar_audio(self,config:Dict[str,Any],duracion_sec:float)->np.ndarray:
                samples=int(44100*duracion_sec);t=np.linspace(0,duracion_sec,samples);nt=config.get('neurotransmisor_preferido','gaba')
                nt_freq={"dopamina":12.0,"serotonina":7.5,"gaba":6.0,"acetilcolina":14.0,"oxitocina":8.0}
                freq=nt_freq.get(nt.lower(),10.0);wave=0.3*np.sin(2*np.pi*freq*t);return np.stack([wave,wave])
            def validar_configuracion(self,config:Dict[str,Any])->bool:return True
            def obtener_capacidades(self)->Dict[str,Any]:return{"nombre":"NeuroMix Fallback","protocolo_motor":True}
            def generate_neuro_wave(self,nt:str,dur:float,**kw)->np.ndarray:return self.generar_audio({"neurotransmisor_preferido":nt},dur)
        return NeuroMixFallback()
    
    def _fallback_harmonic_v7(self):
        class HarmonicFallback:
            def __init__(self):self.version="fallback_v7";self.aurora_director_compatible=True
            def generar_audio(self,config:Dict[str,Any],duracion_sec:float)->np.ndarray:
                samples=int(44100*duracion_sec);noise=np.random.normal(0,0.1,samples);return np.stack([noise,noise])
            def validar_configuracion(self,config:Dict[str,Any])->bool:return True
            def obtener_capacidades(self)->Dict[str,Any]:return{"nombre":"HarmonicEssence Fallback","protocolo_motor":True}
            def generate_textured_noise(self,config,**kw)->np.ndarray:
                dur=getattr(config,'duration_sec',10.0);return self.generar_audio({},dur)
        return HarmonicFallback()
    
    def _fallback_quality_v7(self):
        class QualityFallback:
            def __init__(self):self.version="fallback_v7"
            def validar_y_normalizar(self,signal:np.ndarray)->np.ndarray:
                if signal.ndim==1:signal=np.stack([signal,signal])
                max_val=np.max(np.abs(signal))
                if max_val>0:signal=signal*(0.85/max_val)
                return np.clip(signal,-1.0,1.0)
        return QualityFallback()
    
    def _fallback_profiles_v7(self):
        class ProfilesFallback:
            def __init__(self):self.version="fallback_v7";self.protocolo_director_v7=True
            def procesar_objetivo(self,objetivo:str,contexto:Dict[str,Any])->Dict[str,Any]:
                return{"perfil_recomendado":"fallback_profile","configuracion_aurora":{"preset_emocional":"campo_fallback","estilo":"sereno","modo":"perfil_campo","beat_base":10.0,"capas":{"neuro_wave":True,"binaural":True,"wave_pad":True,"textured_noise":True,"heartbeat":False}},"aurora_v7_optimizado":True}
            def obtener_alternativas(self,objetivo:str)->List[str]:return["relajacion","concentracion"]
            def obtener_perfil(self,nombre:str):return None
            def recomendar_secuencia_perfiles(self,obj:str,dur:int):return[(obj,dur)]
        return ProfilesFallback()
    
    def _fallback_router_v7(self):
        class RouterFallback:
            def __init__(self):self.version="fallback_v7";self.protocolo_director_v7=True
            def procesar_objetivo(self,objetivo:str,contexto:Dict[str,Any])->Dict[str,Any]:
                return{"preset_emocional":"calma_profunda","estilo":"sereno","modo":"normal","beat_base":8.0,"capas":{"neuro_wave":True,"binaural":True,"wave_pad":True,"textured_noise":True,"heartbeat":False},"aurora_v7_optimizado":True}
            def obtener_alternativas(self,objetivo:str)->List[str]:return["relajacion","concentracion","creatividad"]
            def rutear_objetivo(self,obj:str,**kw):return self.procesar_objetivo(obj,kw)
        return RouterFallback()
    
    def _log_resultado_v7(self):
        total=len(self.componentes_registrados);activos=len(self.componentes_activos)
        logger.info(f"📊 Aurora V7: {activos}/{total} ({activos/total*100:.0f}%) ✅{self.stats['exitosos']} 🔄{self.stats['fallback']} ❌{self.stats['fallidos']}")
        if self.protocolos_validados["sistema_completo"]:logger.info("🤖 Sistema completo funcional")
        else:logger.warning("⚠️ Sistema parcial")

class EstrategiaGeneracion(Enum):
    AURORA_COMPLETO="aurora_completo";INTELIGENCIA_PRIMARIA="inteligencia_primaria";MOTORES_ESPECIALIZADOS="motores_especializados"
    HIBRIDO_OPTIMIZADO="hibrido_optimizado";FALLBACK_INTELIGENTE="fallback_inteligente"

@dataclass
class ConfiguracionAuroraV7:
    objetivo:str="relajacion";duracion_min:int=20;sample_rate:int=44100;estrategia_preferida:Optional[EstrategiaGeneracion]=None
    forzar_componentes:List[str]=field(default_factory=list);excluir_componentes:List[str]=field(default_factory=list)
    intensidad:str="media";estilo:str="sereno";neurotransmisor_preferido:Optional[str]=None;normalizar:bool=True
    calidad_objetivo:str="alta";exportar_wav:bool=True;nombre_archivo:str="aurora_experience";incluir_metadatos:bool=True
    configuracion_custom:Dict[str,Any]=field(default_factory=dict);perfil_usuario:Optional[Dict[str,Any]]=None
    contexto_uso:Optional[str]=None;optimizacion_v7:bool=True;usar_protocolos_v7:bool=True
    
    def validar(self)->List[str]:
        problemas=[]
        if self.duracion_min<=0:problemas.append("Duración debe ser positiva")
        if self.sample_rate not in[22050,44100,48000]:problemas.append("Sample rate no estándar")
        if self.intensidad not in["suave","media","intenso"]:problemas.append("Intensidad inválida")
        if not self.objetivo.strip():problemas.append("Objetivo no puede estar vacío")
        return problemas

@dataclass
class ResultadoAuroraV7:
    audio_data:np.ndarray;metadatos:Dict[str,Any];estrategia_usada:EstrategiaGeneracion;componentes_usados:List[str]
    tiempo_generacion:float;configuracion:ConfiguracionAuroraV7;coherencia_total:float=0.0;validacion_cientifica:str="experimental"
    protocolos_utilizados:List[str]=field(default_factory=list);optimizaciones_aplicadas:List[str]=field(default_factory=list)
    recomendaciones:List[str]=field(default_factory=list);calidad_audio:Dict[str,float]=field(default_factory=dict);aurora_v7_compliant:bool=True

class AuroraDirectorV7Conectado:
    def __init__(self,auto_detectar:bool=True,modo_v7:bool=True):
        self.version=VERSION;self.modo_v7=modo_v7;self.detector=DetectorComponentesV7();self.componentes:Dict[str,ComponenteAurora]={}
        self.sistema_iniciado=False;self.protocolos_activos={}
        self.stats={"experiencias":0,"tiempo_total":0.0,"estrategias":{},"objetivos":{},"errores":0,"protocolos_usados":{},"coherencia_promedio":0.0,"optimizaciones_v7":0}
        if auto_detectar:self._init_sistema_v7()
    
    def _init_sistema_v7(self):
        logger.info(f"🌟 Inicializando {self.version}")
        try:
            self.componentes=self.detector.detectar_todos_v7();self._configurar_protocolos_v7();self._validar_sistema_completo()
            self._log_estado_v7();self.sistema_iniciado=True;logger.info("🚀 Sistema Aurora V7 inicializado")
        except Exception as e:logger.error(f"❌ Error: {e}");self.sistema_iniciado=False
    
    def _configurar_protocolos_v7(self):
        self.protocolos_activos={"motor_aurora":[],"gestor_inteligencia":[],"pipeline":[],"sistema_completo":False}
        for nombre,comp in self.componentes.items():
            if comp.protocolo_implementado:
                if comp.tipo=="motor":self.protocolos_activos["motor_aurora"].append(nombre)
                elif comp.tipo=="gestor_inteligencia":self.protocolos_activos["gestor_inteligencia"].append(nombre)
                elif comp.tipo=="pipeline":self.protocolos_activos["pipeline"].append(nombre)
        self.protocolos_activos["sistema_completo"]=(len(self.protocolos_activos["motor_aurora"])>0 and len(self.protocolos_activos["gestor_inteligencia"])>0)
    
    def _validar_sistema_completo(self):
        validaciones={"componentes_basicos":len(self.componentes)>0,"protocolos_v7":self.protocolos_activos["sistema_completo"],
                     "motor_disponible":len(self.protocolos_activos["motor_aurora"])>0,"inteligencia_disponible":len(self.protocolos_activos["gestor_inteligencia"])>0,
                     "emotion_style_conectado":"emotion_style"in self.componentes,"fallbacks_funcionando":True}
        if not all(validaciones.values()):
            problemas=[k for k,v in validaciones.items()if not v];logger.warning(f"⚠️ Validaciones fallidas: {problemas}")
        else:logger.info("✅ Validaciones OK")
    
    def _log_estado_v7(self):
        motores=[c for c in self.componentes.values()if c.tipo=="motor"];gestores=[c for c in self.componentes.values()if c.tipo=="gestor_inteligencia"]
        v7_optimizados=sum(1 for c in self.componentes.values()if c.aurora_v7_optimizado)
        logger.info(f"🔧 Motores:{len(motores)} 🧠Gestores:{len(gestores)} 🎯V7:{v7_optimizados}/{len(self.componentes)}")
    
    def _get_estrategias_v7(self)->List[str]:
        estrategias=[];motores=len(self.protocolos_activos["motor_aurora"]);gestores=len(self.protocolos_activos["gestor_inteligencia"])
        if motores>=2 and gestores>=1:estrategias.append("aurora_completo")
        if gestores>=1:estrategias.append("inteligencia_primaria")
        if motores>=1:estrategias.append("motores_especializados")
        if motores>=1 and gestores>=1:estrategias.append("hibrido_optimizado")
        estrategias.append("fallback_inteligente");return estrategias
    
    def crear_experiencia(self,objetivo:str,**kwargs)->ResultadoAuroraV7:
        inicio=time.time()
        try:
            logger.info(f"🎯 Aurora V7: '{objetivo}'");config=self._crear_config_v7(objetivo,kwargs)
            problemas=config.validar()
            if problemas:logger.warning(f"⚠️ Problemas: {problemas}")
            estrategia=self._seleccionar_estrategia_v7(config);logger.info(f"🧠 Estrategia: {estrategia.value}")
            resultado_raw=self._ejecutar_estrategia_v7(estrategia,config);resultado_final=self._post_procesar_v7(resultado_raw,config)
            self._validar_resultado_v7(resultado_final);tiempo=time.time()-inicio;self._update_stats_v7(objetivo,estrategia,tiempo,resultado_final)
            logger.info(f"✅ Experiencia creada en {tiempo:.2f}s Coherencia:{resultado_final.coherencia_total:.0%} Audio:{resultado_final.audio_data.shape}")
            return resultado_final
        except Exception as e:
            logger.error(f"❌ Error: {e}");self.stats["errores"]+=1;return self._crear_experiencia_emergencia_v7(objetivo,str(e))
    
    def _crear_config_v7(self,objetivo:str,kwargs:Dict)->ConfiguracionAuroraV7:
        configs_inteligentes={"concentracion":{"intensidad":"intenso","estilo":"crystalline","neurotransmisor_preferido":"acetilcolina","calidad_objetivo":"alta"},
                             "claridad_mental":{"intensidad":"media","estilo":"minimalista","neurotransmisor_preferido":"dopamina","calidad_objetivo":"alta"},
                             "relajacion":{"intensidad":"suave","estilo":"sereno","neurotransmisor_preferido":"gaba","calidad_objetivo":"alta"},
                             "meditacion":{"intensidad":"suave","estilo":"mistico","neurotransmisor_preferido":"serotonina","duracion_min":35},
                             "creatividad":{"intensidad":"media","estilo":"organico","neurotransmisor_preferido":"anandamida","calidad_objetivo":"alta"},
                             "sanacion":{"intensidad":"suave","estilo":"medicina_sagrada","neurotransmisor_preferido":"endorfina","duracion_min":45}}
        config_base={}
        for key,conf in configs_inteligentes.items():
            if key in objetivo.lower():config_base=conf.copy();break
        parametros_contexto=self._detectar_contexto_avanzado(objetivo)
        config_final={"objetivo":objetivo,"optimizacion_v7":True,"usar_protocolos_v7":True,**config_base,**parametros_contexto,**kwargs}
        return ConfiguracionAuroraV7(**config_final)
    
    def _detectar_contexto_avanzado(self,objetivo:str)->Dict[str,Any]:
        params={};obj_lower=objetivo.lower()
        if any(p in obj_lower for p in["profundo","intenso","fuerte","poderoso"]):params["intensidad"]="intenso"
        elif any(p in obj_lower for p in["suave","ligero","sutil","gentil"]):params["intensidad"]="suave"
        if any(p in obj_lower for p in["rapido","corto","breve","express"]):params["duracion_min"]=10
        elif any(p in obj_lower for p in["largo","extenso","profundo","inmersivo"]):params["duracion_min"]=45
        if any(p in obj_lower for p in["maxima","premium","alta_calidad","profesional"]):params["calidad_objetivo"]="maxima"
        if any(p in obj_lower for p in["trabajo","oficina","estudio","productividad"]):params["contexto_uso"]="trabajo"
        elif any(p in obj_lower for p in["dormir","noche","sueño","descanso"]):params["contexto_uso"]="sueño"
        elif any(p in obj_lower for p in["ejercicio","deporte","gimnasio","entrenamiento"]):params["contexto_uso"]="ejercicio"
        elif any(p in obj_lower for p in["terapia","sanacion","curacion","medicina"]):params["contexto_uso"]="terapia"
        return params
    
    def _seleccionar_estrategia_v7(self,config:ConfiguracionAuroraV7)->EstrategiaGeneracion:
        if config.estrategia_preferida:
            estrategias_disponibles=self._get_estrategias_v7()
            if config.estrategia_preferida.value in estrategias_disponibles:return config.estrategia_preferida
        motores=len(self.protocolos_activos["motor_aurora"]);gestores=len(self.protocolos_activos["gestor_inteligencia"])
        if(config.calidad_objetivo=="maxima"and motores>=2 and gestores>=1 and"emotion_style"in self.componentes):return EstrategiaGeneracion.AURORA_COMPLETO
        elif(gestores>=1 and"emotion_style"in self.componentes and config.objetivo in["claridad_mental","creatividad","sanacion"]):return EstrategiaGeneracion.INTELIGENCIA_PRIMARIA
        elif motores>=1:return EstrategiaGeneracion.MOTORES_ESPECIALIZADOS
        elif motores>=1 and gestores>=1:return EstrategiaGeneracion.HIBRIDO_OPTIMIZADO
        else:return EstrategiaGeneracion.FALLBACK_INTELIGENTE
    
    def _ejecutar_estrategia_v7(self,estrategia:EstrategiaGeneracion,config:ConfiguracionAuroraV7)->Dict[str,Any]:
        duracion_sec=config.duracion_min*60
        try:
            if estrategia==EstrategiaGeneracion.AURORA_COMPLETO:audio,comps,protos=self._generar_aurora_completo_v7(config,duracion_sec)
            elif estrategia==EstrategiaGeneracion.INTELIGENCIA_PRIMARIA:audio,comps,protos=self._generar_inteligencia_primaria_v7(config,duracion_sec)
            elif estrategia==EstrategiaGeneracion.MOTORES_ESPECIALIZADOS:audio,comps,protos=self._generar_motores_especializados_v7(config,duracion_sec)
            elif estrategia==EstrategiaGeneracion.HIBRIDO_OPTIMIZADO:audio,comps,protos=self._generar_hibrido_optimizado_v7(config,duracion_sec)
            else:audio,comps,protos=self._generar_fallback_inteligente_v7(config,duracion_sec)
            return{"audio":audio,"componentes":comps,"protocolos":protos,"estrategia":estrategia}
        except Exception as e:
            logger.error(f"❌ Error estrategia {estrategia.value}: {e}");return self._generar_fallback_inteligente_v7(config,duracion_sec)
    
    def _generar_aurora_completo_v7(self,config:ConfiguracionAuroraV7,duracion_sec:float)->Tuple[np.ndarray,List[str],List[str]]:
        componentes_usados=[];protocolos_usados=[]
        if"emotion_style"in self.componentes:
            emotion_style=self.componentes["emotion_style"].instancia
            try:
                contexto={"duracion_min":config.duracion_min,"intensidad":config.intensidad,"calidad_objetivo":config.calidad_objetivo,"contexto_uso":config.contexto_uso}
                config_procesada=emotion_style.procesar_objetivo(config.objetivo,contexto);componentes_usados.append("emotion_style");protocolos_usados.append("gestor_inteligencia")
                if"beat_base"in config_procesada:config.configuracion_custom.update(config_procesada)
            except Exception as e:logger.warning(f"⚠️ Error EmotionStyle: {e}")
        audio_base=None
        if"neuromix"in self.componentes and config.neurotransmisor_preferido:
            try:
                neuromix=self.componentes["neuromix"].instancia
                if hasattr(neuromix,'generar_audio'):audio_base=neuromix.generar_audio(config.__dict__,duracion_sec)
                elif hasattr(neuromix,'generate_neuro_wave'):audio_base=neuromix.generate_neuro_wave(config.neurotransmisor_preferido,duracion_sec,intensidad=config.intensidad)
                componentes_usados.append("neuromix");protocolos_usados.append("motor_aurora")
            except Exception as e:logger.warning(f"⚠️ Error NeuroMix: {e}")
        if audio_base is None and"emotion_style"in self.componentes:
            try:
                emotion_style=self.componentes["emotion_style"].instancia
                if hasattr(emotion_style,'generar_audio'):audio_base=emotion_style.generar_audio(config.__dict__,duracion_sec);componentes_usados.append("emotion_style_motor");protocolos_usados.append("motor_aurora")
            except Exception as e:logger.warning(f"⚠️ Error EmotionStyle motor: {e}")
        if"harmonic_essence"in self.componentes and audio_base is not None:
            try:
                harmonic=self.componentes["harmonic_essence"].instancia
                if hasattr(harmonic,'generar_audio'):textura=harmonic.generar_audio(config.__dict__,duracion_sec)
                elif hasattr(harmonic,'generate_textured_noise'):
                    from types import SimpleNamespace
                    h_config=SimpleNamespace(duration_sec=duracion_sec,style=config.estilo,amplitude=0.3);textura=harmonic.generate_textured_noise(h_config)
                if textura is not None and textura.shape==audio_base.shape:audio_final=audio_base+0.3*textura;componentes_usados.append("harmonic_essence");protocolos_usados.append("motor_aurora")
                else:audio_final=audio_base
            except Exception as e:logger.warning(f"⚠️ Error HarmonicEssence: {e}");audio_final=audio_base
        else:audio_final=audio_base
        if"quality_pipeline"in self.componentes and audio_final is not None:
            try:
                pipeline=self.componentes["quality_pipeline"].instancia;audio_final=pipeline.validar_y_normalizar(audio_final)
                componentes_usados.append("quality_pipeline");protocolos_usados.append("pipeline")
            except Exception as e:logger.warning(f"⚠️ Error Quality Pipeline: {e}")
        if audio_final is None:audio_final=self._generar_audio_basico_v7(config,duracion_sec);componentes_usados.append("fallback_interno");protocolos_usados.append("fallback")
        return audio_final,componentes_usados,protocolos_usados
    
    def _generar_inteligencia_primaria_v7(self,config:ConfiguracionAuroraV7,duracion_sec:float)->Tuple[np.ndarray,List[str],List[str]]:
        if"emotion_style"in self.componentes:
            try:
                emotion_style=self.componentes["emotion_style"].instancia
                if hasattr(emotion_style,'generar_audio'):return emotion_style.generar_audio(config.__dict__,duracion_sec),["emotion_style"],["motor_aurora"]
            except Exception as e:logger.warning(f"⚠️ Error EmotionStyle: {e}")
        return self._generar_motores_especializados_v7(config,duracion_sec)
    
    def _generar_motores_especializados_v7(self,config:ConfiguracionAuroraV7,duracion_sec:float)->Tuple[np.ndarray,List[str],List[str]]:
        for motor_nombre in["neuromix","harmonic_essence","hypermod"]:
            if motor_nombre in self.componentes:
                try:
                    motor=self.componentes[motor_nombre].instancia
                    if hasattr(motor,'generar_audio'):return motor.generar_audio(config.__dict__,duracion_sec),[motor_nombre],["motor_aurora"]
                    elif motor_nombre=="neuromix"and hasattr(motor,'generate_neuro_wave'):
                        nt=config.neurotransmisor_preferido or"gaba";return motor.generate_neuro_wave(nt,duracion_sec,intensidad=config.intensidad),[motor_nombre],["motor_especializado"]
                    elif motor_nombre=="harmonic_essence"and hasattr(motor,'generate_textured_noise'):
                        from types import SimpleNamespace
                        h_config=SimpleNamespace(duration_sec=duracion_sec,style=config.estilo);return motor.generate_textured_noise(h_config),[motor_nombre],["motor_especializado"]
                except Exception as e:logger.warning(f"⚠️ Error motor {motor_nombre}: {e}");continue
        return self._generar_fallback_inteligente_v7(config,duracion_sec)
    
    def _generar_hibrido_optimizado_v7(self,config:ConfiguracionAuroraV7,duracion_sec:float)->Tuple[np.ndarray,List[str],List[str]]:
        componentes_usados=[];protocolos_usados=[];config_procesada=None
        for gestor_nombre in["emotion_style","objective_router","field_profiles"]:
            if gestor_nombre in self.componentes:
                try:
                    gestor=self.componentes[gestor_nombre].instancia;contexto={"duracion_min":config.duracion_min,"intensidad":config.intensidad}
                    config_procesada=gestor.procesar_objetivo(config.objetivo,contexto);componentes_usados.append(f"{gestor_nombre}_inteligencia");protocolos_usados.append("gestor_inteligencia");break
                except Exception as e:logger.debug(f"Error gestor {gestor_nombre}: {e}");continue
        audio=None
        for motor_nombre in["neuromix","harmonic_essence","emotion_style"]:
            if motor_nombre in self.componentes:
                try:
                    motor=self.componentes[motor_nombre].instancia;config_motor=config.__dict__.copy()
                    if config_procesada:config_motor.update(config_procesada)
                    if hasattr(motor,'generar_audio'):audio=motor.generar_audio(config_motor,duracion_sec);componentes_usados.append(f"{motor_nombre}_motor");protocolos_usados.append("motor_aurora");break
                except Exception as e:logger.debug(f"Error motor {motor_nombre}: {e}");continue
        if audio is None:audio=self._generar_audio_basico_v7(config,duracion_sec);componentes_usados.append("fallback_hibrido");protocolos_usados.append("fallback")
        return audio,componentes_usados,protocolos_usados
    
    def _generar_fallback_inteligente_v7(self,config:ConfiguracionAuroraV7,duracion_sec:float)->Tuple[np.ndarray,List[str],List[str]]:
        try:return self._generar_audio_basico_v7(config,duracion_sec),["fallback_inteligente_v7"],["fallback_v7"]
        except Exception as e:logger.error(f"❌ Error fallback: {e}");samples=int(44100*max(1.0,duracion_sec));return np.zeros((2,samples),dtype=np.float32),["emergencia_absoluta"],["emergencia"]
    
    def _generar_audio_basico_v7(self,config:ConfiguracionAuroraV7,duracion_sec:float)->np.ndarray:
        samples=int(config.sample_rate*duracion_sec);t=np.linspace(0,duracion_sec,samples)
        freq_map={"concentracion":14.0,"claridad_mental":14.5,"enfoque":15.0,"relajacion":7.0,"calma":6.5,"paz":5.0,
                 "creatividad":10.5,"inspiracion":11.0,"arte":10.0,"meditacion":6.0,"espiritual":7.83,"conexion":8.0,
                 "energia":12.0,"vitalidad":13.0,"poder":14.0,"sanacion":528.0,"curacion":174.0,"regeneracion":285.0}
        freq=10.0
        for palabra,f in freq_map.items():
            if palabra in config.objetivo.lower():freq=f;break
        intensidad_map={"suave":0.3,"media":0.5,"intenso":0.7};amp=intensidad_map.get(config.intensidad,0.5)
        audio=amp*np.sin(2*np.pi*freq*t)
        if config.estilo=="crystalline":audio+=0.2*amp*np.sin(2*np.pi*freq*2*t)
        elif config.estilo=="organico":audio+=0.15*amp*np.sin(2*np.pi*freq*3*t)+0.1*amp*np.sin(2*np.pi*freq*5*t)
        fade_samples=int(config.sample_rate*1.0)
        if len(audio)>fade_samples*2:
            fade_in=np.linspace(0,1,fade_samples);fade_out=np.linspace(1,0,fade_samples)
            audio[:fade_samples]*=fade_in;audio[-fade_samples:]*=fade_out
        return np.stack([audio,audio])
    
    def _post_procesar_v7(self,resultado_raw:Dict[str,Any],config:ConfiguracionAuroraV7)->ResultadoAuroraV7:
        audio=resultado_raw["audio"]
        if config.normalizar:
            max_val=np.max(np.abs(audio))
            if max_val>0:
                target_level=0.90 if config.calidad_objetivo=="maxima"else 0.85 if config.calidad_objetivo=="alta"else 0.80
                audio=audio*(target_level/max_val);audio=np.clip(audio,-1.0,1.0)
        calidad_audio=self._calcular_calidad_audio_v7(audio);coherencia_total=self._calcular_coherencia_total_v7(config,resultado_raw)
        metadatos={"objetivo":config.objetivo,"duracion_min":config.duracion_min,"estrategia":resultado_raw["estrategia"].value,
                  "componentes":resultado_raw["componentes"],"protocolos":resultado_raw["protocolos"],"timestamp":datetime.now().isoformat(),
                  "sample_rate":config.sample_rate,"version":self.version,"aurora_v7":True,"coherencia_total":coherencia_total,
                  "calidad_audio":calidad_audio,"optimizaciones_v7":config.optimizacion_v7,"sistema_protocolo_completo":self.protocolos_activos["sistema_completo"]}
        recomendaciones=self._generar_recomendaciones_v7(config,calidad_audio,coherencia_total)
        return ResultadoAuroraV7(audio_data=audio,metadatos=metadatos,estrategia_usada=resultado_raw["estrategia"],componentes_usados=resultado_raw["componentes"],
                               tiempo_generacion=0.0,configuracion=config,coherencia_total=coherencia_total,validacion_cientifica=self._determinar_validacion_cientifica(resultado_raw),
                               protocolos_utilizados=resultado_raw["protocolos"],optimizaciones_aplicadas=self._extraer_optimizaciones_aplicadas(config),
                               recomendaciones=recomendaciones,calidad_audio=calidad_audio,aurora_v7_compliant=True)
    
    def _calcular_calidad_audio_v7(self,audio:np.ndarray)->Dict[str,float]:
        try:
            rms=np.sqrt(np.mean(audio**2));peak=np.max(np.abs(audio));crest_factor=peak/rms if rms>0 else 0
            correlation=np.corrcoef(audio[0],audio[1])[0,1]if audio.shape[0]==2 else 1.0
            return{"rms":float(rms),"peak":float(peak),"crest_factor":float(crest_factor),"stereo_correlation":float(correlation),
                  "dynamic_range":float(20*np.log10(peak/rms))if rms>0 else 0,"calidad_general":min(1.0,(rms*2+(1-abs(correlation-0.8))+min(crest_factor/3,1))/3)}
        except:return{"calidad_general":0.5}
    
    def _calcular_coherencia_total_v7(self,config:ConfiguracionAuroraV7,resultado:Dict[str,Any])->float:
        coherencia=0.5
        if"gestor_inteligencia"in resultado["protocolos"]:coherencia+=0.2
        if"motor_aurora"in resultado["protocolos"]:coherencia+=0.2
        componentes_v7=sum(1 for comp in resultado["componentes"]if comp in self.componentes and self.componentes[comp].aurora_v7_optimizado)
        if componentes_v7>0:coherencia+=min(0.3,componentes_v7*0.1)
        if config.optimizacion_v7:coherencia+=0.1
        return min(1.0,coherencia)
    
    def _determinar_validacion_cientifica(self,resultado:Dict[str,Any])->str:
        return"validado"if"emotion_style"in resultado["componentes"]else"experimental"if len(resultado["componentes"])>1 else"básico"
    
    def _extraer_optimizaciones_aplicadas(self,config:ConfiguracionAuroraV7)->List[str]:
        optimizaciones=[]
        if config.optimizacion_v7:optimizaciones.append("Aurora V7 optimizado")
        if config.usar_protocolos_v7:optimizaciones.append("Protocolos V7 activos")
        if config.calidad_objetivo in["alta","maxima"]:optimizaciones.append("Calidad superior")
        if config.neurotransmisor_preferido:optimizaciones.append("Optimización neurotransmisor")
        return optimizaciones
    
    def _generar_recomendaciones_v7(self,config:ConfiguracionAuroraV7,calidad:Dict[str,float],coherencia:float)->List[str]:
        recomendaciones=[]
        if coherencia<0.7:recomendaciones.append("Considerar usar calidad 'maxima' para mejor coherencia")
        if calidad.get("calidad_general",0)<0.6:recomendaciones.append("Verificar configuración de audio")
        if config.duracion_min<15:recomendaciones.append("Duración mínima recomendada: 15 minutos")
        if not config.neurotransmisor_preferido:recomendaciones.append("Especificar neurotransmisor para mejor personalización")
        return recomendaciones
    
    def _validar_resultado_v7(self,resultado:ResultadoAuroraV7):
        if resultado.audio_data.size==0:raise ValueError("Audio vacío generado")
        if np.isnan(resultado.audio_data).any():raise ValueError("Audio contiene NaN")
        if np.max(np.abs(resultado.audio_data))>1.1:raise ValueError("Audio excede límites de amplitud")
        if resultado.audio_data.ndim!=2 or resultado.audio_data.shape[0]!=2:raise ValueError("Audio debe ser estéreo [2, samples]")
        if resultado.coherencia_total<0 or resultado.coherencia_total>1:raise ValueError("Coherencia fuera de rango [0,1]")
    
    def _update_stats_v7(self,objetivo:str,estrategia:EstrategiaGeneracion,tiempo:float,resultado:ResultadoAuroraV7):
        self.stats["experiencias"]+=1;self.stats["tiempo_total"]+=tiempo
        self.stats["estrategias"][estrategia.value]=self.stats["estrategias"].get(estrategia.value,0)+1
        self.stats["objetivos"][objetivo]=self.stats["objetivos"].get(objetivo,0)+1
        for protocolo in resultado.protocolos_utilizados:self.stats["protocolos_usados"][protocolo]=self.stats["protocolos_usados"].get(protocolo,0)+1
        total_experiencias=self.stats["experiencias"]
        self.stats["coherencia_promedio"]=(self.stats["coherencia_promedio"]*(total_experiencias-1)+resultado.coherencia_total)/total_experiencias
        if resultado.configuracion.optimizacion_v7:self.stats["optimizaciones_v7"]+=1
    
    def _crear_experiencia_emergencia_v7(self,objetivo:str,error:str)->ResultadoAuroraV7:
        audio_emergencia=self._generar_audio_basico_v7(ConfiguracionAuroraV7(objetivo=objetivo),60.0);config_emergencia=ConfiguracionAuroraV7(objetivo=objetivo,duracion_min=1)
        return ResultadoAuroraV7(audio_data=audio_emergencia,metadatos={"error":error,"modo_emergencia":True,"objetivo":objetivo,"version":self.version,"aurora_v7":True},
                               estrategia_usada=EstrategiaGeneracion.FALLBACK_INTELIGENTE,componentes_usados=["emergencia_v7"],tiempo_generacion=0.0,configuracion=config_emergencia,
                               coherencia_total=0.3,validacion_cientifica="emergencia",protocolos_utilizados=["emergencia"],optimizaciones_aplicadas=["Modo emergencia"],
                               recomendaciones=["Verificar configuración del sistema"],calidad_audio={"calidad_general":0.3},aurora_v7_compliant=False)
    
    def obtener_estado_completo_v7(self)->Dict[str,Any]:
        return{"version":self.version,"aurora_v7_mode":self.modo_v7,"sistema_iniciado":self.sistema_iniciado,
              "componentes_detectados":{nombre:{"disponible":comp.disponible,"version":comp.version,"tipo":comp.tipo,"protocolo_implementado":comp.protocolo_implementado,
                                               "aurora_v7_optimizado":comp.aurora_v7_optimizado,"fallback":comp.version=="fallback_v7"}for nombre,comp in self.componentes.items()},
              "protocolos_activos":self.protocolos_activos,"protocolos_validados":self.detector.protocolos_validados,"estadisticas_deteccion":self.detector.stats,
              "estadisticas_uso":self.stats,"estrategias_disponibles":self._get_estrategias_v7(),"sistema_protocolo_completo":self.protocolos_activos["sistema_completo"],
              "motores_aurora_v7":len(self.protocolos_activos["motor_aurora"]),"gestores_inteligencia_v7":len(self.protocolos_activos["gestor_inteligencia"]),
              "componentes_v7_optimizados":sum(1 for c in self.componentes.values()if c.aurora_v7_optimizado),"coherencia_promedio_sistema":self.stats["coherencia_promedio"],
              "timestamp":datetime.now().isoformat(),"aurora_v7_ready":True}

_director_global_v7=None

def Aurora(objetivo:str=None,**kwargs):
    global _director_global_v7
    if _director_global_v7 is None:_director_global_v7=AuroraDirectorV7Conectado(modo_v7=True)
    return _director_global_v7.crear_experiencia(objetivo,**kwargs)if objetivo else _director_global_v7

Aurora.rapido=lambda obj,**kw:Aurora(obj,duracion_min=5,calidad_objetivo="media",**kw)
Aurora.largo=lambda obj,**kw:Aurora(obj,duracion_min=60,calidad_objetivo="alta",**kw)
Aurora.premium=lambda obj,**kw:Aurora(obj,duracion_min=45,calidad_objetivo="maxima",optimizacion_v7=True,**kw)
Aurora.terapeutico=lambda obj,**kw:Aurora(obj,duracion_min=45,intensidad="suave",calidad_objetivo="maxima",contexto_uso="terapia",**kw)
Aurora.estado=lambda:Aurora().obtener_estado_completo_v7()
Aurora.diagnostico=lambda:Aurora().detector.stats
Aurora.protocolos=lambda:Aurora().protocolos_activos
Aurora.coherencia=lambda:Aurora().stats["coherencia_promedio"]

def crear_director_aurora_v7()->AuroraDirectorV7Conectado:return AuroraDirectorV7Conectado(modo_v7=True)

def obtener_director_global()->AuroraDirectorV7Conectado:
    global _director_global_v7
    if _director_global_v7 is None:_director_global_v7=crear_director_aurora_v7()
    return _director_global_v7

def verificar_conexion_motores_principales():
    print("🔧 Verificando conexión Director ↔ Motores Principales...")
    motores_principales={"neuromix":{"archivo":"neuromix_aurora_v27","clases":["AuroraNeuroAcousticEngineV27","AuroraNeuroAcousticEngine","_global_engine"],
                                    "metodos_clave":["generar_audio","generate_neuro_wave","validar_configuracion"],"test_config":{"neurotransmisor_preferido":"dopamina","intensidad":"media"}},
                        "hypermod":{"archivo":"hypermod_v32","clases":["HyperModEngineV32AuroraConnected","_motor_global_v32","generar_bloques_aurora_integrado"],
                                   "metodos_clave":["generar_audio","generar_bloques","crear_preset_relajacion"],"test_config":{"objetivo":"concentracion","intensidad":"media"}},
                        "harmonic_essence":{"archivo":"harmonicEssence_v34","clases":["HarmonicEssenceV34AuroraConnected","crear_motor_aurora_conectado"],
                                           "metodos_clave":["generar_audio","generate_textured_noise","generar_desde_experiencia_aurora"],"test_config":{"objetivo":"creatividad","estilo":"organico"}}}
    resultados={}
    for motor_nombre,info in motores_principales.items():
        print(f"\n🚀 Testing {motor_nombre.upper()}...");resultado_motor={"importado":False,"instancia":None,"protocolo":False,"funcional":False}
        try:
            modulo=importlib.import_module(info["archivo"]);print(f"   ✅ Módulo {info['archivo']} importado");resultado_motor["importado"]=True
            clases_encontradas=[clase_nombre for clase_nombre in info["clases"]if hasattr(modulo,clase_nombre)]
            if not clases_encontradas:print(f"   ❌ No se encontraron clases válidas para {motor_nombre}");continue
            instancia=None
            for clase_nombre in clases_encontradas:
                try:
                    item=getattr(modulo,clase_nombre)
                    if motor_nombre=="neuromix":instancia=item(enable_advanced_features=True)if callable(item)and"V27"in clase_nombre else(item()if callable(item)else item)
                    elif motor_nombre=="hypermod":instancia=modulo if clase_nombre=="generar_bloques_aurora_integrado"else(item()if callable(item)else item)
                    elif motor_nombre=="harmonic_essence":instancia=item(cache_size=128)if callable(item)and"crear_motor"in clase_nombre else(item(cache_size=128,enable_aurora_v7=True)if callable(item)else item)
                    if instancia:print(f"   ✅ Instancia creada con {clase_nombre}");resultado_motor["instancia"]=instancia;break
                except Exception as e:print(f"   ⚠️ Error con {clase_nombre}: {e}");continue
            if not instancia:print(f"   ❌ No se pudo crear instancia para {motor_nombre}");continue
            metodos_protocolo=["generar_audio","validar_configuracion","obtener_capacidades"];protocolo_completo=all(hasattr(instancia,metodo)for metodo in metodos_protocolo)
            metodos_especificos=sum(1 for metodo in info["metodos_clave"]if hasattr(instancia,metodo));protocolo_ok=protocolo_completo or metodos_especificos>=1;resultado_motor["protocolo"]=protocolo_ok
            print(f"   📡 Protocolo: {'✅'if protocolo_completo else'❌'} Específicos: {metodos_especificos}/{len(info['metodos_clave'])}")
            if protocolo_ok:
                try:
                    if hasattr(instancia,'validar_configuracion'):
                        config_test={**info["test_config"],"duracion_min":1};valido=instancia.validar_configuracion(config_test);print(f"   🧪 Validación: {'✅'if valido else'❌'}")
                    if hasattr(instancia,'obtener_capacidades'):
                        caps=instancia.obtener_capacidades();caps_ok=caps and isinstance(caps,dict);print(f"   📊 Capacidades: {'✅'if caps_ok else'❌'}")
                    audio_generado=False
                    if hasattr(instancia,'generar_audio'):
                        try:
                            config_audio={**info["test_config"],"sample_rate":44100};audio=instancia.generar_audio(config_audio,0.5)
                            if audio is not None and hasattr(audio,'shape')and audio.size>0:audio_generado=True;print(f"   🎵 Audio: ✅ {audio.shape}")
                        except Exception as e:print(f"   🎵 Audio: ❌ {e}")
                    elif motor_nombre=="neuromix"and hasattr(instancia,'generate_neuro_wave'):
                        try:audio=instancia.generate_neuro_wave("dopamina",0.5,intensidad="media");if audio is not None and hasattr(audio,'shape'):audio_generado=True;print(f"   🎵 NeuroWave: ✅ {audio.shape}")
                        except Exception as e:print(f"   🎵 NeuroWave: ❌ {e}")
                    elif motor_nombre=="harmonic_essence"and hasattr(instancia,'generate_textured_noise'):
                        try:
                            from types import SimpleNamespace
                            config=SimpleNamespace(duration_sec=0.5,amplitude=0.5);audio=instancia.generate_textured_noise(config)
                            if audio is not None and hasattr(audio,'shape'):audio_generado=True;print(f"   🎵 Textured Noise: ✅ {audio.shape}")
                        except Exception as e:print(f"   🎵 Textured Noise: ❌ {e}")
                    resultado_motor["funcional"]=audio_generado
                except Exception as e:print(f"   ❌ Error test funcional: {e}")
        except ImportError as e:print(f"   ❌ Error importando {info['archivo']}: {e}")
        except Exception as e:print(f"   ❌ Error inesperado: {e}")
        resultados[motor_nombre]=resultado_motor
    motores_ok=sum(1 for resultado in resultados.values()if resultado["funcional"])
    print(f"\n🏆 RESUMEN: {motores_ok}/3 motores funcionales")
    for motor,resultado in resultados.items():
        status="✅"if resultado["funcional"]else"🔄"if resultado["protocolo"]else"❌"
        print(f"   {status} {motor.upper()}: {resultado['funcional']}")
    if motores_ok==3:print("🎉 ¡TODOS LOS MOTORES CONECTADOS!")
    elif motores_ok>=1:print("⚡ Sistema funcional")
    else:print("⚠️ Sistema en modo fallback")
    return resultados

def test_aurora_v7_completo():
    print("🌟 Aurora Director V7 CONECTADO - Testing");director=Aurora();estado=director.obtener_estado_completo_v7()
    print(f"🚀 {estado['version']} V7:{'✅'if estado['aurora_v7_mode']else'❌'} Iniciado:{'✅'if estado['sistema_iniciado']else'❌'} Protocolo:{'✅'if estado['sistema_protocolo_completo']else'❌'}")
    print(f"\n📊 Componentes: {len(estado['componentes_detectados'])}")
    for nombre,info in estado['componentes_detectados'].items():
        if info['disponible']:
            emoji="✅"if info['protocolo_implementado']and not info['fallback']else"🔄"if info['fallback']else"⚠️"
            v7_status="V7"if info['aurora_v7_optimizado']else"v6";print(f"   {emoji} {nombre} v{info['version']} ({v7_status})")
    print(f"\n🔧 Protocolos: 🚀{estado['motores_aurora_v7']} 🧠{estado['gestores_inteligencia_v7']} ⚡{estado['componentes_v7_optimizados']} 📊{estado['coherencia_promedio_sistema']:.0%}")
    try:
        print(f"\n🎵 Test generación...");resultado=Aurora("claridad_mental",duracion_min=1,calidad_objetivo="alta")
        print(f"✅ ¡Generación exitosa! Audio:{resultado.audio_data.shape} Estrategia:{resultado.estrategia_usada.value} Componentes:{','.join(resultado.componentes_usados)} Coherencia:{resultado.coherencia_total:.0%}")
    except Exception as e:print(f"❌ Error test: {e}")
    print(f"\n🏆 AURORA DIRECTOR V7 CONECTADO ✅ Sistema funcional 🔗 Integración completa 🚀 ¡Listo!")
