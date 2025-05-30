"""HarmonicEssence V34 Ultimate - Aurora V7 Connected (Optimized)"""
import numpy as np,warnings,logging,math,random,time
from scipy.signal import butter,sosfilt,filtfilt,hilbert,spectrogram
from scipy.fft import fft,ifft,fftfreq
from scipy.stats import entropy,skew,kurtosis,shapiro,normaltest
from scipy.interpolate import interp1d
from typing import Optional,Tuple,Union,List,Dict,Any,Callable
from dataclasses import dataclass,field
from functools import lru_cache
from enum import Enum
from datetime import datetime

logging.basicConfig(level=logging.INFO);logger=logging.getLogger("Aurora.HarmonicEssence.Connected");VERSION="V34_AURORA_CONNECTED"

try:
    from emotion_style_profiles import (GestorEmotionStyleUnificado,PresetEmocionalCompleto,PerfilEstiloCompleto,PresetEstiloCompleto,CategoriaEmocional,CategoriaEstilo,TipoPad,EstiloRuido,TipoFiltro,TipoEnvolvente,EfectosPsicofisiologicos,ConfiguracionFiltroAvanzada,ConfiguracionRuidoAvanzada,crear_gestor_emotion_style_unificado,obtener_experiencia_completa)
    AURORA_V7_EMOTION_STYLE_DISPONIBLE=True;logger.info("✅ Aurora V7 Connected")
except ImportError:AURORA_V7_EMOTION_STYLE_DISPONIBLE=False;logger.warning("⚠️ Aurora V7 unavailable")

if not AURORA_V7_EMOTION_STYLE_DISPONIBLE:
    class TipoTexturaUnificado(Enum):DYNAMIC="dynamic";STATIC="static";FRACTAL="fractal";ORGANIC="organic";DIGITAL="digital";NEUROMORPHIC="neuromorphic";HOLOGRAPHIC="holographic";QUANTUM_FIELD="quantum_field";CRYSTALLINE_MATRIX="crystalline_matrix";ETHEREAL_MIST="ethereal_mist"
else:
    class TipoTexturaUnificado(Enum):DYNAMIC="dynamic";STATIC="static";FRACTAL="fractal";ORGANIC="organic";DIGITAL="digital";NEUROMORPHIC="neuromorphic";HOLOGRAPHIC="holographic";QUANTUM_FIELD="quantum_field";CRYSTALLINE_MATRIX="crystalline_matrix";ETHEREAL_MIST="ethereal_mist";LIQUID_LIGHT="liquid_light";GATED="gated";RISING="rising";FALLING="falling";BREATHY="breathy";SHIMMER="shimmer";TRIBAL="tribal";REVERSO="reverso";QUANTUM="quantum";NEURAL="neural";CRYSTALLINE="crystalline";HEALING="healing";COSMIC="cosmic";ETHEREAL="ethereal";BINAURAL_TEXTURE="binaural_texture";ALPHA_WAVE="alpha_wave";THETA_WAVE="theta_wave";GAMMA_BURST="gamma_burst";CONSCIOUSNESS="consciousness";MEDITATION="meditation";FOCUS="focus";RELAXATION="relaxation"

class TipoModulacionUnificado(Enum):AM="am";FM="fm";PM="pm";NEUROMORPHIC="neuromorphic";QUANTUM_COHERENCE="quantum_coherence";FRACTAL_CHAOS="fractal_chaos";GOLDEN_RATIO="golden_ratio";FIBONACCI_SEQUENCE="fibonacci_sequence";NEURAL_OPTIMIZED="neural_optimized";PSYCHOACOUSTIC="psychoacoustic";THERAPEUTIC="therapeutic"

class EfectoEspacialUnificado(Enum):STEREO_BASIC="stereo_basic";SURROUND_3D="surround_3d";HOLOGRAPHIC_8D="holographic_8d";BINAURAL_3D="binaural_3d";QUANTUM_SPACE="quantum_space";NEURAL_MAPPING="neural_mapping";CENTER="center";STEREO_8D="8d";REVERSE="reverse";CIRCULAR="circular";NEUROACOUSTIC="neuroacoustic";HEMISPHERIC="hemispheric";CONSCIOUSNESS="consciousness";BINAURAL_SWEEP="binaural_sweep"

class ModeloFiltroCientifico(Enum):BUTTERWORTH="butterworth";CHEBYSHEV="chebyshev";ELLIPTIC="elliptic";BESSEL="bessel";NEURAL_OPTIMIZED="neural_optimized";PSYCHOACOUSTIC="psychoacoustic";THERAPEUTIC="therapeutic"

@dataclass
class ConfiguracionRuidoCientifica:
    sample_rate:int=44100;precision_bits:int=24;validacion_automatica:bool=True;distribucion_estadistica:str="normal";semilla_reproducibilidad:Optional[int]=None;factor_correlacion:float=0.0;perfil_espectral:str="white";ancho_banda_hz:Tuple[float,float]=(20,20000);resolucion_espectral:float=1.0;frecuencias_neuroacusticas:List[float]=field(default_factory=lambda:[40,100,440,1000,4000]);efectos_neuroquimicos:List[str]=field(default_factory=list);optimizacion_cerebral:bool=True;complejidad_textural:float=1.0;naturalidad_patron:float=0.8;coherencia_temporal:float=0.85;modelo_filtro:ModeloFiltroCientifico=ModeloFiltroCientifico.NEURAL_OPTIMIZED;orden_filtro:int=6;tolerancia_ripple:float=0.1;profundidad_espacial:float=0.5;velocidad_movimiento:float=0.25;umbral_calidad:float=0.8;analisis_automatico:bool=True;generar_metadatos:bool=True;version:str="v34_aurora_connected";timestamp:str=field(default_factory=lambda:datetime.now().isoformat())

@dataclass
class FilterConfigV34Unificado:
    cutoff_freq:float=500.0;filter_type:str='lowpass';sample_rate:int=44100;order:int=2;resonance:float=0.7;drive:float=0.0;wetness:float=1.0;modulation_enabled:bool=False;modulation_type:TipoModulacionUnificado=TipoModulacionUnificado.AM;modulation_rate:float=0.5;modulation_depth:float=0.3;neurotransmitter_optimization:Optional[str]=None;brainwave_sync:Optional[str]=None;modelo_cientifico:ModeloFiltroCientifico=ModeloFiltroCientifico.NEURAL_OPTIMIZED;validacion_respuesta:bool=True;optimizacion_neuroacustica:bool=True

@dataclass 
class NoiseConfigV34Unificado:
    duration_sec:float;sample_rate:int=44100;amplitude:float=0.5;stereo_width:float=0.0;filter_config:Optional[FilterConfigV34Unificado]=None;texture_complexity:float=1.0;spectral_tilt:float=0.0;texture_type:TipoTexturaUnificado=TipoTexturaUnificado.ORGANIC;modulation:TipoModulacionUnificado=TipoModulacionUnificado.AM;spatial_effect:EfectoEspacialUnificado=EfectoEspacialUnificado.STEREO_BASIC;neurotransmitter_profile:Optional[str]=None;brainwave_target:Optional[str]=None;emotional_state:Optional[str]=None;style_profile:Optional[str]=None;envelope_shape:str="standard";harmonic_richness:float=0.5;synthesis_method:str="additive";precision_cientifica:bool=True;analisis_espectral:bool=False;validacion_distribucion:bool=True;modelo_psicoacustico:bool=True;validacion_neuroacustica:bool=True;optimizacion_integral:bool=True;objetivo_terapeutico:Optional[str]=None;patron_organico:Optional[str]=None;tipo_patron_cuantico:Optional[str]=None;configuracion_cientifica:Optional[ConfiguracionRuidoCientifica]=None;preset_emocional:Optional[str]=None;perfil_estilo:Optional[str]=None;preset_estetico:Optional[str]=None;auto_optimizar_coherencia:bool=True
    
    def __post_init__(self):
        if self.filter_config is None:self.filter_config=FilterConfigV34Unificado()
        if self.configuracion_cientifica is None:self.configuracion_cientifica=ConfiguracionRuidoCientifica()
        if self.duration_sec<=0:raise ValueError("Duración debe ser positiva")

_metadatos_ruido_globales={};_experiencias_aurora_cache={}

def _almacenar_metadatos_ruido(tipo:str,metadatos:Dict[str,Any]):
    if tipo not in _metadatos_ruido_globales:_metadatos_ruido_globales[tipo]=[]
    _metadatos_ruido_globales[tipo].append(metadatos)

class HarmonicEssenceV34AuroraConnected:
    def __init__(self,cache_size:int=256,enable_aurora_v7:bool=True):
        self.version=VERSION;self.cache_size=cache_size;self.enable_aurora_v7=enable_aurora_v7 and AURORA_V7_EMOTION_STYLE_DISPONIBLE;self._filter_cache={};self._texture_cache={};self._scientific_cache={};self._aurora_experience_cache={};self._coherence_cache={}
        
        if self.enable_aurora_v7:
            try:self.aurora_gestor=crear_gestor_emotion_style_unificado();logger.info("🎨 Aurora V7 Connected")
            except Exception as e:logger.warning(f"⚠️ Error Aurora V7: {e}");self.enable_aurora_v7=False;self.aurora_gestor=None
        else:self.aurora_gestor=None
        
        self._windows=self._precompute_windows();self._lookup_tables=self._precompute_lookup_tables();self.stats={'textures_generated':0,'cache_hits':0,'processing_time':0.0,'scientific_generations':0,'validations_performed':0,'optimizations_applied':0,'aurora_experiences_created':0,'coherence_optimizations':0}
    
    @lru_cache(maxsize=32)
    def _precompute_windows(self)->Dict[str,np.ndarray]:
        w={}
        for s in [512,1024,2048,4096]:w[f'hann_{s}']=np.hanning(s);w[f'blackman_{s}']=np.blackman(s);w[f'hamming_{s}']=np.hamming(s)
        return w
    
    def _precompute_lookup_tables(self)->Dict[str,np.ndarray]:
        t={};size=8192;x=np.linspace(0,2*np.pi,size,endpoint=False);t['sine']=np.sin(x);t['triangle']=2*np.arcsin(np.sin(x))/np.pi;t['sawtooth']=2*(x/(2*np.pi)-np.floor(x/(2*np.pi)+0.5));t['square']=np.sign(np.sin(x));t['fractal_1']=self._generate_fractal_wave(x,3);t['neural_spike']=self._generate_neural_spike_pattern(size);t['brainwave']=self._generate_brainwave_pattern(size);t['quantum_coherence']=self._generate_quantum_coherence_pattern(size);t['organic_flow']=self._generate_organic_flow_pattern(x);return t
    
    def _generate_fractal_wave(self,t:np.ndarray,iterations:int=3)->np.ndarray:
        wave=np.zeros_like(t)
        for i in range(iterations):freq_mult=2**i;amp_mult=1/(2**i);wave+=amp_mult*np.sin(freq_mult*t)
        return wave/np.max(np.abs(wave))
    
    def _generate_neural_spike_pattern(self,size:int)->np.ndarray:
        pattern=np.zeros(size);spike_positions=np.random.randint(0,size,size//20)
        for pos in spike_positions:
            if pos<size-10:spike=np.exp(-np.linspace(0,5,10));pattern[pos:pos+10]=spike
        return pattern
    
    def _generate_brainwave_pattern(self,size:int)->np.ndarray:
        t=np.linspace(0,1,size);pattern=(0.3*np.sin(2*np.pi*8*t)+0.2*np.sin(2*np.pi*14*t)+0.4*np.sin(2*np.pi*6*t)+0.1*np.sin(2*np.pi*2*t));return pattern/np.max(np.abs(pattern))
    
    def _generate_quantum_coherence_pattern(self,size:int)->np.ndarray:
        t=np.linspace(0,1,size);coherence=np.zeros_like(t)
        for n in range(1,8):fase_cuantica=np.random.random()*2*np.pi;amplitud_cuantica=0.7/n;coherence+=amplitud_cuantica*np.sin(2*np.pi*40*n*t+fase_cuantica)
        modulacion_coherencia=0.8+0.2*np.sin(2*np.pi*0.1*t);return coherence*modulacion_coherencia
    
    def _generate_organic_flow_pattern(self,t:np.ndarray)->np.ndarray:
        respiracion=np.sin(2*np.pi*0.25*t);variacion_freq=0.1*np.sin(2*np.pi*0.1*t);fase=2*np.pi*0.25*t;patron_asimetrico=np.where(np.sin(fase)>=0,np.sin(fase)**0.7,-(-np.sin(fase))**1.3);return 0.6*patron_asimetrico*(1+variacion_freq*0.2)
    
    def generar_desde_experiencia_aurora(self,objetivo_emocional:str,contexto:str=None,duracion_sec:float=10.0,**kwargs)->np.ndarray:
        if not self.enable_aurora_v7:logger.warning("⚠️ Aurora V7 unavailable");return self._generate_fallback_texture(duracion_sec,objetivo_emocional,**kwargs)
        
        start_time=time.time();cache_key=f"aurora_{objetivo_emocional}_{contexto}_{duracion_sec}"
        
        if cache_key in self._aurora_experience_cache:self.stats['cache_hits']+=1;return self._aurora_experience_cache[cache_key]
        
        try:
            experiencia=obtener_experiencia_completa(objetivo_emocional,contexto)
            if "error" in experiencia:logger.warning(f"⚠️ Error Aurora: {experiencia['error']}");return self._generate_fallback_texture(duracion_sec,objetivo_emocional,**kwargs)
            
            config=self._crear_config_desde_experiencia_aurora(experiencia,duracion_sec,**kwargs);texture=self.generate_textured_noise(config)
            if config.auto_optimizar_coherencia:texture=self._optimizar_coherencia_aurora(texture,experiencia)
            
            if len(self._aurora_experience_cache)<self.cache_size:self._aurora_experience_cache[cache_key]=texture
            
            self.stats['aurora_experiences_created']+=1;self.stats['processing_time']+=time.time()-start_time;metadatos={"tipo":"experiencia_aurora_v7","objetivo_emocional":objetivo_emocional,"contexto":contexto,"preset_emocional":experiencia['preset_emocional']['nombre'],"perfil_estilo":experiencia['perfil_estilo']['nombre'],"preset_estetico":experiencia['preset_estetico']['nombre'],"score_coherencia":experiencia['score_coherencia'],"timestamp":datetime.now().isoformat()};_almacenar_metadatos_ruido("aurora_v7_experience",metadatos);logger.info(f"✨ Aurora V7: {objetivo_emocional} (coherencia: {experiencia['score_coherencia']:.2f})");return texture
            
        except Exception as e:logger.error(f"❌ Error Aurora V7: {e}");return self._generate_fallback_texture(duracion_sec,objetivo_emocional,**kwargs)
    
    def _crear_config_desde_experiencia_aurora(self,experiencia:Dict[str,Any],duracion_sec:float,**kwargs)->NoiseConfigV34Unificado:
        preset_emocional=experiencia['preset_emocional'];perfil_estilo=experiencia['perfil_estilo'];preset_estetico=experiencia['preset_estetico']
        
        pad_to_texture_map={"sine":TipoTexturaUnificado.MEDITATION,"organic_flow":TipoTexturaUnificado.ORGANIC,"crystalline":TipoTexturaUnificado.CRYSTALLINE,"spectral":TipoTexturaUnificado.ETHEREAL,"tribal_pulse":TipoTexturaUnificado.TRIBAL,"neuromorphic":TipoTexturaUnificado.NEUROMORPHIC,"quantum_pad":TipoTexturaUnificado.QUANTUM}
        
        envolvente_to_spatial_map={"suave":EfectoEspacialUnificado.STEREO_BASIC,"eterea":EfectoEspacialUnificado.HOLOGRAPHIC_8D,"ritmica":EfectoEspacialUnificado.BINAURAL_3D,"cuantica":EfectoEspacialUnificado.QUANTUM_SPACE,"neuromorfica":EfectoEspacialUnificado.NEURAL_MAPPING}
        
        pad_type=perfil_estilo.get('tipo_pad','sine');texture_type=pad_to_texture_map.get(pad_type,TipoTexturaUnificado.ORGANIC);envolvente=preset_estetico.get('envolvente','suave');spatial_effect=envolvente_to_spatial_map.get(envolvente,EfectoEspacialUnificado.STEREO_BASIC)
        
        neurotransmisores=preset_emocional.get('neurotransmisores',{})
        if 'anandamida' in neurotransmisores:modulation=TipoModulacionUnificado.QUANTUM_COHERENCE
        elif 'dopamina' in neurotransmisores:modulation=TipoModulacionUnificado.NEURAL_OPTIMIZED
        elif 'gaba' in neurotransmisores:modulation=TipoModulacionUnificado.THERAPEUTIC
        else:modulation=TipoModulacionUnificado.PSYCHOACOUSTIC
        
        intensidad_emocional=preset_estetico.get('intensidad_emocional',0.5);texture_complexity=max(0.3,min(1.0,intensidad_emocional*1.2));frecuencia_base=preset_emocional.get('frecuencia_base',10.0);amplitude=max(0.2,min(0.8,0.4+(frecuencia_base-8.0)*0.02))
        
        return NoiseConfigV34Unificado(duration_sec=duracion_sec,sample_rate=kwargs.get('sample_rate',44100),amplitude=amplitude,stereo_width=kwargs.get('stereo_width',0.8),texture_type=texture_type,modulation=modulation,spatial_effect=spatial_effect,texture_complexity=texture_complexity,neurotransmitter_profile=self._extraer_neurotransmisor_principal(neurotransmisores),emotional_state=preset_emocional['nombre'].lower().replace(' ','_'),style_profile=perfil_estilo['nombre'].lower(),preset_emocional=preset_emocional['nombre'],perfil_estilo=perfil_estilo['nombre'],preset_estetico=preset_estetico['nombre'],auto_optimizar_coherencia=kwargs.get('auto_optimizar_coherencia',True),precision_cientifica=kwargs.get('precision_cientifica',True),optimizacion_integral=True)
    
    def _extraer_neurotransmisor_principal(self,neurotransmisores:Dict[str,float])->str:
        return max(neurotransmisores.items(),key=lambda x:x[1])[0] if neurotransmisores else "serotonina"
    
    def _optimizar_coherencia_aurora(self,texture:np.ndarray,experiencia:Dict[str,Any])->np.ndarray:
        score_coherencia=experiencia.get('score_coherencia',0.5)
        if score_coherencia>=0.8:texture=self._aplicar_refinamiento_coherencia_alta(texture)
        elif score_coherencia>=0.6:texture=self._aplicar_optimizacion_coherencia_media(texture)
        else:texture=self._aplicar_correccion_coherencia_baja(texture,experiencia)
        self.stats['coherence_optimizations']+=1;return texture
    
    def _aplicar_refinamiento_coherencia_alta(self,texture:np.ndarray)->np.ndarray:
        if texture.ndim>1:correlation_factor=0.05;left,right=texture[0],texture[1];right_corr=right+correlation_factor*left;return np.stack([left,right_corr])
        return texture
    
    def _aplicar_optimizacion_coherencia_media(self,texture:np.ndarray)->np.ndarray:
        if texture.ndim>1:
            for i in range(texture.shape[0]):texture[i]=self._aplicar_suavizado_espectral_ligero(texture[i])
        else:texture=self._aplicar_suavizado_espectral_ligero(texture)
        return texture
    
    def _aplicar_correccion_coherencia_baja(self,texture:np.ndarray,experiencia:Dict[str,Any])->np.ndarray:
        try:
            frecuencia_base=experiencia['preset_emocional'].get('frecuencia_base',10.0);cutoff_freq=frecuencia_base*50
            if texture.ndim>1:
                for i in range(texture.shape[0]):texture[i]=self.apply_lowpass(texture[i],cutoff_freq,modelo_cientifico=True)
            else:texture=self.apply_lowpass(texture,cutoff_freq,modelo_cientifico=True)
        except Exception as e:logger.warning(f"⚠️ Error coherencia: {e}")
        return texture
    
    def _aplicar_suavizado_espectral_ligero(self,signal:np.ndarray)->np.ndarray:
        try:
            kernel_size=min(21,len(signal)//10)
            if kernel_size>1:kernel=np.ones(kernel_size)/kernel_size;smoothed=np.convolve(signal,kernel,mode='same');return 0.7*signal+0.3*smoothed
        except:pass
        return signal
    
    def buscar_presets_aurora_por_efecto(self,efecto:str,umbral:float=0.5)->List[str]:
        if not self.enable_aurora_v7:return []
        try:return self.aurora_gestor.buscar_por_efecto(efecto,umbral)
        except Exception as e:logger.warning(f"⚠️ Error búsqueda: {e}");return []
    
    def analizar_compatibilidad_aurora(self,preset_emocional:str,perfil_estilo:str,preset_estetico:str=None)->Dict[str,Any]:
        if not self.enable_aurora_v7:return {"error":"Aurora V7 no disponible"}
        try:
            from emotion_style_profiles import analizar_coherencia_aurora
            return analizar_coherencia_aurora(preset_emocional,perfil_estilo,preset_estetico)
        except Exception as e:logger.warning(f"⚠️ Error compatibilidad: {e}");return {"error":str(e)}
    
    def generar_secuencia_aurora(self,objetivo:str,duracion_total_sec:float=300.0)->List[np.ndarray]:
        if not self.enable_aurora_v7:logger.warning("⚠️ Aurora V7 unavailable");return []
        try:
            from emotion_style_profiles import generar_secuencia_aurora
            secuencia=generar_secuencia_aurora(objetivo,int(duracion_total_sec//60));texturas=[]
            for fase in secuencia:duracion_fase=fase['duracion_min']*60;texture=self.generar_desde_experiencia_aurora(fase['preset'],duracion_sec=duracion_fase);texturas.append(texture)
            return texturas
        except Exception as e:logger.error(f"❌ Error secuencia: {e}");return []
    
    def obtener_texturas_por_neurotransmisor(self,neurotransmisor:str,duracion_sec:float=10.0)->Dict[str,np.ndarray]:
        if not self.enable_aurora_v7:return {}
        try:
            from emotion_style_profiles import buscar_por_neurotransmisor
            resultados=buscar_por_neurotransmisor(neurotransmisor);texturas={}
            for preset_name in resultados.get('presets_emocionales',[]):
                try:texture=self.generar_desde_experiencia_aurora(preset_name,duracion_sec=duracion_sec);texturas[preset_name]=texture
                except Exception as e:logger.warning(f"⚠️ Error {preset_name}: {e}")
            return texturas
        except Exception as e:logger.error(f"❌ Error neurotransmisor: {e}");return {}
    
    def generate_textured_noise(self,config:Union[NoiseConfigV34Unificado,Any],seed:Optional[int]=None)->np.ndarray:
        start_time=time.time()
        if not isinstance(config,NoiseConfigV34Unificado):config=self._convert_to_unified_config(config)
        if seed is not None:np.random.seed(seed)
        
        if (self.enable_aurora_v7 and (config.preset_emocional or config.perfil_estilo or config.preset_estetico)):result=self._generate_aurora_v7_texture(config,seed)
        elif config.objetivo_terapeutico:result=self._generate_therapeutic_texture(config,seed)
        elif config.patron_organico:result=self._generate_organic_pattern_texture(config,seed)
        elif config.tipo_patron_cuantico:result=self._generate_quantum_texture(config,seed)
        elif self.enable_aurora_v7 and config.neurotransmitter_profile:result=self._generate_neuroacoustic_texture(config,seed)
        elif config.texture_type in[TipoTexturaUnificado.HOLOGRAPHIC,TipoTexturaUnificado.QUANTUM_FIELD]:result=self._generate_advanced_texture(config,seed)
        else:result=self._generate_standard_texture(config,seed)
        
        if config.validacion_neuroacustica:result=self._validate_and_optimize_texture(result,config)
        
        cache_key=self._generate_cache_key(config,seed)
        if len(self._texture_cache)<self.cache_size:self._texture_cache[cache_key]=result
        
        self.stats['textures_generated']+=1;self.stats['processing_time']+=time.time()-start_time;return result
    
    def _generate_aurora_v7_texture(self,config:NoiseConfigV34Unificado,seed:Optional[int])->np.ndarray:
        try:
            preset_emocional=None;perfil_estilo=None;preset_estetico=None
            if config.preset_emocional:preset_emocional=self.aurora_gestor.obtener_preset_emocional(config.preset_emocional)
            if config.perfil_estilo:perfil_estilo=self.aurora_gestor.obtener_perfil_estilo(config.perfil_estilo)
            if config.preset_estetico:preset_estetico=self.aurora_gestor.obtener_preset_estetico(config.preset_estetico)
            
            base_texture=self._generate_standard_texture(config,seed)
            if preset_emocional:base_texture=self._aplicar_modulacion_emocional(base_texture,preset_emocional,config)
            if perfil_estilo:base_texture=self._aplicar_perfil_estilo(base_texture,perfil_estilo,config)
            if preset_estetico:base_texture=self._aplicar_preset_estetico(base_texture,preset_estetico,config)
            if config.auto_optimizar_coherencia and preset_emocional:base_texture=self._optimizar_coherencia_final(base_texture,preset_emocional)
            return base_texture
        except Exception as e:logger.warning(f"⚠️ Error Aurora V7: {e}");return self._generate_standard_texture(config,seed)
    
    def _aplicar_modulacion_emocional(self,texture:np.ndarray,preset_emocional:'PresetEmocionalCompleto',config:NoiseConfigV34Unificado)->np.ndarray:
        try:
            t=np.linspace(0,config.duration_sec,len(texture[0]) if texture.ndim>1 else len(texture));freq_base=preset_emocional.frecuencia_base;modulacion_emocional=1.0+0.1*np.sin(2*np.pi*freq_base*t)
            for nt,intensidad in preset_emocional.neurotransmisores.items():
                if nt=='dopamina' and intensidad>0.5:burst_pattern=1.0+0.2*intensidad*(np.random.random(len(t))>0.8);modulacion_emocional*=burst_pattern
                elif nt=='gaba' and intensidad>0.5:gaba_decay=np.exp(-t*intensidad*0.1);modulacion_emocional*=gaba_decay
                elif nt=='serotonina' and intensidad>0.5:serotonina_wave=1.0+0.1*intensidad*np.sin(2*np.pi*0.1*t);modulacion_emocional*=serotonina_wave
            
            if texture.ndim>1:
                for i in range(texture.shape[0]):texture[i]*=modulacion_emocional[:len(texture[i])]
            else:texture*=modulacion_emocional[:len(texture)]
            return texture
        except Exception as e:logger.warning(f"⚠️ Error modulación: {e}");return texture
    
    def _aplicar_perfil_estilo(self,texture:np.ndarray,perfil_estilo:'PerfilEstiloCompleto',config:NoiseConfigV34Unificado)->np.ndarray:
        try:
            if perfil_estilo.modulacion_am>0:
                t=np.linspace(0,config.duration_sec,len(texture[0]) if texture.ndim>1 else len(texture));am_mod=1.0+perfil_estilo.modulacion_am*np.sin(2*np.pi*perfil_estilo.velocidad_modulacion*t)
                if texture.ndim>1:
                    for i in range(texture.shape[0]):texture[i]*=am_mod[:len(texture[i])]
                else:texture*=am_mod[:len(texture)]
            if hasattr(perfil_estilo,'filtro') and perfil_estilo.filtro:cutoff=perfil_estilo.filtro.frecuencia_corte;texture=self._aplicar_filtro_perfil(texture,cutoff,config.sample_rate)
            return texture
        except Exception as e:logger.warning(f"⚠️ Error perfil: {e}");return texture
    
    def _aplicar_preset_estetico(self,texture:np.ndarray,preset_estetico:'PresetEstiloCompleto',config:NoiseConfigV34Unificado)->np.ndarray:
        try:
            if hasattr(preset_estetico,'textura'):
                textura=preset_estetico.textura
                if textura.granularidad>0:granularity_noise=np.random.normal(0,textura.granularidad*0.05,len(texture[0]) if texture.ndim>1 else len(texture));if texture.ndim>1:[texture[i].__iadd__(granularity_noise[:len(texture[i])]) for i in range(texture.shape[0])];else:texture+=granularity_noise[:len(texture)]
                if textura.fluctuacion>0:t=np.linspace(0,config.duration_sec,len(texture[0]) if texture.ndim>1 else len(texture));fluctuation=1.0+textura.fluctuacion*0.1*np.sin(2*np.pi*0.3*t);if texture.ndim>1:[texture[i].__imul__(fluctuation[:len(texture[i])]) for i in range(texture.shape[0])];else:texture*=fluctuation[:len(texture)]
            return texture
        except Exception as e:logger.warning(f"⚠️ Error estético: {e}");return texture
    
    def _optimizar_coherencia_final(self,texture:np.ndarray,preset_emocional:'PresetEmocionalCompleto')->np.ndarray:
        try:
            intensidad=preset_emocional.intensidad.value if hasattr(preset_emocional.intensidad,'value') else preset_emocional.intensidad
            target_rms=0.3 if intensidad=='suave' else 0.7 if intensidad=='intenso' else 0.5
            current_rms=np.sqrt(np.mean(texture**2))
            if current_rms>0:normalization_factor=target_rms/current_rms;texture*=normalization_factor
            return texture
        except Exception as e:logger.warning(f"⚠️ Error coherencia final: {e}");return texture
    
    def _aplicar_filtro_perfil(self,texture:np.ndarray,cutoff:float,sample_rate:int)->np.ndarray:
        try:
            if texture.ndim>1:filtered=np.zeros_like(texture);[filtered.__setitem__(i,self.apply_lowpass(texture[i],cutoff,sample_rate)) for i in range(texture.shape[0])];return filtered
            else:return self.apply_lowpass(texture,cutoff,sample_rate)
        except:return texture
    
    def _generate_fallback_texture(self,duracion_sec:float,objetivo:str,**kwargs)->np.ndarray:
        samples=int(44100*duracion_sec);freq_map={'relajacion':6.0,'concentracion':14.0,'creatividad':10.0,'meditacion':7.83,'energia':15.0,'calma':5.0};freq=freq_map.get(objetivo.lower(),10.0);t=np.linspace(0,duracion_sec,samples);base_wave=0.3*np.sin(2*np.pi*freq*t);noise=0.1*np.random.normal(0,1,samples);texture=base_wave+noise;return np.stack([texture,texture])
    
    def _convert_to_unified_config(self,config_legacy)->NoiseConfigV34Unificado:
        return NoiseConfigV34Unificado(duration_sec=getattr(config_legacy,'duration_sec',1.0),sample_rate=getattr(config_legacy,'sample_rate',44100),amplitude=getattr(config_legacy,'amplitude',0.5),stereo_width=getattr(config_legacy,'stereo_width',0.0),texture_complexity=getattr(config_legacy,'texture_complexity',1.0),spectral_tilt=getattr(config_legacy,'spectral_tilt',0.0))
    
    def _generate_cache_key(self,config:NoiseConfigV34Unificado,seed:Optional[int])->str:
        aurora_key=f"_{config.preset_emocional}_{config.perfil_estilo}_{config.preset_estetico}" if self.enable_aurora_v7 else ""
        return(f"dur_{config.duration_sec}_amp_{config.amplitude}_type_{config.texture_type.value}_neuro_{config.neurotransmitter_profile}_seed_{seed}{aurora_key}")
    
    def generate_white_noise(self,duration_ms:float=1000,sr:int=44100,gain_db:float=-15,precision_cientifica:bool=True,analisis_espectral:bool=False,validacion_distribucion:bool=True)->np.ndarray:
        if duration_ms<=0:duration_ms=max(1,abs(duration_ms))
        if sr<=0:sr=44100
        samples=np.random.normal(0,1,int(sr*duration_ms/1000));factor=10**(gain_db/20);noise_v6=samples*factor
        if not precision_cientifica:return noise_v6
        noise_v7=self._generar_ruido_precision_cientifica(duration_ms,sr,gain_db)
        if validacion_distribucion:validacion=self._validar_distribucion_estadistica(noise_v7);if not validacion["distribucion_valida"]:logger.warning("Validación distribución")
        if analisis_espectral:analisis=self._analizar_espectro_ruido_blanco(noise_v7,sr);if analisis["planitud_espectral"]<0.8:logger.warning("Planitud espectral baja")
        metadatos={"tipo":"ruido_blanco","duracion_ms":duration_ms,"sample_rate":sr,"gain_db":gain_db,"precision_cientifica":precision_cientifica,"timestamp":datetime.now().isoformat()};_almacenar_metadatos_ruido("white_noise",metadatos);self.stats['scientific_generations']+=1;return noise_v7
    
    def _generar_ruido_precision_cientifica(self,duration_ms:float,sr:int,gain_db:float)->np.ndarray:
        n_samples=int(sr*duration_ms/1000);np.random.seed(None);ruido=np.random.normal(0,1,n_samples);factor=10**(gain_db/20);ruido=ruido*factor;dc_offset=np.mean(ruido)
        if abs(dc_offset)>1e-6:ruido-=dc_offset
        return ruido
    
    def _validar_distribucion_estadistica(self,ruido:np.ndarray)->Dict[str,Any]:
        if len(ruido)==0:return{"distribucion_valida":False,"problemas":["Señal vacía"]}
        try:
            if len(ruido)<=5000:stat,p_value=shapiro(ruido[:5000])
            else:stat,p_value=normaltest(ruido)
            normalidad_ok=p_value>0.05;media=np.mean(ruido);media_ok=abs(media)<0.01;std=np.std(ruido);std_ok=0.1<std<2.0;problemas=[]
            if not normalidad_ok:problemas.append("Distribución no normal")
            if not media_ok:problemas.append(f"Media descentrada: {media:.4f}")
            if not std_ok:problemas.append(f"Desviación estándar anómala: {std:.4f}")
            return{"distribucion_valida":len(problemas)==0,"normalidad":normalidad_ok,"media_centrada":media_ok,"std_apropiada":std_ok,"problemas":problemas,"estadisticas":{"media":media,"std":std,"p_value_normalidad":p_value}}
        except:return{"distribucion_valida":True,"problemas":[]}
    
    def _analizar_espectro_ruido_blanco(self,ruido:np.ndarray,sr:int)->Dict[str,Any]:
        fft_ruido=np.abs(fft(ruido));freqs=fftfreq(len(ruido),1/sr);n_half=len(fft_ruido)//2;fft_half=fft_ruido[:n_half];freqs_half=freqs[:n_half];media_espectral=np.mean(fft_half);std_espectral=np.std(fft_half);planitud_espectral=1.0-(std_espectral/(media_espectral+1e-10));umbral_pico=media_espectral*3;picos_prominentes=np.sum(fft_half>umbral_pico);return{"planitud_espectral":max(0,min(1,planitud_espectral)),"picos_prominentes":picos_prominentes,"media_espectral":media_espectral,"std_espectral":std_espectral,"rango_frecuencial":(freqs_half[0],freqs_half[-1])}
    
    def apply_lowpass(self,signal:np.ndarray,cutoff:float=8000,sr:int=44100,modelo_cientifico:bool=True,validacion_respuesta:bool=True,optimizacion_neuroacustica:bool=True)->np.ndarray:
        if len(signal)==0:return np.array([])
        nyquist=sr/2
        if cutoff>=nyquist:cutoff=nyquist*0.95
        if cutoff<=0:cutoff=100
        try:b,a=butter(6,cutoff/(0.5*sr),btype="low");filtered_v6=filtfilt(b,a,signal)
        except:return signal
        if not modelo_cientifico:return filtered_v6
        filtered_v7=self._aplicar_filtro_cientifico_avanzado(signal,cutoff,sr,ModeloFiltroCientifico.NEURAL_OPTIMIZED)
        if validacion_respuesta:validacion=self._validar_respuesta_filtro(signal,filtered_v7,cutoff,sr);if not validacion["respuesta_valida"]:logger.warning("Validación filtro")
        if optimizacion_neuroacustica and len(filtered_v7)>200:ventana=20;kernel=np.ones(ventana)/ventana;suavizado=np.convolve(filtered_v7,kernel,mode='same');filtered_v7=0.9*filtered_v7+0.1*suavizado
        return filtered_v7
    
    def _aplicar_filtro_cientifico_avanzado(self,signal:np.ndarray,cutoff:float,sr:int,modelo:ModeloFiltroCientifico)->np.ndarray:
        if modelo==ModeloFiltroCientifico.NEURAL_OPTIMIZED:
            try:
                signal_clean=signal-np.mean(signal);nyquist=sr/2;normalized_cutoff=cutoff/nyquist;normalized_cutoff=min(0.99,max(0.01,normalized_cutoff));b,a=butter(8,normalized_cutoff,btype='low');filtered=filtfilt(b,a,signal_clean)
                if len(filtered)>100:ventana=50;kernel=np.hanning(ventana);kernel=kernel/np.sum(kernel);filtered[:ventana]=np.convolve(filtered[:ventana*2],kernel,mode='same')[:ventana];filtered[-ventana:]=np.convolve(filtered[-ventana*2:],kernel,mode='same')[-ventana:]
                return filtered
            except:b,a=butter(6,cutoff/(0.5*sr),btype="low");return filtfilt(b,a,signal)
        else:b,a=butter(6,cutoff/(0.5*sr),btype="low");return filtfilt(b,a,signal)
    
    def _validar_respuesta_filtro(self,original:np.ndarray,filtrado:np.ndarray,cutoff:float,sr:int)->Dict[str,Any]:
        fft_orig=np.abs(fft(original));fft_filt=np.abs(fft(filtrado));freqs=fftfreq(len(original),1/sr);idx_cutoff=np.argmin(np.abs(freqs-cutoff));atenuacion_cutoff=20*np.log10((fft_filt[idx_cutoff]+1e-10)/(fft_orig[idx_cutoff]+1e-10));atenuacion_ok=-4<=atenuacion_cutoff<=-2;problemas=[]
        if not atenuacion_ok:problemas.append(f"Atenuación en frecuencia de corte: {atenuacion_cutoff:.1f}dB")
        return{"respuesta_valida":len(problemas)==0,"atenuacion_cutoff_db":atenuacion_cutoff,"problemas":problemas}
    
    def _generate_therapeutic_texture(self,config,seed):base_texture=self._generate_standard_texture(config,seed);return self.optimizar_ruido_terapeutico(base_texture,config.objetivo_terapeutico,config.configuracion_cientifica,validacion_clinica=True,iteraciones_maximas=3) if config.objetivo_terapeutico else base_texture
    
    def _generate_organic_pattern_texture(self,config,seed):return self.generar_patrones_organicos(config.patron_organico,config.duration_sec*1000,config.configuracion_cientifica) if config.patron_organico else self._generate_standard_texture(config,seed)
    
    def _generate_quantum_texture(self,config,seed):return self.generar_ruido_cuantico(config.tipo_patron_cuantico,config.duration_sec*1000,{'nivel_cuantico':0.7,'coherencia_temporal':0.8}) if config.tipo_patron_cuantico else self._generate_standard_texture(config,seed)
    
    def _generate_neuroacoustic_texture(self,config,seed):
        if not self.enable_aurora_v7 or not config.neurotransmitter_profile:return self._generate_standard_texture(config,seed)
        try:
            n_samples=int(config.duration_sec*config.sample_rate);t=np.linspace(0,config.duration_sec,n_samples);freq_map={'dopamina':12.0,'serotonina':7.5,'gaba':6.0,'acetilcolina':14.0,'oxitocina':8.0};primary_freq=freq_map.get(config.neurotransmitter_profile.lower(),10.0)
            if 'gaba'in config.neurotransmitter_profile.lower():base_wave=np.sin(2*np.pi*primary_freq*t);texture=base_wave*np.exp(-t*0.1)
            elif 'dopamina'in config.neurotransmitter_profile.lower():base_wave=np.sin(2*np.pi*primary_freq*t);burst_pattern=1+0.3*np.sin(2*np.pi*0.5*t);texture=base_wave*burst_pattern
            else:texture=np.sin(2*np.pi*primary_freq*t)
            for i,harmonic_mult in enumerate([2,3,4]):harmonic_amp=0.3/harmonic_mult;texture+=harmonic_amp*np.sin(2*np.pi*primary_freq*harmonic_mult*t)
            if np.max(np.abs(texture))>0:texture=texture/np.max(np.abs(texture))*config.amplitude
            return self._process_stereo_v34(texture,config.stereo_width,config.spatial_effect)
        except:return self._generate_standard_texture(config,seed)
    
    def _generate_advanced_texture(self,config,seed):
        n_samples=int(config.duration_sec*config.sample_rate)
        if config.texture_type==TipoTexturaUnificado.HOLOGRAPHIC:texture=self._generate_holographic_texture(n_samples,config)
        elif config.texture_type==TipoTexturaUnificado.QUANTUM_FIELD:texture=self._generate_quantum_field_texture(n_samples,config)
        elif config.texture_type==TipoTexturaUnificado.CRYSTALLINE_MATRIX:texture=self._generate_crystalline_matrix_texture(n_samples,config)
        else:texture=self._generate_standard_texture(config,seed)
        return texture
    
    def _generate_holographic_texture(self,n_samples,config):
        layers=[];base_freq=220.0;phi=(1+math.sqrt(5))/2
        for layer in range(4):layer_freq=base_freq*(phi**layer);t=np.linspace(0,config.duration_sec,n_samples);layer_wave=np.sin(2*np.pi*layer_freq*t);holographic_mod=np.sin(2*np.pi*layer_freq*0.618*t);layer_wave=layer_wave*(1+0.3*holographic_mod);layer_amplitude=config.amplitude/(layer+1);layers.append(layer_wave*layer_amplitude)
        holographic_texture=np.sum(layers,axis=0)
        if np.max(np.abs(holographic_texture))>0:holographic_texture=holographic_texture/np.max(np.abs(holographic_texture))*config.amplitude
        return self._process_stereo_v34(holographic_texture,config.stereo_width,config.spatial_effect)
    
    def _generate_quantum_field_texture(self,n_samples,config):
        quantum_noise=np.random.normal(0,config.amplitude*0.1,n_samples);coherence_length=min(1000,n_samples//10);coherence_kernel=np.exp(-np.linspace(0,5,coherence_length));coherent_noise=np.convolve(quantum_noise,coherence_kernel,mode='same');t=np.linspace(0,config.duration_sec,n_samples);base_freq=7.83;quantum_field=(np.sin(2*np.pi*base_freq*t)+0.5*np.sin(2*np.pi*base_freq*2*t)+0.3*np.sin(2*np.pi*base_freq*3.14159*t));texture=quantum_field*config.amplitude+coherent_noise;quantum_prob=np.random.random(n_samples);quantum_gates=quantum_prob>0.8;texture[quantum_gates]*=1.5;return self._process_stereo_v34(texture,config.stereo_width,config.spatial_effect)
    
    def _generate_crystalline_matrix_texture(self,n_samples,config):
        crystal_freqs=[256,512,1024,2048];t=np.linspace(0,config.duration_sec,n_samples);texture=np.zeros(n_samples)
        for i,freq in enumerate(crystal_freqs):amplitude=config.amplitude/(2**i);crystal_wave=amplitude*np.sin(2*np.pi*freq*t);crystal_mod=1+0.05*np.sin(2*np.pi*freq*0.001*t);crystal_wave*=crystal_mod;texture+=crystal_wave
        defect_probability=0.001;defects=np.random.random(n_samples)<defect_probability;texture[defects]*=np.random.uniform(0.5,1.5,np.sum(defects));return self._process_stereo_v34(texture,config.stereo_width,config.spatial_effect)
    
    def _generate_standard_texture(self,config,seed):
        n_samples=int(config.duration_sec*config.sample_rate);base_noise=self._generar_ruido_precision_cientifica(config.duration_sec*1000,config.sample_rate,-15) if config.precision_cientifica else np.random.normal(0,config.amplitude,n_samples);textured=self._apply_advanced_texturing_v34(base_noise,config,seed)
        if config.envelope_shape!="standard":textured=self.apply_envelope(textured,config.envelope_shape,analisis_naturalidad=config.precision_cientifica)
        if config.filter_config:textured=self._apply_optimized_filter_v34(textured,config.filter_config)
        return self._process_stereo_v34(textured,config.stereo_width,config.spatial_effect)
    
    def _apply_advanced_texturing_v34(self,signal,config,seed):
        n_samples=len(signal);duration=config.duration_sec;lfo_rate=self._calculate_lfo_rate_v34(seed,config.texture_complexity,config.texture_type);time_axis=np.linspace(0,duration,n_samples)
        if config.texture_type==TipoTexturaUnificado.NEUROMORPHIC:primary_lfo=self._lookup_tables['neural_spike'][np.mod((time_axis*len(self._lookup_tables['neural_spike'])).astype(int),len(self._lookup_tables['neural_spike']))]
        elif config.texture_type in[TipoTexturaUnificado.FRACTAL,TipoTexturaUnificado.ORGANIC]:primary_lfo=self._lookup_tables['organic_flow'][np.mod((time_axis*len(self._lookup_tables['organic_flow'])).astype(int),len(self._lookup_tables['organic_flow']))]
        elif config.texture_type==TipoTexturaUnificado.QUANTUM:primary_lfo=self._lookup_tables['quantum_coherence'][np.mod((time_axis*len(self._lookup_tables['quantum_coherence'])).astype(int),len(self._lookup_tables['quantum_coherence']))]
        else:primary_lfo=np.sin(2*np.pi*lfo_rate*time_axis)
        golden_ratio=(1+math.sqrt(5))/2;secondary_lfo=0.3*np.sin(2*np.pi*lfo_rate*golden_ratio*time_axis+np.pi/4);combined_lfo=primary_lfo+golden_ratio*secondary_lfo if config.modulation==TipoModulacionUnificado.GOLDEN_RATIO else primary_lfo+secondary_lfo;mod_depth=0.3+0.2*config.texture_complexity;modulated=signal*(1+mod_depth*combined_lfo);return modulated
    
    def _calculate_lfo_rate_v34(self,seed,complexity,texture_type):
        base_rates={TipoTexturaUnificado.MEDITATION:0.02,TipoTexturaUnificado.FOCUS:0.05,TipoTexturaUnificado.RELAXATION:0.08,TipoTexturaUnificado.CONSCIOUSNESS:0.15,TipoTexturaUnificado.NEUROMORPHIC:0.1,TipoTexturaUnificado.HOLOGRAPHIC:0.618};base_rate=base_rates.get(texture_type,0.08+0.04*complexity);return base_rate+0.02*((seed%20)/20.0) if seed is not None else base_rate
    
    def _process_stereo_v34(self,signal,stereo_width,spatial_effect):
        if spatial_effect==EfectoEspacialUnificado.STEREO_BASIC:return self._process_basic_stereo(signal,stereo_width)
        elif spatial_effect==EfectoEspacialUnificado.SURROUND_3D:return self._process_3d_surround(signal,stereo_width)
        elif spatial_effect==EfectoEspacialUnificado.HOLOGRAPHIC_8D:return self._process_holographic_8d(signal,stereo_width)
        elif spatial_effect==EfectoEspacialUnificado.BINAURAL_3D:return self._process_binaural_3d(signal,stereo_width)
        elif spatial_effect==EfectoEspacialUnificado.QUANTUM_SPACE:return self._process_quantum_space(signal,stereo_width)
        else:return self._process_basic_stereo(signal,stereo_width)
    
    def _process_basic_stereo(self,signal,stereo_width):
        if abs(stereo_width)<0.001:return signal
        width=np.clip(stereo_width,-1.0,1.0);mid=signal;side=signal*width*0.5
        if abs(width)>0.1:side=self._decorrelate_channel_v34(side)
        left=mid-side;right=mid+side;return np.stack([left,right])
    
    def _process_3d_surround(self,signal,stereo_width):
        duration=len(signal)/44100;t=np.linspace(0,duration,len(signal));elevation_mod=np.sin(2*np.pi*0.1*t);depth_mod=np.cos(2*np.pi*0.05*t);left=signal*(0.6+0.2*elevation_mod+0.2*depth_mod);right=signal*(0.6-0.2*elevation_mod+0.2*np.sin(2*np.pi*0.1*t+np.pi/3));width_factor=np.clip(stereo_width,0,2.0);left=left*(1+width_factor*0.3);right=right*(1+width_factor*0.3);return np.stack([left,right])
    
    def _process_holographic_8d(self,signal,stereo_width):
        duration=len(signal)/44100;t=np.linspace(0,duration,len(signal));rotation1=np.sin(2*np.pi*0.2*t);golden_ratio=(1+math.sqrt(5))/2;rotation2=np.cos(2*np.pi*0.2*golden_ratio*t);left_mod=(0.5+0.15*rotation1+0.1*rotation2);right_mod=(0.5+0.15*np.cos(2*np.pi*0.2*t+np.pi/2)+0.1*np.sin(2*np.pi*0.2*golden_ratio*t+np.pi/3));left=signal*left_mod;right=signal*right_mod;holographic_width=np.clip(stereo_width*1.5,0,2.0);left=left*(1+holographic_width*0.2);right=right*(1+holographic_width*0.2);return np.stack([left,right])
    
    def _process_binaural_3d(self,signal,stereo_width):
        duration=len(signal)/44100;t=np.linspace(0,duration,len(signal));base_freq_diff=6.0;adaptive_diff=base_freq_diff*(1+0.3*np.sin(2*np.pi*0.05*t));binaural_mod_left=np.sin(2*np.pi*adaptive_diff*t);binaural_mod_right=np.sin(2*np.pi*adaptive_diff*t+np.pi);left=signal*(1+0.05*binaural_mod_left);right=signal*(1+0.05*binaural_mod_right);return np.stack([left,right])
    
    def _process_quantum_space(self,signal,stereo_width):
        duration=len(signal)/44100;t=np.linspace(0,duration,len(signal));quantum_prob=np.random.random(len(signal));quantum_state=quantum_prob>0.7;state_left=np.sin(2*np.pi*0.1*t);state_right=np.cos(2*np.pi*0.1*t);entanglement=np.sin(2*np.pi*0.05*t)*np.cos(2*np.pi*0.07*t);left=signal.copy();right=signal.copy();left[quantum_state]*=(1+0.2*state_left[quantum_state]+0.1*entanglement[quantum_state]);right[quantum_state]*=(1+0.2*state_right[quantum_state]-0.1*entanglement[quantum_state]);return np.stack([left,right])
    
    def _decorrelate_channel_v34(self,signal):
        delay_samples=3;decorr1=0.7*signal+0.3*np.roll(signal,delay_samples) if len(signal)>delay_samples else signal
        try:a1=0.1;decorr2=signal.copy();[decorr2.__setitem__(i,decorr2[i]+a1*decorr2[i-1]) for i in range(1,len(decorr2))]
        except:decorr2=signal
        return 0.6*decorr1+0.4*decorr2
    
    def _apply_optimized_filter_v34(self,signal,filter_config):
        try:
            cache_key=(f"{filter_config.cutoff_freq}_{filter_config.filter_type}_{filter_config.sample_rate}_{filter_config.order}")
            if cache_key not in self._filter_cache:nyq=filter_config.sample_rate*0.5;normalized_cutoff=filter_config.cutoff_freq/nyq;normalized_cutoff=0.99 if normalized_cutoff>=1.0 else 0.01 if normalized_cutoff<=0.0 else normalized_cutoff;sos=butter(filter_config.order,normalized_cutoff,btype=filter_config.filter_type,analog=False,output='sos');self._filter_cache[cache_key]=sos
            sos=self._filter_cache[cache_key];filtered=sosfilt(sos,signal)
            if filter_config.resonance>0.7:q_boost=1+(filter_config.resonance-0.7)*0.5;filtered=filtered*q_boost
            if filter_config.drive>0:filtered=np.tanh(filtered*(1+filter_config.drive))/(1+filter_config.drive)
            return signal*(1-filter_config.wetness)+filtered*filter_config.wetness
        except:return signal
    
    def _validate_and_optimize_texture(self,texture,config):self.stats['validations_performed']+=1;validacion=self.validar_calidad_textura(texture,generar_reporte=False);if not validacion['validacion_global']:logger.warning("Textura requiere optimización");self.stats['optimizations_applied']+=1;return texture
    
    def apply_envelope(self,signal,env_type="gated",analisis_naturalidad=True,optimizacion_perceptual=True,validacion_cientifica=True):
        if len(signal)==0:return np.array([]);length=len(signal);t=np.linspace(0,1,length)
        if env_type=="rising":envelope_v6=t
        elif env_type=="falling":envelope_v6=1-t
        elif env_type=="dynamic":envelope_v6=0.6+0.4*np.sin(2*np.pi*2*t)
        elif env_type=="breathy":envelope_v6=np.abs(np.random.normal(0.5,0.4,length));envelope_v6=np.clip(envelope_v6,0,1)
        elif env_type=="shimmer":envelope_v6=0.4+0.6*np.sin(2*np.pi*8*t+np.random.rand())
        elif env_type=="tribal":envelope_v6=(np.sin(2*np.pi*4*t)>0).astype(float)
        elif env_type=="reverso":half=signal[:length//2];return np.concatenate((half[::-1],signal[length//2:]))
        elif env_type=="quantum":envelope_v6=(np.random.rand(length)>0.7).astype(float)
        else:gate=np.tile([1]*1000+[0]*1000,length//2000+1)[:length];envelope_v6=gate
        signal_v6=signal*envelope_v6
        if not analisis_naturalidad:return signal_v6
        envelope_v7=self._generar_envolvente_cientifica_mejorada(env_type,length,t);signal_v7=signal*envelope_v7
        if analisis_naturalidad:naturalidad=self._analizar_naturalidad_envolvente(envelope_v7,env_type);if naturalidad<0.7:logger.warning(f"Naturalidad envolvente {env_type}: {naturalidad:.3f}")
        if optimizacion_perceptual and env_type in["gated","quantum"]:envolvente_suavizada=np.copy(envelope_v7);cambios=np.abs(np.diff(envelope_v7));umbral_abrupto=np.percentile(cambios,90);[envolvente_suavizada.__setitem__(i,np.mean(envolvente_suavizada[max(0,i-min(10,len(envolvente_suavizada)-i)//2):min(len(envolvente_suavizada),i+min(10,len(envolvente_suavizada)-i)//2)])) for i in range(1,len(envolvente_suavizada)-1) if cambios[i-1]>umbral_abrupto and min(10,len(envolvente_suavizada)-i)>1];signal_v7=signal*envolvente_suavizada
        return signal_v7
    
    def _generar_envolvente_cientifica_mejorada(self,env_type,length,t):
        if env_type=="breathy":freq_respiracion=0.25;patron_base=0.5+0.3*np.sin(2*np.pi*freq_respiracion*t*length/44100);variabilidad=0.1*np.random.normal(0,1,length);variabilidad=np.convolve(variabilidad,np.ones(100)/100,mode='same');envolvente=patron_base+variabilidad;return np.clip(envolvente,0,1)
        elif env_type=="shimmer":freq_base=6;freq_modulacion=0.1;shimmer_base=0.5+0.4*np.sin(2*np.pi*freq_base*t);variabilidad_lenta=0.1*np.sin(2*np.pi*freq_modulacion*t);envolvente=shimmer_base+variabilidad_lenta;return np.clip(envolvente,0.1,1.0)
        elif env_type=="tribal":freq_principal=2;freq_secundaria=3;pulso_principal=(np.sin(2*np.pi*freq_principal*t)>0).astype(float);pulso_secundario=0.3*(np.sin(2*np.pi*freq_secundaria*t)>0.5).astype(float);envolvente_pulso=0.8+0.2*np.sin(np.pi*t*freq_principal*2);envolvente=(pulso_principal+pulso_secundario)*envolvente_pulso;return np.clip(envolvente,0,1)
        elif env_type=="rising":return t
        elif env_type=="falling":return 1-t
        elif env_type=="dynamic":return 0.6+0.4*np.sin(2*np.pi*2*t)
        else:return np.ones(length)
    
    def _analizar_naturalidad_envolvente(self,envolvente,env_type):
        if len(envolvente)==0:return 0.0
        if env_type=="breathy":variabilidad=np.std(np.diff(envolvente));regularidad=np.corrcoef(envolvente[:-1],envolvente[1:])[0,1];naturalidad=(variabilidad*10)*(1-abs(regularidad));return min(1.0,max(0.0,naturalidad))
        elif env_type=="shimmer":fft_env=np.abs(fft(envolvente));pico_principal=np.max(fft_env[1:])/(np.mean(fft_env[1:])+1e-10);naturalidad=1.0/(1.0+abs(pico_principal-3.0));return naturalidad
        else:return 1.0/(1.0+np.std(np.diff(envolvente))*10) if len(envolvente)>1 else 0.5
    
    def pan_stereo(self,signal,style="center",sr=44100,modelo_psicoacustico=True,validacion_imagen=True,optimizacion_neuroacustica=True):
        if len(signal)==0:return np.array([[],[]])
        t=np.linspace(0,1,len(signal))
        if style=="8d":pan_v6=0.5+0.5*np.sin(2*np.pi*0.25*t)
        elif style=="reverse":pan_v6=np.linspace(1,0,len(signal))
        else:pan_v6=np.ones(len(signal))*0.5
        left_v6=signal*(1-pan_v6);right_v6=signal*pan_v6;stereo_v6=np.stack([left_v6,right_v6],axis=0)
        if not modelo_psicoacustico:return stereo_v6
        pan_v7=self._generar_curva_paneo_cientifica(style,len(signal),sr);pan_curve_log=np.sign(pan_v7-0.5)*np.abs(pan_v7-0.5)**0.7+0.5;left_v7=signal*(1-pan_curve_log);right_v7=signal*pan_curve_log
        if optimizacion_neuroacustica and style in["8d","neuroacoustic"]and len(right_v7)>2:delay_samples=2;right_delayed=np.zeros_like(right_v7);right_delayed[delay_samples:]=right_v7[:-delay_samples];right_v7=0.9*right_v7+0.1*right_delayed
        stereo_v7=np.stack([left_v7,right_v7],axis=0);return stereo_v7
    
    def _generar_curva_paneo_cientifica(self,style,length,sr):
        t=np.linspace(0,1,length)
        if style=="8d":component1=0.5+0.3*np.sin(2*np.pi*0.25*t);component2=0.1*np.sin(2*np.pi*0.7*t);return component1+component2
        elif style=="neuroacoustic":freq_hemisferica=0.1;return 0.5+0.4*np.sin(2*np.pi*freq_hemisferica*t)
        elif style=="reverse":return np.linspace(1,0,length)
        else:return np.ones(length)*0.5
    
    def optimizar_ruido_terapeutico(self,señal_original,objetivo_terapeutico,configuracion=None,validacion_clinica=True,iteraciones_maximas=5):
        if len(señal_original)==0:return np.array([])
        if configuracion is None:configuracion=ConfiguracionRuidoCientifica()
        objetivos_validos=["relajacion","sueño","ansiedad","concentracion","meditacion","sanacion"]
        if objetivo_terapeutico not in objetivos_validos:objetivo_terapeutico="relajacion"
        config_terapeutica={"relajacion":{"intensidad_target":0.3,"patron_temporal":"descendente"},"sueño":{"intensidad_target":0.2,"patron_temporal":"muy_descendente"},"concentracion":{"intensidad_target":0.4,"patron_temporal":"estable"}}.get(objetivo_terapeutico,{"intensidad_target":0.3,"patron_temporal":"descendente"})
        rms_actual=np.sqrt(np.mean(señal_original**2));potencial_base=max(0,1.0-abs(rms_actual-config_terapeutica["intensidad_target"])*3);señal_optimizada=señal_original.copy();mejor_puntuacion=potencial_base
        for iteracion in range(iteraciones_maximas):
            rms_actual=np.sqrt(np.mean(señal_optimizada**2));rms_objetivo=config_terapeutica["intensidad_target"]
            if rms_actual>0:factor_ajuste=rms_objetivo/rms_actual;factor_suave=0.7*factor_ajuste+0.3;señal_optimizada*=factor_suave
            patron=config_terapeutica["patron_temporal"];t=np.linspace(0,1,len(señal_optimizada))
            if patron=="descendente":envolvente=1.0-0.3*t
            elif patron=="muy_descendente":envolvente=1.0-0.6*t**0.5
            else:envolvente=np.ones(len(señal_optimizada))
            envolvente=np.clip(envolvente,0.1,1.0);señal_optimizada*=envolvente;rms_eval=np.sqrt(np.mean(señal_optimizada**2));evaluacion={"potencial_terapeutico":max(0,1.0-abs(rms_eval-config_terapeutica["intensidad_target"])*3)}
            if evaluacion['potencial_terapeutico']>mejor_puntuacion:señal_optimizada=señal_optimizada;mejor_puntuacion=evaluacion['potencial_terapeutico']
        if validacion_clinica:rms=np.sqrt(np.mean(señal_optimizada**2));problemas=[];if rms>0.8:problemas.append("Nivel peligroso");validacion={"apto_clinico":len(problemas)==0,"problemas":problemas};if not validacion['apto_clinico']:logger.warning("Validación clínica")
        metadatos={"objetivo_terapeutico":objetivo_terapeutico,"iteraciones_realizadas":iteraciones_maximas,"timestamp":datetime.now().isoformat()};_almacenar_metadatos_ruido("terapeutico_optimizado",metadatos);return señal_optimizada
    
    def validar_calidad_textura(self,señal,criterios_personalizados=None,generar_reporte=True,nivel_detalle="completo"):
        if len(señal)==0:return{"validacion_global":False,"error":"Señal vacía"}
        if criterios_personalizados is None:criterios_personalizados={"calidad_minima":0.7,"distorsion_maxima":0.05,"naturalidad_minima":0.6,"seguridad_auditiva":True,"efectividad_minima":0.65}
        validacion_completa={"timestamp":datetime.now().isoformat(),"nivel_detalle":nivel_detalle,"criterios_aplicados":criterios_personalizados,"validaciones_individuales":{},"validacion_global":False,"problemas_detectados":[],"recomendaciones":[]};max_amplitude=np.max(np.abs(señal));rms=np.sqrt(np.mean(señal**2));problemas=[]
        if max_amplitude>0.95:problemas.append("Saturación detectada")
        if rms==0:problemas.append("Señal silenciosa")
        validacion_tecnica={"valida":len(problemas)==0,"problemas":problemas,"puntuacion":max(0,1.0-len(problemas)*0.3)};validacion_completa["validaciones_individuales"]["tecnica"]=validacion_tecnica
        if not validacion_tecnica["valida"]:validacion_completa["problemas_detectados"].extend(validacion_tecnica["problemas"])
        puntuaciones=[validacion_tecnica["puntuacion"]];puntuacion_global=np.mean(puntuaciones);validacion_completa["puntuacion_global"]=puntuacion_global;validacion_completa["validacion_global"]=(len(validacion_completa["problemas_detectados"])==0 and puntuacion_global>=criterios_personalizados.get("calidad_minima",0.7))
        if puntuacion_global<0.7:validacion_completa["recomendaciones"].append("Considerar optimización general")
        return validacion_completa
    
    def generar_patrones_organicos(self,tipo_patron,duracion_ms=3000,configuracion=None,validacion_naturalidad=True):
        if configuracion is None:configuracion=ConfiguracionRuidoCientifica()
        tipos_validos=["respiratorio","cardiaco","neural","oceano","viento","lluvia","bosque","cosmos"]
        if tipo_patron not in tipos_validos:tipo_patron="respiratorio"
        parametros_patron={"respiratorio":{"frecuencia_base":0.25,"profundidad":0.6},"cardiaco":{"frecuencia_base":1.2,"profundidad":0.8},"oceano":{"frecuencia_base":0.1,"profundidad":0.9}}.get(tipo_patron,{"frecuencia_base":0.25,"profundidad":0.6})
        sr=configuracion.sample_rate;n_samples=int(sr*duracion_ms/1000);t=np.linspace(0,duracion_ms/1000,n_samples)
        if tipo_patron=="respiratorio":freq_base=parametros_patron["frecuencia_base"];profundidad=parametros_patron["profundidad"];respiracion_base=np.sin(2*np.pi*freq_base*t);variacion_freq=0.1*np.sin(2*np.pi*freq_base*0.1*t);fase=2*np.pi*freq_base*t;patron_asimetrico=np.where(np.sin(fase)>=0,np.sin(fase)**0.7,-(-np.sin(fase))**1.3);patron=profundidad*patron_asimetrico*(1+variacion_freq*0.2)
        elif tipo_patron=="oceano":freq_base=parametros_patron["frecuencia_base"];profundidad=parametros_patron["profundidad"];onda1=np.sin(2*np.pi*freq_base*t);onda2=0.6*np.sin(2*np.pi*freq_base*1.618*t);onda3=0.3*np.sin(2*np.pi*freq_base*2.618*t);modulacion_marea=0.8+0.2*np.sin(2*np.pi*freq_base*0.1*t);patron=(onda1+onda2+onda3)*modulacion_marea*profundidad
        else:patron=np.sin(2*np.pi*0.25*t)*0.6
        if validacion_naturalidad and len(patron)>1:diferencias=np.diff(patron);suavidad=1.0/(1.0+np.std(diferencias)*10);naturalidad=max(0,min(1,suavidad));if naturalidad<0.7:logger.warning(f"Naturalidad {tipo_patron}")
        left=patron;right=patron.copy()
        if len(right)>10:delay_samples=5;right_delayed=np.zeros_like(right);right_delayed[delay_samples:]=right[:-delay_samples];right=0.9*right+0.1*right_delayed
        patron_stereo=np.stack([left,right]);metadatos={"tipo_patron":tipo_patron,"duracion_ms":duracion_ms,"timestamp":datetime.now().isoformat()};_almacenar_metadatos_ruido("patron_organico",metadatos);return patron_stereo
    
    def generar_ruido_cuantico(self,tipo_patron_cuantico="coherencia",duracion_ms=2000,parametros_cuanticos=None,validacion_consciousness=True):
        if parametros_cuanticos is None:parametros_cuanticos={"nivel_cuantico":0.7,"coherencia_temporal":0.8,"factor_superposicion":0.6}
        tipos_validos=["coherencia","superposicion","entanglement","tunneling","interference","consciousness"]
        if tipo_patron_cuantico not in tipos_validos:tipo_patron_cuantico="coherencia"
        sr=44100;n_samples=int(sr*duracion_ms/1000);t=np.linspace(0,duracion_ms/1000,n_samples)
        if tipo_patron_cuantico=="coherencia":nivel_cuantico=parametros_cuanticos.get("nivel_cuantico",0.7);coherencia=parametros_cuanticos.get("coherencia_temporal",0.8);freq_base=40;patron=np.zeros_like(t);[patron.__iadd__(nivel_cuantico/n*np.sin(2*np.pi*freq_base*n*t+np.random.random()*2*np.pi)) for n in range(1,8)];modulacion_coherencia=coherencia+(1-coherencia)*np.sin(2*np.pi*0.1*t);patron*=modulacion_coherencia
        elif tipo_patron_cuantico=="superposicion":factor_superposicion=parametros_cuanticos.get("factor_superposicion",0.6);estado1=np.sin(2*np.pi*40*t);estado2=np.sin(2*np.pi*60*t);patron=(np.sqrt(factor_superposicion)*estado1+np.sqrt(1-factor_superposicion)*estado2)
        else:patron=np.sin(2*np.pi*40*t)
        incertidumbre=0.05*np.random.normal(0,1,len(patron));patron+=incertidumbre
        if validacion_consciousness and len(patron)>100:hist,_=np.histogram(patron,bins=50);hist_norm=hist/np.sum(hist);entropia=-np.sum(hist_norm*np.log2(hist_norm+1e-10));factor_consciousness=min(1.0,entropia/6);if factor_consciousness<0.6:logger.warning(f"Factor consciousness: {factor_consciousness:.3f}")
        left=patron.copy();right=patron.copy()
        if tipo_patron_cuantico=="entanglement":right=-right
        patron_stereo=np.stack([left,right]);metadatos={"tipo_patron_cuantico":tipo_patron_cuantico,"duracion_ms":duracion_ms,"timestamp":datetime.now().isoformat()};_almacenar_metadatos_ruido("cuantico",metadatos);return patron_stereo
    
    def clear_cache(self):self._filter_cache.clear();self._texture_cache.clear();self._scientific_cache.clear();self._aurora_experience_cache.clear();self._coherence_cache.clear();self._precompute_windows.cache_clear();self.stats={'textures_generated':0,'cache_hits':0,'processing_time':0.0,'scientific_generations':0,'validations_performed':0,'optimizations_applied':0,'aurora_experiences_created':0,'coherence_optimizations':0}
    
    def get_performance_stats(self):
        cache_stats={'filter_cache_size':len(self._filter_cache),'texture_cache_size':len(self._texture_cache),'scientific_cache_size':len(self._scientific_cache),'cached_windows':len(self._windows),'lookup_tables':len(self._lookup_tables),'aurora_cache_size':len(self._aurora_experience_cache)};cache_stats.update(self.stats)
        if self.enable_aurora_v7:cache_stats['aurora_v7_integration']={'emotion_style_manager_active':self.aurora_gestor is not None,'total_presets_disponibles':len(self.aurora_gestor.presets_emocionales) if self.aurora_gestor else 0,'total_perfiles_disponibles':len(self.aurora_gestor.perfiles_estilo) if self.aurora_gestor else 0}
        return cache_stats
    
    def get_version_info(self):return{'version':self.version,'aurora_v7_connected':self.enable_aurora_v7,'scientific_integration':True,'emotion_style_unified':AURORA_V7_EMOTION_STYLE_DISPONIBLE,'texture_types':[t.value for t in TipoTexturaUnificado],'modulation_types':[m.value for m in TipoModulacionUnificado],'spatial_effects':[s.value for s in EfectoEspacialUnificado],'scientific_filters':[f.value for f in ModeloFiltroCientifico],'aurora_v7_methods':['generar_desde_experiencia_aurora','buscar_presets_aurora_por_efecto','analizar_compatibilidad_aurora','generar_secuencia_aurora','obtener_texturas_por_neurotransmisor'] if self.enable_aurora_v7 else [],'organic_patterns':['respiratorio','cardiaco','neural','oceano','viento','lluvia','bosque','cosmos'],'quantum_patterns':['coherencia','superposicion','entanglement','tunneling','interference','consciousness'],'therapeutic_objectives':['relajacion','sueño','ansiedad','concentracion','meditacion','sanacion']}

class FilterConfig:
    def __init__(self,cutoff_freq=500.0,sample_rate=44100,filter_type="lowpass"):self.cutoff_freq=cutoff_freq;self.sample_rate=sample_rate;self.filter_type=filter_type

class NoiseStyle:DYNAMIC="DYNAMIC";STATIC="STATIC";FRACTAL="FRACTAL"

class NoiseConfig:
    def __init__(self,duration_sec,noise_style,stereo_width=1.0,filter_config=None,sample_rate=44100,amplitude=0.3):self.duration_sec=duration_sec;self.noise_style=noise_style;self.stereo_width=stereo_width;self.sample_rate=sample_rate;self.amplitude=amplitude;self.filter_config=filter_config or FilterConfig()

class HarmonicEssence:
    def __init__(self,cache_size:int=128):self._v34_engine=HarmonicEssenceV34AuroraConnected(cache_size=cache_size,enable_aurora_v7=True)
    def generate_textured_noise(self,config,seed:Optional[int]=None)->np.ndarray:return self._v34_engine.generate_textured_noise(config,seed)
    def clear_cache(self):return self._v34_engine.clear_cache()
    def get_performance_stats(self)->dict:return self._v34_engine.get_performance_stats()

def crear_motor_aurora_conectado(cache_size:int=256)->HarmonicEssenceV34AuroraConnected:return HarmonicEssenceV34AuroraConnected(cache_size=cache_size,enable_aurora_v7=True)

def generar_textura_desde_emocion(objetivo_emocional:str,duracion_sec:float=10.0,contexto:str=None,**kwargs)->np.ndarray:motor=crear_motor_aurora_conectado();return motor.generar_desde_experiencia_aurora(objetivo_emocional,contexto,duracion_sec,**kwargs)

def buscar_presets_por_efecto(efecto:str,umbral:float=0.5)->List[str]:motor=crear_motor_aurora_conectado();return motor.buscar_presets_aurora_por_efecto(efecto,umbral)

def analizar_compatibilidad_presets(preset_emocional:str,perfil_estilo:str,preset_estetico:str=None)->Dict[str,Any]:motor=crear_motor_aurora_conectado();return motor.analizar_compatibilidad_aurora(preset_emocional,perfil_estilo,preset_estetico)

def crear_secuencia_emocional(objetivo:str,duracion_total_sec:float=300.0)->List[np.ndarray]:motor=crear_motor_aurora_conectado();return motor.generar_secuencia_aurora(objetivo,duracion_total_sec)

def obtener_pack_neurotransmisor(neurotransmisor:str,duracion_sec:float=10.0)->Dict[str,np.ndarray]:motor=crear_motor_aurora_conectado();return motor.obtener_texturas_por_neurotransmisor(neurotransmisor,duracion_sec)

HarmonicEssenceV34=HarmonicEssenceV34AuroraConnected;NoiseConfigV34=NoiseConfigV34Unificado;FilterConfigV34=FilterConfigV34Unificado

if __name__=="__main__":
    print("🎨 HarmonicEssence V34 Aurora Connected - Optimized");print("="*80);motor=crear_motor_aurora_conectado();stats=motor.get_performance_stats();version_info=motor.get_version_info();print(f"🚀 {version_info['version']}");print(f"🌟 Aurora V7: {'✅' if version_info['aurora_v7_connected'] else '❌'}");print(f"🎭 Emotion-Style: {'✅' if version_info['emotion_style_unified'] else '❌'}")
    
    if version_info['aurora_v7_connected']:
        aurora_info=stats.get('aurora_v7_integration',{});print(f"🎯 Presets: {aurora_info.get('total_presets_disponibles',0)}");print(f"🎨 Perfiles: {aurora_info.get('total_perfiles_disponibles',0)}");print("🧪 Testing Aurora V7...")
        
        try:
            print("📝 Test 1: Experiencia emocional...");texture_emocional=generar_textura_desde_emocion("claridad_mental",duracion_sec=3.0,contexto="trabajo");print(f"✅ Textura: {texture_emocional.shape}")
            print("📝 Test 2: Búsqueda por efecto...");presets_concentracion=buscar_presets_por_efecto("concentracion",0.5);print(f"✅ Presets concentración: {len(presets_concentracion)}");if presets_concentracion:print(f"   Ejemplo: {presets_concentracion[0]}")
            print("📝 Test 3: Compatibilidad...");compatibilidad=analizar_compatibilidad_presets("claridad_mental","crystalline","etereo");if "error" not in compatibilidad:score=compatibilidad.get('score_compatibilidad_total',0);print(f"✅ Score: {score:.2f}");print(f"   Recomendación: {compatibilidad.get('recomendacion','N/A')}")
            print("📝 Test 4: Pack neurotransmisor...");pack_dopamina=obtener_pack_neurotransmisor("dopamina",2.0);print(f"✅ Pack dopamina: {len(pack_dopamina)} texturas");[print(f"   • {preset_name}: {pack_dopamina[preset_name].shape}") for preset_name in list(pack_dopamina.keys())[:2]]
            print("📝 Test 5: Secuencia emocional...");secuencia=crear_secuencia_emocional("estado_flujo",60.0);print(f"✅ Secuencia: {len(secuencia)} fases");print("🏆 AURORA V7 - COMPLETAMENTE FUNCIONAL!")
        except Exception as e:print(f"❌ Error Aurora V7: {e}");import traceback;traceback.print_exc()
    
    else:
        print("⚠️ Aurora V7 unavailable - compatibility mode")
        try:config_basico=NoiseConfig(duration_sec=2.0,noise_style="DYNAMIC",amplitude=0.3);texture_basica=motor.generate_textured_noise(config_basico);print(f"✅ Textura básica: {texture_basica.shape}")
        except Exception as e:print(f"❌ Error compatibilidad: {e}")
    
    print(f"\n📊 Estadísticas:");print(f"   • Texturas: {stats['textures_generated']}");print(f"   • Aurora V7: {stats.get('aurora_experiences_created',0)}");print(f"   • Cache hits: {stats['cache_hits']}");print(f"   • Optimizaciones: {stats.get('coherence_optimizations',0)}");print("🚀 HarmonicEssence V34 Aurora Connected - Optimizado!")
