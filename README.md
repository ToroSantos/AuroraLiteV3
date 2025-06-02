# 🌌 Aurora V7 Integral – Sistema Neuroacústico Modular

**Aurora V7** es un sistema avanzado de generación de audio neuroacústico terapéutico, diseñado para inducir estados mentales, emocionales y espirituales específicos mediante la síntesis dinámica de frecuencias, estructuras y texturas emocionales. Esta versión integra la gestión inteligente de objetivos y enrutamiento mediante el nuevo archivo `objective_manager.py`, optimizando así la experiencia auditiva y su personalización.

---

## 🧠 ¿Qué es Aurora?

Aurora es un sistema orquestado que crea pistas de audio personalizadas basadas en:
- Un **objetivo funcional** (por ejemplo: claridad, gratitud, enfoque, integración).
- Una **estructura emocional y progresiva**, dividida en fases.
- Motores que generan capas neurofuncionales (neuro wave), emocionales (pads), estéticas (ruido, efectos) y sincronización.

El sistema produce archivos `.wav` estéreos listos para reproducción con audífonos o altavoces, dependiendo del objetivo terapéutico.

---

## 🧩 Arquitectura Modular del Sistema

```
🌐 AURORA DIRECTOR (Orquestador principal)
├── aurora_director_v7.py            → Cerebro maestro que integra objetivos, fases y motores

⚙️ MOTORES PRINCIPALES
├── neuromix_aurora_v27.py           → Motor neuroacústico (neuro wave, AM/FM, neurotransmisores)
├── hypermod_v32.py                  → Motor estructural (fases, intensidad, duración)
├── harmonicEssence_v34.py           → Motor emocional-estético (pads, ruido, estilo, paneo)

🔎 ANÁLISIS Y CALIDAD
├── Carmine_Analyzer.py              → Análisis emocional y estructural
├── aurora_quality_pipeline.py       → Normalización, compresión y mastering
├── verify_structure.py              → Validación técnica (bloques, capas, transiciones)

🎯 OBJETIVOS Y ENRUTAMIENTO
├── objective_manager.py             → NUEVO: unifica plantillas de objetivos y enrutamiento inteligente

🎨 EMOCIÓN Y ESTILO
├── emotion_style_profiles.py        → Perfiles emocionales y de estilo auditivo
├── field_profiles.py                → Configuraciones de campo y ambientación
├── psychedelic_effects_tables.json  → Efectos psicodélicos y modulaciones
├── presets_fases.py                 → Configuración de fases y estructura narrativa

📐 CONTROL DE CAPAS
├── layer_scheduler.py               → Activación de capas por fase
├── sync_manager.py                  → Sincronización entre motores

🎵 UTILIDADES
├── harmony_generator.py             → Generación de pads armónicos

📚 DOCUMENTACIÓN
├── README.md                        → Este archivo
```

---

## 🔄 Lógica de Funcionamiento

1. **Input del usuario:** objetivo deseado, emoción, duración, estilo.
2. `aurora_director_v7.py` consulta `objective_manager.py` para enrutar y configurar el objetivo seleccionado, obteniendo la plantilla y los presets asociados.
3. `hypermod_v32.py` estructura las fases (ej.: preparación → intención → clímax → resolución).
4. `neuromix_aurora_v27.py` genera la capa funcional (neurotransmisores, AM/FM, pulsos).
5. `harmonicEssence_v34.py` agrega emoción y estética (pads, ruido texturizado, paneo).
6. `Carmine_Analyzer.py` evalúa la calidad emocional y técnica.
7. `aurora_quality_pipeline.py` finaliza el audio con mastering y protección auditiva.
8. Resultado final: archivo `.wav` profesional y funcional.

---

## ✅ ¿Qué genera Aurora?

- Audio estereofónico terapéutico o de exploración emocional
- Estructura narrativa por fases
- Modulación cerebral (binaural, isocrónica, AM/FM)
- Configuración personalizada según el objetivo elegido:
  - Claridad mental
  - Gratitud
  - Regulación emocional
  - Integración psicodélica
  - Sueño profundo
  - Estado de flujo
- Uso con audífonos recomendado para objetivos específicos.

---

## 📌 Mejoras y Actualizaciones Clave

- **objective_manager.py:** Nueva unificación de plantillas de objetivos y enrutamiento, reemplazando a `objective_router_v7.py` y `objective_templates.py`.
- Integración directa con `aurora_director_v7.py` para una experiencia más fluida.
- Uso de enums, dataclasses y análisis semántico inteligente para personalización avanzada.
- Modularidad total: permite integrar nuevos estilos, presets o capas en el futuro.

---

## 📌 Requisitos para ejecución local

- Python 3.10 o superior
- `numpy`, `wave`, `scipy` (para versiones extendidas)
- Ejecutar `aurora_director_v7.py` o integrarlo con GUI

---

## 🧭 Siguientes pasos

- Personaliza tus propios objetivos y estilos en `objective_manager.py`.
- Explora nuevos efectos psicodélicos desde `psychedelic_effects_tables.json`.
- Ajusta presets emocionales y de campo según tus necesidades terapéuticas.

---

**Aurora V7 – Generación auditiva para el bienestar emocional y mental.**  
**Desarrollado con propósito terapéutico y expansión consciente.**
