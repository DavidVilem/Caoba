# Medical Error Detection

Este documento presenta la descripción de la solución y la arquitectura para desarrollar una herramienta automatizada que emplea tecnologías avanzadas de procesamiento de lenguaje natural (PLN) y aprendizaje automático con el fin de detectar eficientemente y con alta precisión errores en las notas clínicas. La iniciativa de este proyecto es apoyar a los profesionales de la salud en la identificación rápida de posibles inexactitudes o errores en la documentación clínica, lo cual es fundamental para mejorar la calidad de la atención médica y garantizar la seguridad de los pacientes. La arquitectura del sistema se centra en la integración de modelos de procesamiento de lenguaje natural de vanguardia y técnicas de aprendizaje automático para analizar el texto de las notas clínicas, identificando patrones que puedan indicar la presencia de errores. Se presta especial atención a la precisión de los algoritmos utilizados, así como a su capacidad para integrarse con los sistemas de información clínica existentes sin interrumpir los flujos de trabajo habituales. Este enfoque no solo facilita un entorno de atención más seguro para los pacientes, sino que también contribuye a la eficiencia operativa al reducir el tiempo necesario para la revisión manual de documentos.

Además, la información documental relacionada con el proyecto se puede encontrar en los siguientes enlaces:

[Carpeta Asociada asociada a Protocolo de Informacion de Equipo](**Se debe habilitar enlace de OneDrive en donde tenga la informacion**)

[Informe Proyecto](https://docs.google.com/document/d/1JLWx9k1nKSNmiUi5aEe9pj2OXv_I_AML/edit?usp=drive_link&ouid=108978349794988793613&rtpof=true&sd=true)

[Manual de Usuario](**Se debe habilitar enlace en donde tenga la informacion**)

[Video Demo](**Se debe habilitar enlace en donde esta el demo establecido*)

[Plantilla Comunicaciones](https://docs.google.com/document/d/1JLWx9k1nKSNmiUi5aEe9pj2OXv_I_AML/edit?usp=drive_link)

## Tabla de Contenidos
* [Descripción de la solución](#descripción-de-la-solución)
* [Screenshots](#screenshots)
* [Requerimientos](#requerimientos)
* [Instalacion](#instalación)
* [Ejemplos de Codigo](#ejemplos-de-codigo)
* [Pruebas Automatizadas](#pruebas-automatizadas)
* [Autores](#autores)

## Descripción de la solución

El lenguaje natural es el pilar sobre el cual se construye nuestra capacidad para comunicarnos, compartir ideas y resolver problemas complejos. A lo largo de la historia, este lenguaje se ha refinado y adaptado, permitiéndonos no solo interactuar entre nosotros sino también con la tecnología. En el corazón de esta interacción se encuentra el Procesamiento de Lenguaje Natural (PLN), una disciplina de la Inteligencia Artificial (IA) que busca cerrar la brecha entre la comunicación humana y la comprensión computacional. A través del PLN, las máquinas pueden interpretar, analizar y responder a lenguajes humanos de una manera que antes se consideraba exclusiva del entendimiento humano.

En el ámbito clínico, la aplicación del PLN y, más específicamente, de los Modelos de Lenguaje de Gran Escala (LLM), ha inaugurado una era de transformaciones significativas. Estos modelos aprovechan el lenguaje natural para analizar extensos volúmenes de notas clínicas, identificando patrones, inconsistencias y errores potenciales con una precisión y eficiencia sin precedentes. Al hacerlo, los LLM no solo apoyan a los profesionales de la salud en la detección de anomalías que podrían pasar desapercibidas, sino que también facilitan un entorno más seguro y confiable para la atención al paciente.

La integración de los LLM en la medicina representa un avance notable en la forma en que se procesa y utiliza la información clínica. Al interpretar el lenguaje natural contenido en las notas clínicas, estos modelos ofrecen insights críticos que pueden mejorar la toma de decisiones médicas y optimizar los resultados para los pacientes. Esta capacidad de los LLM para trabajar con lenguaje natural abre nuevas posibilidades para el desarrollo de herramientas clínicas que no solo comprenden el texto, sino que también razonan y proporcionan soluciones basadas en un análisis detallado de los datos disponibles.

En este proyecto se busca comprender las complejidades y patrones lingüísticos presentes en las notas clínicas utilizadas dentro del entorno de atención médica, empleando algoritmos avanzados de procesamiento de lenguaje natural.

Aunque la tarea principal es Detectar si el texto incluye algún error médico, el objetivo de este proyecto es desarrollar una herramienta automatizada que, mediante el uso de tecnologías avanzadas de procesamiento de lenguaje natural y aprendizaje automático, sea capaz de detectar de manera eficiente y precisa la presencia de errores en las notas clínicas. Este sistema buscará apoyar a los profesionales de la salud en la identificación rápida de posibles inexactitudes en la documentación clínica, contribuyendo así a la mejora de la calidad de la atención médica y a la seguridad de los pacientes.

### Reto del cliente

El reto viene de parte de "MEDIQA-CORR @ NAACL-ClinicalNLP 2024" y tiene como tareas:

En esta tarea, buscamos abordar el problema de identificar y corregir (sentido común ) errores médicos en las notas clínicas. Desde una perspectiva humana, estos errores requieren experiencia y conocimientos médicos para identificarlos y corregirlos. 

Tasks

Participants will be given a snippet of clinical text and asked to:
- Detect whether the text includes a medical error. (Binary Classification)
- Identify the text span associated with the error, if a medical error exists. (Span Identification)
- Provide a free text correction, if a medical error exists. (Natural Language Generation)

### Solución Alianza CAOBA
**Nota:** Puede poner la solucion propuesta desde el equipo
### Impacto potencial esperado en el Negocio
**Nota:** Puede poner el impacto que espera el negocio

**Ejemplos: Como debe estar la descripción de la solución**
![](structure_example/docs/readme/ejemplo_descripcion_proyecto.png)

### Screenshots / Demo
**Nota:** Obligatorio: Debe poner una imagen, .gif o otros de la solucion entregada desplegada para el negocio
![screenshot](https://www.eclipsemediasolutions.com/sites/default/files/Audience-web-traffic-fluctuations1.jpg)

## Arquitectura logica de la solución

La arquitectura empleada para esta tarea combina Redes Neuronales Recurrentes (RNN) y Modelos BERT (Bidirectional Encoder Representations from Transformers), dos poderosas tecnologías de procesamiento de lenguaje natural que se complementan para analizar y comprender el texto de las notas clínicas.

Las Redes Neuronales Recurrentes (RNN) son una clase de redes neuronales que resultan especialmente eficaces para procesar secuencias de datos, como lo son las series temporales o las secuencias de texto. Su característica principal es la capacidad de mantener un estado (o memoria) que captura la información de los datos previos en la secuencia, lo que permite que la red haga uso de esta información para procesar los siguientes elementos de la secuencia. Esto las hace ideales para tareas donde el contexto o la dependencia temporal entre los datos es crucial para su comprensión.

Por otro lado, los Modelos BERT son una revolución en el campo del PLN, proporcionando una forma de preentrenar modelos de lenguaje en enormes corpus de texto de manera bidireccional. Esto significa que BERT no solo aprende de la secuencia de palabras que preceden a una palabra dada sino también de las que le siguen, lo cual resulta en una comprensión contextual mucho más rica del lenguaje. BERT ha demostrado ser excepcionalmente bueno en tareas que requieren comprensión del lenguaje, incluyendo la clasificación de texto, la extracción de entidades, y la comprensión de la intención detrás de las preguntas.

La combinación de RNN y Modelos BERT en esta arquitectura aprovecha lo mejor de ambos mundos: la habilidad de las RNN para manejar secuencias de datos y su dependencia temporal, junto con la poderosa capacidad de BERT para entender el contexto y la semántica del lenguaje a un nivel profundo. Al integrar estas tecnologías, el sistema es capaz de analizar de manera eficiente y precisa las notas clínicas, identificando errores y extrayendo insights valiosos que pueden mejorar la atención al paciente y los procesos clínicos. La arquitectura está diseñada para ser robusta, escalable y capaz de manejar el volumen y la complejidad de los datos clínicos, asegurando que los profesionales de la salud reciban apoyo de la más alta calidad en la revisión y análisis de la documentación clínica.

**Ejemplo: Se espera un diagrama como la siguiente figura:** 
![](structure_example/docs/readme/docs_Arquitectura.png)



## Estructura del proyecto

**Ejemplo: Forma de poner estructura en Markdown**

-- Se recomiendo uso del comanddo y su uso dependera de su Sistema Operativo:
[tree](https://www.geeksforgeeks.org/tree-command-unixlinux/) 

```
.
├── README.md
├── data/
│   ├── raw/
│   │   ├── data.txt
│   │   └── data_ignorar.txt
│   └── stage/
│       └── todo.txt
│   ├── analytics/
│   │   └── todo.txt
├── datalab/
│   └── todo.txt
├── src/
│   └── todo.txt
├── conf/
│   └── todo.txt
├── docs/
│   └── readme/
│       ├── docs_Arquitectura.png
│       ├── docs_modeloDatosPowerBI.png
│       ├── ejemplo_descripcion_proyecto.png
│       └── ejemplo_estructura_proyecto.png
├── dashboard/
│   └── todo.txt
├── deploy/
│   └── todo.txt
└── temp/
    ├── 00Standard_Ref_to_make_Experiments.ipynb
    └── root.py
```

**Ejemplo:Se espera un diagrama como la siguiente figura: ** 
![](structure_example/docs/readme/ejemplo_estructura_proyecto.png)



## Proceso de ejecucion y despliegue

## Requerimientos
**Nota:** Obligatorio: Minimo debe escribir los requerimientos por cada lenguaje de programacion usado tanto en el back-end (Ej: Python, R) como en el front-end, si aplica. Tambien, es importante que ponga las versiones correspondientes 
### Librerias Empleadas 

•	Pandas: Para la manipulación y análisis de datos.
•	NumPy: Biblioteca de soporte para matrices y operaciones matemáticas de alto nivel.
•	re: Módulo de expresiones regulares para procesamiento de texto.
•	Optuna: Framework de optimización de hiperparámetros para mejorar el rendimiento de los modelos.
•	Random: Generación de números aleatorios para la creación de conjuntos de datos y validación cruzada.
•	Sklearn (scikit-learn): Para realizar tareas de aprendizaje automático como la división de datos y evaluación de modelos mediante KFold y métricas de precisión, recall, fscore y accuracy.
•	Datasets: De la biblioteca Hugging Face, para facilitar la carga y manipulación de conjuntos de datos en formato adecuado para modelos de PLN.
•	Transformers: También de Hugging Face, proporciona modelos preentrenados para clasificación de secuencias, tokenización y entrenamiento de modelos.
•	RobertaTokenizer y RobertaForSequenceClassification: Tokenizador y modelo para clasificación de secuencias basados en RoBERTa.
•	TrainingArguments y Trainer: Para configurar y ejecutar el entrenamiento de modelos de transformers.
•	TrainerCallback: Para personalizar callbacks durante el entrenamiento.

### Requerimientos Hardware
Para llenar
### Requerimientos Software
Para llenar

## Instalación: 
**Nota:** Obligatorio: Minimo debe haber en el proyecto el archivo que permita instalar el ambiente necesario para el despliegue de la solución y los comandos ejecutados para la instalacion. Por ejemplo, si es Python un requeriments.txt o un archivo de DESCRIPTION en R. 

## Configuracion
**Nota:** Para llenar

## Ejemplos de Codigo
**Nota:** Para llenar

## Errores conocidos
**Nota:** Para llenar

## Pruebas Automatizadas
**Nota:** Si aplica puede poner como correr las pruebas

## Imagenes
**Nota:** Si aplica puede poner cuales fueron las imagenes usadas (Ejemplo: Docker)

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
![Road_Map](src/Road.png)


## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.

## Autores
**Nota:** Obligatorio: Minimo debe llenar los autores tanto de analitica como del negocio,su organizacion, su nombre con el nombre del papel que tomo en el equipo, su respectivo correo electronico

| Organización   | Nombre del Miembro | Correo electronico | 
|----------|-------------|-------------|
| PUJ-Bogota |  Persona 1: Cientific@ de Datos | ejemplo@XXXX |
| Organizacion  |  Persona 2:Lider del negocio  | ejemplo@XXXX |
