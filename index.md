---
layout: default
---

## Objetivo

_El objetivo del curso es introducir los fundamentos teóricos y los diferentes modelos y algoritmos del Aprendizaje Automático._ 
_El estudiante adquirirá conocimientos generales y prácticos del estado del arte en el tema, para su aplicación en la práctica particularmente en un contexto de Ciencias de Datos._


### Comunicación

* [Aula PEDCO](https://pedco.uncoma.edu.ar/course/view.php?id=9477)
* [Grupo TELEGRAM](https://t.me/+O4K7jtf60Qw5ODIx)

### Administrativo

* [Programa Aprendizaje Automático](/docs/Administrativa/02%20Aprendizaje%20Automatico.docx.pdf)
* Clase Administrativa [(pdf)](/docs/Administrativa/EIDA_Admin.pdf) [(tex)](/docs/Administrativa/EIDA_Admin.zip)

## Clases y Prácticos

* **Unidad I**: Introducción al Aprendizaje Automático
  * Teoría [(pdf)](/docs/Administrativa/EIDA_Admin.pdf) [(tex)](/docs/Administrativa/EIDA_Admin.zip)
  * Práctica [(pdf)](https://pedco.uncoma.edu.ar/course/view.php?id=9477) [(tex)](https://pedco.uncoma.edu.ar/course/view.php?id=9477) [(notebooks)](https://pedco.uncoma.edu.ar/course/view.php?id=9477)
* **Unidad II**:  Clasificación del Aprendizaje Automático y
algoritmos
  * Teoría [(pdf)](https://pedco.uncoma.edu.ar/course/view.php?id=9477) [(tex)](https://pedco.uncoma.edu.ar/course/view.php?id=9477)
  * Práctica [(pdf)](https://pedco.uncoma.edu.ar/course/view.php?id=9477) [(tex)](https://pedco.uncoma.edu.ar/course/view.php?id=9477) [(notebooks)](https://pedco.uncoma.edu.ar/course/view.php?id=9477)
* **Unidad III**: Preprocesamiento y generación de características
  * Teoría [(pdf)](https://pedco.uncoma.edu.ar/course/view.php?id=9477) [(tex)](https://pedco.uncoma.edu.ar/course/view.php?id=9477) 
  * Práctica [(pdf)](https://pedco.uncoma.edu.ar/course/view.php?id=9477) [(tex)](https://pedco.uncoma.edu.ar/course/view.php?id=9477) [(notebooks)](https://pedco.uncoma.edu.ar/course/view.php?id=9477)
* **Unidad IV**: Regresión
  * Teoría [(pdf)](https://pedco.uncoma.edu.ar/course/view.php?id=9477) [(tex)](https://pedco.uncoma.edu.ar/course/view.php?id=9477)
  * Práctica [(pdf)](https://pedco.uncoma.edu.ar/course/view.php?id=9477) [(tex)](https://pedco.uncoma.edu.ar/course/view.php?id=9477) [(notebooks)](https://pedco.uncoma.edu.ar/course/view.php?id=9477)
* **Unidad V**: Máquinas de Soporte Vectorial (SVM)
  * Teoría [(pdf)](https://pedco.uncoma.edu.ar/course/view.php?id=9477) [(tex)](https://pedco.uncoma.edu.ar/course/view.php?id=9477)
  * Práctica [(pdf)](https://pedco.uncoma.edu.ar/course/view.php?id=9477) [(tex)](https://pedco.uncoma.edu.ar/course/view.php?id=9477) [(notebooks)](https://pedco.uncoma.edu.ar/course/view.php?id=9477)
* **Unidad VI**: Redes Neuronales
  * Teoría [(pdf)](https://pedco.uncoma.edu.ar/course/view.php?id=9477) [(tex)](https://pedco.uncoma.edu.ar/course/view.php?id=9477)
  * Práctica [(pdf)](https://pedco.uncoma.edu.ar/course/view.php?id=9477) [(tex)](https://pedco.uncoma.edu.ar/course/view.php?id=9477) [(notebooks)](https://pedco.uncoma.edu.ar/course/view.php?id=9477)
* **Unidad VII**: Aprendizaje No supervisado
  * Teoría [(pdf)](https://pedco.uncoma.edu.ar/course/view.php?id=9477) [(tex)](https://pedco.uncoma.edu.ar/course/view.php?id=9477)
  * Práctica [(pdf)](https://pedco.uncoma.edu.ar/course/view.php?id=9477) [(tex)](https://pedco.uncoma.edu.ar/course/view.php?id=9477) [(notebooks)](https://pedco.uncoma.edu.ar/course/view.php?id=9477)


## Material

### Ambiente
[Entorno de Trabajo Jupyter](https://jupyter.fi.uncoma.edu.ar/hub/login?next=%2Fhub%2F) (usar credenciales provistas por TICs)

## Tu primer modelo

_(*) Adaptado de Andrew Ng course_

Para correr el ejemplo, usa el ambiente Jupyter o crea un Google Colab y copia
el pega el siguente código:

```python
# Mi primer modelo!

# Paso 1: Importar librerías
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Paso 2: Crear datos con ruido
np.random.seed(42)

# Superficies en m²
metros2 = np.array([50, 60, 70, 80, 90, 100, 110, 120, 130]).reshape(-1, 1)

# Precios base (aprox 1.5 mil USD por m²) + ruido aleatorio
precio_base = 1.5 * metros2.flatten()
ruido = np.random.normal(loc=0.0, scale=10.0, size=metros2.shape[0])
precio = precio_base + ruido

# Paso 3: Crear y entrenar el modelo
modelo = LinearRegression()
modelo.fit(metros2, precio)

# Paso 4: Mostrar coeficientes
print(f"Pendiente (coeficiente): {modelo.coef_[0]:.2f}")
print(f"Intersección (ordenada al origen): {modelo.intercept_:.2f}")
print(f"R² (score): {modelo.score(metros2, precio):.2f}")

# Paso 5: Visualización
plt.figure(figsize=(8, 6))
plt.scatter(metros2, precio, color='blue', label='Datos reales (con ruido)')
plt.plot(metros2, modelo.predict(metros2), color='red', label='Línea de regresión')
plt.xlabel("Superficie (m²)")
plt.ylabel("Precio (miles de USD)")
plt.title("Regresión lineal con datos ruidosos")
plt.legend()
plt.grid(True)
plt.show()

# Paso 6: Predicción de una casa nueva
nueva_casa = np.array([[105]])  # 105 m²
prediccion = modelo.predict(nueva_casa)
print(f"Predicción para una casa de 105 m²: {prediccion[0]:.2f} mil USD")
```
![Regresión](/assets/img/regression.png)